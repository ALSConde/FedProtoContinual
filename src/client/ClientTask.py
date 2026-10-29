from typing import Callable, Mapping, Optional, Sequence, Union
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn.functional as F
from src.model.Models import FCLModel
from src.model.blocks.Adapter import Adapter
from src.model.layers.PrototypeMemory import PrototypeMemory
from src.utils.losses.Losses import (
    distillation_loss,
    distillation_loss_kl,
    local_class_prototypes,
    prototype_alignment_loss,
    split_by_know,
)


def _device_batches(
    loader: DataLoader, device: torch.device, max_classes: Optional[int] = None
):
    """Same batches, in the same order, as iterating `loader`, but moved to `device`
    with ONE host->device copy per tensor instead of one per batch (every pageable
    copy synchronises the stream, which is very slow when the GPU is shared)."""
    xs, ys, sizes = [], [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        sizes.append(len(y))
    if not xs:
        return []
    X, Y = torch.cat(xs), torch.cat(ys)  # still on CPU: this check costs no GPU sync
    if max_classes is not None and int(Y.max()) >= max_classes:
        raise ValueError(
            f"label {int(Y.max())} >= {max_classes} classes: expand the classifier and "
            "the prototype memory before training"
        )
    X, Y = X.to(device), Y.to(device)
    out, i = [], 0
    for n in sizes:
        out.append((X[i : i + n], Y[i : i + n]))
        i += n
    return out


def _build_ce_index(
    local_class_ids: Optional[Sequence[int]], num_classes: int, device: torch.device
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Index tensors that restrict the cross-entropy to the classes present in the
    client's local data (flag ce-local-classes).

    Returns (ids, remap): `ids` are the kept logit columns, `remap` maps a global class
    id to its position among them (-1 for classes that are not kept). None means the CE
    is computed over every class, as before. Inference is unaffected: the classifier
    still scores every seen prototype, since no task identity exists at test time.
    """
    if local_class_ids is None:
        return None
    ids = sorted({int(c) for c in local_class_ids})
    if not ids:
        return None
    if ids[-1] >= num_classes:
        raise ValueError(
            f"local class id {ids[-1]} >= classifier size {num_classes}: expand the "
            "classifier before building the CE class index"
        )
    ids_t = torch.tensor(ids, dtype=torch.long, device=device)
    remap = torch.full((num_classes,), -1, dtype=torch.long, device=device)
    remap[ids_t] = torch.arange(len(ids), dtype=torch.long, device=device)
    return ids_t, remap


def _class_ce(
    logits: torch.Tensor,
    y: torch.Tensor,
    ce_index: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if ce_index is None:
        return F.cross_entropy(logits, y)
    ids, remap = ce_index
    return F.cross_entropy(logits.index_select(1, ids), remap[y])


@torch.no_grad()
def provisional_prototypes(
    model: FCLModel,
    loader: DataLoader,
    device: torch.device,
    known_consolidated: set,
) -> list[int]:
    """Pre-training pass (flag proto-prepass).

    Classes that have no server prototype yet (first round of a class) sit at zero in the
    classifier, so their logits are constant and the CE gives no usable signal. This
    pass embeds the local data with the model received for the round (eval mode, no
    gradient) and writes the per-class mean of the shared embedding, the same quantity
    the server aggregates, into the LOCAL classifier only. Nothing is sent to the
    server, the classes are not added to `known_consolidated` (so L_proto still treats
    them as new), and the values are overwritten when the server's prototypes arrive.
    Returns the class ids that received a provisional prototype.
    """
    model.to(device)
    num_classes = model.classifier.num_classes
    if num_classes == 0:
        return []
    was_training = model.training
    model.eval()
    try:
        sums = torch.zeros(num_classes, model.classifier.embedding_dim, device=device)
        counts = torch.zeros(num_classes, device=device)
        for x, y in _device_batches(loader, device, max_classes=num_classes):
            _, h_shared = model.embed_both(x)
            sums.index_add_(0, y, h_shared.to(sums.dtype))
            counts.index_add_(0, y, torch.ones(len(y), device=device))
        ids = [
            c
            for c in torch.nonzero(counts > 0).flatten().tolist()
            if c not in known_consolidated
        ]
        if ids:
            idx = torch.tensor(ids, dtype=torch.long, device=device)
            protos = model.classifier.prototypes
            protos[idx] = (sums[idx] / counts[idx].unsqueeze(1)).to(protos.dtype)
        return ids
    finally:
        model.train(was_training)


@torch.no_grad()
def collect_prototype_stats(
    model: FCLModel,
    loader: DataLoader,
    memory: PrototypeMemory,
    device: torch.device,
) -> None:
    """Post-training pass (flag proto-postpass).

    Replaces the statistics accumulated during training (a mix of embeddings from a model
    that kept changing, in train mode) with statistics of the trained model in eval mode,
    which is the model the client actually returns. The sums stay un-normalised and the
    counts exact, so the server-side aggregation is unchanged.
    """
    model.to(device)
    was_training = model.training
    model.eval()
    try:
        memory.expand(model.classifier.num_classes)
        memory.reset()
        for x, y in _device_batches(loader, device, max_classes=memory.num_classes):
            _, h_shared = model.embed_both(x)
            memory.update(h_shared, y, check_bounds=False)
    finally:
        model.train(was_training)


_LR_SCALE_KEYS = ("extractor", "adapter_global", "local")


def _build_param_groups(
    model: FCLModel, lr: float, lr_scales: Optional[Mapping[str, float]] = None
) -> dict[str, tuple[list, float]]:
    """Every model parameter in exactly one named group, with its learning rate.

    extractor       feature_extractor
    adapter_global  adapter_global + incorporated adapters + shadow adapters (all shared)
    local           adapter_local + alpha_gate + classifier.scale (never leave the client)

    lr_scales multiplies the round lr per group (default 1.0 everywhere, which is
    numerically identical to a single Adam over model.parameters()).
    """
    scales = {k: 1.0 for k in _LR_SCALE_KEYS}
    for key, value in (lr_scales or {}).items():
        if key not in scales:
            raise ValueError(
                f"Unknown lr scale '{key}'. Expected one of {_LR_SCALE_KEYS}."
            )
        scales[key] = float(value)
    for key, value in scales.items():
        if value <= 0.0:
            raise ValueError(f"lr scale '{key}' must be > 0 (got {value}).")

    extractor = list(model.feature_extractor.parameters())
    shared_adapters = [
        p
        for module in (
            model.adapter_global,
            model.incorporated_adapters,
            model.shadow_adapters,
        )
        for p in module.parameters()
    ]
    local = [
        p
        for module in (model.adapter_local, model.alpha_gate)
        for p in module.parameters()
    ] + [model.classifier.scale]

    covered = {id(p) for p in extractor + shared_adapters + local}
    missing = [n for n, p in model.named_parameters() if id(p) not in covered]
    if missing:
        raise RuntimeError(
            f"Parameters not assigned to any optimizer group: {missing}. Update "
            "_build_param_groups when adding modules to FCLModel."
        )
    return {
        "extractor": (extractor, lr * scales["extractor"]),
        "adapter_global": (shared_adapters, lr * scales["adapter_global"]),
        "local": (local, lr * scales["local"]),
    }


def _build_optimizers(
    model: FCLModel,
    lr: float,
    lr_scales: Optional[Mapping[str, float]],
    split: bool,
) -> tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:
    """split=False: one Adam over all groups. split=True: (shared optimizer, local
    optimizer) over disjoint parameter sets."""
    groups = _build_param_groups(model, lr, lr_scales)

    def _pg(names):
        return [
            {"params": groups[n][0], "lr": groups[n][1]} for n in names if groups[n][0]
        ]

    if not split:
        return torch.optim.Adam(_pg(("extractor", "adapter_global", "local"))), None
    return (
        torch.optim.Adam(_pg(("extractor", "adapter_global"))),
        torch.optim.Adam(_pg(("local",))),
    )


def _logits_frozen_scale(classifier, h: torch.Tensor) -> torch.Tensor:
    """Classifier logits with the learnable scale treated as a constant, so the shared
    objective never updates a parameter that belongs to the local optimizer."""
    return (
        F.linear(F.normalize(h, dim=1), F.normalize(classifier.prototypes, dim=1))
        * classifier.scale.detach()
    )


def _kd_term(
    model: FCLModel,
    h: torch.Tensor,
    h_teacher: torch.Tensor,
    known_sorted: list,
    kd_mode: str,
    kd_temperature: float,
    kd_scale_mode: str = "const",
    kd_logit_scale: float = 1.0,
) -> Optional[torch.Tensor]:
    """One KD term. For kd_mode="kl" the logit scale of the similarity distributions is
    set by kd_scale_mode:
      const   teacher and student both use kd_logit_scale (default; with 1.0 this is the
              previous behaviour, with ~20 the distributions match the classifier's);
      local   both use the classifier's own learned scale, detached (strength then varies
              per client and over the rounds, because the scale is local and trained);
      hybrid  teacher: kd_logit_scale, student: local scale (kept for comparison only: it
              adds a sharpness-matching term whenever the local scale drifts away from
              kd_logit_scale).
    """
    if kd_mode == "embedding_mse":
        return distillation_loss(h, h_teacher)
    reference = (
        model.classifier.prototypes[known_sorted] if len(known_sorted) >= 2 else None
    )
    if kd_scale_mode == "const":
        s_student = s_teacher = kd_logit_scale
    elif kd_scale_mode == "local":
        s_student = s_teacher = model.classifier.scale.detach()
    elif kd_scale_mode == "hybrid":
        s_student, s_teacher = model.classifier.scale.detach(), kd_logit_scale
    else:
        raise ValueError(f"Unknown kd_scale_mode '{kd_scale_mode}'.")
    return distillation_loss_kl(
        h,
        h_teacher,
        reference,
        temperature=kd_temperature,
        scale_student=s_student,
        scale_teacher=s_teacher,
    )


def _split_train_step(
    model: FCLModel,
    x: torch.Tensor,
    y: torch.Tensor,
    memory: PrototypeMemory,
    opt_g: torch.optim.Optimizer,
    opt_l: torch.optim.Optimizer,
    *,
    ce_index,
    known_consolidated: set,
    known_sorted: list,
    lambda_proto_shared: float,
    lambda_proto_local: float,
    lambda_kd_local: float,
    lambda_kd_shared: float,
    embed_global: Optional[Callable],
    kd_mode: str,
    kd_temperature: float,
    proximal_mu: float,
    prox_params: list,
    prox_ref: list,
    lambda_mofe: float,
    mofe_margin: float,
    kd_scale_mode: str = "const",
    kd_logit_scale: float = 1.0,
) -> Optional[torch.Tensor]:
    """One optimisation step with two disjoint objectives.

        L_g = CE(h_g) + lambda_proto_shared * L_proto(h_g) [+ lambda_kd_shared * KD]
              [+ mu/2 * ||w - w_round_start||^2]            -> opt_g (shared params)
        L_l = CE(h_l) + lambda_proto_local  * L_proto(h_l) [+ lambda_kd_local  * KD]
                                                            -> opt_l (local params)

    h_l is built from detached shared features (FCLModel.embed_split), so the two
    backward passes touch disjoint parameters. With a shadow adapter on probation the
    paired CE / MoFe hinge acts on the shared side (the shadow adapter is a shared module).
    Returns the (detached) sum of both losses, or None on the cold-start skip.
    """
    has_shadow = len(model.shadow_adapters) > 0
    h_g, h_g_with, h_l = model.embed_split(x, with_shadow=has_shadow)
    memory.update(h_g, y, check_bounds=False)  # h_g is the deployed shared embedding
    if model.classifier.num_classes == 0:
        return None  # cold start: only statistics

    protos = model.classifier.prototypes
    opt_g.zero_grad()
    opt_l.zero_grad()

    # ---- shared objective -------------------------------------------------------
    ce_g = _class_ce(_logits_frozen_scale(model.classifier, h_g), y, ce_index)
    lp_g = prototype_alignment_loss(h_g, y, protos, known_consolidated)
    if has_shadow:
        ce_with = _class_ce(
            _logits_frozen_scale(model.classifier, h_g_with), y, ce_index
        )
        loss_g = 0.5 * (ce_g + ce_with)
        if lambda_mofe > 0.0:
            loss_g = loss_g + lambda_mofe * F.relu(
                ce_with - ce_g.detach() + mofe_margin
            )
        lp_with = prototype_alignment_loss(h_g_with, y, protos, known_consolidated)
        if lp_g is not None and lp_with is not None:
            loss_g = loss_g + lambda_proto_shared * 0.5 * (lp_g + lp_with)
        elif lp_g is not None:
            loss_g = loss_g + lambda_proto_shared * lp_g
    else:
        loss_g = ce_g
        if lp_g is not None:
            loss_g = loss_g + lambda_proto_shared * lp_g

    # ---- local objective ---------------------------------------------------------
    ce_l = _class_ce(model.classifier(h_l), y, ce_index)
    loss_l = ce_l
    lp_l = prototype_alignment_loss(h_l, y, protos, known_consolidated)
    if lp_l is not None:
        loss_l = loss_l + lambda_proto_local * lp_l

    # ---- knowledge distillation (teacher: frozen shared branch of the step) -----
    if embed_global is not None and (lambda_kd_local > 0.0 or lambda_kd_shared > 0.0):
        h_teacher = embed_global(x)
        if lambda_kd_shared > 0.0:
            kd_g = _kd_term(
                model,
                h_g,
                h_teacher,
                known_sorted,
                kd_mode,
                kd_temperature,
                kd_scale_mode,
                kd_logit_scale,
            )
            if kd_g is not None:
                loss_g = loss_g + lambda_kd_shared * kd_g
        if lambda_kd_local > 0.0:
            kd_l = _kd_term(
                model,
                h_l,
                h_teacher,
                known_sorted,
                kd_mode,
                kd_temperature,
                kd_scale_mode,
                kd_logit_scale,
            )
            if kd_l is not None:
                loss_l = loss_l + lambda_kd_local * kd_l

    if proximal_mu > 0.0:
        prox = sum((p - r).pow(2).sum() for p, r in zip(prox_params, prox_ref))
        loss_g = loss_g + 0.5 * proximal_mu * prox

    loss_g.backward()
    loss_l.backward()
    opt_g.step()
    opt_l.step()
    return (loss_g + loss_l).detach()


# Function to load data -- With sintetic data for testing purposes
def load_data(
    partition_id: int,
    num_partitions: int,
    input_dim: int,
    num_classes_total: int,
    batch_size: int,
    n_samples: int = 200,
):
    torch.manual_seed(partition_id)
    classes_per_client = max(2, num_classes_total // num_partitions)
    start = (partition_id * classes_per_client) % num_classes_total
    client_classes = [
        (start + i) % num_classes_total for i in range(classes_per_client)
    ]

    y = torch.tensor(
        [client_classes[i % len(client_classes)] for i in range(n_samples)]
    )
    class_centers = (
        torch.randn(
            num_classes_total, input_dim, generator=torch.Generator().manual_seed(0)
        )
        * 3.0
    )
    X = torch.randn(n_samples, input_dim) + class_centers[y]

    dataset = TensorDataset(X, y)
    n_train = int(0.8 * n_samples)
    train_ds, val_ds = random_split(dataset, [n_train, n_samples - n_train])
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size)
    return trainloader, valloader, client_classes


def train_fn(
    model: FCLModel,
    trainloader: DataLoader,
    memory: PrototypeMemory,
    epochs: int,
    lr: float,
    device: torch.device,
    known_consolidated: Optional[set] = None,
    lambda_proto: float = 1.0,
    lambda_kd: float = 0.5,
    kd_mode: str = "kl",
    kd_temperature: float = 2.0,
    frozen_embed_fn: Optional[Callable] = None,
    proximal_mu: float = 0.0,
    lambda_mofe: float = 0.5,
    mofe_margin: float = 0.0,
    local_class_ids: Optional[Sequence[int]] = None,
    proto_prepass: bool = False,
    lr_scales: Optional[Mapping[str, float]] = None,
    split_optimizers: bool = False,
    lambda_proto_shared: Optional[float] = None,
    lambda_proto_local: Optional[float] = None,
    lambda_kd_shared: float = 0.0,
    kd_scale_mode: str = "const",
    kd_logit_scale: float = 1.0,
) -> float:
    """`local_class_ids`: when given, the CE is restricted to those classes (flag
    ce-local-classes). `proto_prepass`: build provisional local prototypes for classes
    without a server prototype before the first step (flag proto-prepass).

    `lr_scales`: per-group multipliers of `lr` (keys: extractor, adapter_global, local).
    `split_optimizers`: two optimizers/objectives (see _split_train_step); it needs the
    local adapter and is otherwise ignored. In that mode `lambda_kd` is the weight of the
    KD term on the LOCAL embedding and `lambda_kd_shared` the one on the SHARED embedding;
    `lambda_proto_shared/local` default to `lambda_proto`.
    `kd_scale_mode` / `kd_logit_scale`: logit scale of the KL distillation (see _kd_term).
    """
    if kd_mode not in ("kl", "embedding_mse"):
        raise ValueError(
            f"Invalid kd_mode: {kd_mode}. Must be 'kl' or 'embedding_mse'."
        )

    model.to(device)
    model.train()
    known_consolidated = known_consolidated or set()
    known_sorted = sorted(known_consolidated)

    split_active = bool(split_optimizers and model.use_local_adapter)
    use_kd = lambda_kd > 0.0 or (split_active and lambda_kd_shared > 0.0)
    embed_global = None
    if use_kd:
        embed_global = (
            frozen_embed_fn
            if frozen_embed_fn is not None
            else model.frozen_global_embed_fn()
        )

    prox_params, prox_ref = [], []
    if proximal_mu > 0.0:
        prox_params = model.global_branch_parameters()
        prox_ref = [p.detach().clone() for p in prox_params]

    optmizer, opt_local = _build_optimizers(model, lr, lr_scales, split_active)
    lp_shared = lambda_proto if lambda_proto_shared is None else lambda_proto_shared
    lp_local = lambda_proto if lambda_proto_local is None else lambda_proto_local
    running_loss, n_batches = (
        0.0,
        0,
    )  # running_loss becomes a device tensor (no per-step sync)
    memory.expand(model.classifier.num_classes)
    ce_index = _build_ce_index(local_class_ids, model.classifier.num_classes, device)
    if proto_prepass:
        provisional_prototypes(model, trainloader, device, known_consolidated)

    for _ in range(epochs):
        for x, y in _device_batches(
            trainloader, device, max_classes=memory.num_classes
        ):
            if split_active:
                step_loss = _split_train_step(
                    model,
                    x,
                    y,
                    memory,
                    optmizer,
                    opt_local,
                    ce_index=ce_index,
                    known_consolidated=known_consolidated,
                    known_sorted=known_sorted,
                    lambda_proto_shared=lp_shared,
                    lambda_proto_local=lp_local,
                    lambda_kd_local=lambda_kd,
                    lambda_kd_shared=lambda_kd_shared,
                    embed_global=embed_global,
                    kd_mode=kd_mode,
                    kd_temperature=kd_temperature,
                    proximal_mu=proximal_mu,
                    prox_params=prox_params,
                    prox_ref=prox_ref,
                    lambda_mofe=lambda_mofe,
                    mofe_margin=mofe_margin,
                    kd_scale_mode=kd_scale_mode,
                    kd_logit_scale=kd_logit_scale,
                )
                if step_loss is not None:
                    running_loss = running_loss + step_loss
                    n_batches += 1
                continue

            has_shadow = len(model.shadow_adapters) > 0
            if has_shadow:
                # Candidate on probation: both views of the same batch. Prototypes are
                # accumulated from the deployed (shadow-free) shared embedding.
                h, h_shared, h_with = model.embed_pair(x)
            else:
                h, h_shared = model.embed_both(x)
                h_with = None
            memory.update(h_shared, y, check_bounds=False)

            if model.classifier.num_classes == 0:
                continue  # cold start: without prototypes yet, just accumulate statistics

            optmizer.zero_grad()

            logits = model.classifier(h)
            ce = _class_ce(logits, y, ce_index)

            l_proto = prototype_alignment_loss(
                h, y, model.classifier.prototypes, known_consolidated
            )
            if has_shadow:
                # Both views train (nothing is frozen). Dropout-like pairing keeps the
                # model good without the candidate, so the later with/without
                # comparison is not biased by co-adaptation. The MoFe hinge only
                # pushes the 'with' side (the 'without' loss is detached).
                ce_with = _class_ce(model.classifier(h_with), y, ce_index)
                loss = 0.5 * (ce + ce_with)
                if lambda_mofe > 0.0:
                    loss = loss + lambda_mofe * F.relu(
                        ce_with - ce.detach() + mofe_margin
                    )
                l_proto_with = prototype_alignment_loss(
                    h_with, y, model.classifier.prototypes, known_consolidated
                )
                if l_proto is not None and l_proto_with is not None:
                    loss = loss + lambda_proto * 0.5 * (l_proto + l_proto_with)
                elif l_proto is not None:
                    loss = loss + lambda_proto * l_proto
            else:
                loss = ce
                if l_proto is not None:
                    loss += lambda_proto * l_proto

            if use_kd:
                h_global = embed_global(x)
                l_kd = _kd_term(
                    model,
                    h,
                    h_global,
                    known_sorted,
                    kd_mode,
                    kd_temperature,
                    kd_scale_mode,
                    kd_logit_scale,
                )
                if l_kd is not None:
                    loss += lambda_kd * l_kd

            if proximal_mu > 0.0:
                prox = sum((p - r).pow(2).sum() for p, r in zip(prox_params, prox_ref))
                loss += 0.5 * proximal_mu * prox

            loss.backward()
            optmizer.step()

            running_loss = running_loss + loss.detach()
            n_batches += 1

    return float(running_loss) / max(n_batches, 1)


def embed_with_extra_incorporated(
    model: FCLModel, x: torch.Tensor, extra_adapter: Optional[Adapter] = None
):
    feats = model.feature_extractor(x)
    x_global = model.adapter_global(feats)
    incorporated = model.incorporated_delta(x_global)
    if extra_adapter is not None:
        incorporated += extra_adapter.forward_delta(x_global)
    delta_local = model.adapter_local.forward_delta(x_global)
    x_local = model.alpha_gate(x_global, delta_local)
    x_final = x_local + incorporated
    x_shared = x_global + incorporated
    return x_final, x_shared


def evaluate_with_candidate(
    model: FCLModel,
    candidate: Optional[Adapter],
    loader: DataLoader,
    device: torch.device,
):
    model.to(device)
    model.eval()
    correct, total, loss_sum, n_batches = 0, 0, 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if model.classifier.num_classes == 0:
                continue
            h, _ = embed_with_extra_incorporated(model, x, candidate)
            logits = model.classifier(h)
            loss_sum += F.cross_entropy(logits, y).detach()
            correct += (logits.argmax(dim=1) == y).sum()
            total += y.size(0)
            n_batches += 1

    if total == 0:
        return 0.0, 0.0
    return float(loss_sum) / max(n_batches, 1), float(correct) / total


def short_local_adaptation(
    model: FCLModel,
    candidate: Adapter,
    adapt_loader: DataLoader,
    lr: float,
    device: torch.device,
    max_steps: int,
    ce_class_ids: Optional[Sequence[int]] = None,
):
    """Adapt adapter_local, alpha_gate and the candidate (a private copy of it, which
    is discarded after the vote) for exactly `max_steps` optimizer steps (mini-batches).
    The loader is cycled when it has fewer than `max_steps` batches."""
    model.to(device)
    model.train()
    candidate.to(device)
    candidate.train()

    trainable_params = (
        list(model.adapter_local.parameters())
        + list(model.alpha_gate.parameters())
        + list(candidate.parameters())
    )
    frozen_modules = [
        model.feature_extractor,
        model.adapter_global,
        model.incorporated_adapters,
        model.classifier,
    ]
    saved_requires_grad = []
    for module in frozen_modules:
        for p in module.parameters():
            saved_requires_grad.append((p, p.requires_grad))
            p.requires_grad_(False)
    for p in candidate.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    ce_index = _build_ce_index(ce_class_ids, model.classifier.num_classes, device)
    steps_done = 0
    try:
        if model.classifier.num_classes != 0 and len(adapt_loader) > 0:
            while steps_done < max_steps:
                for x, y in adapt_loader:
                    if steps_done >= max_steps:
                        break
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    h, _ = embed_with_extra_incorporated(model, x, candidate)
                    logits = model.classifier(h)
                    loss = _class_ce(logits, y, ce_index)
                    loss.backward()
                    optimizer.step()
                    steps_done += 1
    finally:
        for p, requires_grad in saved_requires_grad:
            p.requires_grad_(requires_grad)
        candidate.eval()

    return steps_done


def vote_on_candidate(
    model: FCLModel,
    candidate: Adapter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    adapt_steps: int,
    vote_margin: float,
    ce_class_ids: Optional[Sequence[int]] = None,
) -> tuple[float, float, float]:
    _, acc_before = evaluate_with_candidate(model, None, val_loader, device)
    short_local_adaptation(
        model,
        candidate,
        train_loader,
        lr=lr,
        device=device,
        max_steps=adapt_steps,
        ce_class_ids=ce_class_ids,
    )
    _, acc_after = evaluate_with_candidate(model, candidate, val_loader, device)

    vote = 1.0 if (acc_after - acc_before) > vote_margin else 0.0
    return vote, acc_before, acc_after


@torch.no_grad()
def per_sample_ce_by_config(
    model: FCLModel,
    loader: DataLoader,
    device: torch.device,
    candidate: Optional[Adapter],
    configs: dict,
) -> dict:
    """Per-sample cross-entropy of the shared embedding (x_global + incorporated
    deltas, no local branch) under several incorporated-adapter configurations,
    all computed on the very same batches (so differences are paired).

    configs: name -> (set of incorporated slots to drop, add_candidate)
    Returns name -> 1-D CPU tensor with one CE value per validation sample.
    """
    model.to(device)
    model.eval()
    if candidate is not None:
        candidate.to(device)
        candidate.eval()
    out = {name: [] for name in configs}
    if model.classifier.num_classes == 0:
        return {name: torch.empty(0) for name in configs}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_global = model.adapter_global(model.feature_extractor(x))
        deltas = [a.forward_delta(x_global) for a in model.incorporated_adapters]
        cand = candidate.forward_delta(x_global) if candidate is not None else None
        for name, (drop, add_candidate) in configs.items():
            h = x_global
            for i, d in enumerate(deltas):
                if i not in drop:
                    h = h + d
            if add_candidate and cand is not None:
                h = h + cand
            logits = model.classifier(h)
            out[name].append(F.cross_entropy(logits, y, reduction="none").cpu())
    return {name: (torch.cat(v) if v else torch.empty(0)) for name, v in out.items()}


def build_mofe_payload(
    model: FCLModel,
    loader: DataLoader,
    device: torch.device,
    partition_id: int,
    kind: str,
    candidate: Optional[Adapter] = None,
) -> dict:
    """Sufficient statistics for the server-side MoFe decisions with no raw data.
    kind="vote": configurations 'plus' (current + candidate) and 'rep_j'
                 (adapter j replaced by the candidate), against 'minus' (current).
    kind="loo" : configurations 'drop_j' (adapter j removed), against 'minus'.
    For every configuration the payload carries the sum and sum of squares of the
    paired per-sample gain  d = CE(minus) - CE(config)  (positive = config better).
    """
    k = len(model.incorporated_adapters)
    configs: dict = {"minus": (frozenset(), False)}
    if kind == "vote":
        configs["plus"] = (frozenset(), True)
        for j in range(k):
            configs[f"rep_{j}"] = (frozenset({j}), True)
    elif kind == "loo":
        for j in range(k):
            configs[f"drop_{j}"] = (frozenset({j}), False)
    else:
        raise ValueError(f"Unknown MoFe payload kind '{kind}'.")

    ce = per_sample_ce_by_config(model, loader, device, candidate, configs)
    base = ce["minus"]
    stats = {}
    for name, values in ce.items():
        d = base - values
        stats[name] = {
            "d_sum": float(d.sum()),
            "d_sq": float((d * d).sum()),
            "ce_mean": float(values.mean()) if values.numel() else 0.0,
        }
    return {
        "kind": kind,
        "partition_id": int(partition_id),
        "k": k,
        "n": int(base.numel()),
        "stats": stats,
    }


def test_fn(
    model: FCLModel,
    valloader: DataLoader,
    device: torch.device,
    branch: str = "local",
):
    """Accuracy/loss on a client loader.
    branch="local": personalized embedding (adapter_local + alpha gate).
    branch="global": shared embedding (global adapter + incorporated adapters),
    i.e. what the server-side model sees; the difference between the two is the
    per-client personalization gain.
    """
    if branch not in ("local", "global"):
        raise ValueError(f"Invalid branch '{branch}'. Must be 'local' or 'global'.")
    model.to(device)
    model.eval()
    correct, total, loss_sum, n_batches = 0, 0, 0.0, 0

    with torch.no_grad():
        for x, y in valloader:
            x, y = x.to(device), y.to(device)
            h = model.embed(x) if branch == "local" else model.embed_both(x)[1]

            if model.classifier.num_classes == 0:
                continue

            logits = model.classifier(h)
            loss_sum += F.cross_entropy(logits, y).detach()
            correct += (logits.argmax(dim=1) == y).sum()
            total += y.size(0)
            n_batches += 1

    if total == 0:
        return 0.0, 0.0
    return float(loss_sum) / max(n_batches, 1), float(correct) / total


def compute_local_contribution_ratio(
    model: FCLModel, loader: DataLoader, device: torch.device
) -> Optional[float]:
    model.eval()
    total_ratio, total_n = 0.0, 0
    with torch.no_grad():
        for x, y in _device_batches(loader, device):
            ratios = model.local_contribution_ratio(x)
            total_ratio += ratios.sum()
            total_n += ratios.numel()
    if total_n == 0:
        return None
    return float(total_ratio) / total_n


def compute_interference(
    model: FCLModel,
    loader: DataLoader,
    known_consolidated: set,
    device: torch.device,
) -> Optional[float]:
    """Crowding of the classes present in the local data against the old global prototypes.

    For every local class c: p_c is its local prototype (mean of the normalised shared
    embeddings, what the pre-pass builds) while the server has no prototype for it yet, and
    the global prototype afterwards. "Old" classes are the known ones that are NOT in the
    local data. Returns the mean over local classes of max_old cos(p_c, p_old): close to 1
    when the classes being learned sit where old classes already are (the situation in
    which learning them conflicts with keeping the old geometry). None when there is no
    local data or no old class with a prototype (e.g. the first step).
    """
    model.eval()
    with torch.no_grad():
        sums: dict = {}
        counts: dict = {}
        for x, y in _device_batches(loader, device):
            _, h_shared = model.embed_both(x)
            h_shared = F.normalize(h_shared, dim=1)
            for c in y.unique().tolist():
                m = y == c
                sums[c] = sums.get(c, 0) + h_shared[m].sum(0)
                counts[c] = counts.get(c, 0) + int(m.sum())
        if not sums:
            return None
        protos = model.classifier.prototypes

        def _has_proto(c: int) -> bool:
            return c < protos.shape[0] and float(protos[c].norm()) > 0.0

        local = {
            c: (
                F.normalize(protos[c], dim=0)
                if (c in known_consolidated and _has_proto(c))
                else F.normalize(sums[c] / counts[c], dim=0)
            )
            for c in sums
        }
        old = [c for c in known_consolidated if c not in sums and _has_proto(c)]
        if not old:
            return None
        p_old = F.normalize(protos[old], dim=1)
        values = [float((p_old @ p).max()) for p in local.values()]
        return sum(values) / len(values)


def compute_expansion_signal(
    model: FCLModel,
    loader: DataLoader,
    known_consolidated: set,
    scale: Union[float, torch.Tensor],
    device: torch.device,
) -> Optional[dict]:
    model.eval()

    with torch.no_grad():
        h_all, y_all = [], []
        for x, y in _device_batches(loader, device):
            h_all.append(model.embed(x))
            y_all.append(y)
        if not h_all:
            return None

        # split once instead of per batch
        H, Y = torch.cat(h_all), torch.cat(y_all)
        cons_mask = split_by_know(Y, known_consolidated)
        h_cons, y_cons = H[cons_mask], Y[cons_mask]
        h_new, y_new = H[~cons_mask], Y[~cons_mask]
        if h_cons.shape[0] == 0:
            h_cons = y_cons = None
        if h_new.shape[0] == 0:
            h_new = y_new = None
        if h_cons is None and h_new is None:
            return None
        proto_cons = (
            model.classifier.prototypes[y_cons]
            if y_cons is not None and len(y_cons) > 0
            else None
        )

        proto_new = (
            local_class_prototypes(h_new, y_new)
            if h_new is not None and len(h_new) > 0 and y_new is not None
            else None
        )

        logits, labels_for_ce = None, None
        if h_cons is not None and len(h_cons) > 0 and model.classifier.num_classes > 1:
            logits = model.classifier(h_cons)
            labels_for_ce = y_cons

        return {
            "h_cons": h_cons,
            "proto_cons": proto_cons,
            "h_new": h_new,
            "proto_new": proto_new,
            "logits": logits,
            "labels": labels_for_ce,
            "scale": scale,
            "num_seen_classes": max(model.classifier.num_classes, 1),
        }
