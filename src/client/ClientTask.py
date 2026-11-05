from dataclasses import dataclass
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
from src.utils.losses.PrototypeLosses import (
    embedding_geometry,
    ppp_loss,
    proto_nce_loss,
    relation_distillation_loss,
    valid_prototype_mask,
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


VALID_CLS_LOSSES = ("ce", "triplet", "contrastive", "ppp", "nce")
VALID_KD_MODES = ("kl", "embedding_mse", "prd", "kl+prd")


def _prototype_metric_loss(
    model: FCLModel,
    h: torch.Tensor,
    y: torch.Tensor,
    ce_index: Optional[tuple[torch.Tensor, torch.Tensor]],
    kind: str,
    margin: float,
) -> torch.Tensor:
    """Margin-based alternative to the prototype cross-entropy (flag cls-loss).

    Distances are cosine distances d = 1 - cos(h, prototype) in [0, 2]; the prototypes are the
    classifier's (constants here, as in the CE) and act as proxies, so no pairs/triplets of
    samples are needed. Negatives are the other prototypes that exist (non-zero) and, with
    ce-local-classes, only the classes present in the local data (same scope as the CE).
      triplet      batch-hard proxy triplet: relu(d(h, p_y) - min_k d(h, p_k) + margin)
      contrastive  d(h, p_y)^2 + mean_k relu(margin - d(h, p_k))^2
    Unlike the softmax, a hinge stops pushing a sample once its margin is met, so the push
    away from other (e.g. old) prototypes is bounded.
    """
    protos = model.classifier.prototypes
    dist = 1.0 - F.normalize(h, dim=1) @ F.normalize(protos, dim=1).t()
    valid = protos.norm(dim=1) > 0
    if ce_index is not None:
        keep = torch.zeros_like(valid)
        keep[ce_index[0]] = True
        valid = valid & keep
    rows = torch.arange(len(y), device=h.device)
    pos = dist[rows, y]
    neg_mask = valid.unsqueeze(0).expand_as(dist).clone()
    neg_mask[rows, y] = False
    has_neg = neg_mask.any(dim=1)
    if kind == "triplet":
        hardest = (
            torch.where(neg_mask, dist, torch.full_like(dist, 4.0)).min(dim=1).values
        )
        per_sample = F.relu(pos - hardest + margin) * has_neg
        return per_sample.sum() / has_neg.sum().clamp_min(1)
    if kind == "contrastive":
        push = (F.relu(margin - dist).pow(2) * neg_mask).sum(dim=1) / neg_mask.sum(
            dim=1
        ).clamp_min(1)
        return (pos.pow(2) + push).mean()
    raise ValueError(f"Unknown cls loss '{kind}'. Expected one of {VALID_CLS_LOSSES}.")


def _cls_loss(
    model: FCLModel,
    h: torch.Tensor,
    y: torch.Tensor,
    logits: torch.Tensor,
    ce_index: Optional[tuple[torch.Tensor, torch.Tensor]],
    cls_loss: str = "ce",
    cls_margin: float = 0.2,
    cls_tau: float = 0.1,
) -> torch.Tensor:
    """The classification objective: the prototype cross-entropy (default), a margin loss
    on the same prototypes (triplet / contrastive), or one of the batch-aware prototype
    objectives that handle intra-class compactness and inter-class separation together
    (ppp = CoPE's pseudo-prototypical proxy loss, nce = FPL-style prototype InfoNCE, see
    src/utils/losses/PrototypeLosses.py; both use `cls_tau` as temperature)."""
    if cls_loss == "ce":
        return _class_ce(logits, y, ce_index)
    if cls_loss in ("ppp", "nce"):
        return _prototype_batch_loss(model, h, y, ce_index, cls_loss, cls_tau)
    return _prototype_metric_loss(model, h, y, ce_index, cls_loss, cls_margin)


def _prototype_batch_loss(
    model: FCLModel,
    h: torch.Tensor,
    y: torch.Tensor,
    ce_index: Optional[tuple[torch.Tensor, torch.Tensor]],
    kind: str,
    tau: float,
) -> torch.Tensor:
    """ppp / nce on the classifier's prototypes. Scope of the competitors follows
    ce-local-classes exactly like the CE and the margin losses (ce_index). Falls back to a
    zero that stays attached to the graph when no term is defined (e.g. all prototypes
    still zero), so callers can always add/backward it."""
    protos = model.classifier.prototypes
    valid = valid_prototype_mask(protos, None if ce_index is None else ce_index[0])
    if kind == "ppp":
        loss = ppp_loss(h, y, protos, valid, tau)
    elif kind == "nce":
        loss = proto_nce_loss(h, y, protos, valid, tau)
    else:
        raise ValueError(f"Unknown batch cls loss '{kind}'.")
    return (h * 0.0).sum() if loss is None else loss


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


@torch.no_grad()
def prototype_geometry_metrics(
    model: FCLModel,
    loader: DataLoader,
    device: torch.device,
    known_consolidated: set,
    prefix: str = "geo_",
) -> dict[str, float]:
    """Intra/inter-class diagnostics of the TRAINED client (eval mode, no gradient), on the
    embedding the classifier actually uses (the personalised one) and on the shared one.
    Keys: <prefix>{local,shared}_{intra_cos,margin,nn_error,leak_old,leak_old_rate} plus
    <prefix>proto_cos_{max,mean} (prototype-to-prototype) and <prefix>defined (1.0 when the
    sample-level numbers could be computed, else 0.0). The key set is ALWAYS the same: Flower
    skips the whole aggregation of a round when the clients' MetricRecords have different
    keys, so a number that is undefined for one client (no prototype yet, no old class...)
    is reported as 0.0 and `defined` tells how many clients really measured it (the
    federated mean of a number mixes those zeros: read it together with `defined`).
    "Old" for the leak numbers = classes in `known_consolidated` that are ABSENT from this
    client's training labels (the same rule as OldPrototypeState). `known_consolidated` alone
    would be wrong: after the first round of a step it already contains the classes being
    learned, so no sample would be left to measure (the leak numbers came out as a constant
    0.0). Any mass of the current samples near an old prototype is interference."""
    keys = [
        f"{prefix}{view}_{name}"
        for view in ("local", "shared")
        for name in ("intra_cos", "margin", "nn_error", "leak_old", "leak_old_rate")
    ] + [f"{prefix}proto_cos_max", f"{prefix}proto_cos_mean"]
    zeros = {key: 0.0 for key in keys}
    zeros[f"{prefix}defined"] = 0.0
    measured = _prototype_geometry(model, loader, device, known_consolidated, prefix)
    zeros.update(measured)
    if f"{prefix}local_intra_cos" in measured:
        zeros[f"{prefix}defined"] = 1.0
    return zeros


@torch.no_grad()
def _prototype_geometry(
    model: FCLModel,
    loader: DataLoader,
    device: torch.device,
    known_consolidated: set,
    prefix: str,
) -> dict[str, float]:
    model.to(device)
    num_classes = model.classifier.num_classes
    if num_classes == 0:
        return {}
    was_training = model.training
    model.eval()
    try:
        hs_local, hs_shared, ys = [], [], []
        for x, y in _device_batches(loader, device, max_classes=num_classes):
            h_l, h_s = model.embed_both(x)
            hs_local.append(h_l)
            hs_shared.append(h_s)
            ys.append(y)
        if not ys:
            return {}
        y_all = torch.cat(ys)
        protos = model.classifier.prototypes
        present = {int(c) for c in y_all.unique().tolist()}
        old = sorted(set(known_consolidated) - present)
        old_ids = torch.tensor(old, dtype=torch.long, device=device) if old else None
        out: dict[str, float] = {}
        for name, hs in (("local", hs_local), ("shared", hs_shared)):
            geo = embedding_geometry(torch.cat(hs), y_all, protos, old_ids)
            for key, value in geo.items():
                if key.startswith("proto_cos"):
                    out[f"{prefix}{key}"] = value  # same for both views: keep one
                else:
                    out[f"{prefix}{name}_{key}"] = value
        return out
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


class LossTermProbe:
    """Per-term diagnostics of the REAL training: the value and the gradient norm of every
    loss term, as they enter the objective (already multiplied by their weight).

    The gradient is taken with respect to the embedding the term is computed on
    (autograd.grad on h only, so it is cheap and does not touch the encoder). It answers the
    question the loss values cannot: which term steers the update. Reported (mean over the
    recorded batches of the round, 0.0 when a term never ran):
        lt_<name>        weighted value        lt_<name>_grad   weighted ||dterm/dh||_F
    names: cls (CE / ppp / nce / margin loss), proto (L_proto), kl (per-sample KD), prd
    (relation distillation); with split optimizers each is recorded for the shared (_sh) and
    the local (_lo) objective. The key set depends only on the run mode, never on the client,
    because Flower skips the aggregation when the clients' metric keys differ.
    """

    TERMS = ("cls", "proto", "kl", "prd")

    def __init__(self) -> None:
        self.names: list[str] = list(self.TERMS)
        self._val: dict = {}
        self._grad: dict = {}
        self._n: dict = {}
        self.configure(False)

    def configure(self, split: bool) -> None:
        self.names = (
            [f"{t}_{b}" for b in ("sh", "lo") for t in self.TERMS]
            if split
            else list(self.TERMS)
        )
        self._val = {n: 0.0 for n in self.names}
        self._grad = {n: 0.0 for n in self.names}
        self._n = {n: 0 for n in self.names}

    def record(
        self,
        name: str,
        term: Optional[torch.Tensor],
        wrt: Optional[torch.Tensor],
        weight: float = 1.0,
    ) -> None:
        if (
            term is None
            or wrt is None
            or name not in self._val
            or not term.requires_grad
            or not wrt.requires_grad
        ):
            return
        weighted = term * weight
        (g,) = torch.autograd.grad(weighted, wrt, retain_graph=True, allow_unused=True)
        self._val[name] = self._val[name] + weighted.detach()
        if g is not None:
            self._grad[name] = self._grad[name] + g.norm().detach()
        self._n[name] += 1

    def summary(self, prefix: str = "lt_") -> dict[str, float]:
        out: dict[str, float] = {}
        for name in self.names:
            n = max(self._n[name], 1)
            out[f"{prefix}{name}"] = float(self._val[name]) / n
            out[f"{prefix}{name}_grad"] = float(self._grad[name]) / n
        return out


@dataclass
class OldPrototypeState:
    """Learnable copy of the prototypes of the OLD classes for the PRD term (paper: P^t_k).

    In PRD the prototypes of past classes are parameters that the relation-distillation loss
    itself updates, so they follow the moving representation; P^{t-1}_k is the frozen snapshot
    taken before training. The FPC client otherwise treats prototypes as server constants
    (P^t == P^{t-1}), which can only stabilise the embedding. `student` is trained locally
    and exported to the server (see `export_old_prototypes`), which folds it into the global
    prototypes with an EMA (PrototypeAggregator.aggregate_old).

    "Old" = known (consolidated) classes that are absent from the client's training labels:
    samples of a present class are tied to its prototype by the CE / L_proto terms, as in the
    paper where P_o holds only the classes of previous tasks.
    """

    class_ids: list
    student: torch.nn.Parameter  # (K, d) P^t, trained when `learnable`, else a constant
    teacher: torch.Tensor  # (K, d) P^{t-1}, frozen snapshot
    init_norm: torch.Tensor  # (K,) norm of the received prototypes
    learnable: bool = (
        True  # False: only restricts the PRD to the old classes (prd_old_only)
    )


def _local_label_set(
    loader: DataLoader, local_class_ids: Optional[Sequence[int]]
) -> set:
    """Labels present in the client's training data (reuses `local_class_ids` if given)."""
    if local_class_ids is not None:
        return {int(c) for c in local_class_ids}
    labels: set = set()
    for batch in loader:
        labels.update(int(c) for c in batch[1].unique().tolist())
    return labels


def make_old_prototype_state(
    model: FCLModel,
    known_sorted: Sequence[int],
    local_labels: set,
    learnable: bool = True,
) -> OldPrototypeState:
    protos = model.classifier.prototypes
    ids = [
        c
        for c in known_sorted
        if c not in local_labels and c < protos.shape[0] and float(protos[c].norm()) > 0
    ]
    init = protos[ids].detach().clone()
    return OldPrototypeState(
        class_ids=ids,
        student=torch.nn.Parameter(init.clone(), requires_grad=learnable),
        teacher=init,
        init_norm=init.norm(dim=1),
        learnable=learnable,
    )


@torch.no_grad()
def export_old_prototypes(
    state: OldPrototypeState,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """(class ids, learned prototypes, drift). The learned vectors are rescaled to the norm
    of the received ones: the loss only sees cosines, so the norm is free to wander, and the
    server averages in the same raw space as its prototypes. drift = mean(1 - cos(learned,
    received)), a diagnostic of how far the PRD moved the old prototypes."""
    learned = state.student.detach()
    unit = learned / learned.norm(dim=1, keepdim=True).clamp_min(1e-12)
    restored = unit * state.init_norm.unsqueeze(1)
    drift = float((1.0 - (unit * F.normalize(state.teacher, dim=1)).sum(dim=1)).mean())
    return (
        torch.tensor(state.class_ids, dtype=torch.long),
        restored.cpu(),
        drift,
    )


@torch.no_grad()
def compute_old_prototype_drift(
    model: FCLModel,
    start_model: FCLModel,
    loader: DataLoader,
    device: torch.device,
    known_consolidated: set,
    sigma_factor: float = 1.0,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Semantic-drift compensation of the OLD prototypes (SDC, Yu et al. 2020), from the
    client's own data and without any old sample.

    `start_model` is the model as received at the start of the round, `model` the trained
    one. Both embed the same local batches (shared embedding, the space of the prototypes).
    For every old class k (known, absent from the local labels, prototype exists):
        w_ik  = exp(-||f_start(x_i) - p_k||^2 / (2 sigma_k^2))
        delta = sum_i w_ik (f_trained(x_i) - f_start(x_i))        (K, d)
        mass  = sum_i w_ik                                          (K,)
    so delta / mass is the drift of the local samples that sit near p_k, which is where the
    old samples of class k probably moved too. sigma_k^2 = sigma_factor^2 * mean_i of the
    squared distance of the local samples to p_k (adaptive bandwidth). The sums, not the
    ratio, are returned: the server adds them across clients (a weighted mean) and uses
    `mass` as the evidence of its EMA (PrototypeAggregator.aggregate_drift).
    Returns None when there is no old class or no data.
    """
    protos = model.classifier.prototypes
    num_classes = protos.shape[0]
    if num_classes == 0:
        return None
    was_training = model.training
    model.eval()
    start_model.eval()
    try:
        h0, h1, ys = [], [], []
        for x, y in _device_batches(loader, device, max_classes=num_classes):
            h0.append(start_model.embed_both(x)[1])
            h1.append(model.embed_both(x)[1])
            ys.append(y)
    finally:
        model.train(was_training)
    if not ys:
        return None
    h0, h1, y_all = torch.cat(h0), torch.cat(h1), torch.cat(ys)
    present = {int(c) for c in y_all.unique().tolist()}
    old = [
        c
        for c in sorted(known_consolidated)
        if c not in present and c < num_classes and float(protos[c].norm()) > 0
    ]
    if not old:
        return None
    ref = protos[old].to(h0.device)  # (K, d) in the space of the start model
    d2 = torch.cdist(h0, ref) ** 2  # (N, K)
    sigma2 = (sigma_factor**2) * d2.mean(dim=0, keepdim=True).clamp_min(1e-12)
    w = torch.exp(-d2 / (2.0 * sigma2))  # (N, K)
    sum_delta = w.t() @ (h1 - h0)  # (K, d)
    mass = w.sum(dim=0)  # (K,)
    return torch.tensor(old, dtype=torch.long), sum_delta.cpu(), mass.cpu()


def _prd_kwargs(
    state: Optional[OldPrototypeState],
    h_shared: Optional[torch.Tensor],
    learn: bool = True,
) -> dict:
    """Extra `_kd_term` arguments for the learnable old prototypes. learn=False (the local
    branch of the split step) keeps the prototypes but cuts their gradient: they live in the
    shared space, only the shared embedding is allowed to move them."""
    if state is None:
        return {}
    # a constant (old-only) reference keeps the embedding the PRD always used; only the
    # learnable prototypes live in, and are moved through, the shared embedding space
    use_shared = learn and state.learnable
    return dict(
        prd_prototypes_student=state.student if learn else state.student.detach(),
        prd_prototypes_teacher=state.teacher,
        h_prd=h_shared if use_shared else None,
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
    prd_prototypes_student: Optional[torch.Tensor] = None,
    prd_prototypes_teacher: Optional[torch.Tensor] = None,
    h_prd: Optional[torch.Tensor] = None,
    prd_reduction: str = "sum",
    prd_weight: float = 1.0,
    probe: Optional[LossTermProbe] = None,
    probe_weight: float = 1.0,
    probe_suffix: str = "",
) -> Optional[torch.Tensor]:
    """One KD term. kd_mode: "embedding_mse" | "kl" (per-sample distribution over the old
    prototypes) | "prd" (per-prototype distribution over the batch, PRD) | "kl+prd" (sum).
    For the similarity-based modes the logit scale of the distributions is set by
    kd_scale_mode:
      const   teacher and student both use kd_logit_scale (default; with 1.0 this is the
              previous behaviour, with ~20 the distributions match the classifier's);
      local   both use the classifier's own learned scale, detached (strength then varies
              per client and over the rounds, because the scale is local and trained);
      hybrid  teacher: kd_logit_scale, student: local scale (kept for comparison only: it
              adds a sharpness-matching term whenever the local scale drifts away from
              kd_logit_scale).
    For the PRD modes, `prd_prototypes_student` / `prd_prototypes_teacher` (K, d) are the
    paper's P^t_k / P^{t-1}_k (row k = same class). Default (None): the student uses the
    classifier's prototypes of `known_sorted` and the teacher reuses them, i.e. P^t = P^{t-1},
    which is what happens when the prototypes are server constants (see
    relation_distillation_loss). `h_prd` is the student embedding of the PRD term when it
    must differ from `h` (the learnable old prototypes live in the SHARED embedding space,
    so the PRD term is evaluated on the shared embedding, see OldPrototypeState).
    `prd_reduction` ("sum" | "mean" over the old prototypes) and `prd_weight` (multiplies the
    PRD term only; the caller's lambda_kd then weights KL + prd_weight * PRD, so the PRD
    weight is lambda_kd * prd_weight and the KL one stays lambda_kd). `probe` records the
    KL and PRD components with `probe_weight` (the caller's lambda) as `kl`/`prd` + suffix.
    """
    if kd_mode == "embedding_mse":
        return distillation_loss(h, h_teacher)
    if kd_scale_mode == "const":
        s_student = s_teacher = kd_logit_scale
    elif kd_scale_mode == "local":
        s_student = s_teacher = model.classifier.scale.detach()
    elif kd_scale_mode == "hybrid":
        s_student, s_teacher = model.classifier.scale.detach(), kd_logit_scale
    else:
        raise ValueError(f"Unknown kd_scale_mode '{kd_scale_mode}'.")
    terms = []
    if kd_mode in ("kl", "kl+prd"):
        reference = (
            model.classifier.prototypes[known_sorted]
            if len(known_sorted) >= 2
            else None
        )
        kl_term = distillation_loss_kl(
            h,
            h_teacher,
            reference,
            temperature=kd_temperature,
            scale_student=s_student,
            scale_teacher=s_teacher,
        )
        if probe is not None:
            probe.record(f"kl{probe_suffix}", kl_term, h, probe_weight)
        terms.append(kl_term)
    if kd_mode in ("prd", "kl+prd"):
        # PRD needs a single old prototype, unlike the per-sample KL (>= 2)
        if prd_prototypes_student is not None:
            reference = prd_prototypes_student
        else:
            reference = (
                model.classifier.prototypes[known_sorted]
                if len(known_sorted) >= 1
                else None
            )
        h_student_prd = h if h_prd is None else h_prd
        prd_term = relation_distillation_loss(
            h_student_prd,
            h_teacher,
            reference,
            prd_prototypes_teacher,
            temperature=kd_temperature,
            scale_student=s_student,
            scale_teacher=s_teacher,
            reduction=prd_reduction,
        )
        if probe is not None:
            probe.record(
                f"prd{probe_suffix}", prd_term, h_student_prd, probe_weight * prd_weight
            )
        if prd_term is not None:
            prd_term = prd_term * prd_weight
        terms.append(prd_term)
    terms = [t for t in terms if t is not None]
    if not terms:
        return None
    return sum(terms)


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
    cls_loss: str = "ce",
    cls_margin: float = 0.2,
    cls_tau: float = 0.1,
    prd_state: Optional[OldPrototypeState] = None,
    prd_reduction: str = "sum",
    prd_weight: float = 1.0,
    probe: Optional[LossTermProbe] = None,
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
    ce_g = _cls_loss(
        model,
        h_g,
        y,
        _logits_frozen_scale(model.classifier, h_g),
        ce_index,
        cls_loss,
        cls_margin,
        cls_tau,
    )
    lp_g = prototype_alignment_loss(h_g, y, protos, known_consolidated)
    if probe is not None:
        probe.record("cls_sh", ce_g, h_g)
        probe.record("proto_sh", lp_g, h_g, lambda_proto_shared)
    if has_shadow:
        ce_with = _cls_loss(
            model,
            h_g_with,
            y,
            _logits_frozen_scale(model.classifier, h_g_with),
            ce_index,
            cls_loss,
            cls_margin,
            cls_tau,
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
    ce_l = _cls_loss(
        model, h_l, y, model.classifier(h_l), ce_index, cls_loss, cls_margin, cls_tau
    )
    loss_l = ce_l
    lp_l = prototype_alignment_loss(h_l, y, protos, known_consolidated)
    if probe is not None:
        probe.record("cls_lo", ce_l, h_l)
        probe.record("proto_lo", lp_l, h_l, lambda_proto_local)
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
                prd_reduction=prd_reduction,
                prd_weight=prd_weight,
                probe=probe,
                probe_weight=lambda_kd_shared,
                probe_suffix="_sh",
                **_prd_kwargs(prd_state, h_g),
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
                prd_reduction=prd_reduction,
                prd_weight=prd_weight,
                probe=probe,
                probe_weight=lambda_kd_local,
                probe_suffix="_lo",
                **_prd_kwargs(prd_state, None, learn=False),
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
    cls_loss: str = "ce",
    cls_margin: float = 0.2,
    cls_tau: float = 0.1,
    prd_learn_protos: bool = False,
    prd_proto_lr: Optional[float] = None,
    prd_out: Optional[dict] = None,
    prd_reduction: str = "sum",
    prd_weight: float = 1.0,
    loss_probe: Optional[LossTermProbe] = None,
    prd_old_only: bool = False,
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
    `cls_loss` / `cls_margin` / `cls_tau`: classification objective, "ce" (default), a margin
    loss on the prototypes ("triplet" / "contrastive", see _prototype_metric_loss) or a
    batch-aware intra+inter-class objective ("ppp" / "nce" with temperature `cls_tau`, see
    _prototype_batch_loss). `kd_mode` also accepts "prd" and "kl+prd" (see _kd_term).
    `prd_learn_protos`: the prototypes of the old classes become learnable (see
    OldPrototypeState), moved by the PRD term evaluated on the SHARED embedding, with an Adam
    group of learning rate `prd_proto_lr` (default `lr`). Needs kd_mode in {prd, kl+prd}; it
    is inactive while there is no KD teacher, and in the split mode only the shared KD term
    (lambda_kd_shared > 0) moves them. When active and `prd_out` is a dict, it is filled with
    {"class_ids", "prototypes", "num_samples", "drift"} for the upload to the server.
    `prd_reduction` ("sum" | "mean" over the old prototypes) and `prd_weight` (relative to
    lambda_kd, PRD term only) control the magnitude of the PRD term. `loss_probe` records the
    value and the embedding-gradient norm of every loss term (see LossTermProbe); off by
    default, and it never changes the training.
    `prd_old_only`: the PRD term uses only the known classes that are ABSENT from the client's
    training labels (the paper's P_o: classes of previous tasks), with constant prototypes,
    instead of every known class (which also freezes the relations of the classes being
    learned). The KL term is unaffected. Implied when `prd_learn_protos` is on."""
    if cls_loss not in VALID_CLS_LOSSES:
        raise ValueError(
            f"Unknown cls_loss '{cls_loss}'. Expected one of {VALID_CLS_LOSSES}."
        )
    if kd_mode not in VALID_KD_MODES:
        raise ValueError(
            f"Invalid kd_mode: {kd_mode}. Must be one of {VALID_KD_MODES}."
        )
    if prd_reduction not in ("sum", "mean"):
        raise ValueError(
            f"Invalid prd_reduction: {prd_reduction}. Must be 'sum' or 'mean'."
        )
    if (prd_learn_protos or prd_old_only) and kd_mode not in ("prd", "kl+prd"):
        raise ValueError(
            "prd_learn_protos / prd_old_only need kd_mode 'prd' or 'kl+prd' "
            f"(got '{kd_mode}'): they act on the PRD term."
        )

    model.to(device)
    model.train()
    known_consolidated = known_consolidated or set()
    known_sorted = sorted(known_consolidated)

    split_active = bool(split_optimizers and model.use_local_adapter)
    if loss_probe is not None:
        loss_probe.configure(split_active)
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

    prd_state: Optional[OldPrototypeState] = None
    if (prd_learn_protos or prd_old_only) and use_kd and known_sorted:
        prd_state = make_old_prototype_state(
            model,
            known_sorted,
            _local_label_set(trainloader, local_class_ids),
            learnable=prd_learn_protos,
        )
        if prd_learn_protos and prd_state.class_ids:
            optmizer.add_param_group(
                {
                    "params": [prd_state.student],
                    "lr": lr if prd_proto_lr is None else float(prd_proto_lr),
                }
            )

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
                    cls_loss=cls_loss,
                    cls_margin=cls_margin,
                    cls_tau=cls_tau,
                    prd_state=prd_state,
                    prd_reduction=prd_reduction,
                    prd_weight=prd_weight,
                    probe=loss_probe,
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
            ce = _cls_loss(model, h, y, logits, ce_index, cls_loss, cls_margin, cls_tau)

            l_proto = prototype_alignment_loss(
                h, y, model.classifier.prototypes, known_consolidated
            )
            if loss_probe is not None:
                loss_probe.record("cls", ce, h)
                loss_probe.record("proto", l_proto, h, lambda_proto)
            if has_shadow:
                # Both views train (nothing is frozen). Dropout-like pairing keeps the
                # model good without the candidate, so the later with/without
                # comparison is not biased by co-adaptation. The MoFe hinge only
                # pushes the 'with' side (the 'without' loss is detached).
                ce_with = _cls_loss(
                    model,
                    h_with,
                    y,
                    model.classifier(h_with),
                    ce_index,
                    cls_loss,
                    cls_margin,
                    cls_tau,
                )
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
                    prd_reduction=prd_reduction,
                    prd_weight=prd_weight,
                    probe=loss_probe,
                    probe_weight=lambda_kd,
                    **_prd_kwargs(prd_state, h_shared),
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

    if (
        prd_state is not None
        and prd_state.learnable
        and prd_state.class_ids
        and prd_out is not None
    ):
        ids, protos, drift = export_old_prototypes(prd_state)
        prd_out.update(
            class_ids=ids,
            prototypes=protos,
            num_samples=len(trainloader.dataset),
            drift=drift,
        )

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

    This is a diagnostic/trigger pass: it iterates the (shuffling) training loader, which
    would advance the global RNG and change the batch order of the training that follows.
    The RNG state is therefore saved and restored, so switching the diagnostics on does not
    change the training trajectory.
    """
    model.eval()
    devices = [device] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
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
