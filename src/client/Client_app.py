import copy
import hashlib
import pickle
import io
import random
from typing import Optional
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
import numpy as np
import torch
from src.client.CandidacyCriterion import CandidacyCriterion
from src.client.ExpansionCriterion import ExpansionCriterion
from src.model.Models import FCLModel
from src.model.blocks.Adapter import Adapter, adapter_topology, promote_to_incorporated
from src.model.layers.WDStats import WDStats
from src.model.layers.PrototypeMemory import PrototypeMemory
from src.utils.ablation import AblationFlags, resolve_ablation_flags
from .ClientTask import (
    collect_prototype_stats,
    compute_local_contribution_ratio,
    train_fn,
    test_fn,
    compute_expansion_signal,
    compute_interference,
    vote_on_candidate,
    build_mofe_payload,
)
from ..utils.data.utd_mahd_dataset import (
    build_class_schedule,
    classes_seen_until_round,
    load_data,
    parse_int_list_config,
    resolve_classes_per_step,
    resolve_dirichlet_mode,
)

app = ClientApp()


_LOCAL_STATE_KEY = "local_modules"
_CANDIDACY_STATE_KEY = "candidacy_state"
_LAST_EXPANSION_ROUND_KEY = "expansion_last_round"
_PROMOTED_CHECKPOINT_KEY = "promoted_local_checkpoint"
_VOTE_ADAPT_CHECKPOINT_KEY = "vote_adapt_checkpoint"
_KD_TEACHER_SNAPSHOT_KEY = "kd_teacher_snapshot"
_KD_TEACHER_MARKER_KEY = "kd_teacher_marker"  # class-count or block index
_KD_TEACHER_INCORP_SIG_KEY = (
    "kd_teacher_incorp_sig"  # topology of the teacher's incorporated adapters
)


def _seed_everything(base_seed: int, partition_id: int, current_round: int) -> None:
    local_seed = (base_seed * 1_000_003 + partition_id * 9_973 + current_round) % (
        2**31 - 1
    )
    random.seed(local_seed)
    np.random.seed(local_seed % (2**32 - 1))
    torch.manual_seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_seed)


def _build_model(context: Context) -> FCLModel:
    input_dim = int(context.run_config["input-dim"])
    flags = resolve_ablation_flags(context.run_config)
    return FCLModel(
        input_dim=input_dim,
        hidden_dim=int(context.run_config["hidden-dim"]),
        d_hat_global=int(context.run_config["d-hat-global"]),
        d_hat_local=int(context.run_config["d-hat-local"]),
        a_max=int(context.run_config.get("a-max", 3)),
        use_local_adapter=flags.use_local_adapter,
        near_identity_eps_local=float(
            context.run_config.get("near-identity-eps-local", 1e-3)
        ),
        alpha_init=float(context.run_config.get("alpha-init", 0.0)),
    )


def _expansion_term_metrics(**values: Optional[float]) -> dict:
    """Per-round expansion diagnostics with a fixed key set (Flower aggregates the metrics
    of all clients, so every client must emit the same keys). A missing value is emitted as
    0.0 with <key>_valid = 0.0; the weighted mean over the clients that have it is then
    mean(<key>) / mean(<key>_valid), because both use the same weights."""
    names = (
        "g_cosine_old",
        "g_cosine_new",
        "g_entropy",
        "interference",
        "expansion_signal",
    )
    out: dict = {}
    for name in names:
        value = values.get(name)
        out[name] = 0.0 if value is None else float(value)
        out[f"{name}_valid"] = 0.0 if value is None else 1.0
    return out


def _inherit(context: Context, key: str, fallback_key: str, default: float) -> float:
    """Run-config value of `key`; a negative value (the pyproject default) or a missing
    key means "same as `fallback_key`"."""
    value = float(context.run_config.get(key, -1.0))
    if value >= 0.0:
        return value
    return float(context.run_config.get(fallback_key, default))


def _param_metrics(model: FCLModel, base_total: int) -> dict:
    """Parameter counters reported to the server every round.

    params_base               : architecture with NO expansion (fresh model from the config)
    params_total              : current model (base + local expansions + incorporated adapters)
    params_expansion_overhead : params_total - params_base
    """
    report = model.parameter_report()
    return {
        "params_total": report["total"],
        "params_base": base_total,
        "params_expansion_overhead": report["total"] - base_total,
        "params_shared": report["shared"],
        "params_adapter_local": report["adapter_local"],
        "params_incorporated": report["incorporated_adapters"],
    }


def _resolve_proximal_mu(config: ConfigRecord, flags: AblationFlags) -> float:
    """FedProx weight for this round; 0.0 (no proximal term) under FedAvg.

    Flower's FedProx strategy injects "proximal-mu" into the train config, so the
    server value wins when present; flags.proximal_mu is the run-config fallback.
    """
    if flags.fl_algorithm != "fedprox":
        return 0.0
    return (
        float(config["proximal-mu"]) if "proximal-mu" in config else flags.proximal_mu
    )


def _apply_incorporated_topology(model: FCLModel, config: ConfigRecord) -> None:
    if "incorporated_topologies" in config:
        topologies = pickle.loads(config["incorporated_topologies"])
        model.load_incorporated_topology(topologies)
    if "shadow_topology" in config:
        model.load_shadow_topology(pickle.loads(config["shadow_topology"]))


def _reset_wd_stats(module: torch.nn.Module) -> None:
    for m in module.modules():
        stats = getattr(m, "stats", None)
        if isinstance(stats, WDStats) and stats is not None:
            stats.reset()


def _load_local_state(context: Context, model: FCLModel, device: torch.device) -> set:
    if _LOCAL_STATE_KEY not in context.state:
        return set()

    record = context.state[_LOCAL_STATE_KEY]
    blob = record["blob"]
    if not isinstance(blob, bytes):
        raise ValueError(f"Expected bytes for local state blob, got {type(blob)}")
    bundle = torch.load(io.BytesIO(blob), map_location=device, weights_only=False)
    model.adapter_local = bundle["adapter_local"].to(device)
    model.alpha_gate = bundle["alpha_gate"].to(device)
    model.classifier = bundle["classifier"].to(device)
    _reset_wd_stats(model.adapter_local)
    return bundle["known_consolidated"]


def _save_local_state(
    context: Context, model: FCLModel, known_consolidated: set
) -> None:
    bundle = {
        "adapter_local": model.adapter_local,
        "alpha_gate": model.alpha_gate,
        "classifier": model.classifier,
        "known_consolidated": known_consolidated,
    }
    buffer = io.BytesIO()
    torch.save(bundle, buffer)
    context.state[_LOCAL_STATE_KEY] = ConfigRecord({"blob": buffer.getvalue()})


def _load_candidacy_criterion(context: Context) -> CandidacyCriterion:
    criterion = CandidacyCriterion(
        theta_alpha=float(context.run_config.get("theta-alpha", 0.32)),
        patience=int(context.run_config.get("candidacy-patience", 3)),
        cooldown_rounds=int(context.run_config.get("candidacy-cooldown", 3)),
        min_local_acc=(
            float(context.run_config["candidacy-min-local-acc"])
            if "candidacy-min-local-acc" in context.run_config
            else None
        ),
    )
    if _CANDIDACY_STATE_KEY in context.state:
        state = context.state[_CANDIDACY_STATE_KEY]
        criterion.load_state(int(state["rounds_above"]), int(state["cooldown"]))
    return criterion


def _save_candidacy_criterion(context: Context, criterion: CandidacyCriterion) -> None:
    rounds_above, cooldown = criterion.state()
    context.state[_CANDIDACY_STATE_KEY] = ConfigRecord(
        {"rounds_above": rounds_above, "cooldown": cooldown}
    )


def _stash_local_checkpoint(context: Context, model: FCLModel, key: str) -> None:
    buffer = io.BytesIO()
    torch.save(
        {"adapter_local": model.adapter_local, "alpha_gate": model.alpha_gate}, buffer
    )
    context.state[key] = ConfigRecord({"blob": buffer.getvalue()})


def _restore_local_checkpoint(
    context: Context, model: FCLModel, key: str, device: torch.device
) -> bool:
    if key not in context.state:
        return False
    blob = context.state[key]["blob"]
    if not isinstance(blob, bytes):
        raise ValueError(f"Expected bytes for local checkpoint blob, got {type(blob)}")
    bundle = torch.load(io.BytesIO(blob), map_location=device, weights_only=False)
    model.adapter_local = bundle["adapter_local"].to(device)
    model.alpha_gate = bundle["alpha_gate"].to(device)
    del context.state[key]
    return True


def _clear_checkpoint(context: Context, key: str) -> None:
    if key in context.state:
        del context.state[key]


def _apply_incorporation_outcome(
    context: Context,
    model: FCLModel,
    config: ConfigRecord,
    own_partition_id: int,
    candidacy_criterion: CandidacyCriterion,
    device: torch.device,
) -> None:
    status = config.get("incorporation_outcome_status")
    outcome_pid = config.get("candidate_outcome_partition_id")
    if status != "none" and int(outcome_pid) == own_partition_id:
        if status == "accepted":
            _stash_local_checkpoint(context, model, _PROMOTED_CHECKPOINT_KEY)
            model.reset_local_branch()
            candidacy_criterion.notify_outcome(incorporated=True)
        elif status == "reverted":
            _restore_local_checkpoint(context, model, _PROMOTED_CHECKPOINT_KEY, device)
            candidacy_criterion.notify_outcome(incorporated=False)
        elif status == "confirmed":
            _clear_checkpoint(context, _PROMOTED_CHECKPOINT_KEY)

    if config.get("last_incorporation_reverted", False):
        _restore_local_checkpoint(context, model, _VOTE_ADAPT_CHECKPOINT_KEY, device)
    if config.get("last_incorporation_confirmed", False):
        _clear_checkpoint(context, _VOTE_ADAPT_CHECKPOINT_KEY)


def _vote_mode(context: Context) -> str:
    return str(context.run_config.get("vote-mode", "legacy")).strip().lower()


def _reply_with_payload(msg: Message, metrics_reply: MetricRecord, payload) -> Message:
    content = {"metrics": metrics_reply}
    if payload is not None:
        content["config"] = ConfigRecord({"mofe_stats": pickle.dumps(payload)})
    return Message(content=RecordDict(content), reply_to=msg)


def _load_global_prototypes(
    model: FCLModel, config: ConfigRecord, known_consolidated: set
) -> None:
    if "global_prototypes" in config:
        mu_global, class_ids = pickle.loads(config["global_prototypes"])
        model.classifier.update_from_global(mu_global, class_ids)
        known_consolidated.update([int(c) for c in class_ids.tolist()])


def _load_client_data(msg: Message, context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    current_round = int(msg.content["config"].get("server_round", 1))

    scenario = str(context.run_config.get("training-scenario", "federated")).lower()
    raw_classes_per_step = context.run_config.get("classes-per-step", None)
    if raw_classes_per_step is not None:
        raw_classes_per_step = int(raw_classes_per_step)
    classes_per_step = resolve_classes_per_step(scenario, raw_classes_per_step)

    raw_dirichlet_mode = str(context.run_config.get("dirichlet-mode", "static")).lower()
    dirichlet_mode = resolve_dirichlet_mode(scenario, raw_dirichlet_mode)

    held_out_subjects = parse_int_list_config(
        context.run_config.get("server-eval-subjects")
    )

    return (
        load_data(
            partition_id,
            num_partitions,
            root=str(context.run_config["data-root"]),
            window_size=int(context.run_config["window-size"]),
            stride=int(context.run_config["stride"]),
            dirichlet_alpha=float(context.run_config["dirichlet-alpha"]),
            batch_size=int(context.run_config["batch-size"]),
            current_round=current_round,
            classes_per_step=classes_per_step,
            rounds_per_step=int(context.run_config.get("rounds-per-step", 1)),
            num_classes_total=(
                int(context.run_config["num-classes-total"])
                if "num-classes-total" in context.run_config
                else None
            ),
            dirichlet_mode=dirichlet_mode,
            held_out_subjects=held_out_subjects,
            partition_mode=str(context.run_config.get("partition-mode", "subject")),
            val_split=str(context.run_config.get("val-split", "recording")),
            class_retention=str(context.run_config.get("class-retention", "dirichlet"))
            .strip()
            .lower(),
            seed=int(context.run_config.get("seed", 0)),
        ),
        partition_id,
    )


def _current_num_classes_seen(context: Context, current_round: int) -> Optional[int]:
    scenario = str(context.run_config.get("training-scenario", "federated")).lower()

    raw_classes_per_step = context.run_config.get("classes-per-step", None)
    if raw_classes_per_step is None:
        return None

    classes_per_step = resolve_classes_per_step(scenario, int(raw_classes_per_step))
    if classes_per_step is None:
        return None

    num_classes_total = int(context.run_config["num-classes-total"])
    schedule = build_class_schedule(num_classes_total, classes_per_step)
    rounds_per_step = int(context.run_config.get("rounds-per-step", 1))
    return len(classes_seen_until_round(current_round, rounds_per_step, schedule))


def _incorporated_signature(model: FCLModel) -> str:
    topologies = [adapter_topology(a) for a in model.incorporated_adapters]
    return hashlib.sha1(repr(topologies).encode("utf-8")).hexdigest()


def _snapshot_global_branch(context: Context, model: FCLModel) -> None:
    buffer = io.BytesIO()
    torch.save(
        {
            "feature_extractor": model.feature_extractor,
            "adapter_global": model.adapter_global,
            "incorporated_adapters": model.incorporated_adapters,
        },
        buffer,
    )
    context.state[_KD_TEACHER_SNAPSHOT_KEY] = ConfigRecord({"blob": buffer.getvalue()})
    context.state[_KD_TEACHER_INCORP_SIG_KEY] = ConfigRecord(
        {"sig": _incorporated_signature(model)}
    )


def _sync_kd_teacher_incorporated(context: Context, model: FCLModel) -> None:
    if _KD_TEACHER_SNAPSHOT_KEY not in context.state:
        return

    signature = _incorporated_signature(model)
    stored = (
        context.state[_KD_TEACHER_INCORP_SIG_KEY]["sig"]
        if _KD_TEACHER_INCORP_SIG_KEY in context.state
        else None
    )
    if stored == signature:
        return

    blob = context.state[_KD_TEACHER_SNAPSHOT_KEY]["blob"]
    bundle = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    bundle["incorporated_adapters"] = copy.deepcopy(model.incorporated_adapters).cpu()
    buffer = io.BytesIO()
    torch.save(bundle, buffer)
    context.state[_KD_TEACHER_SNAPSHOT_KEY] = ConfigRecord({"blob": buffer.getvalue()})
    context.state[_KD_TEACHER_INCORP_SIG_KEY] = ConfigRecord({"sig": signature})
    print(
        f"[kd teacher synced: now holds {len(model.incorporated_adapters)} "
        "incorporated adapter(s)]"
    )


def _load_kd_teacher_embed_fn(context: Context, device: torch.device):
    if _KD_TEACHER_SNAPSHOT_KEY not in context.state:
        return None
    blob = context.state[_KD_TEACHER_SNAPSHOT_KEY]["blob"]
    bundle = torch.load(io.BytesIO(blob), map_location=device, weights_only=False)
    frozen_fe = bundle["feature_extractor"].to(device)
    frozen_ag = bundle["adapter_global"].to(device)
    frozen_incorp = bundle["incorporated_adapters"].to(device)
    for module in (frozen_fe, frozen_ag, frozen_incorp):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    def _embed_global(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = frozen_fe(x)
            x_global = frozen_ag(feats)
            if len(frozen_incorp) > 0:
                x_global = x_global + sum(
                    a.forward_delta(x_global)
                    for a in frozen_incorp
                    if isinstance(a, Adapter)
                )
            return x_global

    return _embed_global


def _resolve_kd_teacher(
    context: Context,
    model: FCLModel,
    current_round: int,
    lambda_kd: float,
    device: torch.device,
):
    num_classes = _current_num_classes_seen(context, current_round)

    if num_classes is None:
        refresh_every = int(context.run_config.get("kd-refresh-rounds", 10))
        block_idx = (current_round - 1) // refresh_every
        last_block = (
            int(context.state[_KD_TEACHER_MARKER_KEY]["n"])
            if _KD_TEACHER_MARKER_KEY in context.state
            else None
        )
        if last_block is None:
            context.state[_KD_TEACHER_MARKER_KEY] = ConfigRecord({"n": block_idx})
            return 0.0, None  # first block: nothing old to distill against yet
        if block_idx > last_block:
            _snapshot_global_branch(context, model)
            context.state[_KD_TEACHER_MARKER_KEY] = ConfigRecord({"n": block_idx})
        _sync_kd_teacher_incorporated(context, model)
        return lambda_kd, _load_kd_teacher_embed_fn(context, device)

    last_num_classes = (
        int(context.state[_KD_TEACHER_MARKER_KEY]["n"])
        if _KD_TEACHER_MARKER_KEY in context.state
        else None
    )
    if last_num_classes is None:
        # First step: no old-class knowledge exists yet, so KD stays off —
        # just remember the current class count to detect the next change.
        context.state[_KD_TEACHER_MARKER_KEY] = ConfigRecord({"n": num_classes})
        return 0.0, None
    if num_classes > last_num_classes:
        _snapshot_global_branch(context, model)
        context.state[_KD_TEACHER_MARKER_KEY] = ConfigRecord({"n": num_classes})
    _sync_kd_teacher_incorporated(context, model)
    return lambda_kd, _load_kd_teacher_embed_fn(context, device)


@app.train()
def train(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    partition_id = int(context.node_config["partition-id"])
    config = msg.content["config"]
    current_round = int(config.get("server_round", 1))
    base_seed = int(context.run_config.get("seed", 0))
    _seed_everything(base_seed, partition_id, current_round)
    flags = resolve_ablation_flags(context.run_config)

    model = _build_model(context)
    base_params_total = model.parameter_report()["total"]  # no expansion yet
    model.to(device)
    _apply_incorporated_topology(model, config)
    model.set_global_arrays(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    known_consolidated = _load_local_state(context, model, device)
    model.to(device)
    candidacy_criterion = _load_candidacy_criterion(context)
    _apply_incorporation_outcome(
        context, model, config, partition_id, candidacy_criterion, device
    )
    model.to(device)
    _load_global_prototypes(model, config, known_consolidated=known_consolidated)

    (train_loader, _, _), partition_id = _load_client_data(msg, context)

    if len(train_loader.dataset) == 0:
        _save_candidacy_criterion(context, candidacy_criterion)
        arrays_reply = ArrayRecord(model.get_global_arrays())
        metrics_reply = MetricRecord(
            {
                "train_loss": 0.0,
                "num-examples": 0,
                "expanded_width": 0,
                "expanded_depth": 0,
                "expansion_g": 0.0,
                **_expansion_term_metrics(),
                "alpha_mean": 0.0,
                "contribution_ratio": 0.0,
                "classifier_scale": float(model.classifier.scale.detach()),
                **_param_metrics(model, base_params_total),
            }
        )
        config_reply = ConfigRecord({"proto_stats": pickle.dumps((None, None, []))})

        content = RecordDict(
            {
                "arrays": arrays_reply,
                "metrics": metrics_reply,
                "config": config_reply,
            }
        )

        return Message(content=content, reply_to=msg)

    new_class_ids = set()
    local_class_ids = set()
    for _, c in train_loader:
        for unique_class in c.unique():
            cid = int(unique_class.item())
            local_class_ids.add(cid)
            if cid not in known_consolidated:
                new_class_ids.add(cid)

    if new_class_ids:
        required_num_classes = max(model.classifier.num_classes, max(new_class_ids) + 1)
        if required_num_classes > model.classifier.num_classes:
            model.classifier._expand(required_num_classes)

    expanded_width, expanded_depth, expansion_g = 0, 0, 0.0
    term_metrics = _expansion_term_metrics()
    # The saturation signal g, its terms and the interference are always logged when a
    # local adapter exists (useful to calibrate theta-exp and interference-floor); the
    # expansion itself only happens with enable-expansion.
    if flags.use_local_adapter and model.classifier.num_classes > 0:
        signal = compute_expansion_signal(
            model,
            train_loader,
            known_consolidated,
            scale=model.classifier.scale.item(),
            device=device,
        )
        interference = compute_interference(
            model, train_loader, known_consolidated, device
        )

        criterion = ExpansionCriterion(
            theta_exp=context.run_config["theta-exp"],
            signal_mode=flags.expansion_signal,
            interference_floor=float(context.run_config.get("interference-floor", 0.5)),
        )
        result = criterion.compute(**signal) if signal is not None else None
        g_value = float(result["g"]) if result is not None else None
        signal_value = criterion.signal(g_value, interference)
        if result is not None:
            expansion_g = g_value
        term_metrics = _expansion_term_metrics(
            g_cosine_old=result["g_cosine_old"] if result is not None else None,
            g_cosine_new=result["g_cosine_new"] if result is not None else None,
            g_entropy=result["g_entropy"] if result is not None else None,
            interference=interference,
            expansion_signal=(
                signal_value
                if (result is not None or interference is not None)
                else None
            ),
        )

        # With expansion-signal=g the decision only exists when g does (previous behaviour,
        # including not touching the adapter's attempt counters otherwise).
        can_decide = result is not None or (
            flags.expansion_signal != "g" and interference is not None
        )
        if can_decide:
            cooldown = int(context.run_config.get("expansion-cooldown", 0))
            last_expansion = (
                int(context.state[_LAST_EXPANSION_ROUND_KEY]["round"])
                if _LAST_EXPANSION_ROUND_KEY in context.state
                else -(10**9)
            )
            expansion_allowed = (
                cooldown <= 0 or current_round - last_expansion >= cooldown
            )

            if flags.enable_expansion and expansion_allowed:
                params_local_before = model.parameter_report()["adapter_local"]
                kind = criterion.step(
                    model.adapter_local, signal_value, g_reduced_below_threshold=False
                )
                if kind is not None:
                    context.state[_LAST_EXPANSION_ROUND_KEY] = ConfigRecord(
                        {"round": current_round}
                    )
                    params_local_after = model.parameter_report()["adapter_local"]
                    print(
                        f"[client {partition_id} expands in {kind} mode "
                        f"(signal={signal_value:.4f}, g={expansion_g:.4f}, "
                        f"interference={'n/a' if interference is None else f'{interference:.4f}'}) "
                        f"| adapter_local params {params_local_before} -> {params_local_after}]"
                    )
                    expanded_width = int(kind == "width")
                    expanded_depth = int(kind == "depth")

    memory = PrototypeMemory(
        embedding_dim=int(context.run_config["hidden-dim"]),
        num_classes=max(model.classifier.num_classes, 1),
        device=device,
    )

    if flags.enable_kd:
        lambda_kd, frozen_embed_fn = _resolve_kd_teacher(
            context,
            model,
            current_round,
            float(context.run_config.get("lambda-kd", 0.5)),
            device,
        )
    else:
        lambda_kd, frozen_embed_fn = 0.0, None

    # Split optimizers: KD may act on the local embedding, the shared one, or both. The
    # teacher only exists after the first step (frozen_embed_fn is None before), so the
    # presence of the teacher gates both weights.
    split_active = flags.split_optimizers and flags.use_local_adapter
    lambda_kd_shared = 0.0
    if split_active:
        if flags.kd_on in ("shared", "both") and frozen_embed_fn is not None:
            lambda_kd_shared = _inherit(context, "lambda-kd-shared", "lambda-kd", 0.5)
        if flags.kd_on == "shared":
            lambda_kd = 0.0

    train_loss = train_fn(
        model,
        train_loader,
        memory,
        epochs=int(context.run_config["local-epochs"]),
        lr=float(config["lr"]),
        device=device,
        known_consolidated=known_consolidated,
        lambda_proto=float(context.run_config.get("lambda-proto", 1.0)),
        lambda_kd=lambda_kd,
        kd_mode=str(context.run_config.get("kd-mode", "kl")),
        kd_temperature=float(context.run_config.get("kd-temperature", 2.0)),
        frozen_embed_fn=frozen_embed_fn,
        proximal_mu=_resolve_proximal_mu(config, flags),
        lambda_mofe=float(context.run_config.get("lambda-mofe", 0.5)),
        mofe_margin=float(context.run_config.get("mofe-train-margin", 0.0)),
        local_class_ids=sorted(local_class_ids) if flags.ce_local_classes else None,
        proto_prepass=flags.proto_prepass,
        lr_scales={
            "extractor": float(context.run_config.get("lr-scale-extractor", 1.0)),
            "adapter_global": float(
                context.run_config.get("lr-scale-adapter-global", 1.0)
            ),
            "local": float(context.run_config.get("lr-scale-local", 1.0)),
        },
        split_optimizers=flags.split_optimizers,
        lambda_proto_shared=_inherit(
            context, "lambda-proto-shared", "lambda-proto", 1.0
        ),
        lambda_proto_local=_inherit(context, "lambda-proto-local", "lambda-proto", 1.0),
        lambda_kd_shared=lambda_kd_shared,
        kd_scale_mode=flags.kd_scale_mode,
        kd_logit_scale=float(context.run_config.get("kd-logit-scale", 1.0)),
        cls_loss=flags.cls_loss,
        cls_margin=float(context.run_config.get("cls-margin", 0.2)),
    )

    if flags.proto_postpass:
        # Statistics of the trained model (eval mode) replace the ones accumulated while
        # the weights were still changing; same format, so the server is unaffected.
        collect_prototype_stats(model, train_loader, memory, device)

    sum_h, counts, class_ids = memory.get_stats()
    memory.reset()

    _save_local_state(context, model, known_consolidated)

    config_reply_data = {"proto_stats": pickle.dumps((sum_h, counts, class_ids))}

    alpha_mean = model.alpha_gate.mean_alpha() if flags.use_local_adapter else 0.0
    contribution_ratio = (
        compute_local_contribution_ratio(model, train_loader, device)
        if flags.use_local_adapter
        else None
    )
    should_propose = False
    if flags.enable_incorporation:
        should_propose = candidacy_criterion.step(
            contribution_ratio if contribution_ratio is not None else 0.0,
            local_acc=None,
        )
    if should_propose and not config.get("candidacy_locked", False):
        candidate = promote_to_incorporated(model.adapter_local, model.alpha_gate)
        buffer = io.BytesIO()
        torch.save(candidate, buffer)
        config_reply_data["propose_candidate"] = True
        config_reply_data["candidate_adapter"] = buffer.getvalue()
        config_reply_data["candidate_partition_id"] = partition_id
        print(
            f"[client {partition_id} proposes incorporation"
            f"(contribution_ratio={contribution_ratio:.4f}, alpha_mean={alpha_mean:.4f})]"
        )

    _save_candidacy_criterion(context, candidacy_criterion)

    arrays_reply = ArrayRecord(model.get_global_arrays())
    metrics_reply = MetricRecord(
        {
            "train_loss": train_loss,
            "num-examples": len(train_loader.dataset),
            "expanded_width": expanded_width,
            "expanded_depth": expanded_depth,
            "expansion_g": expansion_g,
            **term_metrics,
            "alpha_mean": alpha_mean,
            # local, learned per client (never aggregated): logged to see how far it drifts from 20
            "classifier_scale": float(model.classifier.scale.detach()),
            "contribution_ratio": (
                contribution_ratio if contribution_ratio is not None else 0.0
            ),
            **_param_metrics(model, base_params_total),
        }
    )
    config_reply = ConfigRecord(config_reply_data)
    content = RecordDict(
        {
            "arrays": arrays_reply,
            "metrics": metrics_reply,
            "config": config_reply,
        }
    )

    return Message(content=content, reply_to=msg)


def _handle_vote_round(
    msg: Message,
    context: Context,
    model: FCLModel,
    known_consolidated: set,
    device: torch.device,
    flags: AblationFlags,
) -> Message:
    config = msg.content["config"]
    own_partition_id = int(context.node_config["partition-id"])
    proposer_partition_id = int(config.get("candidate_partition_id", -1))

    (train_loader, valloader, _), _ = _load_client_data(msg, context)
    num_examples = len(valloader.dataset)

    # Regular evaluation first (model without candidate, before any adaptation), so
    # vote rounds report the same eval_* metrics as every other round.
    if num_examples > 0:
        eval_loss, eval_acc = test_fn(model, valloader, device)
        if flags.use_local_adapter:
            _, eval_acc_global = test_fn(model, valloader, device, branch="global")
        else:
            eval_acc_global = eval_acc
    else:
        eval_loss, eval_acc, eval_acc_global = 0.0, 0.0, 0.0
    eval_metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_acc_global": eval_acc_global,
    }

    if own_partition_id == proposer_partition_id or num_examples == 0:
        metrics_reply = MetricRecord(
            {
                **eval_metrics,
                "vote": 0.0,
                "acc_before": 0.0,
                "acc_after": 0.0,
                "partition_id": own_partition_id,
                "num-examples": num_examples,
            }
        )
        return Message(content=RecordDict({"metrics": metrics_reply}), reply_to=msg)

    candidate = torch.load(
        io.BytesIO(config["candidate_adapter"]), map_location=device, weights_only=False
    )
    candidate.to(device)

    if _vote_mode(context) == "mofe":
        # MoFe vote: paired comparison of the validation loss of the shared
        # embedding with vs. without the candidate (and with the candidate in place
        # of each incorporated adapter). Nothing is trained or adapted here, so the
        # two arms differ only by the candidate.
        payload = build_mofe_payload(
            model, valloader, device, own_partition_id, "vote", candidate=candidate
        )
        gain = payload["stats"]["plus"]["d_sum"]
        metrics_reply = MetricRecord(
            {
                **eval_metrics,
                "vote": float(gain > 0.0),
                "acc_before": 0.0,
                "acc_after": 0.0,
                "partition_id": own_partition_id,
                "num-examples": num_examples,
            }
        )
        return _reply_with_payload(msg, metrics_reply, payload)

    _stash_local_checkpoint(context, model, _VOTE_ADAPT_CHECKPOINT_KEY)
    vote_ce_class_ids = None
    if flags.ce_local_classes:
        vote_ce_class_ids = sorted(
            {int(c) for _, ys in train_loader for c in ys.unique().tolist()}
        )
    vote, acc_before, acc_after = vote_on_candidate(
        model,
        candidate,
        train_loader,
        valloader,
        device=device,
        lr=float(
            context.run_config.get(
                "vote-adapt-lr", context.run_config.get("learning-rate", 0.001)
            )
        ),
        adapt_steps=int(context.run_config.get("vote-adapt-steps", 10)),
        vote_margin=float(context.run_config.get("vote-margin", 0.03)),
        ce_class_ids=vote_ce_class_ids,
    )

    if vote == 1.0:
        _save_local_state(context, model, known_consolidated)
    else:
        _clear_checkpoint(context, _VOTE_ADAPT_CHECKPOINT_KEY)

    metrics_reply = MetricRecord(
        {
            **eval_metrics,
            "vote": vote,
            "acc_before": acc_before,
            "acc_after": acc_after,
            "partition_id": own_partition_id,
            "num-examples": num_examples,
        }
    )
    return Message(content=RecordDict({"metrics": metrics_reply}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = msg.content["config"]
    partition_id = int(context.node_config["partition-id"])
    current_round = int(config.get("server_round", 1))
    base_seed = int(context.run_config.get("seed", 0))
    _seed_everything(base_seed, partition_id, current_round)
    flags = resolve_ablation_flags(context.run_config)

    model = _build_model(context)
    model.to(device)
    _apply_incorporated_topology(model, config)
    model.set_global_arrays(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    known_consolidated = _load_local_state(context, model, device)
    model.to(device)
    _load_global_prototypes(model, config, known_consolidated=known_consolidated)

    if config.get("vote_round", False):
        return _handle_vote_round(
            msg, context, model, known_consolidated, device, flags
        )

    (_, valloader, _), _ = _load_client_data(msg, context)

    if len(valloader.dataset) == 0:
        metrics_reply = MetricRecord(
            {
                "eval_loss": 0.0,
                "eval_acc": 0.0,
                "eval_acc_global": 0.0,
                "num-examples": 0,
            }
        )
        content = RecordDict({"metrics": metrics_reply})
        return Message(content=content, reply_to=msg)

    eval_loss, eval_acc = test_fn(model, valloader, device)
    if flags.use_local_adapter:
        _, eval_acc_global = test_fn(model, valloader, device, branch="global")
    else:
        eval_acc_global = eval_acc

    metrics_reply = MetricRecord(
        {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "eval_acc_global": eval_acc_global,
            "num-examples": len(valloader.dataset),
        }
    )
    payload = None
    if _vote_mode(context) == "mofe" and flags.enable_incorporation:
        if len(model.shadow_adapters) > 0:
            # Candidate on probation: the vote statistics are computed with the
            # co-trained shadow adapter (with vs. without), every evaluation round.
            payload = build_mofe_payload(
                model,
                valloader,
                device,
                partition_id,
                "vote",
                candidate=model.shadow_adapters[0],
            )
        elif len(model.incorporated_adapters) > 0:
            payload = build_mofe_payload(model, valloader, device, partition_id, "loo")
    return _reply_with_payload(msg, metrics_reply, payload)
