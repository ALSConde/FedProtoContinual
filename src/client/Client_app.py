import pickle
import io
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
import torch
from src.client.CandidacyCriterion import CandidacyCriterion
from src.client.ExpansionCriterion import ExpansionCriterion
from src.model.Models import FCLModel
from src.model.blocks.Adapter import promote_to_incorporated
from src.model.layers.PrototypeMemory import PrototypeMemory
from .ClientTask import train_fn, test_fn, compute_expansion_signal, vote_on_candidate
from ..utils.data.utd_mahd_dataset import (
    load_data,
    parse_int_list_config,
    resolve_classes_per_step,
    resolve_dirichlet_mode,
)

app = ClientApp()


_LOCAL_STATE_KEY = "local_modules"
_CANDIDACY_STATE_KEY = "candidacy_state"
_PROMOTED_CHECKPOINT_KEY = "promoted_local_checkpoint"
_VOTE_ADAPT_CHECKPOINT_KEY = "vote_adapt_checkpoint"


def _build_model(context: Context) -> FCLModel:
    input_dim = int(context.run_config["input-dim"])
    return FCLModel(
        input_dim=input_dim,
        hidden_dim=int(context.run_config["hidden-dim"]),
        d_hat_global=int(context.run_config["d-hat-global"]),
        d_hat_local=int(context.run_config["d-hat-local"]),
        a_max=int(context.run_config.get("a-max", 3)),
    )


def _apply_incorporated_topology(model: FCLModel, config: ConfigRecord) -> None:
    if "incorporated_topology" in config:
        topologies = pickle.loads(config["incorporated_topologies"])
        model.load_incorporated_topology(topologies)


def _load_local_state(context: Context, model: FCLModel) -> set:
    if _LOCAL_STATE_KEY not in context.state:
        return set()

    record = context.state[_LOCAL_STATE_KEY]
    blob = record["blob"]
    if not isinstance(blob, bytes):
        raise ValueError(f"Expected bytes for local state blob, got {type(blob)}")
    bundle = torch.load(io.BytesIO(blob), weights_only=False)
    model.adapter_local = bundle["adapter_local"]
    model.alpha_gate = bundle["alpha_gate"]
    model.classifier = bundle["classifier"]
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


def _restore_local_checkpoint(context: Context, model: FCLModel, key: str) -> bool:
    if key not in context.state:
        return False
    blob = context.state[key]["blob"]
    if not isinstance(blob, bytes):
        raise ValueError(f"Expected bytes for local checkpoint blob, got {type(blob)}")
    bundle = torch.load(io.BytesIO(blob), weights_only=False)
    model.adapter_local = bundle["adapter_local"]
    model.alpha_gate = bundle["alpha_gate"]
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
) -> None:
    status = config.get("incorporation_outcome_status")
    outcome_pid = config.get("candidate_outcome_partition_id")
    if (
        status is not None
        and outcome_pid is not None
        and int(outcome_pid) == own_partition_id
    ):
        if status == "accepted":
            _stash_local_checkpoint(context, model, _PROMOTED_CHECKPOINT_KEY)
            model.reset_local_branch()
            candidacy_criterion.notify_outcome(incorporated=True)
        elif status == "reverted":
            _restore_local_checkpoint(context, model, _PROMOTED_CHECKPOINT_KEY)
            candidacy_criterion.notify_outcome(incorporated=False)
        elif status == "confirmed":
            _clear_checkpoint(context, _PROMOTED_CHECKPOINT_KEY)

    if config.get("last_incorporation_reverted", False):
        _restore_local_checkpoint(context, model, _VOTE_ADAPT_CHECKPOINT_KEY)
    if config.get("last_incorporation_confirmed", False):
        _clear_checkpoint(context, _VOTE_ADAPT_CHECKPOINT_KEY)


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
        ),
        partition_id,
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    partition_id = int(context.node_config["partition-id"])
    config = msg.content["config"]

    model = _build_model(context)
    _apply_incorporated_topology(model, config)
    model.set_global_arrays(msg.content["arrays"].to_torch_state_dict())

    known_consolidated = _load_local_state(context, model)
    candidacy_criterion = _load_candidacy_criterion(context)
    _apply_incorporation_outcome(
        context, model, config, partition_id, candidacy_criterion
    )
    _load_global_prototypes(
        model, msg.content["config"], known_consolidated=known_consolidated
    )

    (train_loader, _, _), partition_id = _load_client_data(msg, context)

    if len(train_loader.dataset) == 0:
        arrays_reply = ArrayRecord(model.get_global_arrays())
        metrics_reply = MetricRecord({"train_loss": 0.0, "num-examples": 0})
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
    for _, c in train_loader:
        for unique_class in c.unique():
            cid = int(unique_class.item())
            if cid not in known_consolidated:
                new_class_ids.add(cid)

    if new_class_ids:
        required_num_classes = max(model.classifier.num_classes, max(new_class_ids) + 1)
        if required_num_classes > model.classifier.num_classes:
            model.classifier._expand(required_num_classes)

    if model.classifier.num_classes > 0:
        signal = compute_expansion_signal(
            model,
            train_loader,
            known_consolidated,
            scale=model.classifier.scale.item(),
            device=device,
        )

        if signal is not None:
            criterion = ExpansionCriterion(theta_exp=context.run_config["theta-exp"])
            result = criterion.compute(**signal)

            kind = criterion.step(
                model.adapter_local, result["g"], g_reduced_below_threshold=False
            )
            if kind is not None:
                print(
                    f"[client {partition_id} expands in {kind} mode (g={result['g']:.4f})]"
                )

    memory = PrototypeMemory(
        embedding_dim=int(context.run_config["hidden-dim"]),
        num_classes=max(model.classifier.num_classes, 1),
        device=device,
    )

    train_loss = train_fn(
        model,
        train_loader,
        memory,
        epochs=int(context.run_config["local-epochs"]),
        lr=float(msg.content["config"]["lr"]),
        device=device,
        known_consolidated=known_consolidated,
        lambda_proto=float(context.run_config.get("lambda-proto", 1.0)),
        lambda_kd=float(context.run_config.get("lambda-kd", 0.5)),
        kd_mode=str(context.run_config.get("kd-mode", "kl")),
        kd_temperature=float(context.run_config.get("kd-temperature", 2.0)),
    )

    sum_h, counts, class_ids = memory.get_stats()
    memory.reset()

    _save_local_state(context, model, known_consolidated)

    config_reply_data = {"proto_stats": pickle.dumps((sum_h, counts, class_ids))}

    alpha_mean = model.alpha_gate.mean_alpha()
    should_propose = candidacy_criterion.step(alpha_mean, local_acc=None)
    if should_propose and not config.get("candidacy_locked", False):
        candidate = promote_to_incorporated(model.adapter_local, model.alpha_gate)
        buffer = io.BytesIO()
        torch.save(candidate, buffer)
        config_reply_data["propose_candidate"] = True
        config_reply_data["candidate_adapter"] = buffer.getvalue()
        config_reply_data["candidate_partition_id"] = partition_id
        print(
            f"[client {partition_id} proposes incorporation"
            f"(alpha_mean={alpha_mean:.4f})]"
        )

    _save_candidacy_criterion(context, candidacy_criterion)

    arrays_reply = ArrayRecord(model.get_global_arrays())
    metrics_reply = MetricRecord(
        {"train_loss": train_loss, "num-examples": len(train_loader.dataset)}
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
    msg: Message, context: Context, model: FCLModel, known_consolidated: set
) -> Message:
    config = msg.content["config"]
    own_partition_id = int(context.node_config["partition-id"])
    proposer_partition_id = int(config.get("candidate_partition_id", -1))
    (train_loader, valloader, _), _ = _load_client_data(msg, context)
    num_examples = len(valloader.dataset)

    if own_partition_id == proposer_partition_id:
        metrics_reply = MetricRecord(
            {
                "vote": 0.0,
                "acc_before": 0.0,
                "acc_after": 0.0,
                "partition_id": own_partition_id,
                "num-examples": num_examples,
            }
        )
        return Message(content=RecordDict({"metrics": metrics_reply}), reply_to=msg)

    candidate = torch.load(io.BytesIO(config["candidate_adapter"]), weights_only=False)

    _stash_local_checkpoint(context, model, _VOTE_ADAPT_CHECKPOINT_KEY)
    vote, acc_before, acc_after = vote_on_candidate(
        model,
        candidate,
        train_loader,
        valloader,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        lr=float(
            context.run_config.get(
                "vote-adapt-lr", context.run_config.get("learning-rate", 0.001)
            )
        ),
        adapt_steps=int(context.run_config.get("vote-adapt-steps", 10)),
        vote_margin=float(context.run_config.get("vote-margin", 0.03)),
    )

    if vote == 1.0:
        _save_local_state(context, model, known_consolidated)
    else:
        _clear_checkpoint(context, _VOTE_ADAPT_CHECKPOINT_KEY)

    metrics_reply = MetricRecord(
        {
            "vote": vote,
            "acc_before": acc_before,
            "acc_after": acc_after,
            "partition_id": own_partition_id,
            "num_examples": num_examples,
        }
    )
    return Message(content=RecordDict({"metrics": metrics_reply}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = msg.content["config"]

    model = _build_model(context)
    _apply_incorporated_topology(model, config)
    model.set_global_arrays(msg.content["arrays"].to_torch_state_dict())

    known_consolidated = _load_local_state(context, model)
    _load_global_prototypes(model, config, known_consolidated=known_consolidated)

    if config.get("vote_round", False):
        return _handle_vote_round(msg, context, model, known_consolidated)

    (_, valloader, _), _ = _load_client_data(msg, context)

    if len(valloader.dataset) == 0:
        metrics_reply = MetricRecord(
            {"eval_loss": 0.0, "eval_acc": 0.0, "num-examples": 0}
        )
        content = RecordDict({"metrics": metrics_reply})
        return Message(content=content, reply_to=msg)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics_reply = MetricRecord(
        {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(valloader.dataset),
        }
    )
    content = RecordDict({"metrics": metrics_reply})
    return Message(content=content, reply_to=msg)
