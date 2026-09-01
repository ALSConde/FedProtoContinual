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
from src.client.ExpansionCriterion import ExpansionCriterion
from src.model.Models import FCLModel
from src.model.layers.PrototypeMemory import PrototypeMemory
from .ClientTask import train_fn, test_fn, compute_expansion_signal
from ..utils.utd_mahd_dataset import (
    load_data,
    parse_int_list_config,
    resolve_classes_per_step,
    resolve_dirichlet_mode,
)

app = ClientApp()


_LOCAL_STATE_KEY = "local_modules"


def _build_model(context: Context) -> FCLModel:
    input_dim = int(context.run_config["input-dim"])
    return FCLModel(
        input_dim=input_dim,
        hidden_dim=int(context.run_config["hidden-dim"]),
        d_hat_global=int(context.run_config["d-hat-global"]),
        d_hat_local=int(context.run_config["d-hat-local"]),
    )


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

    model = _build_model(context)
    model.set_global_arrays(msg.content["arrays"].to_torch_state_dict())

    known_consolidated = _load_local_state(context, model)
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

    new_classes = 0
    for _, c in train_loader:
        for unique_class in c.unique():
            if int(unique_class.item()) not in known_consolidated:
                new_classes += 1
    if new_classes > 0:
        model.classifier._expand(new_classes)

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
    )

    sum_h, counts, class_ids = memory.get_stats()
    memory.reset()

    _save_local_state(context, model, known_consolidated)

    arrays_reply = ArrayRecord(model.get_global_arrays())
    metrics_reply = MetricRecord(
        {"train_loss": train_loss, "num-examples": len(train_loader.dataset)}
    )
    config_reply = ConfigRecord(
        {"proto_stats": pickle.dumps((sum_h, counts, class_ids))}
    )
    content = RecordDict(
        {
            "arrays": arrays_reply,
            "metrics": metrics_reply,
            "config": config_reply,
        }
    )

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(context)
    model.set_global_arrays(msg.content["arrays"].to_torch_state_dict())

    known_consolidated = _load_local_state(context, model)
    _load_global_prototypes(
        model, msg.content["config"], known_consolidated=known_consolidated
    )

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
