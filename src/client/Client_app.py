import pickle

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
from src.model.Models import FCLModel
from src.model.layers.PrototypeMemory import PrototypeMemory
from .ClientTask import train_fn, test_fn
from ..utils.utd_mahd_dataset import load_data

app = ClientApp()


def _build_model(context: Context) -> FCLModel:
    input_dim = int(context.run_config["input-dim"])
    return FCLModel(
        input_dim=input_dim,
        hidden_dim=int(context.run_config["hidden-dim"]),
        d_hat_global=int(context.run_config["d-hat-global"]),
        d_hat_local=int(context.run_config["d-hat-local"]),
    )


def _load_global_prototypes(model: FCLModel, config: ConfigRecord) -> None:
    if "global_prototypes" in config:
        mu_global, class_ids = pickle.loads(config["global_prototypes"])
        model.classifier.update_from_global(mu_global, class_ids)


@app.train()
def train(msg: Message, context: Context) -> Message:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(context)
    model.set_global_arrays(msg.content["arrays"].to_torch_state_dict())
    _load_global_prototypes(model, msg.content["config"])

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    train_loader, _, _ = load_data(
        partition_id,
        num_partitions,
        root=str(context.run_config["data-root"]),
        window_size=int(context.run_config["window-size"]),
        stride=int(context.run_config["stride"]),
        dirichlet_alpha=float(context.run_config["dirichlet-alpha"]),
        batch_size=int(context.run_config["batch-size"]),
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

    arrays_reply = ArrayRecord(model.get_global_arrays())
    metrics_reply = MetricRecord(
        {"train_loss": train_loss, "num-examples": len(train_loader.dataset), "prototypes_updated": prototypes_updated}
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
    _load_global_prototypes(model, msg.content["config"])

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    _, valloader, _ = load_data(
        partition_id,
        num_partitions,
        root=str(context.run_config["data-root"]),
        window_size=int(context.run_config["window-size"]),
        stride=int(context.run_config["stride"]),
        dirichlet_alpha=float(context.run_config["dirichlet-alpha"]),
        batch_size=int(context.run_config["batch-size"]),
    )

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
