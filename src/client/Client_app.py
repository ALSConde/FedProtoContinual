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
import torch.nn as nn

from src.model.layers.PrototypeMemory import PrototypeMemory
from .ClientTask import load_data, train_fn, test_fn

app = ClientApp()


def _build_model(context: Context) -> nn.Module:
    some_model = nn.Module()
    return some_model


def _load_global_prototypes(model: nn.Module, config: ConfigRecord) -> None:
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
        input_dim=int(context.run_config["input-dim"]),
        num_classes_total=int(context.run_config["num-classes-total"]),
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
    _load_global_prototypes(model, msg.content["config"])

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    _, valloader, _ = load_data(
        partition_id,
        num_partitions,
        input_dim=int(context.run_config["input-dim"]),
        num_classes_total=int(context.run_config["num-classes-total"]),
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
