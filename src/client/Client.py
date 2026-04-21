from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torchvision
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torchvision.transforms import Compose, Normalize, ToTensor
from .plugins.AlphaGateLossPlugin import AlphaGateLossPlugin
from .plugins.CompactionLossPlugin import CompactionLossPlugin
from .plugins.GlobalLossPlugin import GlobalLossPlugin
from .plugins.LogPlugin import LogPlugin
from .plugins.LwFPlugin import LwFPlugin
from .plugins.OrthoLossPlugin import OrthoLossPlugin
from .strategies.NaiveStrategy import NaiveStrategy
from ..core.Strategy import Strategy
from ..model.Models import Model
from ..model.layers.PrototypeMemory import PrototypeMemory


def _num_examples_from_dataloader(dataloader) -> int:
    return int(len(dataloader.dataset))


def _build_strategy(context: Context) -> Strategy:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)

    batch_size = int(context.run_config.get("batch-size", 8))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    learning_rate = float(context.run_config.get("learning-rate", 1e-3))

    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_test = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./storage/test_data",
            train=False,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    data_train = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./storage/train_data",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    plugins = [
        LwFPlugin(),
        GlobalLossPlugin(beta=1.5),
        CompactionLossPlugin(alpha=1.0),
        OrthoLossPlugin(alpha=1.0),
        AlphaGateLossPlugin(alpha=1.0),
        LogPlugin(evaluate_every_epoch=False),
    ]

    return NaiveStrategy(
        model,
        optimizer,
        loss_fn,
        epochs=local_epochs,
        device=device,
        data_test=data_test,
        data_train=data_train,
        plugins=plugins,
    )


@dataclass
class TrainResult:
    state_dict: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    proto_sum_h: torch.Tensor
    proto_counts: torch.Tensor
    proto_class_ids: torch.Tensor


class Client:
    def __init__(
        self,
        strategy: Strategy,
        train_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.strategy = strategy
        self.train_config = train_config or {}

    def _load_global_state(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        local_state = self.strategy.model.state_dict()
        for key, value in global_state_dict.items():
            if (
                key.startswith("shared_encoder.")
                or key.startswith("global_features.")
                or key == "classifier.prototypes"
            ):
                if key in local_state:
                    local_state[key] = value.to(
                        device=local_state[key].device, dtype=local_state[key].dtype
                    )
        self.strategy.model.load_state_dict(local_state, strict=True)

    def _set_round_lr(self, lr: float | None) -> None:
        if lr is None:
            return
        for group in self.strategy.optimizer.param_groups:
            group["lr"] = float(lr)

    def _collect_prototype_stats(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_classes = max(1, int(self.strategy.model.classifier.num_classes))
        embedding_dim = int(self.strategy.model.classifier.embedding_dim)
        memory = PrototypeMemory(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            device=self.strategy.device,
        )

        self.strategy.model.eval()
        with torch.no_grad():
            for images, labels in self.strategy.dataset:
                images = images.to(self.strategy.device)
                labels = labels.to(self.strategy.device)
                _, fused_feat, _, _ = self.strategy.model(images)
                memory.update(fused_feat, labels)

        return memory.get_stats()

    def execute_round(
        self,
        server_round: int,
        global_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        lr: float | None = None,
    ) -> TrainResult:
        if global_state_dict is not None:
            self._load_global_state(global_state_dict)

        round_lr = lr if lr is not None else self.train_config.get("lr")
        self._set_round_lr(round_lr)

        self.strategy.start()
        self.strategy.evaluation()
        proto_sum_h, proto_counts, proto_class_ids = self._collect_prototype_stats()

        result = TrainResult(
            state_dict=self.strategy.model.state_dict(),
            metrics={
                "round": float(server_round),
                "train_loss": float(self.strategy.total_loss),
                "train_acc": float(self.strategy.total_correct),
                "eval_loss": float(self.strategy.eval_loss),
                "eval_acc": float(self.strategy.eval_acc),
                "num-examples": float(
                    _num_examples_from_dataloader(self.strategy.dataset)
                ),
            },
            proto_sum_h=proto_sum_h,
            proto_counts=proto_counts,
            proto_class_ids=proto_class_ids,
        )

        return result

    def evaluate_round(
        self,
        server_round: int,
        global_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        if global_state_dict is not None:
            self._load_global_state(global_state_dict)

        self.strategy.evaluation()
        return {
            "round": float(server_round),
            "eval_loss": float(self.strategy.eval_loss),
            "eval_acc": float(self.strategy.eval_acc),
            "num-examples": float(
                _num_examples_from_dataloader(self.strategy.data_test)
            ),
        }


_CLIENTS: Dict[int, Client] = {}


def _get_or_create_client(context: Context) -> Client:
    node_id = int(context.node_id)
    if node_id not in _CLIENTS:
        strategy = _build_strategy(context)
        _CLIENTS[node_id] = Client(strategy=strategy)
    return _CLIENTS[node_id]


def _get_first_array_record(content: RecordDict) -> ArrayRecord:
    if len(content.array_records) == 0:
        raise ValueError("Incoming message does not contain an ArrayRecord.")
    return next(iter(content.array_records.values()))


def _get_first_config(content: RecordDict) -> Dict[str, Any]:
    if len(content.config_records) == 0:
        return {}
    cfg = next(iter(content.config_records.values()))
    return dict(cfg)


app = ClientApp()


@app.train()
def train(message: Message, context: Context) -> Message:
    client = _get_or_create_client(context)
    content = message.content

    global_state_dict = _get_first_array_record(content).to_torch_state_dict()
    config = _get_first_config(content)
    server_round = int(config.get("server-round", 0))
    lr = float(config["lr"]) if "lr" in config else None

    result = client.execute_round(
        server_round=server_round,
        global_state_dict=global_state_dict,
        lr=lr,
    )

    reply = RecordDict()
    reply.array_records["arrays"] = ArrayRecord(result.state_dict)
    reply.array_records["proto_stats"] = ArrayRecord(
        {
            "proto_sum_h": result.proto_sum_h,
            "proto_counts": result.proto_counts,
            "proto_class_ids": result.proto_class_ids,
        }
    )
    reply.metric_records["metrics"] = MetricRecord(result.metrics)
    return Message(reply, reply_to=message)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    client = _get_or_create_client(context)
    content = message.content

    global_state_dict = _get_first_array_record(content).to_torch_state_dict()
    config = _get_first_config(content)
    server_round = int(config.get("server-round", 0))

    metrics = client.evaluate_round(
        server_round=server_round,
        global_state_dict=global_state_dict,
    )

    reply = RecordDict()
    reply.metric_records["metrics"] = MetricRecord(metrics)
    return Message(reply, reply_to=message)
