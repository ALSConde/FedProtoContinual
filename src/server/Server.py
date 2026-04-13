from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import Strategy
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

from model.Models import Model
from src.utils.load_centralized_data import load_centralized_data
from src.utils.test import test


class Server:
    def __init__(self, strategy: Strategy, grid: Grid, context: Context):
        self.grid = grid
        self.strategy = strategy

        # configs
        self.num_rounds: int = int(context.run_config["num_rounds"])
        self.lr: float = float(context.run_config["learning-rate"])

    def execute(self):
        global_model = Model()
        arrays = ArrayRecord(global_model.state_dict())

        result = self.strategy.start(
            grid=self.grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": self.lr}),
            num_rounds=self.num_rounds,
            evaluate_fn=self.global_evaluate,
        )

        state_dict = result.arrays.to_torch_state_dict()
        model_name = (
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_model.pth"
        )
        torch.save(state_dict, f"./storage/{model_name}")

    def global_evaluate(self, server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = Model()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        test_loader = load_centralized_data()

        test_loss, test_acc = test(
            model, test_loader, criterion=torch.nn.CrossEntropyLoss(), device=device
        )

        return MetricRecord({"accuracy": test_acc, "loss": test_loss})
