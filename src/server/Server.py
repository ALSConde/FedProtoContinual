from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import Strategy
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

from src.utils.load_centralized_data import load_centralized_data
from src.utils.test import test


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Server:
    def __init__(self, strategy: Strategy, grid: Grid, context: Context):
        self.grid = grid
        self.strategy = strategy

        # configs
        self.num_rounds: int = context.run_config["num_rounds"]
        self.lr: float = context.run_config["learning-rate"]

    def execute(self):
        global_model = SimpleNet()
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
        torch.save(state_dict, f"./src/storage/{model_name}")

    def global_evaluate(self, server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = SimpleNet()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        test_loader = load_centralized_data()

        test_loss, test_acc = test(model, test_loader, criterion=torch.nn.CrossEntropyLoss(), device=device)

        return MetricRecord({"accuracy": test_acc, "loss": test_loss})
