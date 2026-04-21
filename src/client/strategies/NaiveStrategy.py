from ...core.Strategy import Strategy
from ...model.Models import Model
import torch


class NaiveStrategy(Strategy):
    fused_feat = None
    global_feat = None
    local_feat = None

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        epochs,
        plugins=None,
        device=torch.device("cpu"),
        data_test=None,
        data_train=None,
    ):
        super().__init__(plugins)
        self.model = model
        self.data_test = data_test
        self.dataset = data_train
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device

        self.model.to(self.device)

    def forward(self):
        self.mb_output, self.fused_feat, self.global_feat, self.local_feat = self.model(
            self.mb_x
        )
        loss = self.loss_fn(self.mb_output, self.mb_y)
        return self.mb_output, loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in self.data_test:  # type: ignore
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, _ = self.model.global_forward(images)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()

        avg_loss = total_loss / len(self.data_test.dataset)  # type: ignore
        avg_acc = correct / len(self.data_test.dataset)  # type: ignore
        self.eval_loss = avg_loss
        self.eval_acc = avg_acc
