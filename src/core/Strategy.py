from abc import ABC, abstractmethod
from typing import Any, Tuple


class Clock:
    epoch = 0
    iteration = 0
    total_iterations = 0


class Strategy(ABC):
    model = None
    model_old = None
    dataset = None
    data_test = None
    epochs: int = 1
    optimizer = None
    device = None
    loss_fn = None

    clock: Clock

    loss = None
    total_loss = None
    total_correct = None

    mb_x = None
    mb_y = None
    mb_output = None

    eval_loss = None
    eval_acc = None

    def __init__(self, plugins=None):
        self.plugins = plugins if plugins else []
        self.clock = Clock()

    def _trigger_plugins(self, method_name: str) -> None:
        for plugin in self.plugins:
            method = getattr(plugin, method_name)
            method(self)

    def adaptation(self) -> None:
        self._trigger_plugins("adaptation")

    def before_training(self) -> None:
        self._trigger_plugins("before_training")

    def after_training(self) -> None:
        self._trigger_plugins("after_training")

    def before_training_epoch(self) -> None:
        self._trigger_plugins("before_training_epoch")
        self.total_loss = 0.0
        self.total_correct = 0

    def after_training_epoch(self) -> None:
        self.total_loss /= len(self.dataset.dataset)
        self.total_correct /= len(self.dataset.dataset)
        self._trigger_plugins("after_training_epoch")
        self.clock.epoch += 1
        

    def before_training_iteration(self) -> None:
        self._trigger_plugins("before_training_iteration")

    def before_backward(self) -> None:
        self._trigger_plugins("before_backward")

    def after_training_iteration(self) -> None:
        self._trigger_plugins("after_training_iteration")
        self.clock.iteration += 1
        self.clock.total_iterations += 1

    def before_evaluate(self) -> None:
        self._trigger_plugins("before_evaluate")
        self.eval_loss = 0.0
        self.eval_acc = 0.0

    def after_evaluate(self) -> None:
        self._trigger_plugins("after_evaluate")

    def start(self) -> None:
        self.adaptation()
        self.before_training()
        for epoch in range(self.epochs):
            self.before_training_epoch()

            for mb_x, mb_y in self.dataset:  # type: ignore
                self.mb_x = mb_x
                self.mb_y = mb_y
                self.mb_x, self.mb_y = self.mb_x.to(self.device), self.mb_y.to(
                    self.device
                )

                self.before_training_iteration()
                self.mb_output, self.loss = self.forward()
                self.before_backward()
                self.backward()
                self.total_loss += self.loss.item() * self.mb_x.size(0)
                self.total_correct += (self.mb_output.argmax(dim=1) == self.mb_y).sum().item()
                self.optimizer_step()
                self.after_training_iteration()

            self.after_training_epoch()

        self.after_training()

    def evaluation(self) -> None:
        self.before_evaluate()
        self.evaluate()
        self.after_evaluate()

    @abstractmethod
    def forward(self) -> Tuple[Any, Any]:
        pass

    def backward(self) -> None:
        self.optimizer.zero_grad()  # type: ignore
        self.loss.backward()  # type: ignore

    def optimizer_step(self) -> None:
        self.optimizer.step()  # type: ignore

    @abstractmethod
    def evaluate(self) -> None:
        pass
