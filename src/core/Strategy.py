from abc import ABC, abstractmethod
from typing import Any, Tuple


class Strategy(ABC):
    model = None
    model_old = None
    dataset = None
    epochs = 1
    optimizer = None
    device = None

    loss = None
    mb_x = None
    mb_y = None
    mb_output = None

    def __init__(self, plugins=None):
        self.plugins = plugins if plugins else []

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

    def after_training_epoch(self) -> None:
        self._trigger_plugins("after_training_epoch")

    def before_training_iteration(self) -> None:
        self._trigger_plugins("before_training_iteration")

    def before_backward(self) -> None:
        self._trigger_plugins("before_backward")

    def after_training_iteration(self) -> None:
        self._trigger_plugins("after_training_iteration")

    def before_evaluate(self) -> None:
        self._trigger_plugins("before_evaluate")

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

                self.before_training_iteration()
                self.mb_output, self.loss = self.forward()
                self.before_backward()
                self.backward()
                self.optimizer_step()
                self.after_training_iteration()

            self.after_training_epoch()

        self.after_training()

    @abstractmethod
    def forward(self) -> Tuple[Any, Any]:
        pass

    def backward(self) -> None:
        self.optimizer.zero_grad()  # type: ignore
        self.loss.backward()  # type: ignore

    def optimizer_step(self) -> None:
        self.optimizer.step()  # type: ignore

    @abstractmethod
    def evaluate(self):
        pass
