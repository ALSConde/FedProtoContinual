from abc import ABC

class Plugins(ABC):
    def __init__(self):
        pass

    def adaptation(self, strategy) -> None:
        pass

    def before_training(self, strategy) -> None:
        pass

    def after_training(self, strategy) -> None:
        pass

    def before_training_epoch(self, strategy) -> None:
        pass

    def after_training_epoch(self, strategy) -> None:
        pass

    def before_training_iteration(self, strategy) -> None:
        pass

    def before_backward(self, strategy) -> None:
        pass

    def after_training_iteration(self, strategy) -> None:
        pass

    def before_evaluate(self, strategy) -> None:
        pass

    def after_evaluate(self, strategy) -> None:
        pass
