from ...core.Plugins import Plugins
from ...core.Strategy import Strategy


class LogPlugin(Plugins):
    def __init__(self, evaluate_every_epoch=False):
        self.evaluate_every_epoch = evaluate_every_epoch

    def before_training_epoch(self, strategy: Strategy) -> None:
        print(f"Starting training for epoch {strategy.clock.epoch + 1}...")

    def after_training_epoch(self, strategy: Strategy) -> None:
        print(f"Finished training for epoch {strategy.clock.epoch + 1}.")
        print(f"Loss: {strategy.total_loss:.4f}")
        print(f"Accuracy: {strategy.total_correct:.4f}")

        if self.evaluate_every_epoch:
            strategy.evaluation()

    def before_evaluate(self, strategy: Strategy) -> None:
        print("Starting evaluation...")

    def after_evaluate(self, strategy: Strategy) -> None:
        print("Finished evaluation.")
        print(f"Evaluation Loss: {strategy.eval_loss:.4f}")
        print(f"Evaluation Accuracy: {strategy.eval_acc:.4f}")
