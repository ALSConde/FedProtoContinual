from ...core.Plugins import Plugins
from ...core.Strategy import Strategy
import torch


class AlphaGateLossPlugin(Plugins):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def before_backward(self, strategy: Strategy):
        if strategy.model.gate is not None and hasattr(strategy.model.gate, "alpha"):
            alpha_loss = torch.mean(strategy.model.gate.alpha)
            strategy.loss += self.alpha * alpha_loss
