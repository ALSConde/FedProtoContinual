from core.Plugins import Plugins
from core.Strategy import Strategy
import torch

class GlobalLossPlugin(Plugins):
    def __init__(self, beta: float = 1.5):
        super().__init__()
        self.beta = beta
        
    
    def before_backward(self, strategy: Strategy) -> None:
        with torch.no_grad():
            if hasattr(strategy.model, "global_forward"):
                output = strategy.model.global_forward(strategy.mb_x)[0]
                loss = strategy.loss_fn(output, strategy.mb_y)
                strategy.loss += self.beta * loss
            else:
                raise AttributeError("Model does not have a global_forward method.")