from core.Plugins import Plugins
from core.Strategy import Strategy
from torch.nn import functional as F


class CompactionLossPlugin(Plugins):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def before_backward(self, strategy: Strategy) -> None:
        target_prototypes = strategy.model.classifier.prototypes[strategy.mb_y]
        compact_loss = F.mse_loss(strategy.fused_feat, target_prototypes)
        strategy.loss += self.alpha * compact_loss
