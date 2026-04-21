from ...core.Plugins import Plugins
from ...core.Strategy import Strategy
import torch.nn.functional as F


class OrthoLossPlugin(Plugins):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def before_backward(self, strategy: Strategy):
        ortho_loss = (
            F.cosine_similarity(strategy.global_feat, strategy.local_feat, dim=-1)
            .pow(2)
            .mean()
        )

        strategy.loss += self.alpha * ortho_loss
