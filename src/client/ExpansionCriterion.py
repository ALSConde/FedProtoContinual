import math
from typing import Optional, Union
import torch.nn.functional as F
import torch

from src.model.blocks.Adapter import Adapter


class ExpansionCriterion:
    def __init__(
        self,
        theta_exp: float = 0.3,
        w_cos_old: float = 0.4,
        w_cos_new: float = 0.4,
        w_entropy: float = 0.2,
        max_width_attempts: int = 3,
    ) -> None:
        assert (
            abs(w_cos_old + w_cos_new + w_entropy - 1.0) < 1e-6
        ), "The weights w_cos_old + w_cos_new + w_entropy have to sum 1."

        self.theta_exp = theta_exp
        self.base_weights = {"old": w_cos_old, "new": w_cos_new, "entropy": w_entropy}
        self.max_width_attempts = max_width_attempts

    @staticmethod
    def _cosine_gap(
        h: Optional[torch.Tensor], prototypes: Optional[torch.Tensor]
    ) -> Optional[float]:
        if h is None or prototypes is None or h.numel() == 0 or prototypes.numel() == 0:
            return None
        h_n = F.normalize(h, dim=1)
        p_n = F.normalize(prototypes, dim=1)

        # 0.5 * ||h_hat - p_hat||^2 = (1 - <h_hat, p_hat>)
        # cos_sim = (h_n * p_n)
        # return (1 - cos_sim).mean().item()
        sq_dist = F.mse_loss(h_n, p_n, reduction="none").sum(dim=1)
        return (0.5 * sq_dist).mean().item()

    @staticmethod
    def _scaled_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        scale: Union[float, torch.Tensor],
        num_seen_classes: int,
    ) -> Optional[float]:
        if logits is None or logits.numel() == 0 or num_seen_classes <= 1:
            return None
        ce = F.cross_entropy(logits, labels, reduction="mean")
        denom = scale * math.log(num_seen_classes)
        return (ce / denom).item()

    def compute(
        self,
        h_cons: Optional[torch.Tensor],
        proto_cons: Optional[torch.Tensor],
        h_new: Optional[torch.Tensor],
        proto_new: Optional[torch.Tensor],
        logits: torch.Tensor,
        labels: torch.Tensor,
        scale: Union[float, torch.Tensor],
        num_seen_classes: int,
    ) -> dict:
        terms = {
            "old": self._cosine_gap(h_cons, proto_cons),
            "new": self._cosine_gap(h_new, proto_new),
            "entropy": self._scaled_entropy(logits, labels, scale, num_seen_classes),
        }

        weights = dict(self.base_weights)
        available = {k: v for k, v in terms.items() if v is not None}
        missing_weight = sum(weights[k] for k, v in terms.items() if v is not None)

        if len(available) == 0:
            raise RuntimeError(
                "Terms not computable on this round "
                "(D_cons, D_new, and logits are empty)."
            )

        if missing_weight > 0:
            redistribute = missing_weight / len(available)
            for k in available:
                weights[k] += redistribute
        g = sum(weights[k] * terms[k] for k in available)

        return {
            "g": g,
            "g_cosine_old": terms["old"],
            "g_cosine_new": terms["new"],
            "g_entropy": terms["entropy"],
            "trigger": g > self.theta_exp,
        }

    def decide_expansion_type(
        self, adapter: Adapter, g_reduced_below_threshold: bool
    ) -> str:
        bottleneck_saturated = adapter.bottleneck_dim >= adapter.max_bottleneck
        width_ineffective = (
            adapter.width_expansions_since_reduction >= self.max_width_attempts
            and not g_reduced_below_threshold
        )
        depth_exhausted = (
            adapter.max_depth is not None and adapter.depth >= adapter.max_depth
        )
        if depth_exhausted:
            return "width"
        return "depth" if (bottleneck_saturated or width_ineffective) else "width"

    def step(
        self,
        adapter: Adapter,
        g: float,
        g_reduced_below_threshold: bool,
        **width_kwargs,
    ) -> Optional[str]:
        if g <= self.theta_exp:
            adapter.notify_g_reduced_below_threshold()
            return None

        kind = self.decide_expansion_type(adapter, g_reduced_below_threshold)
        if kind == "width":
            delta = adapter.compute_delta_d(g=g, **width_kwargs)
            adapter.expand_width(delta)
        else:
            adapter.expand_depth(None)
        return kind
