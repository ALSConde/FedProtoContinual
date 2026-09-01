from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.model.layers.WDLayer import WDLayer


class Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        down_features: int,
        max_bottleneck: Optional[int] = None,
        near_identity_eps: float = 1e-3,
        max_depth: Optional[int] = None,
    ):
        super(Adapter, self).__init__()
        self.in_features = in_features
        self.max_bottleneck = (
            max_bottleneck if max_bottleneck is not None else down_features * 4
        )
        self.near_identity_eps = near_identity_eps
        self.max_depth = max_depth if max_depth is not None else 3

        self.down_stages = nn.ModuleList([WDLayer(in_features, down_features)])
        self.up_proj = WDLayer(down_features, in_features)

        self.width_expansions_since_reduction: int = 0

        layer = self.down_stages[0]
        assert isinstance(layer, WDLayer)
        self._near_identity_init(layer)  # self.down_stages[0] == layer
        self._near_identity_init(self.up_proj)

    @property
    def depth(self) -> int:
        return len(self.down_stages)

    @property
    def bottleneck_dim(self) -> int:
        return self.down_stages[-1].out_features

    def forward_delta(self, x) -> torch.Tensor:
        h = x
        for stage in self.down_stages:
            h = F.relu(stage(h))
        return self.up_proj(h)

    def forward(self, x):
        return x + self.forward_delta(x)

    # Width expansion logic
    def compute_delta_d(
        self,
        g: float,
        w_loss: float = 1.0,
        w_local: float = 1.0,
        f_grow: float = 0.25,
        delta_min: int = 1,
        delta_max: int = 16,
        activation_threshold: float = 0.05,
    ) -> int:
        layer = self.down_stages[-1]
        assert isinstance(layer, WDLayer)
        u = layer.stats.usage_ratio(threshold=activation_threshold)
        c_grow = w_loss * g + w_local * u
        raw = f_grow * c_grow * self.bottleneck_dim
        delta = int(round(raw))
        return max(delta_min, min(delta, delta_max))

    def expand_width(self, delta_d: int) -> int:
        if delta_d <= 0:
            return 0

        last_stage = self.down_stages[-1]

        assert isinstance(last_stage, WDLayer)
        last_stage.expand(delta_d)
        self.up_proj.expand_input(delta_d)
        self._near_zero_init_new_rows(last_stage, delta_d)

        self.width_expansions_since_reduction += 1

        return delta_d

    def notify_g_reduced_below_threshold(self) -> None:
        self.width_expansions_since_reduction = 0

    # Depth expansion logic
    def expand_depth(self, new_bottleneck_prime: Optional[int]) -> None:
        if self.max_depth is not None and self.depth >= self.max_depth:
            raise RuntimeError(f"Cannot expand depth beyond max_depth={self.max_depth}")

        last_stage = self.down_stages[-1]
        assert isinstance(last_stage, WDLayer)
        W = last_stage.weight.data
        b = last_stage.bias.data if last_stage.bias is not None else None
        d_hat, d_in = W.shape

        max_rank = min(d_hat, d_in)
        rank = (
            max_rank
            if new_bottleneck_prime is None
            else min(new_bottleneck_prime, max_rank)
        )

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

        min_singular = S.min().item()
        if min_singular < -1e-3:
            raise RuntimeError(
                f"Negative singular value encountered during depth expansion: {min_singular}"
            )

        S = torch.clamp(
            S, min=1e-4
        )  # Avoid numerical issues with very small singular values
        sqrt_S = torch.sqrt(S)
        W_a = sqrt_S.unsqueeze(1) * Vh  # (rank, d_in)
        W_b = U * sqrt_S.unsqueeze(0)  # (d_hat, rank)

        stage_a = WDLayer(
            d_in, rank, bias=(b is not None), device=W.device, dtype=W.dtype
        )
        stage_b = WDLayer(
            rank, d_hat, bias=(b is not None), device=W.device, dtype=W.dtype
        )

        with torch.no_grad():
            stage_a.weight.copy_(W_a)
            stage_b.weight.copy_(W_b)
            if b is not None:
                stage_a.bias.zero_()
                stage_b.bias.copy_(b)

        self.down_stages[-1] = stage_a
        self.down_stages.append(stage_b)

    # Utils
    # Quasi-identity initialization for the down and up projections. This is important to ensure that the adapter starts as a near-identity function, allowing the model to retain its original behavior before fine-tuning.
    def _near_identity_init(self, layer: WDLayer):
        nn.init.normal_(layer.weight, mean=0.0, std=self.near_identity_eps)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _near_zero_init_new_rows(self, layer: WDLayer, n_new: int):
        with torch.no_grad():
            layer.weight[-n_new:].normal_(0.0, self.near_identity_eps)
            if layer.bias is not None:
                layer.bias[-n_new:].zero_()

    def _near_zero_init_new_cols(self, layer: WDLayer, n_new: int):
        with torch.no_grad():
            layer.weight[:, -n_new:].normal_(0.0, self.near_identity_eps)
