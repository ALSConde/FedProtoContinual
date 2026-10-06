import copy
import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.blocks.Adapter import Adapter, build_adapter_from_topology
from src.model.layers.AlphaGate import AlphaGate
from src.model.layers.PrototypeClassifier import PrototypeClassifier


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.0,
        max_len: int = 4096,
    ):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})."
            )
        self.embed = nn.Linear(input_dim, hidden_dim)
        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : hidden_dim // 2]
        self.register_buffer("pe", pe, persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)  # (B, T, hidden_dim)
        h = h + self.pe[: h.size(1)]
        h = self.encoder(h)
        return self.norm(h.mean(dim=1))


class LightweightResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_channels),
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_dim, out_channels=32, kernel_size=7, stride=1, padding=1
        )
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.conv2 = nn.Conv1d(32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.conv3 = nn.Conv1d(64, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(num_groups=8, num_channels=128)
 
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        x = F.relu(self.norm3(self.conv3(x)), inplace=True)

        x = self.dropout(x)

        x = x.permute(
            0, 2, 1
        )  # Change shape to (batch_size, seq_len, features) for transformer

        gru_out, _ = self.gru(x)
        return gru_out.mean(dim=1)


class FCLModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        d_hat_global: int = 16,
        d_hat_local: int = 8,
        classifier_scale_init: float = 20.0,
        a_max: int = 3,
        use_local_adapter: bool = True,
    ):
        super().__init__()
        self.use_local_adapter = use_local_adapter
        self.hidden_dim = hidden_dim
        self.d_hat_local = d_hat_local
        self.a_max = a_max
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim)
        self.adapter_global = Adapter(
            in_features=hidden_dim, down_features=d_hat_global
        )
        self.adapter_local = Adapter(in_features=hidden_dim, down_features=d_hat_local)
        self.alpha_gate = AlphaGate(embedding_dim=hidden_dim)
        self.classifier = PrototypeClassifier(
            embedding_dim=hidden_dim, scale_init=classifier_scale_init
        )

        self.incorporated_adapters = nn.ModuleList()

    def incorporated_delta(self, x_global: torch.Tensor) -> torch.Tensor:
        if len(self.incorporated_adapters) == 0:
            return torch.zeros_like(x_global)
        return sum(a.forward_delta(x_global) for a in self.incorporated_adapters)

    def embed_both(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.feature_extractor(x)
        x_global = self.adapter_global(feats)
        incorporated = self.incorporated_delta(x_global)
        x_shared = x_global + incorporated
        if not self.use_local_adapter:
            # Ablation: no personalization branch, local == shared embedding.
            return x_shared, x_shared
        delta_local = self.adapter_local.forward_delta(x_global)
        x_local = self.alpha_gate(x_global, delta_local) + incorporated
        return x_local, x_shared

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x_local, _ = self.embed_both(x)
        return x_local

    def local_contribution_ratio(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_local_adapter:
            return torch.zeros(x.shape[0], device=x.device)
        feats = self.feature_extractor(x)
        x_global = self.adapter_global(feats)
        incorporated = self.incorporated_delta(x_global)
        x_shared = x_global + incorporated
        delta_local = self.adapter_local.forward_delta(x_global)
        alpha = self.alpha_gate.alpha_vector()
        scaled_local = alpha * delta_local
        return scaled_local.norm(dim=-1) / (x_shared.norm(dim=-1) + 1e-8)

    def global_branch_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (
            self.feature_extractor,
            self.adapter_global,
            self.incorporated_adapters,
        ):
            params.extend(module.parameters())
        return params

    def frozen_global_embed_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        frozen_fe = copy.deepcopy(self.feature_extractor)
        frozen_ag = copy.deepcopy(self.adapter_global)
        frozen_incorp = copy.deepcopy(self.incorporated_adapters)
        for module in (frozen_fe, frozen_ag, frozen_incorp):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

        def _embed_global(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                feats = frozen_fe(x)
                x_global = frozen_ag(feats)
                if len(frozen_incorp) > 0:
                    x_global = x_global + sum(
                        a.forward_delta(x_global)
                        for a in frozen_incorp
                        if isinstance(a, Adapter)
                    )
                return x_global

        return _embed_global

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))

    def reset_local_branch(self) -> None:
        device = next(self.parameters()).device
        self.adapter_local = Adapter(
            in_features=self.hidden_dim, down_features=self.d_hat_local
        ).to(device)
        self.alpha_gate = AlphaGate(embedding_dim=self.hidden_dim).to(device)

    def load_incorporated_topology(self, topologies: list) -> None:
        if len(topologies) > self.a_max:
            raise RuntimeError(
                f"Receved {len(topologies)} topologies for incorporated adapters, "
                f"but a_max is set to {self.a_max}. Cannot load more than a_max incorporated adapters."
                "Plase check the server configuration or incorporation/substitution logic."
            )
        device = next(self.parameters(), None)
        device = device.device if device is not None else None
        self.incorporated_adapters = nn.ModuleList(
            build_adapter_from_topology(topo, device=device) for topo in topologies
        )

    def get_global_arrays(self) -> dict:
        sd = {}
        for k, v in self.feature_extractor.state_dict().items():
            sd[f"feature_extractor.{k}"] = v
        for k, v in self.adapter_global.state_dict().items():
            sd[f"adapter_global.{k}"] = v
        for i, adapter in enumerate(self.incorporated_adapters):
            for k, v in adapter.state_dict().items():
                sd[f"incorporated_adapter.{i}.{k}"] = v
        return sd

    def set_global_arrays(self, state_dict: dict) -> None:
        fe_prefix, ag_prefix, inc_prefix = (
            "feature_extractor.",
            "adapter_global.",
            "incorporated_adapter.",
        )
        fe_sd = {
            k[len(fe_prefix) :]: v
            for k, v in state_dict.items()
            if k.startswith(fe_prefix)
        }
        ag_sd = {
            k[len(ag_prefix) :]: v
            for k, v in state_dict.items()
            if k.startswith(ag_prefix)
        }
        self.feature_extractor.load_state_dict(fe_sd)
        self.adapter_global.load_state_dict(ag_sd)

        grouped: dict[int, dict] = {}
        for k, v in state_dict.items():
            if not k.startswith(inc_prefix):
                continue
            rest = k[len(inc_prefix) :]
            idx_str, sub_key = rest.split(".", 1)
            grouped.setdefault(int(idx_str), {})[sub_key] = v

        for idx in sorted(grouped.keys()):
            if idx >= len(self.incorporated_adapters):
                raise RuntimeError(
                    f"State dict contains weights for incorporated adapter index {idx}, but only "
                    f"{len(self.incorporated_adapters)} incorporated adapters. Call load_incorporated_topology() "
                    "before set_global_arrays()."
                )
            self.incorporated_adapters[idx].load_state_dict(grouped[idx])
