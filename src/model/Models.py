import copy
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.blocks.Adapter import Adapter
from src.model.layers.AlphaGate import AlphaGate
from src.model.layers.PrototypeClassifier import PrototypeClassifier


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
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_dim, out_channels=32, kernel_size=7, stride=1, padding=1
        )
        self.norm1 = nn.GroupNorm(zznum_groups=8, num_channels=32)
        self.conv2 = nn.Conv1d(32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_dim, num_layers=3, batch_first=True
        )

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        x = x.permute(
            0, 2, 1
        )  # Change shape to (batch_size, seq_len, features) for LSTM
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class FCLModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        d_hat_global: int = 16,
        d_hat_local: int = 8,
        classifier_scale_init: float = 20.0,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim)
        self.adapter_global = Adapter(
            in_features=hidden_dim, down_features=d_hat_global
        )
        self.adapter_local = Adapter(in_features=hidden_dim, down_features=d_hat_local)
        self.alpha_gate = AlphaGate(embedding_dim=hidden_dim)
        self.classifier = PrototypeClassifier(
            embedding_dim=hidden_dim, scale_init=classifier_scale_init
        )

    def embed_both(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.feature_extractor(x)
        x_global = self.adapter_global(feats)
        delta_local = self.adapter_local.forward_delta(x_global)
        x_local = self.alpha_gate(x_global, delta_local)
        return x_local, x_global

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x_local, _ = self.embed_both(x)
        return x_local

    def frozen_global_embed_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        frozen_fe = copy.deepcopy(self.feature_extractor)
        frozen_ag = copy.deepcopy(self.adapter_global)
        for p in frozen_fe.parameters():
            p.requires_grad_(False)
        for p in frozen_ag.parameters():
            p.requires_grad_(False)

        def _embed_global(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                feats = frozen_fe(x)
                return frozen_ag(feats)

        return _embed_global

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))

    def get_global_arrays(self) -> dict:
        sd = {}
        for k, v in self.feature_extractor.state_dict().items():
            sd[f"feature_extractor.{k}"] = v
        for k, v in self.adapter_global.state_dict().items():
            sd[f"adapter_global.{k}"] = v
        return sd

    def set_global_arrays(self, state_dict: dict) -> None:
        fe_prefix, ag_prefix = "feature_extractor.", "adapter_global."
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
