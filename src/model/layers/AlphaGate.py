import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaGate(nn.Module):
    def __init__(self, embedding_dim: int = 512, alpha_init=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.beta = nn.Parameter(torch.full((embedding_dim,), alpha_init))

    def forward(self, global_features, local_features):
        self.alpha = torch.sigmoid(self.beta) / 2
        return global_features + self.alpha * local_features

    def alpha_vector(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.beta) / 2

    def mean_alpha(self) -> float:
        return self.alpha_vector().mean().item()
