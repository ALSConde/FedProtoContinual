import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaGate(nn.Module):
    def __init__(self, alpha_init=0.0):
        super().__init__()

        self.beta = nn.Parameter(torch.full((512,), alpha_init))

    def forward(self, global_features, local_features):
        self.alpha = torch.sigmoid(self.beta) / 2
        return global_features + self.alpha * local_features
