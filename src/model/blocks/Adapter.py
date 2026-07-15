import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, in_features, down_features):
        super(Adapter, self).__init__()
        self.down_proj = nn.Linear(in_features, down_features)
        self.up_proj = nn.Linear(down_features, in_features)

    def forward(self, x):
        residual = x
        x = F.relu(self.down_proj(x))
        x = self.up_proj(x)

        return x + residual
