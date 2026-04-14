import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaGate(nn.Module):
    def __init__(self, alpha_dim=512, init_alpha=1.0):
        super().__init__()

        self.alpha_net = nn.Linear(alpha_dim * 2, alpha_dim)
        
        nn.init.constant_(self.alpha_net.bias, init_alpha)

    def forward(self, global_features, local_features):
        self.alpha = torch.sigmoid(self.alpha_net(torch.cat([global_features, local_features], dim=1)))
        self.delta = self.alpha * local_features
        return global_features + self.delta