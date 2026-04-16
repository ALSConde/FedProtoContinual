import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaGate(nn.Module):
    def __init__(self, alpha_input, alpha_dim=512, init_alpha=0.0):
        super().__init__()

        self.alpha_net = nn.Linear(alpha_input, alpha_dim)
        
        nn.init.constant_(self.alpha_net.bias, init_alpha)

    def forward(self, global_features, local_features, shared_rep):
        self.alpha = torch.sigmoid(self.alpha_net(shared_rep))  # (b, 512)
        self.delta = self.alpha * local_features
        return global_features + self.delta