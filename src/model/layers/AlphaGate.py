import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaGate(nn.Module):
    """
    @param init_alpha: initial value for the alpha parameter.

    A lower value will lead to a more impact of the local model, while a higher value lead to a strong influence of the global model.

    The alpha parameter is learnable and will be updated during training. EX:

    - if init_alpha is set to -4.0, the initial alpha will be approximately 0.49, meaning that the local and global models will have equal influence.
    - if init_alpha is set to 1.0, the initial alpha will be approximately 0.21 for the local model and 0.8 for the global model.
    - if init_alpha is set to 7.0, the initial alpha will be approximately 0.0009 for the local model and 0.9991 for the global model.
    """

    def __init__(self, init_alpha=1.0):
        super().__init__()

        self.alpha_param = nn.Parameter(torch.tensor(init_alpha))

    def forward(self):
        B = -F.softplus(self.alpha_param)
        alpha = torch.sigmoid(B)

        return alpha


if __name__ == "__main__":
    alpha_gate = AlphaGate(init_alpha=7.0)

    for _ in range(10):
        alpha = alpha_gate()
        print(f"Alpha: {alpha.item():.4f}")
        with torch.no_grad():
            alpha_gate.alpha_param.data += 0.5
