import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Union
from .WDStats import WDStats


class WDLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.stats: WDStats = WDStats()
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def expand(self, n_new_neurons: int):
        if n_new_neurons <= 0:
            return
        new_weight = torch.empty(
            (n_new_neurons, self.in_features),
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        nn.init.kaiming_uniform_(new_weight)
        self.weight = nn.Parameter(torch.cat([self.weight, new_weight], dim=0))
        if self.bias is not None:
            new_bias = torch.empty(
                n_new_neurons, device=self.bias.device, dtype=self.bias.dtype
            )
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(new_bias, -bound, bound)
            self.bias = nn.Parameter(torch.cat([self.bias, new_bias], dim=0))
        self.out_features += n_new_neurons

    def expand_input(self, n_new_inputs: int):
        if n_new_inputs <= 0:
            return
        
        new_weight = torch.empty(
            (self.out_features, n_new_inputs),
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
        self.weight = nn.Parameter(torch.cat([self.weight, new_weight], dim=1))
        self.in_features += n_new_inputs

    # pytorch methods
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.linear(input, self.weight, self.bias)
        self.stats.record_activations(out)
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
