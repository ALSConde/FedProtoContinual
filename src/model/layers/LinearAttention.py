import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    def __init__(self, d_q, d_kv, d_att, d_out, mem_size=16, eps=1e-6, activation="elu"):
        super().__init__()
        self.proj_Q = nn.Linear(d_q, d_att)
        self.proj_K = nn.Linear(d_kv, d_att)
        self.proj_V = nn.Linear(d_kv, d_att)
        self.out_proj = nn.Linear(d_att, d_out)
        self.eps = eps
        self.activation = activation

        if mem_size > 0:
            self.k_mem = nn.Parameter(torch.empty(mem_size, d_att))
            self.v_mem = nn.Parameter(torch.empty(mem_size, d_att))

            nn.init.orthogonal_(self.k_mem)
            nn.init.orthogonal_(self.v_mem)

    def elu_feature_map(self, x):
        return F.elu(x) + 1

    def relu_feature_map(self, x):
        return F.relu(x) + 1

    def activation_feature_map(self, x):
        if self.activation == "elu":
            return self.elu_feature_map(x)
        elif self.activation == "relu":
            return self.relu_feature_map(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, q, k, v):
        Q = self.activation_feature_map(self.proj_Q(q))  # (n, l, d)
        k = self.activation_feature_map(self.proj_K(k))  # (n, s, d)
        v = self.proj_V(v)

        # Append memory vectors to K and V
        if hasattr(self, "k_mem") and hasattr(self, "v_mem"):
            K = torch.cat([k, self.k_mem.expand(Q.size(0), -1, -1)], dim=1)
            V = torch.cat([v, self.v_mem.expand(Q.size(0), -1, -1)], dim=1)
        else:
            K, V = k, v

        # (n, d, d)
        KV = torch.einsum("nsd,nse->nde", K, V)

        # (n, d)
        K_sum = K.sum(dim=1)

        # (n, l)
        Z = 1.0 / (torch.einsum("nld,nd->nl", Q, K_sum) + self.eps)

        # (n, l, d)
        out = torch.einsum("nld,nde,nl->nle", Q, KV, Z)

        return self.out_proj(out)
        
