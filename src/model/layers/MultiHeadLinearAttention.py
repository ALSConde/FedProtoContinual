import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLinearAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        mem_size=16,
        eps=1e-6,
        activation="elu",
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps
        self.activation = activation

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if mem_size > 0:
            self.k_mem = nn.Parameter(torch.randn(num_heads, mem_size, self.head_dim))

            self.v_mem = nn.Parameter(torch.randn(num_heads, mem_size, self.head_dim))

            nn.init.orthogonal_(self.k_mem)
            nn.init.orthogonal_(self.v_mem)

    def feature_map(self, x):
        if self.activation == "elu":
            return F.elu(x) + 1
        elif self.activation == "relu":
            return F.relu(x) + 1
        else:
            raise ValueError()

    def forward(self, x):

        B, N, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # (B,N,D) -> (B,H,N,Dh)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q = self.feature_map(Q)
        K = self.feature_map(K)

        if hasattr(self, "k_mem"):

            K_mem = self.k_mem.unsqueeze(0).expand(B, -1, -1, -1)
            V_mem = self.v_mem.unsqueeze(0).expand(B, -1, -1, -1)

            K = torch.cat([K, K_mem], dim=2)
            V = torch.cat([V, V_mem], dim=2)

        # KV = Σ φ(K)Vᵀ
        # (B,H,Dh,Dh)
        KV = torch.einsum("bhnd,bhne->bhde", K, V)

        # (B,H,Dh)
        K_sum = K.sum(dim=2)

        # (B,H,N)
        Z = 1.0 / (torch.einsum("bhnd,bhd->bhn", Q, K_sum) + self.eps)

        # (B,H,N,Dh)
        out = torch.einsum("bhnd,bhde,bhn->bhne", Q, KV, Z)

        # (B,N,H,Dh)
        out = out.transpose(1, 2)

        # (B,N,D)
        out = out.reshape(B, N, D)

        return self.out_proj(out)
