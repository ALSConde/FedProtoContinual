import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.AlphaGate import AlphaGate
from .layers.LinearAttention import LinearAttention
from .layers.PrototypeClassifier import PrototypeClassifier
from .layers.MultiHeadLinearAttention import MultiHeadLinearAttention
from .blocks.Adapter import Adapter


class LinearTransformerBlock(nn.Module):
    def __init__(
        self, dim, norm_dim, heads=8, mlp_ratio=2.0, activation="elu", mem_size=64
    ):
        super().__init__()
        self.activation = activation

        self.norm1 = nn.LayerNorm(norm_dim)
        self.attn = MultiHeadLinearAttention(
            embed_dim=dim, num_heads=heads, activation=activation, mem_size=mem_size
        )
        self.norm2 = nn.LayerNorm(norm_dim)

        internal_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, dim),
        )

    def forward(self, x):

        x = x + self.attn(self.norm1(x))

        x = x + self.mlp(self.norm2(x))

        return x


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


class ModelSharedEncoder2(nn.Module):
    def __init__(self, depth=2, heads=2):
        super().__init__()
        self.layer1 = ResidualBlock(3, 128, stride=1)
        self.layer2 = ResidualBlock(128, 256, stride=2)
        self.layer3 = LightweightResidualBlock(256, 512, stride=2)

        self.blocks = nn.ModuleList(
            [
                LinearTransformerBlock(
                    dim=512,
                    norm_dim=[64, 512],
                    heads=heads,
                    mlp_ratio=0.5,
                    activation="relu",
                    mem_size=64,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm([64, 512])
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        b, c, h, w = x.shape
        # (B, C, H, W) → (B, N, C)
        x = x.view(b, c, h * w).transpose(1, 2)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # x = x.mean(dim=1)
        x = F.gelu(self.fc(x))
        return x


class ModelSharedEncoder(nn.Module):
    def __init__(self):
        super(ModelSharedEncoder, self).__init__()

        self.layer1 = ResidualBlock(3, 128, stride=1)  # 32x32
        self.layer2 = ResidualBlock(128, 256, stride=2)  # 16x16
        self.layer3 = LightweightResidualBlock(256, 512, stride=2)  # 8x8

        self.spatial_attention = LinearAttention(
            d_q=512,
            d_kv=512,
            d_att=512,
            d_out=512,
            mem_size=64,
        )
        self.spatial_norm = nn.GroupNorm(num_groups=8, num_channels=512)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Spatial attention
        b, c, h, w = x.size()
        x_reshaped = x.view(b, c, h * w).transpose(1, 2)  # (b, h*w, c)
        attn_out = self.spatial_attention(
            x_reshaped, x_reshaped, x_reshaped
        )  # (b, h*w, c)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)  # (b, c, h, w)
        attn_out = self.spatial_norm(attn_out + x)

        x = self.pooling(attn_out)
        x = x.view(b, c)
        x = F.gelu(self.fc(x))

        return x


class ModelGlobalFeatures(nn.Module):
    def __init__(self):
        super(ModelGlobalFeatures, self).__init__()
        self.global_adapter = Adapter(in_features=512, down_features=128)

        self.attention = LinearAttention(
            d_q=self.global_adapter.up_proj.out_features,
            d_kv=self.global_adapter.up_proj.out_features,
            d_att=128,
            d_out=512,
            mem_size=64,
        )
        self.norm_q = nn.LayerNorm(normalized_shape=[512])
        self.norm_kv = nn.LayerNorm(normalized_shape=[512])

    def forward(self, x):

        q = self.norm_q(x).unsqueeze(1)  # (b, 1, 512)
        k = self.norm_kv(x).unsqueeze(1)  # (b, 1, 512)
        v = k
        h_attn = self.attention(q, k, v).squeeze(1)  # (b, 512)
        x = x + h_attn

        x = x + F.relu(self.global_adapter(x))

        return x


class ModelLocalFeatures(nn.Module):
    def __init__(self):
        super(ModelLocalFeatures, self).__init__()
        self.global_adapter = Adapter(in_features=512, down_features=128)

        self.attention = LinearAttention(
            d_q=self.global_adapter.up_proj.out_features,
            d_kv=self.global_adapter.up_proj.out_features,
            d_att=128,
            d_out=512,
            mem_size=64,
        )
        self.norm_q = nn.LayerNorm(normalized_shape=[512])
        self.norm_kv = nn.LayerNorm(normalized_shape=[512])

    def forward(self, x):
        q = self.norm_q(x).unsqueeze(1)  # (b, 1, 512)
        k = self.norm_kv(x).unsqueeze(1)  # (b, 1, 512)
        v = k
        h_attn = self.attention(q, k, v).squeeze(1)  # (b, 512)
        x = x + h_attn

        x = x + F.relu(self.global_adapter(x))

        return x


class ModelGlobalFeatures2(nn.Module):
    def __init__(self):
        super(ModelGlobalFeatures2, self).__init__()
        self.global_adapter = Adapter(in_features=512, down_features=128)

        self.attention = MultiHeadLinearAttention(
            embed_dim=512,
            num_heads=2,
            activation="relu",
            mem_size=64,
        )
        self.norm = nn.LayerNorm(normalized_shape=[64, 512])

    def forward(self, x):
        x = self.norm(x)  # (b, 64, 512)
        h_attn = self.attention(x)  # (b, 64, 512)
        x = x + h_attn

        x = x + F.relu(self.global_adapter(x))

        return x


class ModelLocalFeatures2(nn.Module):
    def __init__(self):
        super(ModelLocalFeatures2, self).__init__()
        self.global_adapter = Adapter(in_features=512, down_features=128)

        self.attention = MultiHeadLinearAttention(
            embed_dim=512,
            num_heads=2,
            activation="relu",
            mem_size=64,
        )
        self.norm = nn.LayerNorm(normalized_shape=[64, 512])

    def forward(self, x):
        x = self.norm(x)  # (b, 64, 512)
        h_attn = self.attention(x)  # (b, 64, 512)
        x = x + h_attn

        x = x + F.relu(self.global_adapter(x))

        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.shared_encoder = ModelSharedEncoder()
        self.global_features = ModelGlobalFeatures()
        self.local_features = ModelLocalFeatures()
        # self.shared_encoder = ModelSharedEncoder2(depth=1, heads=2)
        # self.global_features = ModelGlobalFeatures2()
        # self.local_features = ModelLocalFeatures2()
        self.gate = AlphaGate(alpha_init=0.0)
        self.classifier = PrototypeClassifier(embedding_dim=512)
        self.classifier.update_from_global(
            mu_global=torch.rand(10, 512), class_ids=torch.arange(10)
        )

    def forward(self, x):
        shared_rep = self.shared_encoder(x)
        global_feat = self.global_features(shared_rep)
        local_feat = self.local_features(global_feat)
        fused_feat = self.gate(global_feat, local_feat)
        # fused_feat = global_feat + local_feat
        out = self.classifier(fused_feat)
        # out = self.classifier(fused_feat.mean(dim=1))

        return out, fused_feat, global_feat, local_feat

    @torch.no_grad()
    def global_forward(self, x):
        shared_rep = self.shared_encoder(x)
        global_feat = self.global_features(shared_rep)
        out = self.classifier(global_feat)
        # out = self.classifier(global_feat.mean(dim=1))
        return out, global_feat
