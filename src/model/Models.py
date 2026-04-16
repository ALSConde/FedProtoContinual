import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.AlphaGate import AlphaGate
from .layers.LinearAttention import LinearAttention
from .layers.PrototypeClassifier import PrototypeClassifier


class ModelSharedEncoder(nn.Module):
    def __init__(self):
        super(ModelSharedEncoder, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 128, kernel_size=3, stride=1, padding=1
        )  # 32x32x3 -> 32x32x128
        self.norm1 = nn.GroupNorm(num_groups=2, num_channels=128)
        self.conv2 = nn.Conv2d(
            128, 256, kernel_size=3, stride=2, padding=2
        )  # 32x32x128 -> 17x17x256
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=256)
        self.conv3 = nn.Conv2d(
            256, 512, kernel_size=3, stride=2, padding=2
        )  # 17x17x256 -> 10x10x512
        self.norm3 = nn.GroupNorm(num_groups=8, num_channels=512)

        self.spatial_attention = LinearAttention(
            d_q=self.conv3.out_channels,
            d_kv=self.conv3.out_channels,
            d_att=512,
            mem_size=64,
        )
        self.spatial_norm = nn.GroupNorm(num_groups=8, num_channels=512)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1024)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))

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
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)

        self.attention = LinearAttention(
            d_q=self.fc2.out_features,
            d_kv=self.fc1.out_features,
            d_att=512,
            mem_size=128,
        )
        self.norm_q = nn.LayerNorm(normalized_shape=[512])
        self.norm_kv = nn.LayerNorm(normalized_shape=[512])

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        q = self.norm_q(h2).unsqueeze(1)  # (b, 1, 512)
        k = self.norm_kv(h1).unsqueeze(1)  # (b, 1, 512)
        v = k
        h_attn = self.attention(q, k, v).squeeze(1)  # (b, 512)
        x = h2 + h_attn

        return x


class ModelLocalFeatures(nn.Module):
    def __init__(self):
        super(ModelLocalFeatures, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)

        self.attention = LinearAttention(
            d_q=self.fc2.out_features,
            d_kv=self.fc1.out_features,
            d_att=512,
            mem_size=128,
        )
        self.norm_q = nn.LayerNorm(normalized_shape=[512])
        self.norm_kv = nn.LayerNorm(normalized_shape=[512])

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        q = self.norm_q(h2).unsqueeze(1)  # (b, 1, 512)
        k = self.norm_kv(h1).unsqueeze(1)  # (b, 1, 512)
        v = k
        h_attn = self.attention(q, k, v).squeeze(1)  # (b, 512)
        x = h2 + h_attn

        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.shared_encoder = ModelSharedEncoder()
        self.global_features = ModelGlobalFeatures()
        self.local_features = ModelLocalFeatures()
        self.gate = AlphaGate(alpha_input=1024,alpha_dim=512, init_alpha=2.0)
        self.fusion_norm = nn.LayerNorm(normalized_shape=[512])
        self.fc = nn.Linear(512, 512)
        self.classifier = PrototypeClassifier(embedding_dim=512)

        self.classifier.update_from_global(
            mu_global=torch.rand(10, 512), class_ids=torch.arange(10)
        )

    def forward(self, x):
        shared_rep = self.shared_encoder(x)
        global_feat = self.global_features(shared_rep)
        local_feat = self.local_features(shared_rep)
        # fused_feat = global_feat
        fused_feat = self.gate(global_feat, local_feat, shared_rep)
        # fused_feat = self.fusion_norm(fused_feat)
        fused_feat = F.gelu(self.fc(fused_feat))
        out = self.classifier(fused_feat)

        return out, fused_feat, global_feat, local_feat

    def global_forward(self, x):
        shared_rep = self.shared_encoder(x)
        global_feat = self.global_features(shared_rep)
        fc_feat = self.fc(global_feat)
        out = self.classifier(fc_feat)
        return out, fc_feat
