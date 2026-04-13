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
            3, 16, kernel_size=3, stride=1, padding=1
        )  # 32x32x3 -> 32x32x16
        self.norm1 = nn.LayerNorm(normalized_shape=[16, 32, 32])
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=2, padding=2
        )  # 32x32x16 -> 17x17x32
        self.norm2 = nn.LayerNorm(normalized_shape=[32, 17, 17])
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=2
        )  # 17x17x32 -> 10x10x64
        self.norm3 = nn.LayerNorm(normalized_shape=[64, 10, 10])

        self.spatial_attention = LinearAttention(
            d_q=self.conv3.out_channels,
            d_kv=self.conv3.out_channels,
            d_att=64,
            mem_size=16,
        )
        self.spatial_norm = nn.LayerNorm(normalized_shape=[64, 10, 10])

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

        return attn_out


class ModelGlobalFeatures(nn.Module):
    def __init__(self):
        super(ModelGlobalFeatures, self).__init__()
        self.fc1 = nn.Linear(64 * 10 * 10, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.attention = LinearAttention(
            d_q=self.fc2.out_features,
            d_kv=self.fc1.out_features,
            d_att=1024,
            mem_size=128,
        )
        self.norm = nn.LayerNorm(normalized_shape=[1024])

    def forward(self, x):
        x = x.flatten(1)  # Flatten
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        q = h2.unsqueeze(1)  # (b, 1, 512)
        k = h1.unsqueeze(1)  # (b, 1, 512)
        v = k
        h_attn = self.attention(q, k, v).squeeze(1)  # (b, 512)
        x = self.norm(h_attn)

        return x


class ModelLocalFeatures(nn.Module):
    def __init__(self):
        super(ModelLocalFeatures, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.attention = LinearAttention(
            d_q=self.fc2.out_features,
            d_kv=self.fc1.out_features,
            d_att=1024,
            mem_size=128,
        )
        self.norm = nn.LayerNorm(normalized_shape=[1024])

    def forward(self, x):
        x = x.flatten(1)  # Flatten
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        q = h2.unsqueeze(1)  # (b, 1, 512)
        k = h1.unsqueeze(1)  # (b, 1, 512)
        v = k
        h_attn = self.attention(q, k, v).squeeze(1)  # (b, 512)
        x = self.norm(h_attn)

        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.shared_encoder = ModelSharedEncoder()
        self.global_features = ModelGlobalFeatures()
        # self.local_features = ModelLocalFeatures()
        self.gate = AlphaGate(init_alpha=1.0)
        self.fc = nn.Linear(1024, 1024)
        self.classifier = PrototypeClassifier(embedding_dim=1024)

        self.classifier.update_from_global(
            mu_global=torch.rand(10, 1024), class_ids=torch.arange(10)
        )

    def forward(self, x):
        shared_rep = self.shared_encoder(x)
        global_feat = self.global_features(shared_rep)
        # local_feat = self.local_features(global_feat)
        # fused_feat = self.gate(global_feat, local_feat)
        fused_feat = self.gate(global_feat, torch.zeros_like(global_feat))
        fused_feat = self.fc(fused_feat)
        out = self.classifier(fused_feat)

        return out, fused_feat


def train(loss, model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        batch_loss = loss(outputs, labels)
        batch_loss.backward()
        optimizer.step()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_loss += batch_loss.item() * images.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = correct / len(train_loader.dataset)
    return avg_loss, avg_acc


def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            total_loss += batch_loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy
