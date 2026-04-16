import torch
import torch.nn as nn
import torchvision
import torchinfo as ti
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    RandomHorizontalFlip,
    RandomCrop,
)
import torch.nn.functional as F
from model.layers.PrototypeMemory import PrototypeMemory
from server.PrototypeAggregator import PrototypeAggregator
from model.Models import Model


def train(model, train_loader, optimizer, loss_fn, memory, device):
    model.train()
    memory.reset()

    total_loss = 0.0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, h, global_feat, local_feat = model(images)
        alpha_loss = (
            torch.mean(model.gate.alpha.pow(2)) * 0.01
            if hasattr(model.gate, "alpha")
            else 0.0
        )
        ortho_loss = torch.mean(
            torch.abs(F.cosine_similarity(global_feat, local_feat, dim=-1))
        )
        loss = (
            loss_fn(outputs, labels) + alpha_loss + ortho_loss
        )  # Regularização para incentivar o uso de local features

        loss.backward()
        optimizer.step()

        # Acumula embeddings para atualização dos protótipos
        # h já está detachado do grafo após o backward
        memory.update(h.detach(), labels)

        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = correct / len(train_loader.dataset)
    return avg_loss, avg_acc, alpha_loss, ortho_loss


def update_prototypes(model, memory, aggregator):
    """
    Simula a agregação federada localmente:
    acumula estatísticas da época e atualiza o classificador.
    No cenário federado este passo ocorre no servidor.
    """
    sum_h, counts, class_ids = memory.get_stats()

    if len(class_ids) == 0:
        return

    aggregator.aggregate([(sum_h, counts, class_ids)])
    mu_global, updated_ids = aggregator.get_prototypes_raw()

    model.classifier.update_from_global(
        mu_global=mu_global[updated_ids],
        class_ids=updated_ids,
    )


def test(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _, _ , _ = model.global_forward(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    avg_acc = correct / len(test_loader.dataset)
    return avg_loss, avg_acc


if __name__ == "__main__":
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # --- Dados ---
    data = torchvision.datasets.CIFAR10(
        root="./storage/train_data",
        train=True,
        download=True,
        transform=train_transforms,
    )
    data_test = torchvision.datasets.CIFAR10(
        root="./storage/test_data",
        train=False,
        download=True,
        transform=test_transforms,
    )
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=512, shuffle=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=512, shuffle=False, pin_memory=True
    )

    # --- Modelo ---
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    ti.summary(
        model,
        input_size=(512, 3, 32, 32),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "trainable",
        ],
    )
    exit(0)
    # --- Memória e agregador de protótipos ---
    # tau ≈ total de samples esperado por classe por época
    # CIFAR-10: 50000 samples / 10 classes = 5000 samples/classe
    memory = PrototypeMemory(embedding_dim=512, num_classes=10, device=device)
    aggregator = PrototypeAggregator(embedding_dim=512, num_classes=10, tau=5000.0)

    # --- Warming up prototypes ---
    print("Warming up prototypes with one pass through the training data...")
    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            (
                _,
                feat,
            ) = model(images.to(device))
            memory.update(feat, labels)  # Acumula estatísticas reais
    update_prototypes(model, memory, aggregator)
    memory.reset()

    # --- Loop de treino ---
    print("Starting training...")
    for epoch in range(10):
        train_loss, train_acc, alpha = train(
            model, train_loader, optimizer, loss_fn, memory, device
        )

        # Atualiza protótipos ao fim de cada época
        update_prototypes(model, memory, aggregator)

        test_loss, test_acc = test(model, test_loader, loss_fn, device)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  Alpha Loss: {alpha:.4f}  Scale: {model.classifier.scale.item():.4f} | "
            f"Test  Loss: {test_loss:.4f}  Acc: {test_acc:.4f} | "
            f"Prototypes: {aggregator.num_initialized_classes}/{aggregator.num_classes}"
        )
