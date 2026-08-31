from typing import Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn.functional as F
from src.model.Models import FCLModel
from src.model.layers.PrototypeMemory import PrototypeMemory


# Function to load data -- With sintetic data for testing purposes
def load_data(
    partition_id: int,
    num_partitions: int,
    input_dim: int,
    num_classes_total: int,
    batch_size: int,
    n_samples: int = 200,
):
    torch.manual_seed(partition_id)
    classes_per_client = max(2, num_classes_total // num_partitions)
    start = (partition_id * classes_per_client) % num_classes_total
    client_classes = [
        (start + i) % num_classes_total for i in range(classes_per_client)
    ]

    y = torch.tensor(
        [client_classes[i % len(client_classes)] for i in range(n_samples)]
    )
    class_centers = (
        torch.randn(
            num_classes_total, input_dim, generator=torch.Generator().manual_seed(0)
        )
        * 3.0
    )
    X = torch.randn(n_samples, input_dim) + class_centers[y]

    dataset = TensorDataset(X, y)
    n_train = int(0.8 * n_samples)
    train_ds, val_ds = random_split(dataset, [n_train, n_samples - n_train])
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size)
    return trainloader, valloader, client_classes


def train_fn(
    model: nn.Module,
    trainloader: DataLoader,
    memory: PrototypeMemory,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    model.to(device)
    model.train()
    optmizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss, n_batches = 0.0, 0

    for _ in range(epochs):
        for (
            x,
            y,
        ) in trainloader:
            x, y = x.to(device), y.to(device)

            h = model.embed(x)
            memory.update(h, y)

            if model.classifier.num_classes == 0:
                continue  # cold start: without prototypes yet, just accumulate statistics

            optmizer.zero_grad()
            logits = model.classifier(h)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optmizer.step()

            running_loss += loss.item()
            n_batches += 1

    return running_loss / max(n_batches, 1)


def test_fn(model: nn.Module, valloader: DataLoader, device: torch.device):
    model.to(device)
    model.eval()
    correct, total, loss_sum, n_batches = 0, 0, 0.0, 0

    with torch.no_grad():
        for x, y in valloader:
            x, y = x.to(device), y.to(device)
            h = model.embed(x)

            if model.classifier.num_classes == 0:
                continue

            logits = model.classifier(h)
            loss_sum += F.cross_entropy(logits, y).item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            n_batches += 1

    if total == 0:
        return 0.0, 0.0
    return loss_sum / max(n_batches, 1), correct / total


def compute_expansion_signal(
    model: FCLModel,
    loader: DataLoader,
    known_consolidated: set,
    scale: Union[float, torch.Tensor],
    device: torch.device,
) -> Optional[dict]:
    model.eval()
    h_cons_list, y_cons_list = [], []
    h_new_list, y_new_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            h = model.embed(x)
            cons_mask = torch.tensor(
                [int(label) in known_consolidated for label in y.tolist()],
                device=device,
            )
            if cons_mask.any():
                h_cons_list.append(h[cons_mask])
                y_cons_list.append(y[cons_mask])
            if (~cons_mask).any():
                h_new_list.append(h[~cons_mask])
                y_new_list.append(y[~cons_mask])

        if not h_cons_list or not h_new_list:
            return None

        h_cons = torch.cat(h_cons_list) if h_cons_list else None
        y_cons = torch.cat(y_cons_list) if y_cons_list else None
        proto_cons = (
            model.classifier.prototypes[y_cons]
            if y_cons is not None and len(y_cons) > 0
            else None
        )

        h_new = torch.cat(h_new_list) if h_new_list else None
        y_new = torch.cat(y_new_list) if y_new_list else None
        proto_new = None

        if h_new is not None and len(h_new) > 0:
            h_new_norm = F.normalize(h_new, dim=1)
            proto_new = torch.zeros_like(h_new)
            for c in y_new.unique():
                mask = y_new == c
                proto_new[mask] = h_new_norm[mask].mean(dim=0, keepdim=True)

        logits, labels_for_ce = None, None
        if h_cons is not None and len(h_cons) > 0 and model.classifier.num_classes > 1:
            logits = model.classifier(h_cons)
            labels_for_ce = y_cons

        return {
            "h_cons": h_cons,
            "proto_cons": proto_cons,
            "h_new": h_new,
            "proto_new": proto_new,
            "logits": logits,
            "labels": labels_for_ce,
            "scale": scale,
            "num_seen_classes": max(model.classifier.num_classes, 1),
        }
