from typing import Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn.functional as F
from src.model.Models import FCLModel
from src.model.layers.PrototypeMemory import PrototypeMemory
from src.utils.losses.Losses import (
    distillation_loss,
    distillation_loss_kl,
    local_class_prototypes,
    prototype_alignment_loss,
    split_by_know,
)


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
    model: FCLModel,
    trainloader: DataLoader,
    memory: PrototypeMemory,
    epochs: int,
    lr: float,
    device: torch.device,
    known_consolidated: Optional[set] = None,
    lambda_proto: float = 1.0,
    lambda_kd: float = 0.5,
    kd_mode: str = "kl",
    kd_temperature: float = 2.0,
) -> float:
    if kd_mode not in ("kl", "embedding_mse"):
        raise ValueError(
            f"Invalid kd_mode: {kd_mode}. Must be 'kl' or 'embedding_mse'."
        )

    model.to(device)
    model.train()
    known_consolidated = known_consolidated or set()
    known_sorted = sorted(known_consolidated)

    embed_global = model.frozen_global_embed_fn()

    optmizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss, n_batches = 0.0, 0

    for _ in range(epochs):
        for (
            x,
            y,
        ) in trainloader:
            x, y = x.to(device), y.to(device)

            h, h_shared = model.embed_both(x)
            memory.update(h_shared, y)

            if model.classifier.num_classes == 0:
                continue  # cold start: without prototypes yet, just accumulate statistics

            optmizer.zero_grad()

            logits = model.classifier(h)
            loss = F.cross_entropy(logits, y)

            l_proto = prototype_alignment_loss(
                h, y, model.classifier.prototypes, known_consolidated
            )
            if l_proto is not None:
                loss += lambda_proto * l_proto

            h_global = embed_global(x)
            if kd_mode == "embedding_mse":
                l_kd = distillation_loss(h, h_global)
            else:
                reference_prototypes = (
                    model.classifier.prototypes[known_sorted]
                    if len(known_sorted) >= 2
                    else None
                )
                l_kd = distillation_loss_kl(
                    h, h_global, reference_prototypes, temperature=kd_temperature
                )
            if l_kd is not None:
                loss += lambda_kd * l_kd

            loss.backward()
            optmizer.step()

            running_loss += loss.item()
            n_batches += 1

    return running_loss / max(n_batches, 1)


def embed_with_extra_incorporated(
    model: FCLModel, x: torch.Tensor, extra_adapter: Optional[nn.Module] = None
):
    feats = model.feature_extractor(x)
    x_global = model.adapter_global(feats)
    incorporated = model.incorporated_delta(x_global)
    if extra_adapter is not None:
        incorporated += extra_adapter.forward_delta(x_global)
    delta_local = model.adapter_local.forward_delta(x_global)
    x_local = model.alpha_gate(x_global, delta_local)
    x_final = x_local + incorporated
    x_shared = x_global + incorporated
    return x_final, x_shared


def evaluate_with_candidate(
    model: FCLModel,
    candidate: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
):
    model.to(device)
    model.eval()
    correct, total, loss_sum, n_batches = 0, 0, 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if model.classifier.num_classes == 0:
                continue
            h, _ = embed_with_extra_incorporated(model, x, candidate)
            logits = model.classifier(h)
            loss_sum += F.cross_entropy(logits, y).item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            n_batches += 1

    if total == 0:
        return 0.0, 0.0
    return loss_sum / max(n_batches, 1), correct / total


def short_local_adaptation(
    model: FCLModel,
    candidate: nn.Module,
    adapt_loader: DataLoader,
    lr: float,
    device: torch.device,
    max_steps: int,
):
    model.to(device)
    model.train()

    trainable_params = list(model.adapter_local.parameters()) + list(
        model.alpha_gate.parameters()
    )
    frozen_modules = [
        model.feature_extractor,
        model.adapter_global,
        model.incorporated_adapters,
        model.classifier,
        candidate,
    ]
    saved_requires_grad = []
    for module in frozen_modules:
        for p in module.parameters():
            saved_requires_grad.append((p, p.requires_grad))
            p.requires_grad_(False)

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    steps_done = 0
    try:
        if model.classifier.num_classes == 0:
            for x, y in adapt_loader:
                if steps_done >= max_steps:
                    break
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                h, _ = embed_with_extra_incorporated(model, x, candidate)
                logits = model.classifier(h)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                steps_done += 1
    finally:
        for p, requires_grad in saved_requires_grad:
            p.requires_grad_(requires_grad)

    return steps_done


def vote_on_candidate(
    model: FCLModel,
    candidate: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    adapt_steps: int,
    vote_margin: float,
) -> tuple[float, float, float]:
    _, acc_before = evaluate_with_candidate(model, candidate, val_loader, device)
    short_local_adaptation(
        model, candidate, train_loader, lr=lr, device=device, max_steps=adapt_steps
    )
    _, acc_after = evaluate_with_candidate(model, candidate, val_loader, device)

    vote = 1.0 if (acc_after - acc_before) > vote_margin else 0.0
    return vote, acc_before, acc_after


def test_fn(model: FCLModel, valloader: DataLoader, device: torch.device):
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
            cons_mask = split_by_know(y, known_consolidated)
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
        proto_new = (
            local_class_prototypes(h_new, y_new)
            if h_new is not None and len(h_new) > 0 and y_new is not None
            else None
        )

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
