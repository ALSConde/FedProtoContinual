from functools import lru_cache
from typing import Optional
import torch.nn.functional as F
import torch


def normalized_sq_distance(
    h: Optional[torch.Tensor], target: Optional[torch.Tensor], reduction: str = "mean"
) -> Optional[torch.Tensor]:
    if h is None or target is None or h.numel() == 0 or target.numel() == 0:
        return None
    h_n = F.normalize(h, dim=1)
    t_n = F.normalize(target, dim=1)
    sq_dist = F.mse_loss(h_n, t_n, reduction="none").sum(dim=1)
    if reduction == "none":
        return sq_dist
    if reduction == "sum":
        return sq_dist.sum()
    return sq_dist.mean()


@lru_cache(maxsize=64)
def _known_ids(known: frozenset, device: torch.device) -> torch.Tensor:
    return torch.tensor(sorted(known), dtype=torch.long, device=device)


def split_by_know(y: torch.Tensor, known_consolidated: set) -> torch.Tensor:
    """Boolean mask: True where the label belongs to an already consolidated class.
    Vectorised (no y.tolist(), no per-sample Python loop, no host sync)."""
    if not known_consolidated:
        return torch.zeros_like(y, dtype=torch.bool)
    return torch.isin(y, _known_ids(frozenset(known_consolidated), y.device))


def local_class_prototypes(h_new: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
    """Per-sample target = mean of the L2-normalised embeddings of the sample's class."""
    h_n = F.normalize(h_new, dim=1)
    uniq, inv = torch.unique(y_new, return_inverse=True)
    sums = torch.zeros(
        uniq.numel(), h_n.size(1), device=h_n.device, dtype=h_n.dtype
    ).index_add_(0, inv, h_n.detach())
    counts = torch.bincount(inv, minlength=uniq.numel()).clamp_min(1)
    proto = sums / counts.unsqueeze(1).to(h_n.dtype)
    return proto[inv].detach()


def prototype_alignment_loss(
    h: torch.Tensor,
    y: torch.Tensor,
    global_prototypes: torch.Tensor,
    known_consolidated: set,
) -> Optional[torch.Tensor]:
    """Same as the previous masked/looped version, without boolean indexing.
    Consolidated classes are pulled towards the server prototype; new classes towards
    the detached mean of the batch's normalised embeddings of that class."""
    if h.numel() == 0:
        return None
    cons = split_by_know(y, known_consolidated)
    h_n = F.normalize(h, dim=1)

    rows = global_prototypes.size(0)
    t_cons = global_prototypes[y.clamp(max=rows - 1)]  # only used where cons is True

    # Group by class id directly. The classifier is expanded to cover every class of the
    # loader before training, so y < rows and no torch.unique is needed.
    idx = y.clamp(max=rows - 1)
    w_new = (~cons).to(h_n.dtype)
    sums = torch.zeros(rows, h_n.size(1), device=h.device, dtype=h_n.dtype).index_add_(
        0, idx, h_n.detach() * w_new.unsqueeze(1)
    )
    cnts = torch.zeros(rows, device=h.device, dtype=h_n.dtype).index_add_(0, idx, w_new)
    t_new = sums[idx] / cnts[idx].clamp_min(1.0).unsqueeze(1)

    target = torch.where(cons.unsqueeze(1), t_cons, t_new)
    d = F.mse_loss(h_n, F.normalize(target, dim=1), reduction="none").sum(dim=1)
    return d.mean()


def distillation_loss(
    h_local: torch.Tensor, h_global: torch.Tensor
) -> Optional[torch.Tensor]:
    return normalized_sq_distance(h_local, h_global.detach())


def distillation_loss_kl(
    h_local: torch.Tensor,
    h_global: torch.Tensor,
    reference_prototypes: Optional[torch.Tensor],
    temperature: float = 2.0,
) -> Optional[torch.Tensor]:
    if reference_prototypes is None or reference_prototypes.shape[0] < 2:
        return None

    p_n = F.normalize(reference_prototypes, dim=1)
    h_local_n = F.normalize(h_local, dim=1)
    h_global_n = F.normalize(h_global, dim=1)

    sim_local = (h_local_n @ p_n.t()) / temperature
    sim_global = (h_global_n @ p_n.t()) / temperature

    log_q_local = F.log_softmax(sim_local, dim=1)  # Student
    q_global = F.softmax(sim_global, dim=1)  # Teacher

    kl_loss = F.kl_div(log_q_local, q_global, reduction="batchmean")
    return kl_loss * (temperature**2)