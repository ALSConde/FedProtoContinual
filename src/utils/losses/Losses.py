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


def split_by_know(y: torch.Tensor, known_consolidated: set) -> torch.Tensor:
    return torch.tensor(
        [int(label) in known_consolidated for label in y.tolist()],
        device=y.device,
    )


def local_class_prototypes(h_new: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
    h_n = F.normalize(h_new, dim=1)
    proto = torch.zeros_like(h_n)
    for c in y_new.unique():
        mask = y_new == c
        proto[mask] = h_n[mask].mean(dim=0, keepdim=True)
    return proto.detach()


def prototype_alignment_loss(
    h: torch.Tensor,
    y: torch.Tensor,
    global_prototypes: torch.Tensor,
    known_consolidated: set,
) -> Optional[torch.Tensor]:
    cons_mask = split_by_know(y, known_consolidated)
    terms = []

    if cons_mask.any():
        h_cons, y_cons = h[cons_mask], y[cons_mask]
        proto_cons = global_prototypes[y_cons]
        d_cons = normalized_sq_distance(h_cons, proto_cons, reduction="none")
        if d_cons is not None:
            terms.append(d_cons)

    if (~cons_mask).any():
        h_new, y_new = h[~cons_mask], y[~cons_mask]
        proto_new = local_class_prototypes(h_new, y_new)
        d_new = normalized_sq_distance(h_new, proto_new, reduction="none")
        if d_new is not None:
            terms.append(d_new)

    if not terms:
        return None
    return torch.cat(terms).mean()


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
