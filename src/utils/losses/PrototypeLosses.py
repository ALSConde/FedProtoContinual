"""Prototype-geometry losses: intra-class compactness and inter-class separation.

Everything here works on L2-normalised embeddings and on the classifier's prototypes,
which are CONSTANTS for the client (they come from the server's aggregation). Only the
embeddings receive gradient, except where a function says otherwise.

Where each idea comes from (see the reference papers of the project):

  ppp_loss               CoPE (De Lange & Tuytelaars, ICCV 2021), Eq. 2-6. Pseudo-
                         Prototypical Proxy loss: besides the class prototype, the OTHER
                         samples of the same class in the batch act as extra (stop-grad)
                         attractors, and samples of other classes act as repellors.
                         Intra-class pull and inter-class push in a single objective.
  proto_nce_loss         FPL / CPCL (Huang et al., CVPR 2023), Eq. 8, with in-batch
                         samples used as additional positives/negatives next to the
                         prototypes. With ONE prototype per class and no batch samples
                         CPCL is exactly the prototype cross-entropy with temperature tau.
  relation_distillation  PRD (Asadi et al., ICML 2023), Eq. 5-6. For every old prototype,
                         the softmax OVER THE BATCH of its similarities to the samples
                         must not change with respect to the teacher (KL student||teacher,
                         as written in the paper): it keeps old
                         prototypes meaningful while the representation moves, using
                         the new-class data as a proxy (no replay).
  embedding_geometry     diagnostics only (no gradient): how compact each class is around
                         its prototype and how much samples leak towards other prototypes.
"""

from typing import Optional, Union

import torch
import torch.nn.functional as F

# Finite stand-in for -inf: logsumexp/logaddexp backward is NaN on all(-inf) rows.
# With tau >= ~0.01 the logits are bounded by 1/tau (<= 100), so exp(-1e4) == 0 in fp32.
_NEG = -1.0e4
_NEG_THRESHOLD = -1.0e3  # a log-sum below this means "no competing class at all"


def valid_prototype_mask(
    prototypes: torch.Tensor, keep_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Boolean (C,) mask of the prototypes that actually exist (non-zero) and, when
    `keep_ids` is given (flag ce-local-classes), belong to the kept classes."""
    valid = prototypes.norm(dim=1) > 0
    if keep_ids is not None:
        keep = torch.zeros_like(valid)
        keep[keep_ids] = True
        valid = valid & keep
    return valid


def _prob_complement(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """1 - a/(a+b) = b/(a+b), computed from log a and log b."""
    return torch.exp(log_b - torch.logaddexp(log_a, log_b))


def ppp_loss(
    h: torch.Tensor,
    y: torch.Tensor,
    prototypes: torch.Tensor,
    valid: torch.Tensor,
    tau: float = 0.1,
) -> Optional[torch.Tensor]:
    """Pseudo-Prototypical Proxy loss (CoPE), cosine version, fully vectorised.

    For a sample i of class c:
      attractors  A_i = {p_c} U {f_j : y_j = c, j != i}   (pseudo-prototypes: stop-grad)
      repellors   R_i = {p_c, f_i}
    and, for a proxy q standing for class c and any embedding f,
      P(c | f, q) = exp(f.q/tau) / (exp(f.q/tau) + sum_{k != c} exp(f.p_k/tau)).
    The objective is the negative log-likelihood of
      P_i = E_{q in A_i}[P(c | f_i, q)]  *  prod_{j : y_j != c} (1 - E_{q in R_i}[P(c | f_j, q)])
    i.e. sample i must be recognised as c through ANY of its attractors (intra-class
    compactness, also against the other windows of the batch) and the samples of other
    classes must NOT be recognised as c through the prototype or through f_i itself
    (inter-class separation, also sample against sample). Loss = -(1/B) * sum of the logs.

    Classes whose prototype does not exist yet (`valid` False) contribute no prototype
    attractor/repellor, and are not competitors either. Returns None when nothing can be
    computed (e.g. no valid prototype at all and no competitor).
    """
    batch = h.size(0)
    if batch == 0 or not bool(valid.any()):
        return None
    hn = F.normalize(h, dim=1)
    pn = F.normalize(prototypes, dim=1)
    num_classes = pn.size(0)

    s_p = (hn @ pn.t()) / tau  # (B, C) similarity to every prototype
    s_p_valid = s_p.masked_fill(~valid.unsqueeze(0), _NEG)

    # log D[j, i] = log sum_{k valid, k != y_i} exp(s_p[j, k])  (competitors of sample j
    # when the proxy stands for the class of sample i)
    not_own = torch.arange(num_classes, device=h.device).unsqueeze(0) != y.unsqueeze(
        1
    )  # (B_i, C)
    s3 = s_p_valid.unsqueeze(1).expand(batch, batch, num_classes)
    s3 = s3.masked_fill(~not_own.unsqueeze(0), _NEG)
    log_d = torch.logsumexp(s3, dim=2)  # (B_j, B_i)
    has_comp = log_d > _NEG_THRESHOLD

    eye = torch.eye(batch, dtype=torch.bool, device=h.device)
    same = y.unsqueeze(0) == y.unsqueeze(1)  # (B, B)
    own_valid = valid[y]  # (B,)

    # ---- attractor term: -log E_{q in A_i}[P(c | f_i, q)] -------------------------
    log_d_self = log_d.diagonal()  # (B,)
    a_proto = s_p.gather(1, y.unsqueeze(1)).squeeze(1)  # f_i . p_{y_i} / tau
    sim_ss = (hn @ hn.detach().t()) / tau  # attractor embeddings are stop-grad
    att_mask = same & ~eye
    p_proto = torch.exp(a_proto - torch.logaddexp(a_proto, log_d_self)) * own_valid
    p_pseudo = torch.exp(
        sim_ss - torch.logaddexp(sim_ss, log_d_self.unsqueeze(1))
    ) * att_mask.to(hn.dtype)
    n_att = own_valid.to(hn.dtype) + att_mask.to(hn.dtype).sum(dim=1)
    has_att = n_att > 0
    p_i = (p_proto + p_pseudo.sum(dim=1)) / n_att.clamp_min(1.0)
    loss_att = -(torch.log(p_i.clamp_min(1e-8)) * has_att).sum()

    # ---- repellor term: -sum_{j : y_j != y_i} log(1 - E_{q in R_i}[P(y_i | f_j, q)]) ---
    log_a_proto = s_p[:, y]  # (B_j, B_i): f_j . p_{y_i} / tau
    log_a_samp = (hn @ hn.t()) / tau  # (B_j, B_i): repellor f_i keeps its gradient
    q_proto = _prob_complement(log_a_proto, log_d) * own_valid.unsqueeze(0).to(hn.dtype)
    q_samp = _prob_complement(log_a_samp, log_d)
    n_rep = own_valid.to(hn.dtype) + 1.0  # (B_i,)
    q_ji = (q_proto + q_samp) / n_rep.unsqueeze(0)  # = 1 - E_R[P]
    pair_mask = (~same) & has_comp
    loss_rep = -(torch.log(q_ji.clamp_min(1e-8)) * pair_mask).sum()

    return (loss_att + loss_rep) / batch


def proto_nce_loss(
    h: torch.Tensor,
    y: torch.Tensor,
    prototypes: torch.Tensor,
    valid: torch.Tensor,
    tau: float = 0.1,
    detach_batch: bool = True,
) -> Optional[torch.Tensor]:
    """Prototype InfoNCE with in-batch proxies (FPL's CPCL, Eq. 8, extended).

        L_i = -log( sum_{q in Pos_i} e^{s(f_i,q)} / ( sum_{q in Pos_i} + sum_{q in Neg_i} ) )

    Pos_i = {p_{y_i}} U {f_j : y_j = y_i, j != i};  Neg_i = {p_k : k != y_i} U {f_j : y_j != y_i}.
    Following FPL the positives are SUMMED inside the log (a sample is happy as long as it
    is close to ANY positive); this is the term that makes the loss tolerant to several
    modes per class (e.g. one per subject), unlike a single-prototype cross-entropy.
    `detach_batch=True` lets the gradient flow only through the anchor f_i (every sample is
    the anchor of its own term, so pair repulsion is still covered symmetrically).
    Samples without any positive are skipped. Returns None if no term is defined.
    """
    batch = h.size(0)
    if batch == 0:
        return None
    hn = F.normalize(h, dim=1)
    pn = F.normalize(prototypes, dim=1)
    num_classes = pn.size(0)
    hb = hn.detach() if detach_batch else hn

    s_p = (hn @ pn.t()) / tau  # (B, C)
    s_b = (hn @ hb.t()) / tau  # (B, B)
    logits = torch.cat([s_p, s_b], dim=1)  # (B, C+B)

    eye = torch.eye(batch, dtype=torch.bool, device=h.device)
    same = y.unsqueeze(0) == y.unsqueeze(1)
    cls = torch.arange(num_classes, device=h.device).unsqueeze(0)
    own = cls == y.unsqueeze(1)  # (B, C)
    valid_row = valid.unsqueeze(0)

    pos_mask = torch.cat([own & valid_row, same & ~eye], dim=1)
    neg_mask = torch.cat([(~own) & valid_row, ~same], dim=1)
    has_pos = pos_mask.any(dim=1)
    if not bool(has_pos.any()):
        return None

    log_pos = torch.logsumexp(logits.masked_fill(~pos_mask, _NEG), dim=1)
    log_all = torch.logsumexp(logits.masked_fill(~(pos_mask | neg_mask), _NEG), dim=1)
    per_sample = -(log_pos - log_all) * has_pos
    return per_sample.sum() / has_pos.sum().clamp_min(1)


def relation_distillation_loss(
    h_student: torch.Tensor,
    h_teacher: torch.Tensor,
    prototypes_student: Optional[torch.Tensor],
    prototypes_teacher: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
    scale_student: Union[float, torch.Tensor] = 1.0,
    scale_teacher: Optional[Union[float, torch.Tensor]] = None,
) -> Optional[torch.Tensor]:
    """Prototype-sample relation distillation (PRD, Eq. 5-6).

    Two prototype sets, as in the paper's notation, row k of both referring to the SAME class:
      prototypes_student   P^t_k      current prototypes, paired with the current model f_t;
      prototypes_teacher   P^{t-1}_k  prototypes frozen at the end of the previous task,
                                      paired with the previous model f_{t-1}.
    For each old class k the similarities to the B samples of the batch are turned into a
    distribution OVER THE BATCH,
        P_t(k)_i     = softmax_i( s * cos(P^t_k,     f_t(x_i))     / T )
        P_{t-1}(k)_i = softmax_i( s * cos(P^{t-1}_k, f_{t-1}(x_i)) / T )
    and the student's must match the teacher's. Contrast with `distillation_loss_kl`, which
    builds a distribution over PROTOTYPES for each sample: PRD preserves the relative
    ordering of the samples around every old prototype, so old prototypes stay in a
    consistent place of the moving embedding space.

    `prototypes_teacher=None` reuses the student's prototypes (detached) for the teacher: the
    case where the prototypes did not move between t-1 and t, e.g. the FPC client, whose
    prototypes are constants received from the server. In the paper the old prototypes are
    learned parameters that this very loss updates (it is how they follow the representation),
    so there P^t_k != P^{t-1}_k after the first step; here that happens whenever
    `prototypes_student` carries gradient (requires_grad), which flows back to it. The teacher
    side never receives gradient.

    Direction: KL(student || teacher) = sum_i q_s(i) [log q_s(i) - log q_t(i)], exactly as in
    the paper (Eq. 6, KL(P_t(k) || P_{t-1}(k)) with t the current model and t-1 the
    previous one). Note this is the reverse of the usual distillation direction
    KL(teacher || student): it is mode-seeking, so the student is penalised mostly where it
    puts mass on samples the teacher considers far from the prototype. SUMMED over the old
    prototypes, as in Eq. 6 of the paper, and scaled by T^2 like the other KD term (the
    paper has no T^2). Because of the sum the magnitude grows with the number of old
    prototypes, so lambda-kd needs to be tuned accordingly.
    """
    if prototypes_student is None or prototypes_student.shape[0] < 1:
        return None
    if h_student.size(0) < 2:
        return None  # a softmax over one sample carries no relation
    if prototypes_teacher is None:
        prototypes_teacher = prototypes_student.detach()
    elif prototypes_teacher.shape != prototypes_student.shape:
        raise ValueError(
            "prototypes_student and prototypes_teacher must hold the same classes in the "
            f"same order, got {tuple(prototypes_student.shape)} vs "
            f"{tuple(prototypes_teacher.shape)}."
        )
    else:
        prototypes_teacher = prototypes_teacher.detach()
    if scale_teacher is None:
        scale_teacher = scale_student
    p_s = F.normalize(prototypes_student, dim=1)  # P^t_k
    p_t = F.normalize(prototypes_teacher, dim=1)  # P^{t-1}_k
    sim_s = (F.normalize(h_student, dim=1) @ p_s.t()) * (scale_student / temperature)
    sim_t = (F.normalize(h_teacher, dim=1) @ p_t.t()) * (scale_teacher / temperature)
    log_q_s = F.log_softmax(sim_s, dim=0)  # normalise over the batch, per prototype
    log_q_t = F.log_softmax(sim_t, dim=0)
    # KL(student || teacher): F.kl_div(input, target) = target * (log target - input)
    kl_per_proto = F.kl_div(log_q_t, log_q_s, reduction="none", log_target=True).sum(
        dim=0
    )
    return kl_per_proto.sum() * (temperature**2)


@torch.no_grad()
def embedding_geometry(
    h: torch.Tensor,
    y: torch.Tensor,
    prototypes: torch.Tensor,
    old_ids: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Intra/inter-class diagnostics of a set of embeddings against the prototypes.

    intra_cos      mean cos(h_i, p_{y_i})                       (compactness; higher = tighter)
    margin         mean [cos(h_i, p_{y_i}) - max_{k != y_i} cos(h_i, p_k)]   (>0 = separated)
    nn_error       fraction of samples whose nearest prototype is not their own class
    leak_old       (if old_ids) mean over samples of max_{k in old} cos(h_i, p_k): how much the
                   current embeddings intrude on prototypes of classes seen before
    leak_old_rate  (if old_ids) fraction of samples whose nearest prototype is an OLD class
                   that is not their own (always 1 for samples of an old class if wrong)
    proto_cos_max / proto_cos_mean   largest / mean cosine between two distinct valid
                   prototypes (inter-class separation of the prototypes themselves)
    Only samples whose own prototype exists enter the sample-level numbers.
    """
    out: dict[str, float] = {}
    valid = prototypes.norm(dim=1) > 0
    n_valid = int(valid.sum())
    if n_valid >= 2:
        pn_v = F.normalize(prototypes[valid], dim=1)
        pair = pn_v @ pn_v.t()
        off = ~torch.eye(n_valid, dtype=torch.bool, device=pair.device)
        out["proto_cos_max"] = float(pair[off].max())
        out["proto_cos_mean"] = float(pair[off].mean())
    if h.numel() == 0 or n_valid == 0:
        return out
    hn = F.normalize(h, dim=1)
    pn = F.normalize(prototypes, dim=1)
    cos = hn @ pn.t()  # (B, C)
    cos_valid = cos.masked_fill(~valid.unsqueeze(0), -2.0)
    rows = torch.arange(len(y), device=h.device)
    has_own = valid[y]
    if not bool(has_own.any()):
        return out
    own = cos[rows, y]
    other = cos_valid.clone()
    other[rows, y] = -2.0
    best_other = other.max(dim=1).values
    sel = has_own & (best_other > -2.0)
    if bool(sel.any()):
        out["intra_cos"] = float(own[has_own].mean())
        out["margin"] = float((own - best_other)[sel].mean())
        out["nn_error"] = float((best_other[sel] > own[sel]).float().mean())
    if old_ids is not None and old_ids.numel() > 0:
        old_mask = torch.zeros_like(valid)
        old_mask[old_ids] = True
        old_mask = old_mask & valid
        if bool(old_mask.any()):
            cos_old = cos.masked_fill(~old_mask.unsqueeze(0), -2.0)
            best_old = cos_old.max(dim=1).values
            not_old_sample = ~old_mask[y]  # samples of classes that are NOT old
            if bool(not_old_sample.any()):
                out["leak_old"] = float(best_old[not_old_sample].mean())
                nearest = cos_valid.argmax(dim=1)
                out["leak_old_rate"] = float(
                    (old_mask[nearest] & (nearest != y))[not_old_sample].float().mean()
                )
    return out
