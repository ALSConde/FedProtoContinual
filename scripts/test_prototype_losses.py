"""Checks for src/utils/losses/PrototypeLosses.py and its wiring in ClientTask.

The vectorised PPP / NCE losses are compared (value AND gradient) against deliberately
naive, loop-based implementations written straight from the definitions in the papers.
Run: python scripts/test_prototype_losses.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.client.ClientTask import (
    VALID_CLS_LOSSES,
    VALID_KD_MODES,
    _cls_loss,
    _kd_term,
    prototype_geometry_metrics,
    provisional_prototypes,
    train_fn,
)
from src.model.Models import FCLModel
from src.model.layers.PrototypeMemory import PrototypeMemory
from src.utils.losses.PrototypeLosses import (
    embedding_geometry,
    ppp_loss,
    proto_nce_loss,
    relation_distillation_loss,
    valid_prototype_mask,
)

DEVICE = torch.device("cpu")
HIDDEN, WINDOW, CHANNELS = 64, 60, 6


# ----------------------------------------------------------------------------------
# naive references (double loops, straight from the definitions)
# ----------------------------------------------------------------------------------
def _naive_ppp(h, y, protos, valid, tau):
    B = h.size(0)
    hn = F.normalize(h, dim=1)
    pn = F.normalize(protos, dim=1)
    C = pn.size(0)

    def posterior(f, proxy, c, cls_ok):
        """P(c | f, proxy) = e^{f.q/tau} / (e^{f.q/tau} + sum_{k != c, valid} e^{f.p_k/tau})"""
        num = torch.exp(f @ proxy / tau)
        den = sum(
            (torch.exp(f @ pn[k] / tau) for k in range(C) if k != c and valid[k]),
            start=torch.zeros((), dtype=h.dtype),
        )
        return num / (num + den) if float(den.detach()) > 0 else None

    total = torch.zeros((), dtype=h.dtype)
    for i in range(B):
        c = int(y[i])
        attractors = []
        if valid[c]:
            attractors.append(pn[c])
        attractors += [hn[j].detach() for j in range(B) if j != i and int(y[j]) == c]
        if attractors:
            probs = [posterior(hn[i], q, c, valid) for q in attractors]
            probs = [
                p if p is not None else torch.ones((), dtype=h.dtype) for p in probs
            ]
            total = total - torch.log(torch.stack(probs).mean().clamp_min(1e-8))
        repellors = ([pn[c]] if valid[c] else []) + [hn[i]]
        for j in range(B):
            if int(y[j]) == c:
                continue
            ps = [posterior(hn[j], q, c, valid) for q in repellors]
            if any(p is None for p in ps):
                continue  # no competitor for sample j: posterior degenerate, skipped
            total = total - torch.log((1 - torch.stack(ps).mean()).clamp_min(1e-8))
    return total / B


def _naive_nce(h, y, protos, valid, tau):
    B = h.size(0)
    hn = F.normalize(h, dim=1)
    hb = hn.detach()
    pn = F.normalize(protos, dim=1)
    C = pn.size(0)
    losses = []
    for i in range(B):
        c = int(y[i])
        pos, neg = [], []
        for k in range(C):
            if valid[k]:
                (pos if k == c else neg).append(hn[i] @ pn[k] / tau)
        for j in range(B):
            if j == i:
                continue
            (pos if int(y[j]) == c else neg).append(hn[i] @ hb[j] / tau)
        if not pos:
            continue
        lp = torch.logsumexp(torch.stack(pos), 0)
        la = torch.logsumexp(torch.stack(pos + neg), 0)
        losses.append(-(lp - la))
    return torch.stack(losses).mean()


def _random_case(seed, B=9, C=5, D=8, n_classes_in_batch=3, invalid=(), ce_keep=None):
    g = torch.Generator().manual_seed(seed)
    # float64: the loop references accumulate float32 rounding of ~1e-4, which would hide
    # (or fake) real logic differences; the logic is what these comparisons are about
    h = torch.randn(B, D, generator=g).double()
    protos = torch.randn(C, D, generator=g).double()
    for k in invalid:
        protos[k] = 0.0
    classes = torch.randperm(C, generator=g)[:n_classes_in_batch]
    y = classes[torch.randint(0, n_classes_in_batch, (B,), generator=g)]
    keep = None if ce_keep is None else torch.tensor(ce_keep)
    return h, y, protos, valid_prototype_mask(protos, keep)


# ----------------------------------------------------------------------------------
def test_ppp_matches_naive():
    cases = [
        dict(seed=0),
        dict(seed=1, invalid=(0, 3)),  # some prototypes not created yet
        dict(seed=2, ce_keep=[0, 1, 2, 3]),  # ce-local-classes scope
        dict(seed=3, B=16, C=7, n_classes_in_batch=4),
        dict(seed=4, B=6, C=6, n_classes_in_batch=5, invalid=(1,)),
    ]
    for kw in cases:
        h, y, protos, valid = _random_case(**kw)
        for tau in (0.1, 0.5):
            h1 = h.clone().requires_grad_(True)
            h2 = h.clone().requires_grad_(True)
            fast = ppp_loss(h1, y, protos, valid, tau)
            slow = _naive_ppp(h2, y, protos, valid, tau)
            assert abs(float(fast) - float(slow)) < 1e-8, (
                kw,
                tau,
                float(fast),
                float(slow),
            )
            fast.backward()
            slow.backward()
            assert torch.allclose(h1.grad, h2.grad, atol=1e-8, rtol=1e-6), (kw, tau)
    print("ppp == naive loops (value + gradient) OK")


def test_nce_matches_naive():
    for kw in (
        dict(seed=0),
        dict(seed=1, invalid=(2,)),
        dict(seed=2, ce_keep=[1, 2, 3]),
    ):
        h, y, protos, valid = _random_case(**kw)
        for tau in (0.1, 0.3):
            h1 = h.clone().requires_grad_(True)
            h2 = h.clone().requires_grad_(True)
            fast = proto_nce_loss(h1, y, protos, valid, tau)
            slow = _naive_nce(h2, y, protos, valid, tau)
            assert abs(float(fast) - float(slow)) < 1e-8, (
                kw,
                tau,
                float(fast),
                float(slow),
            )
            fast.backward()
            slow.backward()
            assert torch.allclose(h1.grad, h2.grad, atol=1e-8, rtol=1e-6), (kw, tau)
    print("nce == naive loops (value + gradient) OK")


def test_geometry_ordering():
    """Both losses must rank a compact+separated batch below a collapsed / mixed one."""
    torch.manual_seed(0)
    C, D = 4, 16
    protos = F.normalize(torch.randn(C, D), dim=1)
    valid = valid_prototype_mask(protos)
    y = torch.arange(C).repeat_interleave(4)
    good = protos[y] + 0.05 * torch.randn(len(y), D)  # tight around own prototype
    mixed = protos[(y + 1) % C] + 0.05 * torch.randn(
        len(y), D
    )  # tight but on the WRONG one
    collapsed = F.normalize(torch.ones(1, D), dim=1).repeat(
        len(y), 1
    )  # all on one point
    for fn in (ppp_loss, proto_nce_loss):
        lg, lm, lc = (
            float(fn(t, y, protos, valid, 0.1)) for t in (good, mixed, collapsed)
        )
        assert lg < lc < lm or lg < lm and lg < lc, (fn.__name__, lg, lm, lc)
        assert lg < 0.5 * min(lm, lc), (fn.__name__, lg, lm, lc)
    print("losses rank good < collapsed/mixed OK")


def test_pseudo_prototypes_matter():
    """The point of PPP: with a stale/wrong prototype, in-batch same-class samples still pull
    the class together; a plain prototype CE would not see that structure."""
    torch.manual_seed(1)
    D = 16
    protos = F.normalize(torch.randn(3, D), dim=1)
    valid = valid_prototype_mask(protos)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    centre = F.normalize(torch.randn(2, D), dim=1)
    h = (centre[y] + 0.02 * torch.randn(8, D)).requires_grad_(
        True
    )  # tight classes, off-prototype
    loss = ppp_loss(h, y, protos, valid, 0.1)
    loss.backward()
    g = h.grad.clone()
    # gradient on a class-0 sample should point (descent direction) towards class-0 mates
    mates = F.normalize(h[1:4].detach().mean(0), dim=0)
    descent = -g[0]
    cos_mates = float(F.normalize(descent, dim=0) @ mates)
    assert cos_mates > 0, cos_mates
    print("ppp pulls towards in-batch mates OK")


def test_degenerate_inputs():
    h = torch.randn(4, 8)
    y = torch.tensor([0, 0, 1, 1])
    no_protos = torch.zeros(3, 8)
    v = valid_prototype_mask(no_protos)
    assert ppp_loss(h, y, no_protos, v) is None
    # nce still defined with zero prototypes (batch proxies only)
    val = proto_nce_loss(h, y, no_protos, v)
    assert val is not None and torch.isfinite(val)
    # a single valid prototype (own class only): no competitor prototype, still finite
    one = torch.zeros(3, 8)
    one[0] = torch.randn(8)
    v1 = valid_prototype_mask(one)
    for fn in (ppp_loss, proto_nce_loss):
        out = fn(h.clone().requires_grad_(True), y, one, v1, 0.1)
        assert out is None or torch.isfinite(out), fn.__name__
        if out is not None:
            out.backward()
    # singletons only: no positive, no pseudo-prototype
    y_unique = torch.tensor([0, 1, 2, 3])
    p4 = torch.randn(4, 8)
    out = ppp_loss(h, y_unique, p4, valid_prototype_mask(p4))
    assert out is not None and torch.isfinite(out)
    print("degenerate inputs OK (no NaN)")


def test_relation_distillation():
    torch.manual_seed(0)
    protos = torch.randn(3, 8)
    h = torch.randn(10, 8)
    same = relation_distillation_loss(
        h, h.clone(), protos, temperature=2.0, scale_student=5.0
    )
    assert abs(float(same)) < 1e-6
    student = (h + 0.5 * torch.randn_like(h)).requires_grad_(True)
    teacher = h.clone()
    loss = relation_distillation_loss(
        student, teacher, protos, temperature=2.0, scale_student=5.0
    )
    assert float(loss) > 0
    loss.backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0
    # direction: must be KL(student || teacher) as in the PRD paper (Eq. 6), not the reverse
    hs = student.detach()
    q_s = F.softmax(
        F.normalize(hs, dim=1) @ F.normalize(protos, dim=1).t() * (5.0 / 2.0), dim=0
    )
    q_t = F.softmax(
        F.normalize(teacher, dim=1) @ F.normalize(protos, dim=1).t() * (5.0 / 2.0),
        dim=0,
    )
    kl_st = (q_s * (q_s.log() - q_t.log())).sum() * 4.0  # student || teacher
    kl_ts = (q_t * (q_t.log() - q_s.log())).sum() * 4.0  # teacher || student
    got = float(
        relation_distillation_loss(
            hs, teacher, protos, temperature=2.0, scale_student=5.0
        )
    )
    assert abs(got - float(kl_st)) < 1e-5, (got, float(kl_st))
    assert abs(float(kl_st) - float(kl_ts)) > 1e-4  # the two directions really differ
    assert abs(got - float(kl_ts)) > 1e-4
    # summed over prototypes (paper Eq. 6): duplicating the prototype set doubles the loss
    one = float(
        relation_distillation_loss(
            hs, teacher, protos, temperature=2.0, scale_student=5.0
        )
    )
    two = float(
        relation_distillation_loss(
            hs, teacher, torch.cat([protos, protos]), temperature=2.0, scale_student=5.0
        )
    )
    assert abs(two - 2 * one) < 1e-5, (one, two)
    # a global rotation of the student around ALL prototypes' relations: scaling h is a no-op
    assert (
        abs(
            float(
                relation_distillation_loss(
                    3.0 * h, h, protos, temperature=2.0, scale_student=5.0
                )
            )
        )
        < 1e-6
    )
    # ordering sensitivity: swapping two samples' embeddings changes the loss
    swapped = h.clone()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert (
        float(
            relation_distillation_loss(
                swapped, h, protos, temperature=2.0, scale_student=5.0
            )
        )
        > 0
    )
    # ---- two prototype sets (paper: P^t_k for the student, P^{t-1}_k for the teacher) ----
    p_prev = protos.clone()
    p_now = (protos + 0.4 * torch.randn_like(protos)).requires_grad_(
        True
    )  # moved since t-1
    f_now = (h + 0.2 * torch.randn_like(h)).requires_grad_(True)
    f_prev = h.clone()
    got2 = relation_distillation_loss(f_now, f_prev, p_now, p_prev, 2.0, 5.0)
    nrm = lambda t: F.normalize(t, dim=1)
    q_s2 = F.softmax(
        nrm(f_now.detach()) @ nrm(p_now.detach()).t() * 2.5, dim=0
    )  # P_t(k)
    q_t2 = F.softmax(nrm(f_prev) @ nrm(p_prev).t() * 2.5, dim=0)  # P_{t-1}(k)
    manual = (q_s2 * (q_s2.log() - q_t2.log())).sum() * 4.0
    assert abs(float(got2) - float(manual)) < 1e-5, (float(got2), float(manual))
    # using P^t for the teacher too (the old single-set behaviour) gives a different value
    same_set = relation_distillation_loss(f_now, f_prev, p_now, None, 2.0, 5.0)
    assert abs(float(got2) - float(same_set)) > 1e-4
    # None == passing the same (detached) set explicitly
    explicit = relation_distillation_loss(
        f_now, f_prev, p_now, p_now.detach(), 2.0, 5.0
    )
    assert torch.allclose(same_set, explicit)
    # gradient reaches the student's prototypes (the paper's "old prototypes follow f") ...
    got2.backward()
    assert p_now.grad is not None and float(p_now.grad.abs().sum()) > 0
    # ... and never the teacher's
    p_prev_g = p_prev.clone().requires_grad_(True)
    relation_distillation_loss(f_now.detach(), f_prev, protos, p_prev_g, 2.0, 5.0)
    assert p_prev_g.grad is None
    try:
        relation_distillation_loss(f_now, f_prev, p_now, p_prev[:2], 2.0, 5.0)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched prototype sets must raise")
    # a learnable student prototype can absorb the drift: moving P^t towards where the
    # teacher's relations are lowers the loss (this is what constants cannot do)
    free = p_prev.clone().requires_grad_(True)
    drifted = (h + 1.0 * torch.randn_like(h)).detach()
    base = float(
        relation_distillation_loss(drifted, h, free.detach(), p_prev, 2.0, 5.0)
    )
    opt = torch.optim.SGD([free], lr=0.5)
    for _ in range(30):
        opt.zero_grad()
        relation_distillation_loss(drifted, h, free, p_prev, 2.0, 5.0).backward()
        opt.step()
    assert float(relation_distillation_loss(drifted, h, free, p_prev, 2.0, 5.0)) < base
    assert relation_distillation_loss(h[:1], h[:1], protos) is None
    assert relation_distillation_loss(h, h, None) is None
    # a single old prototype is enough for PRD (the per-sample KL needs >= 2)
    assert (
        relation_distillation_loss(
            student, teacher, protos[:1], temperature=2.0, scale_student=5.0
        )
        is not None
    )
    print("relation distillation OK")


def test_embedding_geometry():
    protos = torch.eye(4)
    y = torch.tensor([2, 2, 3])
    h = torch.tensor([[0, 0, 1.0, 0], [0, 0, 1.0, 0], [0, 0, 0.6, 0.8]])
    geo = embedding_geometry(h, y, protos, old_ids=torch.tensor([0, 1]))
    assert abs(geo["proto_cos_max"]) < 1e-6 and abs(geo["proto_cos_mean"]) < 1e-6
    # samples 0,1: cos own=1, best other=0 -> margin 1 ; sample 2: own=0.8, other=0.6 -> 0.2
    assert abs(geo["margin"] - (1 + 1 + 0.2) / 3) < 1e-5
    assert abs(geo["intra_cos"] - (1 + 1 + 0.8) / 3) < 1e-5
    assert geo["nn_error"] == 0.0
    assert abs(geo["leak_old"]) < 1e-6 and geo["leak_old_rate"] == 0.0
    # now push the third sample onto an OLD prototype
    h2 = h.clone()
    h2[2] = torch.tensor([0.9, 0.0, 0.0, 0.1])
    geo2 = embedding_geometry(h2, y, protos, old_ids=torch.tensor([0, 1]))
    assert geo2["nn_error"] > 0.3 and geo2["leak_old_rate"] > 0.3
    assert geo2["leak_old"] > 0.2
    # no prototype at all -> empty
    assert embedding_geometry(h, y, torch.zeros(4, 4)) == {}
    print("embedding geometry OK")


# ----------------------------------------------------------------------------------
# wiring through ClientTask
# ----------------------------------------------------------------------------------
def _model(num_classes):
    torch.manual_seed(0)
    m = FCLModel(input_dim=CHANNELS, hidden_dim=HIDDEN)
    m.classifier._expand(num_classes)
    return m


def _loader(classes, per_class=12, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in classes:
        centre = torch.randn(CHANNELS, 1, generator=g) * 2.0
        xs.append(centre + 0.3 * torch.randn(per_class, CHANNELS, WINDOW, generator=g))
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    return DataLoader(
        TensorDataset(torch.cat(xs), torch.cat(ys)), batch_size=batch_size
    )


def _train(model, loader, **kw):
    memory = PrototypeMemory(HIDDEN, max(model.classifier.num_classes, 1), DEVICE)
    args = dict(epochs=2, lr=1e-3, device=DEVICE)
    args.update(kw)
    return train_fn(model, loader, memory, **args), memory


def test_dispatch_and_flags():
    assert "ppp" in VALID_CLS_LOSSES and "nce" in VALID_CLS_LOSSES
    assert "prd" in VALID_KD_MODES and "kl+prd" in VALID_KD_MODES
    from src.utils.ablation import resolve_ablation_flags

    for kind in ("ppp", "nce"):
        f = resolve_ablation_flags({"cls-loss": kind})
        assert f.cls_loss == kind and f"cls-loss={kind}" in f.describe()
    model = _model(5)
    provisional_prototypes(model, _loader([0, 1, 2, 3, 4]), DEVICE, set())
    x, y = next(iter(_loader([3, 4])))
    h, _ = model.embed_both(x)
    logits = model.classifier(h)
    ce = _cls_loss(model, h, y, logits, None, "ce")
    assert torch.allclose(ce, F.cross_entropy(logits, y))  # default path untouched
    for kind in ("ppp", "nce"):
        v = _cls_loss(model, h, y, logits, None, kind, 0.2, 0.1)
        assert torch.isfinite(v) and v.requires_grad
    # zero prototypes: still a graph-attached zero for ppp
    empty = _model(5)
    v = _cls_loss(
        empty, h.detach().requires_grad_(True), y, logits, None, "ppp", 0.2, 0.1
    )
    assert float(v) == 0.0
    v.backward()
    print("dispatch / flags OK")


def test_train_fn_matrix():
    for kind in ("ppp", "nce"):
        for split in (False, True):
            for local in (False, True):
                model = _model(5)
                provisional_prototypes(model, _loader([0, 1, 2, 3, 4]), DEVICE, set())
                loss, _ = _train(
                    model,
                    _loader([3, 4]),
                    known_consolidated={0, 1, 2},
                    cls_loss=kind,
                    cls_tau=0.1,
                    split_optimizers=split,
                    local_class_ids=[3, 4] if local else None,
                )
                assert math.isfinite(loss), (kind, split, local)
    for kd_mode in ("prd", "kl+prd"):
        for split in (False, True):
            model = _model(5)
            provisional_prototypes(model, _loader([0, 1, 2, 3, 4]), DEVICE, set())
            loss, _ = _train(
                model,
                _loader([3, 4]),
                known_consolidated={0, 1, 2},
                lambda_kd=1.0,
                kd_mode=kd_mode,
                kd_logit_scale=20.0,
                split_optimizers=split,
            )
            assert math.isfinite(loss), (kd_mode, split)
    try:
        _train(_model(3), _loader([0, 1, 2]), known_consolidated=set(), kd_mode="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kd_mode must raise")
    print("train_fn OK (ppp/nce x split x local classes; prd / kl+prd x split)")


def test_kd_term_prd_modes():
    model = _model(5)
    provisional_prototypes(model, _loader([0, 1, 2, 3, 4]), DEVICE, set())
    x, _ = next(iter(_loader([3, 4])))
    h, _ = model.embed_both(x)
    h_t = h.detach() + 0.3 * torch.randn_like(h)
    kw = dict(kd_temperature=2.0, kd_scale_mode="const", kd_logit_scale=20.0)
    kl = _kd_term(model, h, h_t, [0, 1, 2], "kl", **kw)
    prd = _kd_term(model, h, h_t, [0, 1, 2], "prd", **kw)
    both = _kd_term(model, h, h_t, [0, 1, 2], "kl+prd", **kw)
    assert abs(float(both) - float(kl) - float(prd)) < 1e-5
    assert _kd_term(model, h, h_t, [], "prd", **kw) is None  # no old prototype
    assert _kd_term(model, h, h_t, [0], "kl", **kw) is None  # KL needs >= 2
    assert _kd_term(model, h, h_t, [0], "prd", **kw) is not None  # PRD needs only 1
    # explicit P^t / P^{t-1}: default (None) == student set reused as teacher set
    ref = model.classifier.prototypes[[0, 1, 2]]
    d0 = _kd_term(model, h, h_t, [0, 1, 2], "prd", **kw)
    d1 = _kd_term(
        model,
        h,
        h_t,
        [0, 1, 2],
        "prd",
        prd_prototypes_student=ref,
        prd_prototypes_teacher=ref.clone(),
        **kw,
    )
    assert torch.allclose(d0, d1)
    moved = (ref + 0.3 * torch.randn_like(ref)).requires_grad_(True)
    d2 = _kd_term(
        model,
        h,
        h_t,
        [0, 1, 2],
        "prd",
        prd_prototypes_student=moved,
        prd_prototypes_teacher=ref,
        **kw,
    )
    assert abs(float(d2) - float(d0)) > 1e-5
    d2.backward()
    assert moved.grad is not None
    print("kd modes OK")


def test_geometry_metrics_hook():
    model = _model(5)
    loader = _loader([0, 1, 2, 3, 4])
    provisional_prototypes(model, loader, DEVICE, set())
    out = prototype_geometry_metrics(model, _loader([3, 4]), DEVICE, {0, 1, 2})
    for key in (
        "geo_local_intra_cos",
        "geo_local_margin",
        "geo_local_leak_old",
        "geo_shared_leak_old_rate",
        "geo_proto_cos_max",
    ):
        assert key in out and math.isfinite(out[key]), (key, out)
    fresh = FCLModel(input_dim=CHANNELS, hidden_dim=HIDDEN)  # no classes yet
    empty = prototype_geometry_metrics(fresh, loader, DEVICE, set())
    assert empty["geo_defined"] == 0.0 and all(v == 0.0 for v in empty.values())
    # constant key set: Flower skips aggregation when clients' metric keys differ
    assert set(empty) == set(out), set(empty) ^ set(out)
    assert out["geo_defined"] == 1.0
    # the hook must leave train/eval mode as it found it
    model.train()
    prototype_geometry_metrics(model, _loader([3, 4]), DEVICE, set())
    assert model.training
    print("geometry metrics hook OK")


def test_learning_signal():
    """Smoke: a few epochs of ppp/nce on separable synthetic classes must reduce the loss."""
    for kind in ("ppp", "nce"):
        model = _model(4)
        loader = _loader([0, 1, 2, 3], per_class=16)
        provisional_prototypes(model, loader, DEVICE, set())
        first, _ = _train(
            model, loader, epochs=1, cls_loss=kind, cls_tau=0.1, lambda_proto=0.0
        )
        later, _ = _train(
            model, loader, epochs=15, cls_loss=kind, cls_tau=0.1, lambda_proto=0.0
        )
        assert later < first, (kind, first, later)
    print("learning signal OK")


if __name__ == "__main__":
    test_ppp_matches_naive()
    test_nce_matches_naive()
    test_geometry_ordering()
    test_pseudo_prototypes_matter()
    test_degenerate_inputs()
    test_relation_distillation()
    test_embedding_geometry()
    test_dispatch_and_flags()
    test_train_fn_matrix()
    test_kd_term_prd_modes()
    test_geometry_metrics_hook()
    test_learning_signal()
    print("\nall prototype-loss checks passed")
