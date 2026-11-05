"""prd-reduction / prd-weight (magnitude of the PRD term) and the LossTermProbe diagnostics.

Run: python scripts/test_prd_controls.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.client.ClientTask import (
    LossTermProbe,
    _kd_term,
    provisional_prototypes,
    train_fn,
)
from src.model.Models import FCLModel
from src.model.layers.PrototypeMemory import PrototypeMemory
from src.utils.losses.PrototypeLosses import relation_distillation_loss

DEVICE = torch.device("cpu")
HIDDEN, WINDOW, CHANNELS = 64, 60, 6


def _model(num_classes=5):
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


def _ready():
    m = _model()
    provisional_prototypes(m, _loader([0, 1, 2, 3, 4]), DEVICE, set())
    return m


def _train(model, loader, **kw):
    memory = PrototypeMemory(HIDDEN, model.classifier.num_classes, DEVICE)
    args = dict(
        epochs=2,
        lr=1e-2,
        device=DEVICE,
        known_consolidated={0, 1, 2},
        lambda_kd=1.0,
        kd_mode="prd",
        kd_logit_scale=20.0,
    )
    args.update(kw)
    return train_fn(model, loader, memory, **args)


# ----------------------------------------------------------------------------------
def test_reduction():
    torch.manual_seed(0)
    h, h_t, protos = torch.randn(10, 8), torch.randn(10, 8), torch.randn(6, 8)
    s = relation_distillation_loss(h, h_t, protos, temperature=2.0, scale_student=5.0)
    m = relation_distillation_loss(
        h, h_t, protos, temperature=2.0, scale_student=5.0, reduction="mean"
    )
    assert abs(float(m) - float(s) / 6) < 1e-6
    # the default is the paper's sum
    assert torch.equal(
        s,
        relation_distillation_loss(
            h, h_t, protos, temperature=2.0, scale_student=5.0, reduction="sum"
        ),
    )
    # one prototype: both reductions coincide
    a = relation_distillation_loss(
        h, h_t, protos[:1], temperature=2.0, scale_student=5.0
    )
    b = relation_distillation_loss(
        h, h_t, protos[:1], temperature=2.0, scale_student=5.0, reduction="mean"
    )
    assert torch.allclose(a, b)
    try:
        relation_distillation_loss(h, h_t, protos, reduction="max")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown reduction must raise")
    # T^2 is part of both (the factor is applied, only K is (not) normalised)
    t1 = relation_distillation_loss(h, h_t, protos, temperature=1.0, scale_student=10.0)
    t2 = relation_distillation_loss(h, h_t, protos, temperature=2.0, scale_student=20.0)
    # same logits scale/T = 10 -> same distributions; the loss differs only by T^2
    assert abs(float(t2) / float(t1) - 4.0) < 1e-4
    print("reduction OK (sum default, mean = sum/K, T^2 applied in both)")


def test_prd_weight_in_kd_term():
    model = _ready()
    x, _ = next(iter(_loader([3, 4])))
    h, _ = model.embed_both(x)
    h_t = h.detach() + 0.3 * torch.randn_like(h)
    kw = dict(kd_temperature=2.0, kd_scale_mode="const", kd_logit_scale=20.0)
    kl = _kd_term(model, h, h_t, [0, 1, 2], "kl", **kw)
    prd = _kd_term(model, h, h_t, [0, 1, 2], "prd", **kw)
    for w in (0.0, 0.25, 3.0):
        both = _kd_term(model, h, h_t, [0, 1, 2], "kl+prd", prd_weight=w, **kw)
        assert abs(float(both) - float(kl) - w * float(prd)) < 1e-5, w
        only = _kd_term(model, h, h_t, [0, 1, 2], "prd", prd_weight=w, **kw)
        assert abs(float(only) - w * float(prd)) < 1e-5
    mean = _kd_term(model, h, h_t, [0, 1, 2], "prd", prd_reduction="mean", **kw)
    assert abs(float(mean) - float(prd) / 3) < 1e-6
    # prd_weight must not touch the KL term
    assert torch.allclose(
        _kd_term(model, h, h_t, [0, 1, 2], "kl", prd_weight=7.0, **kw), kl
    )
    print("prd_weight / prd_reduction in _kd_term OK")


def test_train_fn_controls_change_training():
    base_loss, base_w = None, None
    results = {}
    for name, kw in {
        "sum": {},
        "mean": dict(prd_reduction="mean"),
        "w0": dict(prd_weight=0.0),
        "w2": dict(prd_weight=2.0),
    }.items():
        m = _ready()
        loss = _train(m, _loader([3, 4]), **kw)
        assert math.isfinite(loss), name
        results[name] = [p.detach().clone() for p in m.parameters()]
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a, b))
    assert not same(results["sum"], results["mean"])
    assert not same(results["sum"], results["w2"])
    assert not same(results["sum"], results["w0"])
    # prd_weight=0 == no KD at all on the PRD term: identical to training without KD
    m = _ready()
    _train(m, _loader([3, 4]), lambda_kd=0.0)
    assert same(results["w0"], [p.detach().clone() for p in m.parameters()])
    for bad in ({"prd_reduction": "max"},):
        try:
            _train(_ready(), _loader([3, 4]), **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    del base_loss, base_w
    print("train_fn: reduction / weight change the training; weight 0 == no KD OK")


def test_probe_unit():
    probe = LossTermProbe()
    assert probe.names == ["cls", "proto", "kl", "prd"]
    h = torch.randn(6, 4, requires_grad=True)
    term = (h**2).sum()
    probe.record("cls", term, h, weight=0.5)
    probe.record("cls", term, h, weight=0.5)
    probe.record("prd", None, h)  # a term that did not run: ignored
    probe.record("nope", term, h)  # unknown name: ignored
    out = probe.summary()
    assert abs(out["lt_cls"] - 0.5 * float(term)) < 1e-5
    assert abs(out["lt_cls_grad"] - float((0.5 * 2 * h).norm())) < 1e-5
    assert out["lt_prd"] == 0.0 and out["lt_prd_grad"] == 0.0
    assert set(out) == {
        f"lt_{t}{s}" for t in LossTermProbe.TERMS for s in ("", "_grad")
    }
    probe.configure(True)
    assert len(probe.names) == 8 and all(
        n.endswith(("_sh", "_lo")) for n in probe.names
    )
    assert all(v == 0.0 for v in probe.summary().values())  # configure resets
    print("probe unit OK")


def test_probe_in_training():
    for split in (False, True):
        extra = dict(split_optimizers=split, lambda_kd_shared=1.0 if split else 0.0)
        # the probe never changes the training
        # build and train one model at a time: training consumes the global RNG, so two
        # models built up front would not see the same random state when they train
        plain = _ready()
        _train(plain, _loader([3, 4]), kd_mode="kl+prd", **extra)
        probe = LossTermProbe()
        probed = _ready()
        _train(probed, _loader([3, 4]), kd_mode="kl+prd", loss_probe=probe, **extra)
        assert all(
            torch.equal(a, b) for a, b in zip(plain.parameters(), probed.parameters())
        )
        out = probe.summary()
        expect = (
            {
                f"lt_{t}_{b}{s}"
                for t in LossTermProbe.TERMS
                for b in ("sh", "lo")
                for s in ("", "_grad")
            }
            if split
            else {f"lt_{t}{s}" for t in LossTermProbe.TERMS for s in ("", "_grad")}
        )
        assert set(out) == expect, split
        assert all(math.isfinite(v) for v in out.values())
        suffixes = ("_sh", "_lo") if split else ("",)
        for sfx in suffixes:
            assert out[f"lt_cls{sfx}"] > 0 and out[f"lt_cls{sfx}_grad"] > 0, (
                split,
                sfx,
            )
            assert out[f"lt_prd{sfx}"] > 0 and out[f"lt_prd{sfx}_grad"] > 0, (
                split,
                sfx,
            )
            assert out[f"lt_kl{sfx}"] > 0 and out[f"lt_kl{sfx}_grad"] > 0, (split, sfx)
    # the recorded PRD value is the weighted term: weight 3 -> 3x (same training otherwise
    # would differ, so compare the first-epoch value on identical models via the unit path)
    m = _ready()
    x, _ = next(iter(_loader([3, 4])))
    h, _ = m.embed_both(x)
    h_t = h.detach() + 0.3 * torch.randn_like(h)
    pr = LossTermProbe()
    kw = dict(kd_temperature=2.0, kd_scale_mode="const", kd_logit_scale=20.0)
    _kd_term(
        m, h, h_t, [0, 1, 2], "prd", prd_weight=3.0, probe=pr, probe_weight=2.0, **kw
    )
    ref = _kd_term(m, h, h_t, [0, 1, 2], "prd", **kw)
    assert abs(pr.summary()["lt_prd"] - 6.0 * float(ref)) < 1e-5  # lambda 2 * weight 3
    (g,) = torch.autograd.grad(6.0 * ref, h, retain_graph=True)
    assert abs(pr.summary()["lt_prd_grad"] - float(g.norm())) < 1e-4
    print(
        "probe in training (plain + split): constant keys, no side effect, weighted values OK"
    )


def test_probe_keys_constant_without_kd():
    """No teacher / no KD this round: the keys must still all be there (as zeros)."""
    probe = LossTermProbe()
    _train(_ready(), _loader([3, 4]), lambda_kd=0.0, loss_probe=probe)
    out = probe.summary()
    assert set(out) == {
        f"lt_{t}{s}" for t in LossTermProbe.TERMS for s in ("", "_grad")
    }
    assert out["lt_prd"] == 0.0 and out["lt_kl"] == 0.0 and out["lt_cls"] > 0
    print("probe keys constant without KD OK")


if __name__ == "__main__":
    test_reduction()
    test_prd_weight_in_kd_term()
    test_train_fn_controls_change_training()
    test_probe_unit()
    test_probe_in_training()
    test_probe_keys_constant_without_kd()
    print("\nall prd-control checks passed")
