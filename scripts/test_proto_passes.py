import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.client.ClientTask import (
    _build_ce_index,
    _build_optimizers,
    _build_param_groups,
    _class_ce,
    collect_prototype_stats,
    provisional_prototypes,
    _cls_loss,
    _kd_term,
    _logits_frozen_scale,
    _prototype_metric_loss,
    compute_interference,
    short_local_adaptation,
    train_fn,
)
from src.client.ExpansionCriterion import ExpansionCriterion
from src.model.Models import FCLModel
from src.server.ServerEvaluation import PredictionFlowMonitor, evaluate_global_model
from src.utils.losses.Losses import distillation_loss_kl
from src.model.blocks.Adapter import promote_to_incorporated
from src.model.layers.PrototypeMemory import PrototypeMemory
from src.utils.ablation import resolve_ablation_flags

DEVICE = torch.device("cpu")
HIDDEN, WINDOW, CHANNELS = 64, 60, 6


def _model(num_classes: int) -> FCLModel:
    torch.manual_seed(0)
    m = FCLModel(input_dim=CHANNELS, hidden_dim=HIDDEN)
    m.classifier._expand(num_classes)  # new rows start at zero, as in Client_app
    return m


def _loader(classes, per_class=12, batch_size=8, seed=0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in classes:
        center = torch.randn(CHANNELS, 1, generator=g) * 2.0
        xs.append(center + 0.3 * torch.randn(per_class, CHANNELS, WINDOW, generator=g))
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    return DataLoader(
        TensorDataset(torch.cat(xs), torch.cat(ys)), batch_size=batch_size
    )


def _manual_class_means(model, loader, num_classes):
    model.eval()
    sums = torch.zeros(num_classes, HIDDEN)
    counts = torch.zeros(num_classes)
    with torch.no_grad():
        for x, y in loader:
            _, h_shared = model.embed_both(x)
            sums.index_add_(0, y, h_shared)
            counts.index_add_(0, y, torch.ones(len(y)))
    return sums, counts


def test_flags():
    f = resolve_ablation_flags({})
    assert (f.ce_local_classes, f.proto_prepass, f.proto_postpass) == (False,) * 3
    f = resolve_ablation_flags(
        {"ce-local-classes": "true", "proto-prepass": True, "proto-postpass": "on"}
    )
    assert (f.ce_local_classes, f.proto_prepass, f.proto_postpass) == (True,) * 3
    assert "ce-local-classes=on" in f.describe()
    try:
        resolve_ablation_flags({"proto-prepass": "maybe"})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid boolean must raise")
    print("flags OK")


def test_ce_index():
    torch.manual_seed(1)
    logits = torch.randn(10, 6, requires_grad=True)
    y = torch.tensor([3, 4, 3, 5, 4, 5, 3, 4, 5, 3])

    assert _build_ce_index(None, 6, DEVICE) is None
    assert _build_ce_index([], 6, DEVICE) is None
    assert torch.allclose(_class_ce(logits, y, None), F.cross_entropy(logits, y))

    idx = _build_ce_index([5, 3, 4, 4], 6, DEVICE)  # unsorted + duplicates are fine
    ce = _class_ce(logits, y, idx)
    expected = F.cross_entropy(logits[:, [3, 4, 5]], y - 3)
    assert torch.allclose(ce, expected), (ce, expected)
    ce.backward()
    assert torch.all(logits.grad[:, :3] == 0), "old-class logits must get no gradient"
    assert torch.any(logits.grad[:, 3:] != 0)

    try:
        _build_ce_index([7], 6, DEVICE)
    except ValueError:
        pass
    else:
        raise AssertionError("id outside the classifier must raise")
    print("ce index OK")


def test_provisional_prototypes():
    model = _model(num_classes=5)
    known_vec = torch.randn(HIDDEN)
    model.classifier.prototypes[0] = known_vec  # class 0 already has a server prototype
    known = {0}
    loader = _loader([0, 1, 2])  # class 0 is local too, 3 and 4 are absent
    model.train()
    params_before = [p.detach().clone() for p in model.parameters()]

    ids = provisional_prototypes(model, loader, DEVICE, known)

    assert ids == [1, 2], ids
    assert model.training, "train/eval mode must be restored"
    assert known == {0}, "provisional classes must NOT become consolidated"
    assert torch.equal(model.classifier.prototypes[0], known_vec), "server proto kept"
    assert torch.all(model.classifier.prototypes[3:] == 0), "absent classes stay zero"
    sums, counts = _manual_class_means(model, loader, 5)
    for c in (1, 2):
        expected = sums[c] / counts[c]
        assert torch.allclose(model.classifier.prototypes[c], expected, atol=1e-5)
    for p0, p1 in zip(params_before, model.parameters()):
        assert torch.equal(p0, p1.detach()), "the pre-pass must not touch the weights"
    assert all(p.grad is None for p in model.parameters())
    print("provisional prototypes OK:", ids)


def test_postpass_stats():
    model = _model(num_classes=4)
    loader = _loader([1, 3])
    memory = PrototypeMemory(embedding_dim=HIDDEN, num_classes=1, device=DEVICE)
    memory._sum_h[:] = 0  # sanity
    memory.expand(4)
    memory._sum_h += 123.0  # junk left from "training-time" accumulation
    memory._counts += 7
    model.train()

    collect_prototype_stats(model, loader, memory, DEVICE)

    assert model.training
    sum_h, counts, class_ids = memory.get_stats()
    assert class_ids.tolist() == [1, 3], class_ids
    sums, cnts = _manual_class_means(model, loader, 4)
    assert torch.equal(counts, cnts[[1, 3]].long())
    assert torch.allclose(sum_h, sums[[1, 3]], atol=1e-4)
    print("post-pass stats OK")


def _train(model, loader, **kw):
    memory = PrototypeMemory(
        embedding_dim=HIDDEN,
        num_classes=max(model.classifier.num_classes, 1),
        device=DEVICE,
    )
    kw.setdefault("lambda_kd", 0.0)
    return (
        train_fn(
            model,
            loader,
            memory,
            epochs=2,
            lr=1e-3,
            device=DEVICE,
            **kw,
        ),
        memory,
    )


def test_first_round_signal():
    """Round 1 of a step: every local class has a zero prototype. Without the pre-pass
    the restricted CE is a constant (ln K) and gives no gradient; with it, it learns."""
    loader = _loader([0, 1, 2])
    ids = [0, 1, 2]

    m_off = _model(3)
    ce_idx = _build_ce_index(ids, 3, DEVICE)
    x, y = next(iter(loader))
    h, _ = m_off.embed_both(x)
    ce_const = _class_ce(m_off.classifier(h), y, ce_idx)
    assert torch.allclose(ce_const, torch.log(torch.tensor(3.0)), atol=1e-5)

    m_on = _model(3)
    provisional_prototypes(m_on, loader, DEVICE, set())
    m_on.eval()
    h, _ = m_on.embed_both(x)
    ce_on = _class_ce(m_on.classifier(h), y, ce_idx)
    assert ce_on.item() < ce_const.item() - 0.05, (ce_on.item(), ce_const.item())
    print(
        f"first-round CE: zero prototypes {ce_const.item():.3f} -> pre-pass {ce_on.item():.3f}"
    )


def test_train_fn_flag_matrix():
    old_vec = torch.randn(HIDDEN)
    for ce_local in (False, True):
        for prepass in (False, True):
            model = _model(5)
            model.classifier.prototypes[0] = old_vec
            model.classifier.prototypes[1] = old_vec.flip(0)
            known = {0, 1}
            loader = _loader([2, 3, 4])
            w0 = model.feature_extractor.conv1.weight.detach().clone()
            loss, memory = _train(
                model,
                loader,
                known_consolidated=known,
                local_class_ids=[2, 3, 4] if ce_local else None,
                proto_prepass=prepass,
            )
            assert torch.isfinite(torch.tensor(loss)), (ce_local, prepass, loss)
            assert not torch.equal(w0, model.feature_extractor.conv1.weight.detach())
            assert torch.equal(
                model.classifier.prototypes[0], old_vec
            ), "old proto moved"
            assert model.training
            if prepass:
                assert model.classifier.prototypes[2:].abs().sum() > 0
            else:
                assert torch.all(model.classifier.prototypes[2:] == 0)
    print("train_fn flag matrix OK (ce_local x prepass)")


def test_short_local_adaptation_smoke():
    model = _model(5)
    provisional_prototypes(model, _loader([0, 1, 2, 3, 4]), DEVICE, set())
    candidate = promote_to_incorporated(model.adapter_local, model.alpha_gate)
    loader = _loader([3, 4])
    for ids in (None, [3, 4]):
        steps = short_local_adaptation(
            model,
            candidate,
            loader,
            lr=1e-3,
            device=DEVICE,
            max_steps=4,
            ce_class_ids=ids,
        )
        assert steps == 4
    print("short_local_adaptation OK")


def test_model_init_knobs():
    m = FCLModel(
        input_dim=CHANNELS,
        hidden_dim=HIDDEN,
        near_identity_eps_local=0.05,
        alpha_init=2.0,
    )
    assert torch.allclose(m.alpha_gate.beta, torch.full((HIDDEN,), 2.0))
    assert m.adapter_local.near_identity_eps == 0.05
    std = m.adapter_local.up_proj.weight.std().item()
    assert 0.02 < std < 0.1, std
    m.reset_local_branch()
    assert torch.allclose(m.alpha_gate.beta, torch.full((HIDDEN,), 2.0))
    assert m.adapter_local.near_identity_eps == 0.05
    d = FCLModel(input_dim=CHANNELS, hidden_dim=HIDDEN)  # defaults unchanged
    assert (
        d.adapter_local.near_identity_eps == 1e-3
        and float(d.alpha_gate.beta.detach().sum()) == 0.0
    )
    print("model init knobs OK")


def test_param_groups():
    m = _model(3)
    groups = _build_param_groups(m, 1e-3, {"local": 10.0, "extractor": 0.1})
    ids = [id(p) for ps, _ in groups.values() for p in ps]
    assert (
        len(ids) == len(set(ids)) == len(list(m.parameters()))
    ), "each param exactly once"
    assert groups["extractor"][1] == 1e-3 * 0.1 and groups["local"][1] == 1e-3 * 10.0
    assert groups["adapter_global"][1] == 1e-3
    assert any(p is m.classifier.scale for p in groups["local"][0])
    for bad in ({"local": 0.0}, {"nope": 1.0}):
        try:
            _build_param_groups(m, 1e-3, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} must raise")

    # default scales == a single Adam over model.parameters() (same trajectory)
    def run(make_opt):
        torch.manual_seed(3)
        mm = _model(3)
        opt = make_opt(mm)
        x, y = next(iter(_loader([0, 1, 2], seed=5)))
        for _ in range(3):
            opt.zero_grad()
            h, _ = mm.embed_both(x)
            loss = (h**2).mean() + mm.classifier.scale * 0.0
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in mm.parameters()]

    ref = run(lambda mm: torch.optim.Adam(mm.parameters(), lr=1e-3))
    got = run(lambda mm: _build_optimizers(mm, 1e-3, None, False)[0])
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(ref, got))
    print("param groups OK")


def test_split_gradient_isolation():
    m = _model(3)
    provisional_prototypes(m, _loader([0, 1, 2]), DEVICE, set())
    x, y = next(iter(_loader([0, 1, 2])))
    shared = m.global_branch_parameters()
    local = list(m.adapter_local.parameters()) + list(m.alpha_gate.parameters())

    # local loss only moves local params (and the scale), never shared ones
    m.zero_grad()
    h_g, _, h_l = m.embed_split(x)
    F.cross_entropy(m.classifier(h_l), y).backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in shared)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in local)

    # shared loss with the frozen scale never reaches local params or the scale
    m.zero_grad()
    h_g, _, h_l = m.embed_split(x)
    F.cross_entropy(_logits_frozen_scale(m.classifier, h_g), y).backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in local)
    assert m.classifier.scale.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in shared)

    # the split embeddings match the fused model values
    m.eval()
    with torch.no_grad():
        h_fused, h_shared = m.embed_both(x)
        h_g, _, h_l = m.embed_split(x)
    assert torch.allclose(h_g, h_shared, atol=1e-6) and torch.allclose(
        h_l, h_fused, atol=1e-6
    )
    print("split gradient isolation OK")


def test_split_train_fn():
    old_vec = torch.randn(HIDDEN)
    for kd_on in ("local", "shared", "both"):
        model = _model(5)
        model.classifier.prototypes[0] = old_vec
        model.classifier.prototypes[1] = old_vec.flip(0)
        loader = _loader([2, 3, 4])
        ext0 = model.feature_extractor.conv1.weight.detach().clone()
        loc0 = model.adapter_local.up_proj.weight.detach().clone()
        scale0 = model.classifier.scale.detach().clone()
        loss, _ = _train(
            model,
            loader,
            known_consolidated={0, 1},
            local_class_ids=[2, 3, 4],
            proto_prepass=True,
            split_optimizers=True,
            lambda_kd=1.0 if kd_on in ("local", "both") else 0.0,
            lambda_kd_shared=1.0 if kd_on in ("shared", "both") else 0.0,
            lambda_proto_shared=0.5,
            lambda_proto_local=2.0,
        )
        assert torch.isfinite(torch.tensor(loss)), (kd_on, loss)
        assert not torch.equal(ext0, model.feature_extractor.conv1.weight.detach())
        assert not torch.equal(loc0, model.adapter_local.up_proj.weight.detach())
        assert not torch.equal(
            scale0, model.classifier.scale.detach()
        ), "local opt trains scale"
        assert torch.equal(model.classifier.prototypes[0], old_vec)
    # no local adapter -> the split request falls back to the single objective
    off = FCLModel(input_dim=CHANNELS, hidden_dim=HIDDEN, use_local_adapter=False)
    off.classifier._expand(3)
    loss, _ = _train(
        off, _loader([0, 1, 2]), split_optimizers=True, known_consolidated=set()
    )
    assert torch.isfinite(torch.tensor(loss))
    print("split train_fn OK (kd-on local/shared/both, no-local fallback)")


def test_lr_scale_local_moves_more():
    deltas = {}
    for scale in (1.0, 10.0):
        model = _model(3)
        loc0 = torch.cat(
            [p.detach().flatten().clone() for p in model.adapter_local.parameters()]
        )
        _train(
            model,
            _loader([0, 1, 2], seed=7),
            known_consolidated=set(),
            proto_prepass=True,
            lr_scales={"local": scale},
            split_optimizers=True,
        )
        loc1 = torch.cat(
            [p.detach().flatten() for p in model.adapter_local.parameters()]
        )
        deltas[scale] = (loc1 - loc0).norm().item()
    assert deltas[10.0] > deltas[1.0], deltas
    print(
        f"lr-scale-local moves the local adapter more: {deltas[1.0]:.4f} -> {deltas[10.0]:.4f}"
    )


def _manual_kl(h_s, h_t, protos, T, s_student, s_teacher):
    p = F.normalize(protos, dim=1)
    log_q = F.log_softmax(F.normalize(h_s, dim=1) @ p.t() * (s_student / T), dim=1)
    q = F.softmax(F.normalize(h_t, dim=1) @ p.t() * (s_teacher / T), dim=1)
    return F.kl_div(log_q, q, reduction="batchmean") * T**2


def test_kd_scale():
    torch.manual_seed(2)
    protos = torch.randn(5, HIDDEN)
    h_t = torch.randn(32, HIDDEN)
    h_s = h_t + 0.5 * torch.randn(32, HIDDEN)

    # default (scale 1) is the previous behaviour
    old = distillation_loss_kl(h_s, h_t, protos, temperature=2.0)
    assert torch.allclose(old, _manual_kl(h_s, h_t, protos, 2.0, 1.0, 1.0), atol=1e-6)
    # the previous term is tiny next to the scale-aware one for the same drift
    big = distillation_loss_kl(h_s, h_t, protos, temperature=2.0, scale_student=20.0)
    assert big.item() > 20 * old.item(), (old.item(), big.item())
    assert torch.allclose(big, _manual_kl(h_s, h_t, protos, 2.0, 20.0, 20.0), atol=1e-5)
    # same embeddings: a common scale gives exactly 0, a teacher/student scale mismatch does not
    same = distillation_loss_kl(h_t, h_t, protos, temperature=2.0, scale_student=12.0)
    assert same.item() < 1e-5, same.item()
    mismatch = distillation_loss_kl(
        h_t, h_t, protos, temperature=2.0, scale_student=12.0, scale_teacher=20.0
    )
    assert mismatch.item() > 1e-3, mismatch.item()

    # _kd_term picks the scales by mode (classifier scale 12, kd-logit-scale 20)
    model = _model(5)
    model.classifier.prototypes.copy_(protos)
    with torch.no_grad():
        model.classifier.scale.fill_(12.0)
    known = [0, 1, 2]
    ref = protos[known]
    kw = dict(known_sorted=known, kd_mode="kl", kd_temperature=2.0)
    got = {
        m: _kd_term(model, h_s, h_t, kd_scale_mode=m, kd_logit_scale=20.0, **kw)
        for m in ("const", "local", "hybrid")
    }
    assert torch.allclose(
        got["const"], _manual_kl(h_s, h_t, ref, 2.0, 20.0, 20.0), atol=1e-5
    )
    assert torch.allclose(
        got["local"], _manual_kl(h_s, h_t, ref, 2.0, 12.0, 12.0), atol=1e-5
    )
    assert torch.allclose(
        got["hybrid"], _manual_kl(h_s, h_t, ref, 2.0, 12.0, 20.0), atol=1e-5
    )
    legacy = _kd_term(model, h_s, h_t, **kw)  # defaults = previous behaviour
    assert torch.allclose(legacy, _manual_kl(h_s, h_t, ref, 2.0, 1.0, 1.0), atol=1e-6)
    # the scale is a constant for the KD: no gradient reaches the classifier scale
    model.zero_grad()
    _kd_term(
        model, h_s.clone().requires_grad_(True), h_t, kd_scale_mode="local", **kw
    ).backward()
    assert model.classifier.scale.grad is None
    try:
        _kd_term(model, h_s, h_t, kd_scale_mode="nope", **kw)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kd_scale_mode must raise")
    print("kd scale OK (const / local / hybrid, legacy default preserved)")


def test_kd_scale_in_train_fn():
    for mode in ("const", "local", "hybrid"):
        for split in (False, True):
            model = _model(5)
            provisional_prototypes(model, _loader([0, 1, 2, 3, 4]), DEVICE, set())
            loss, _ = _train(
                model,
                _loader([3, 4]),
                known_consolidated={0, 1, 2},
                lambda_kd=1.0,
                kd_scale_mode=mode,
                kd_logit_scale=20.0,
                split_optimizers=split,
            )
            assert torch.isfinite(torch.tensor(loss)), (mode, split)
    print("kd scale in train_fn OK")


def test_confusion_matrix():
    model = _model(4)
    loader = _loader([0, 1, 2, 3], per_class=15)
    provisional_prototypes(model, loader, DEVICE, set())
    base = evaluate_global_model(model, loader, DEVICE)
    assert len(base) == 4, "default return unchanged"
    loss, acc, per_class_acc, per_class_n, conf = evaluate_global_model(
        model, loader, DEVICE, return_confusion=True
    )
    assert conf.shape == (4, 4) and int(conf.sum()) == sum(per_class_n.values())
    for c, n in per_class_n.items():
        assert int(conf[c].sum()) == n
        assert abs(int(conf[c, c]) / n - per_class_acc[c]) < 1e-9
    assert abs(int(conf.diag().sum()) / int(conf.sum()) - acc) < 1e-9
    assert (base[0], base[1]) == (loss, acc)
    # classes outside allowed_classes are not counted
    _, _, _, n2, conf2 = evaluate_global_model(
        model, loader, DEVICE, {0, 1}, return_confusion=True
    )
    assert set(n2) == {0, 1} and int(conf2[2:].sum()) == 0
    print("confusion matrix OK")


def test_prediction_flow_monitor(tmp_dir="/tmp/_flow_test"):
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)
    # 6 classes: old = {0,1,2}, current = {3,4,5}; 10 samples per class
    conf = torch.zeros(6, 6, dtype=torch.long)
    for c in range(6):
        conf[c, c] = 2
    conf[0, 3] = 6
    conf[1, 4] = 5
    conf[2, 3] = 4
    conf[2, 1] = 2  # old -> current / other old
    conf[3, 0] = 3
    conf[4, 4] = 8
    conf[5, 5] = 5  # current -> old / correct
    conf[3, 3] = 7
    conf[4, 4] = 8
    conf[3, 0] = 3
    conf[5, 1] = 5
    row = PredictionFlowMonitor.summarize(conf, {3, 4, 5}, {0, 1, 2, 3, 4, 5})
    assert row["n_old"] == int(conf[:3].sum()) and row["old_correct"] == 6
    assert (
        row["old_to_current"] == 6 + 5 + 4
        and row["old_to_other_old"] == row["n_old"] - 6 - 15
    )
    assert row["cur_to_old"] == 3 + 5 and row["n_cur"] == int(conf[3:].sum())
    assert (
        abs(row["pred_share_current"] - int(conf[:, 3:].sum()) / int(conf.sum())) < 1e-9
    )
    mon = PredictionFlowMonitor(tmp_dir)
    m1 = mon.update(11, conf, {3, 4, 5}, {0, 1, 2, 3, 4, 5}, step=1, save_matrix=True)
    m2 = mon.update(12, conf, {3, 4, 5}, {0, 1, 2, 3, 4, 5}, step=1)
    assert abs(m1["flow_old_to_current"] - 15 / row["n_old"]) < 1e-9
    assert (Path(tmp_dir) / "prediction_flow.csv").read_text().count(
        "\n"
    ) == 3  # header + 2 rows
    import numpy as np

    saved = np.load(Path(tmp_dir) / "confusion_matrices.npz")
    assert list(saved.keys()) == ["round_11"] and saved["round_11"].shape == (6, 6)
    # no old classes yet (first step): old fractions are simply omitted
    m0 = mon.update(1, conf[3:, 3:], {0, 1, 2}, {0, 1, 2}, step=0)
    assert "flow_old_correct" not in m0
    print("prediction flow monitor OK")


def _unit_loader(vectors_by_class, per_class=6):
    """Loader whose 'embeddings' we control: a stub model maps x to the stored vectors."""
    xs, ys = [], []
    for c, v in vectors_by_class.items():
        xs.append(v.repeat(per_class, 1))
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    return DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)), batch_size=8)


class _StubModel:
    """Minimal model for compute_interference: embed_both is the identity."""

    def __init__(self, protos):
        self.classifier = type("C", (), {})()
        self.classifier.prototypes = protos
        self.training = False

    def eval(self):
        return self

    def embed_both(self, x):
        return x, x


def test_interference():
    e = torch.eye(6)
    protos = torch.zeros(6, 6)
    protos[0], protos[1] = e[0], e[1]  # old classes 0 and 1 have global prototypes
    model = _StubModel(protos)
    # new class 2 sits exactly on old class 0; new class 3 is orthogonal to both
    crowded = compute_interference(model, _unit_loader({2: e[0]}), {0, 1}, DEVICE)
    apart = compute_interference(model, _unit_loader({3: e[3]}), {0, 1}, DEVICE)
    assert abs(crowded - 1.0) < 1e-6 and abs(apart) < 1e-6, (crowded, apart)
    # mean over the local classes: (1 + 0) / 2
    both = compute_interference(model, _unit_loader({2: e[0], 3: e[3]}), {0, 1}, DEVICE)
    assert abs(both - 0.5) < 1e-6
    # a local class the server already knows uses its GLOBAL prototype and is compared with
    # the other known classes only
    protos2 = protos.clone()
    protos2[2] = (e[0] + e[3]) / 2**0.5  # class 2 now consolidated, half-way to class 0
    m2 = _StubModel(protos2)
    got = compute_interference(m2, _unit_loader({2: e[3]}), {0, 1, 2}, DEVICE)
    assert abs(got - 1 / 2**0.5) < 1e-5, got
    # no old class with a prototype (first step) or no data -> None
    assert (
        compute_interference(
            _StubModel(torch.zeros(6, 6)), _unit_loader({2: e[0]}), set(), DEVICE
        )
        is None
    )
    assert (
        compute_interference(
            model,
            DataLoader(
                TensorDataset(torch.zeros(0, 6), torch.zeros(0, dtype=torch.long))
            ),
            {0, 1},
            DEVICE,
        )
        is None
    )
    print("interference OK")


def test_signal_modes():
    g_mode = ExpansionCriterion(theta_exp=0.04)
    assert g_mode.signal(0.03, 0.99) == 0.03 and g_mode.signal(None, 0.9) == 0.0
    i_mode = ExpansionCriterion(
        theta_exp=0.04, signal_mode="interference", interference_floor=0.5
    )
    assert abs(i_mode.signal(0.9, 0.75) - 0.5) < 1e-9
    assert (
        i_mode.signal(0.9, 0.4) == 0.0
        and i_mode.signal(0.9, 1.0) == 1.0
        and i_mode.signal(0.9, None) == 0.0
    )
    c_mode = ExpansionCriterion(
        theta_exp=0.04, signal_mode="combined", interference_floor=0.5
    )
    assert (
        c_mode.signal(0.30, 0.75) == 0.5
        and c_mode.signal(0.8, 0.75) == 0.8
        and c_mode.signal(0.3, None) == 0.3
    )
    for bad in ({"signal_mode": "x"}, {"interference_floor": 1.0}):
        try:
            ExpansionCriterion(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} must raise")
    # the scalar drives the decision: crowded classes expand the adapter, separated ones do not
    model = _model(3)
    crit = ExpansionCriterion(
        theta_exp=0.04, signal_mode="interference", interference_floor=0.5
    )
    before = model.adapter_local.bottleneck_dim
    assert (
        crit.step(
            model.adapter_local, crit.signal(0.0, 0.51), g_reduced_below_threshold=False
        )
        is None
    )
    kind = crit.step(
        model.adapter_local, crit.signal(0.0, 0.95), g_reduced_below_threshold=False
    )
    assert kind == "width" and model.adapter_local.bottleneck_dim > before
    f = resolve_ablation_flags({"expansion-signal": "combined"})
    assert (
        f.expansion_signal == "combined" and "expansion-signal=combined" in f.describe()
    )
    assert resolve_ablation_flags({}).expansion_signal == "g"
    try:
        resolve_ablation_flags({"expansion-signal": "nope"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown expansion-signal must raise")
    print("signal modes OK")


def test_expansion_term_metrics():
    from src.client.Client_app import _expansion_term_metrics

    full = _expansion_term_metrics(
        g_cosine_old=0.1, interference=0.9, expansion_signal=0.8
    )
    empty = _expansion_term_metrics()
    assert set(full) == set(empty) and len(full) == 10, "fixed key set across clients"
    assert full["g_cosine_old"] == 0.1 and full["g_cosine_old_valid"] == 1.0
    assert full["g_entropy"] == 0.0 and full["g_entropy_valid"] == 0.0
    assert all(v == 0.0 for v in empty.values())
    # weighted mean among the clients that have the value = mean(v) / mean(valid)
    a, b = _expansion_term_metrics(interference=0.8), _expansion_term_metrics()
    w = (3.0, 1.0)
    num = (w[0] * a["interference"] + w[1] * b["interference"]) / sum(w)
    den = (w[0] * a["interference_valid"] + w[1] * b["interference_valid"]) / sum(w)
    assert abs(num / den - 0.8) < 1e-9
    print("expansion term metrics OK")


def test_prototype_metric_loss():
    e = torch.eye(4)
    model = _StubModel(e.clone())  # prototypes e0..e3
    model.classifier.prototypes = e.clone()
    y = torch.tensor([0])
    # on its own prototype with every other prototype at distance 1: both hinges are 0
    h_on = e[0:1].clone()
    for kind in ("triplet", "contrastive"):
        assert float(_prototype_metric_loss(model, h_on, y, None, kind, 0.2)) == 0.0
    # half-way between prototypes 0 and 1: d_pos = d(neg 1) = 1 - 1/sqrt(2)
    h_mid = ((e[0] + e[1]) / 2**0.5).unsqueeze(0).clone().requires_grad_(True)
    d = 1 - 1 / 2**0.5
    tri = _prototype_metric_loss(model, h_mid, y, None, "triplet", 0.2)
    assert abs(float(tri) - 0.2) < 1e-6  # d_pos - d_hardest + m
    con = _prototype_metric_loss(model, h_mid, y, None, "contrastive", 0.2)
    assert abs(float(con) - d**2) < 1e-6  # pos^2 + 0 (negatives beyond the margin)
    tri.backward()
    assert h_mid.grad is not None and float(h_mid.grad.abs().sum()) > 0
    # negatives restricted to the local classes: only class 0 -> no negative -> triplet 0
    only0 = (torch.tensor([0]), torch.tensor([0, -1, -1, -1]))
    assert (
        float(_prototype_metric_loss(model, h_mid.detach(), y, only0, "triplet", 0.2))
        == 0.0
    )
    assert (
        abs(
            float(
                _prototype_metric_loss(
                    model, h_mid.detach(), y, only0, "contrastive", 0.2
                )
            )
            - d**2
        )
        < 1e-6
    )
    # classes {0, 1} keep the hardest negative (class 1) and the loss stays 0.2
    both = (torch.tensor([0, 1]), torch.tensor([0, 1, -1, -1]))
    assert (
        abs(
            float(
                _prototype_metric_loss(model, h_mid.detach(), y, both, "triplet", 0.2)
            )
            - 0.2
        )
        < 1e-6
    )
    # a zero (non-existent) prototype is never a negative
    model.classifier.prototypes = e.clone()
    model.classifier.prototypes[1] = 0
    assert (
        float(_prototype_metric_loss(model, h_mid.detach(), y, None, "triplet", 0.2))
        == 0.0
    )
    # dispatch: "ce" is exactly the class cross-entropy
    logits = torch.randn(3, 4)
    yy = torch.tensor([0, 2, 3])
    assert torch.allclose(
        _cls_loss(model, torch.randn(3, 4), yy, logits, None, "ce"),
        F.cross_entropy(logits, yy),
    )
    try:
        _prototype_metric_loss(model, h_mid.detach(), y, None, "nope", 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown loss must raise")
    f = resolve_ablation_flags({"cls-loss": "triplet"})
    assert f.cls_loss == "triplet" and "cls-loss=triplet" in f.describe()
    assert resolve_ablation_flags({}).cls_loss == "ce"
    print("prototype metric loss OK")


def test_cls_loss_in_train_fn():
    for kind in ("ce", "triplet", "contrastive"):
        for split in (False, True):
            for local in (False, True):
                model = _model(5)
                provisional_prototypes(model, _loader([0, 1, 2, 3, 4]), DEVICE, set())
                loss, _ = _train(
                    model,
                    _loader([3, 4]),
                    known_consolidated={0, 1, 2},
                    cls_loss=kind,
                    cls_margin=0.1,
                    split_optimizers=split,
                    local_class_ids=[3, 4] if local else None,
                )
                assert torch.isfinite(torch.tensor(loss)), (kind, split, local)
    try:
        _train(_model(3), _loader([0, 1, 2]), known_consolidated=set(), cls_loss="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown cls_loss must raise in train_fn")
    print(
        "cls loss in train_fn OK (ce / triplet / contrastive x split x local classes)"
    )


def test_interference_does_not_touch_the_rng():
    """The diagnostic iterates a shuffling loader; it must not change the training batch order."""
    e = torch.eye(6)
    protos = torch.zeros(6, 6)
    protos[0], protos[1] = e[0], e[1]
    model = _StubModel(protos)
    xs = torch.cat([e[2].repeat(20, 1), e[3].repeat(20, 1)])
    ys = torch.cat([torch.full((20,), 2), torch.full((20,), 3)])
    loader = DataLoader(TensorDataset(xs, ys), batch_size=8, shuffle=True)

    def order():
        return [tuple(y.tolist()) for _, y in loader]

    torch.manual_seed(123)
    reference = order()  # batch order without the diagnostic
    torch.manual_seed(123)
    state = torch.get_rng_state()
    compute_interference(model, loader, {0, 1}, DEVICE)  # iterates the loader once
    assert torch.equal(state, torch.get_rng_state()), "RNG state changed"
    assert order() == reference, "diagnostic changed the batch order that follows"
    print("interference leaves the RNG untouched OK")


if __name__ == "__main__":
    test_flags()
    test_ce_index()
    test_provisional_prototypes()
    test_postpass_stats()
    test_first_round_signal()
    test_train_fn_flag_matrix()
    test_short_local_adaptation_smoke()
    test_model_init_knobs()
    test_param_groups()
    test_split_gradient_isolation()
    test_split_train_fn()
    test_lr_scale_local_moves_more()
    test_kd_scale()
    test_kd_scale_in_train_fn()
    test_confusion_matrix()
    test_prediction_flow_monitor()
    test_interference()
    test_interference_does_not_touch_the_rng()
    test_signal_modes()
    test_expansion_term_metrics()
    test_prototype_metric_loss()
    test_cls_loss_in_train_fn()
    print("\nall prototype-pass / local-CE checks passed")
