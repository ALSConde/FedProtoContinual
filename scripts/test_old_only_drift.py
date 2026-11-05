"""prd_old_only (PRD restricted to the old classes) and the semantic-drift compensation of the
old prototypes (client: compute_old_prototype_drift, server: aggregate_drift).

Run: python scripts/test_old_only_drift.py
"""

import copy
import math
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import src.client.ClientTask as CT
from src.client.ClientTask import (
    _prd_kwargs,
    compute_old_prototype_drift,
    make_old_prototype_state,
    provisional_prototypes,
    train_fn,
)
from src.model.Models import FCLModel
from src.model.layers.PrototypeMemory import PrototypeMemory
from src.server.PrototypeAggregator import PrototypeAggregator, collect_old_proto_drift

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
# prd_old_only
# ----------------------------------------------------------------------------------
def test_old_only_state_is_constant():
    model = _ready()
    st = make_old_prototype_state(model, [0, 1, 2, 3], {3, 4}, learnable=False)
    assert st.class_ids == [0, 1, 2] and not st.learnable
    assert not st.student.requires_grad
    kw = _prd_kwargs(st, torch.zeros(1))
    assert (
        kw["h_prd"] is None
    )  # a constant reference keeps the embedding the PRD always used
    learn = make_old_prototype_state(model, [0, 1, 2, 3], {3, 4}, learnable=True)
    assert learn.student.requires_grad
    assert _prd_kwargs(learn, torch.zeros(1))["h_prd"] is not None
    print("old-only state is a constant reference OK")


def test_old_only_restricts_the_prd_reference():
    seen = []
    original = CT.relation_distillation_loss

    def spy(h_student, h_teacher, protos, *a, **k):
        seen.append(None if protos is None else int(protos.shape[0]))
        return original(h_student, h_teacher, protos, *a, **k)

    CT.relation_distillation_loss = spy
    try:
        # known {0,1,2,3}, data of classes {3,4}: class 3 is present -> not "old"
        for flag, expect in ((False, 4), (True, 3)):
            seen.clear()
            _train(
                _ready(),
                _loader([3, 4]),
                known_consolidated={0, 1, 2, 3},
                prd_old_only=flag,
            )
            assert seen and set(seen) == {expect}, (flag, set(seen))
    finally:
        CT.relation_distillation_loss = original
    print("old-only drops the present classes from the PRD reference OK")


def test_old_only_without_old_classes_is_no_kd():
    """Every known class is present locally -> nothing to distil: the PRD gets an EMPTY
    reference and returns None every time (so no PRD loss is added)."""
    calls, results = [], []
    original = CT.relation_distillation_loss

    def spy(h_student, h_teacher, protos, *a, **k):
        out = original(h_student, h_teacher, protos, *a, **k)
        calls.append(None if protos is None else int(protos.shape[0]))
        results.append(out)
        return out

    CT.relation_distillation_loss = spy
    try:
        _train(_ready(), _loader([0, 1, 2]), prd_old_only=True)
        assert calls and set(calls) == {0} and all(r is None for r in results)
        calls.clear()
        results.clear()
        # the legacy PRD does constrain the present classes
        _train(_ready(), _loader([0, 1, 2]))
        assert set(calls) == {3} and all(r is not None for r in results)
    finally:
        CT.relation_distillation_loss = original
    print("old-only with no old class adds no PRD loss OK")


def test_old_only_guards_and_no_export():
    try:
        _train(_ready(), _loader([3, 4]), kd_mode="kl", prd_old_only=True)
    except ValueError:
        pass
    else:
        raise AssertionError("prd_old_only needs a PRD kd_mode")
    out = {}
    _train(_ready(), _loader([3, 4]), prd_old_only=True, prd_out=out)
    assert out == {}  # constants: nothing to upload
    # off by default: identical to not passing it
    # one model at a time: training consumes the global RNG
    a = _ready()
    _train(a, _loader([3, 4]))
    b = _ready()
    _train(b, _loader([3, 4]), prd_old_only=False)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    print("old-only guards / no export / off by default OK")


# ----------------------------------------------------------------------------------
# drift compensation: client
# ----------------------------------------------------------------------------------
def _trained_pair(known={0, 1, 2}):
    model = _ready()
    start = copy.deepcopy(model).eval()
    _train(
        model, _loader([3, 4]), known_consolidated=known, lambda_kd=0.0, kd_mode="kl"
    )
    return model, start


def test_drift_zero_when_nothing_moved():
    model = _ready()
    start = copy.deepcopy(model)
    ids, sum_delta, mass = compute_old_prototype_drift(
        model, start, _loader([3, 4]), DEVICE, {0, 1, 2}
    )
    assert ids.tolist() == [0, 1, 2]
    assert sum_delta.shape == (3, HIDDEN) and mass.shape == (3,)
    assert float(sum_delta.abs().max()) == 0.0
    assert bool((mass > 0).all())
    print("drift is zero when the model did not move OK")


def test_drift_selection_and_modes():
    model, start = _trained_pair()
    model.train()
    out = compute_old_prototype_drift(
        model, start, _loader([3, 4]), DEVICE, {0, 1, 2, 3}
    )
    assert out[0].tolist() == [0, 1, 2]  # class 3 is present locally: not old
    assert model.training  # the mode is restored
    assert (
        compute_old_prototype_drift(model, start, _loader([3, 4]), DEVICE, {3, 4})
        is None
    )
    assert (
        compute_old_prototype_drift(model, start, _loader([3, 4]), DEVICE, set())
        is None
    )
    # a class without prototype is skipped
    model.classifier.prototypes[1] = 0.0
    ids = compute_old_prototype_drift(model, start, _loader([3, 4]), DEVICE, {0, 1, 2})[
        0
    ]
    assert ids.tolist() == [0, 2]
    print("drift: old-class selection / modes / None cases OK")


def test_drift_kernel():
    model, start = _trained_pair()
    loader = _loader([3, 4])
    # reference quantities on the same samples
    h0, h1 = [], []
    with torch.no_grad():
        model.eval()
        for x, _ in loader:
            h0.append(start.embed_both(x)[1])
            h1.append(model.embed_both(x)[1])
    h0, h1 = torch.cat(h0), torch.cat(h1)
    ids, sum_delta, mass = compute_old_prototype_drift(
        model, start, loader, DEVICE, {0, 1, 2}, sigma_factor=1e6
    )
    # huge bandwidth: every sample counts the same -> the mean drift of all local samples
    assert torch.allclose(mass, torch.full((3,), float(len(h0))), atol=1e-2)
    assert torch.allclose(
        sum_delta / mass.unsqueeze(1), (h1 - h0).mean(0).expand(3, -1), atol=1e-4
    )
    # tiny bandwidth: only samples very close to the prototype count -> a mass far below N.
    # A class whose samples are ALL far from its prototype ends with mass ~0, i.e. no
    # correction (the server ignores zero mass): that is the intended "no nearby data" case.
    _, sd_small, mass_small = compute_old_prototype_drift(
        model, start, loader, DEVICE, {0, 1, 2}, sigma_factor=0.05
    )
    assert bool((mass_small < 0.2 * len(h0)).all()) and bool((mass_small >= 0).all())
    assert bool((mass_small > 0).any()) and bool((mass_small <= mass).all())
    # mass grows monotonically with the bandwidth
    masses = [
        compute_old_prototype_drift(
            model, start, loader, DEVICE, {0, 1, 2}, sigma_factor=f
        )[2]
        for f in (0.1, 0.5, 1.0, 2.0)
    ]
    assert all(bool((a <= b + 1e-6).all()) for a, b in zip(masses, masses[1:]))
    print("drift kernel (bandwidth limits) OK")


# ----------------------------------------------------------------------------------
# drift compensation: server
# ----------------------------------------------------------------------------------
def _aggregator(tau=10.0, old_tau=50.0, d=4):
    agg = PrototypeAggregator(embedding_dim=d, tau=tau, old_tau=old_tau)
    sum_h = torch.tensor([[2.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 4.0, 0]])
    agg.aggregate([(sum_h, torch.ones(3), torch.tensor([0, 1, 2]))])
    return agg


def test_aggregate_drift_math():
    agg = _aggregator()
    base = agg.mu_global.clone()
    a = (
        torch.tensor([0, 1]),
        torch.tensor([[3.0, 0, 0, 0], [0, 6.0, 0, 0]]),
        torch.tensor([3.0, 6.0]),
    )
    b = (torch.tensor([0]), torch.tensor([[0.0, 2.0, 0, 0]]), torch.tensor([1.0]))
    shifted = agg.aggregate_drift([a, b])
    assert sorted(shifted.tolist()) == [0, 1]
    # class 0: delta_bar = (3,2,0,0)/4 over mass 4
    d0 = torch.tensor([3.0, 2.0, 0, 0]) / 4.0
    alpha0 = 1 - math.exp(-4.0 / 50.0)
    assert torch.allclose(agg.mu_global[0], base[0] + alpha0 * d0, atol=1e-6)
    # class 1: delta_bar = (0,6,0,0)/6 over mass 6
    alpha1 = 1 - math.exp(-6.0 / 50.0)
    assert torch.allclose(
        agg.mu_global[1], base[1] + alpha1 * torch.tensor([0, 1.0, 0, 0]), atol=1e-6
    )
    assert torch.equal(agg.mu_global[2], base[2])  # not reported
    print("aggregate_drift math OK")


def test_aggregate_drift_tau_and_guards():
    moves = []
    for old_tau in (5.0, 50.0, 500.0):
        agg = _aggregator(old_tau=old_tau)
        before = agg.mu_global[0].clone()
        agg.aggregate_drift(
            [(torch.tensor([0]), torch.tensor([[0.0, 0, 0, 8.0]]), torch.tensor([8.0]))]
        )
        moves.append(float((agg.mu_global[0] - before).norm()))
    assert moves[0] > moves[1] > moves[2] > 0, moves  # larger tau -> smaller correction
    # more mass (evidence) at fixed tau -> larger correction, same drift per sample
    small, large = _aggregator(), _aggregator()
    b = small.mu_global[0].clone()
    small.aggregate_drift(
        [(torch.tensor([0]), torch.tensor([[0.0, 0, 0, 1.0]]), torch.tensor([1.0]))]
    )
    large.aggregate_drift(
        [(torch.tensor([0]), torch.tensor([[0.0, 0, 0, 40.0]]), torch.tensor([40.0]))]
    )
    assert float((large.mu_global[0] - b).norm()) > float(
        (small.mu_global[0] - b).norm()
    )
    # unknown / empty / zero-mass entries are ignored
    agg = _aggregator()
    before = agg.mu_global.clone()
    upd = [
        (torch.tensor([9]), torch.ones(1, 4), torch.tensor([5.0])),
        (torch.tensor([1]), torch.ones(1, 4), torch.tensor([0.0])),
        (torch.tensor([], dtype=torch.long), torch.zeros(0, 4), torch.zeros(0)),
    ]
    assert len(agg.aggregate_drift(upd)) == 0 and torch.equal(agg.mu_global, before)
    agg._expand(5)
    assert (
        len(
            agg.aggregate_drift(
                [(torch.tensor([4]), torch.ones(1, 4), torch.tensor([3.0]))]
            )
        )
        == 0
    )
    print("aggregate_drift tau / evidence / guards OK")


def test_collect_old_proto_drift():
    def reply(cfg):
        content = {"config": cfg} if cfg is not None else {}
        return SimpleNamespace(has_content=lambda: True, content=content)

    good = pickle.dumps(
        (torch.tensor([0, 2]), torch.ones(2, 4), torch.tensor([1.0, 2.0]))
    )
    empty = pickle.dumps(
        (torch.tensor([], dtype=torch.long), torch.zeros(0, 4), torch.zeros(0))
    )
    out = collect_old_proto_drift(
        [
            reply({"old_proto_drift": good}),
            reply({"proto_stats": b"x"}),
            reply(None),
            reply({"old_proto_drift": empty}),
            SimpleNamespace(has_content=lambda: False, content={}),
        ]
    )
    assert len(out) == 1 and out[0][0].tolist() == [0, 2]
    print("collect_old_proto_drift OK")


def test_round_trip_client_to_server():
    model, start = _trained_pair()
    protos = start.classifier.prototypes.clone()
    server = PrototypeAggregator(embedding_dim=HIDDEN, tau=10.0, old_tau=20.0)
    server.aggregate([(protos, torch.ones(5), torch.arange(5))])
    base = server.mu_global.clone()
    drift = compute_old_prototype_drift(
        model, start, _loader([3, 4]), DEVICE, {0, 1, 2}
    )
    shifted = server.aggregate_drift([drift])
    assert shifted.tolist() == [0, 1, 2]
    moved = server.mu_global - base
    ids, sum_delta, mass = drift
    expected_dir = sum_delta / mass.unsqueeze(1)
    cos = F.cosine_similarity(moved[:3], expected_dir, dim=1)
    assert bool((cos > 0.999).all())  # the EMA scales the drift, it does not rotate it
    assert (
        float(moved[3:].abs().max()) == 0.0
    )  # present classes never touched by the drift
    print("client -> server drift round trip OK")


def test_strategies_fold_drift():
    from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

    try:
        from flwr.supercore.task_identity import TaskIdentity

        TaskIdentity.run_id, TaskIdentity.task_id, TaskIdentity.node_id = 1, 1, 1
    except ImportError:
        pass

    def make_reply(stats, drift=None, update=None):
        request = Message(
            content=RecordDict({"config": ConfigRecord({})}),
            dst_node_id=0,
            message_type="train",
        )
        cfg = {"proto_stats": pickle.dumps(stats)}
        if drift is not None:
            cfg["old_proto_drift"] = pickle.dumps(drift)
        if update is not None:
            cfg["old_proto_update"] = pickle.dumps(update)
        content = RecordDict(
            {
                "arrays": ArrayRecord({"w": torch.ones(2)}),
                "metrics": MetricRecord({"num-examples": 10, "train_loss": 1.0}),
                "config": ConfigRecord(cfg),
            }
        )
        return Message(content=content, reply_to=request)

    try:
        make_reply(
            (torch.zeros(0, 4), torch.zeros(0), torch.tensor([], dtype=torch.long))
        )
    except (ImportError, AttributeError, TypeError) as exc:
        print(f"strategies check SKIPPED (Flower Message API differs): {exc!r}")
        return
    from src.server.FedAvgStrategy import FedAvgStrategy
    from src.server.FedProxStrategy import FedProxStrategy

    first = (
        torch.tensor([[2.0, 0, 0, 0], [0, 3.0, 0, 0]]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([0, 1]),
    )
    empty_stats = (
        torch.zeros(0, 4),
        torch.zeros(0),
        torch.tensor([], dtype=torch.long),
    )
    for cls, extra in ((FedAvgStrategy, {}), (FedProxStrategy, {"proximal_mu": 0.1})):
        s = cls(embedding_dim=4, tau=10.0, old_tau=50.0, **extra)
        s.aggregate_train(1, [make_reply(first)])
        base = s.proto_aggregator.mu_global.clone()
        drift = (
            torch.tensor([0, 1]),
            torch.tensor([[4.0, 0, 0, 0], [0, 0, 8.0, 0]]),
            torch.tensor([4.0, 8.0]),
        )
        # a round carrying ONLY a drift update (no new samples) must still be folded and broadcast
        s.aggregate_train(2, [make_reply(empty_stats, drift=drift)])
        mu = s.proto_aggregator.mu_global.clone()
        a0, a1 = 1 - math.exp(-4.0 / 50.0), 1 - math.exp(-8.0 / 50.0)
        assert torch.allclose(
            mu[0], base[0] + a0 * torch.tensor([1.0, 0, 0, 0]), atol=1e-6
        )
        assert torch.allclose(
            mu[1], base[1] + a1 * torch.tensor([0, 0, 1.0, 0]), atol=1e-6
        )
        sent, ids = pickle.loads(s._latest_proto_bytes)
        assert torch.allclose(sent[0], mu[0]) and ids.tolist() == [0, 1]
        # learned-prototype update and drift in the same round: EMA first, drift after
        upd = (torch.tensor([0]), torch.tensor([[0.0, 5.0, 0, 0]]), 40)
        before = s.proto_aggregator.mu_global.clone()
        s.aggregate_train(3, [make_reply(empty_stats, drift=drift, update=upd)])
        alpha_u = 1 - math.exp(-40 / 50.0)
        after_old = (1 - alpha_u) * before[0] + alpha_u * upd[1][0]
        expected = after_old + a0 * torch.tensor([1.0, 0, 0, 0])
        assert torch.allclose(s.proto_aggregator.mu_global[0], expected, atol=1e-5)
    print("FedAvg / FedProx fold drift (alone and with a learned update) OK")


if __name__ == "__main__":
    test_old_only_state_is_constant()
    test_old_only_restricts_the_prd_reference()
    test_old_only_without_old_classes_is_no_kd()
    test_old_only_guards_and_no_export()
    test_drift_zero_when_nothing_moved()
    test_drift_selection_and_modes()
    test_drift_kernel()
    test_aggregate_drift_math()
    test_aggregate_drift_tau_and_guards()
    test_collect_old_proto_drift()
    test_round_trip_client_to_server()
    test_strategies_fold_drift()
    print("\nall old-only / drift-compensation checks passed")
