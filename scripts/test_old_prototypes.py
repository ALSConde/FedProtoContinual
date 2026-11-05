"""Learnable old-class prototypes (PRD) and their EMA aggregation on the server.

Client: ClientTask.train_fn(prd_learn_protos=True) -> OldPrototypeState / prd_out.
Server: PrototypeAggregator.aggregate_old, collect_old_proto_updates and the two strategies.
Run: python scripts/test_old_prototypes.py
"""

import math
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.client.ClientTask import (
    export_old_prototypes,
    make_old_prototype_state,
    provisional_prototypes,
    train_fn,
)
from src.model.Models import FCLModel
from src.model.layers.PrototypeMemory import PrototypeMemory
from src.server.PrototypeAggregator import (
    PrototypeAggregator,
    collect_old_proto_updates,
)

DEVICE = torch.device("cpu")
HIDDEN, WINDOW, CHANNELS = 64, 60, 6


# ----------------------------------------------------------------------------------
# server: PrototypeAggregator.aggregate_old
# ----------------------------------------------------------------------------------
def _aggregator(tau=10.0, old_tau=None, d=4):
    agg = PrototypeAggregator(embedding_dim=d, tau=tau, old_tau=old_tau)
    sum_h = torch.tensor([[2.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 4.0, 0]])
    counts = torch.tensor([1.0, 1.0, 1.0])  # first sight: alpha = 1 -> mu = mean
    agg.aggregate([(sum_h, counts, torch.tensor([0, 1, 2]))])
    return agg


def test_aggregate_old_ema():
    agg = _aggregator(tau=10.0, old_tau=50.0)
    before = agg.mu_global.clone()
    p_c1 = torch.tensor(
        [[1.0, 1.0, 0, 0], [0, 0, 2.0, 2.0]]
    )  # classes 0 and 3? only 0 valid
    # client A (n=30) reports classes 0 and 1; client B (n=10) reports class 0 only
    a = (torch.tensor([0, 1]), torch.tensor([[2.0, 2.0, 0, 0], [0, 3.0, 3.0, 0]]), 30)
    b = (torch.tensor([0]), torch.tensor([[0.0, 4.0, 0, 0]]), 10)
    updated = agg.aggregate_old([a, b])
    assert sorted(updated.tolist()) == [0, 1]
    # class 0: weighted mean of the clients, evidence N = 40
    p_bar0 = (30 * a[1][0] + 10 * b[1][0]) / 40
    alpha0 = 1 - math.exp(-40 / 50)
    assert torch.allclose(
        agg.mu_global[0], (1 - alpha0) * before[0] + alpha0 * p_bar0, atol=1e-6
    )
    # class 1: one client, N = 30
    alpha1 = 1 - math.exp(-30 / 50)
    assert torch.allclose(
        agg.mu_global[1], (1 - alpha1) * before[1] + alpha1 * a[1][1], atol=1e-6
    )
    # class 2 was not reported: untouched
    assert torch.equal(agg.mu_global[2], before[2])
    del p_c1
    print("aggregate_old EMA OK")


def test_aggregate_old_tau_controls_amount():
    """Same evidence, larger tau -> the old prototype moves less (the user-facing knob)."""
    moves = []
    for old_tau in (5.0, 50.0, 500.0):
        agg = _aggregator(old_tau=old_tau)
        before = agg.mu_global[0].clone()
        agg.aggregate_old([(torch.tensor([0]), torch.tensor([[0.0, 0, 0, 9.0]]), 40)])
        moves.append(float((agg.mu_global[0] - before).norm()))
    assert moves[0] > moves[1] > moves[2] > 0, moves
    # old_tau defaults to tau
    assert _aggregator(tau=7.0).old_tau == 7.0
    # more evidence -> more movement at fixed tau
    small, large = _aggregator(old_tau=50.0), _aggregator(old_tau=50.0)
    b = small.mu_global[0].clone()
    small.aggregate_old([(torch.tensor([0]), torch.tensor([[0.0, 0, 0, 9.0]]), 5)])
    large.aggregate_old([(torch.tensor([0]), torch.tensor([[0.0, 0, 0, 9.0]]), 100)])
    assert float((large.mu_global[0] - b).norm()) > float(
        (small.mu_global[0] - b).norm()
    )
    try:
        PrototypeAggregator(embedding_dim=4, tau=10.0, old_tau=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("old_tau <= 0 must raise")
    print("tau controls the amount OK")


def test_aggregate_old_ignores_unknown_classes():
    agg = _aggregator()
    before = agg.mu_global.clone()
    # class 7 does not exist on the server; class 1 exists; n=0 contributes nothing
    upd = [
        (torch.tensor([7]), torch.ones(1, 4), 100),
        (torch.tensor([1]), torch.ones(1, 4), 0),
        (torch.tensor([]), torch.zeros(0, 4), 10),
    ]
    assert len(agg.aggregate_old(upd)) == 0
    assert torch.equal(agg.mu_global, before)
    # a class created but never initialised is skipped as well
    agg._expand(5)
    assert len(agg.aggregate_old([(torch.tensor([4]), torch.ones(1, 4), 50)])) == 0
    assert not bool(agg.initialized[4]) and float(agg.mu_global[4].norm()) == 0.0
    print("unknown / uninitialised classes ignored OK")


def test_old_update_does_not_touch_aggregate_path():
    """`aggregate` (classes with samples) is unchanged; aggregate_old follows it."""
    agg = _aggregator(tau=10.0, old_tau=50.0)
    stats = (torch.tensor([[5.0, 5.0, 0, 0]]), torch.tensor([20.0]), torch.tensor([0]))
    agg.aggregate([stats])  # class 0 has samples, EMA with tau=10
    mu0 = agg.mu_global[0].clone()
    alpha = 1 - math.exp(-20 / 10)
    assert torch.allclose(
        mu0,
        (1 - alpha) * torch.tensor([2.0, 0, 0, 0])
        + alpha * torch.tensor([0.25, 0.25, 0, 0]) * 1.0 * 0
        + alpha * (stats[0][0] / 20),
        atol=1e-6,
    )
    agg.aggregate_old([(torch.tensor([1]), torch.tensor([[0.0, 1.0, 0, 0]]), 10)])
    assert torch.equal(agg.mu_global[0], mu0)  # class 0 untouched by the old update
    print("aggregate / aggregate_old separation OK")


def test_collect_old_proto_updates():
    def reply(cfg):
        content = {"config": cfg} if cfg is not None else {}
        return SimpleNamespace(has_content=lambda: True, content=content)

    good = pickle.dumps((torch.tensor([1, 2]), torch.ones(2, 4), 33))
    empty = pickle.dumps((torch.tensor([], dtype=torch.long), torch.zeros(0, 4), 33))
    replies = [
        reply({"old_proto_update": good}),
        reply({"proto_stats": b"x"}),  # no update
        reply(None),  # no config record
        reply({"old_proto_update": empty}),  # nothing learned
        SimpleNamespace(has_content=lambda: False, content={}),
    ]
    out = collect_old_proto_updates(replies)
    assert len(out) == 1 and out[0][2] == 33 and out[0][0].tolist() == [1, 2]
    print("collect_old_proto_updates OK")


# ----------------------------------------------------------------------------------
# client
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


def _ready_model():
    model = _model(5)
    provisional_prototypes(model, _loader([0, 1, 2, 3, 4]), DEVICE, set())
    return model


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


def test_state_selects_old_classes_only():
    model = _ready_model()
    # present classes 3,4; known 0,1,2 -> all three are absent from the data = old
    st = make_old_prototype_state(model, [0, 1, 2], {3, 4})
    assert st.class_ids == [0, 1, 2]
    assert torch.equal(st.teacher, model.classifier.prototypes[[0, 1, 2]])
    assert st.student.requires_grad and not st.teacher.requires_grad
    # a known class that IS in the data is not "old"
    st2 = make_old_prototype_state(model, [0, 1, 2, 3], {3, 4})
    assert st2.class_ids == [0, 1, 2]
    # a class without prototype is skipped
    model.classifier.prototypes[1] = 0.0
    assert make_old_prototype_state(model, [0, 1, 2], {3}).class_ids == [0, 2]
    print("old-class selection OK")


def test_export_restores_norm_and_reports_drift():
    model = _ready_model()
    st = make_old_prototype_state(model, [0, 1, 2], {3, 4})
    ids, protos, drift = export_old_prototypes(st)
    assert ids.tolist() == [0, 1, 2] and abs(drift) < 1e-6  # untouched -> no drift
    assert torch.allclose(protos, st.teacher, atol=1e-5)
    with torch.no_grad():
        st.student.mul_(3.7)  # pure rescale: direction unchanged
    _, p2, d2 = export_old_prototypes(st)
    assert torch.allclose(p2.norm(dim=1), st.init_norm, atol=1e-5) and abs(d2) < 1e-6
    with torch.no_grad():
        st.student[0] = torch.randn_like(st.student[0])
    _, p3, d3 = export_old_prototypes(st)
    assert d3 > 0.05 and torch.allclose(p3.norm(dim=1), st.init_norm, atol=1e-5)
    print("export (norm restored, drift) OK")


def test_train_learns_old_prototypes():
    for split, kd_shared in ((False, 0.0), (True, 1.0)):
        model = _ready_model()
        received = model.classifier.prototypes.clone()
        out = {}
        loss = _train(
            model,
            _loader([3, 4]),
            prd_learn_protos=True,
            prd_out=out,
            split_optimizers=split,
            lambda_kd_shared=kd_shared,
        )
        assert math.isfinite(loss), split
        assert out["class_ids"].tolist() == [0, 1, 2], out
        assert out["num_samples"] == 24
        assert out["drift"] > 1e-6, (split, out["drift"])  # PRD actually moved them
        assert torch.allclose(
            out["prototypes"].norm(dim=1), received[:3].norm(dim=1), atol=1e-4
        )
        # the client does NOT rewrite its own classifier: the server is the only writer
        assert torch.equal(model.classifier.prototypes, received)
    print("train_fn learns and exports old prototypes (plain + split) OK")


def test_split_local_only_kd_does_not_move_prototypes():
    """Split mode: only the SHARED KD term may move the prototypes (they live in the shared
    space); with the KD on the local branch alone the export is a no-op (zero drift)."""
    model = _ready_model()
    out = {}
    _train(
        model,
        _loader([3, 4]),
        prd_learn_protos=True,
        prd_out=out,
        split_optimizers=True,
        lambda_kd_shared=0.0,
    )
    assert out["drift"] < 1e-6, out["drift"]
    print("split: local-only KD leaves prototypes untouched OK")


def test_off_by_default_and_guards():
    # flag off: nothing exported, and training is identical to not passing the arguments
    losses, weights = [], []
    for kwargs in ({}, dict(prd_learn_protos=False, prd_out={})):
        model = _ready_model()
        losses.append(_train(model, _loader([3, 4]), **kwargs))
        weights.append([p.detach().clone() for p in model.parameters()])
    assert losses[0] == losses[1]
    assert all(torch.equal(a, b) for a, b in zip(*weights))
    # needs a PRD kd mode
    try:
        _train(_ready_model(), _loader([3, 4]), kd_mode="kl", prd_learn_protos=True)
    except ValueError:
        pass
    else:
        raise AssertionError("prd_learn_protos with kd_mode=kl must raise")
    # no teacher (lambda_kd = 0 -> no KD) or nothing old: inactive, empty export
    out = {}
    _train(
        _ready_model(),
        _loader([3, 4]),
        lambda_kd=0.0,
        prd_learn_protos=True,
        prd_out=out,
    )
    assert out == {}
    out = {}
    _train(_ready_model(), _loader([0, 1, 2]), prd_learn_protos=True, prd_out=out)
    assert out == {}  # every known class is present locally -> no old class
    # present-class prototypes are never exported
    out = {}
    _train(_ready_model(), _loader([2, 3]), prd_learn_protos=True, prd_out=out)
    assert out["class_ids"].tolist() == [0, 1]
    print("off by default / guards OK")


def test_client_server_round_trip():
    """Two clients learn old prototypes, the server folds them in; new classes still follow
    the usual sample-count path in the same round."""
    server = PrototypeAggregator(embedding_dim=HIDDEN, tau=10.0, old_tau=40.0)
    seed_model = _ready_model()
    mu = seed_model.classifier.prototypes.clone()
    server.aggregate(
        [(mu, torch.ones(5), torch.arange(5))]
    )  # server state = the prototypes both clients receive
    base = server.mu_global.clone()

    updates = []
    for seed in (0, 1):
        model = _model(5)
        model.classifier.prototypes.copy_(server.mu_global)
        out = {}
        _train(model, _loader([3, 4], seed=seed), prd_learn_protos=True, prd_out=out)
        updates.append((out["class_ids"], out["prototypes"], out["num_samples"]))
    updated = server.aggregate_old(updates)
    assert updated.tolist() == [0, 1, 2]
    moved = (server.mu_global - base).norm(dim=1)
    assert bool((moved[:3] > 0).all()) and bool((moved[3:] == 0).all())
    # direction moved, norm kept (the client restored it; EMA of equal-norm vectors <= norm)
    cos = F.cosine_similarity(server.mu_global[:3], base[:3], dim=1)
    assert bool((cos < 1.0).all()) and bool((cos > 0.5).all())
    print("client -> server round trip OK")


# ----------------------------------------------------------------------------------
# strategies (real flwr objects)
# ----------------------------------------------------------------------------------
def _make_reply(partition, proto_stats, old_update=None, params=None):
    from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
    from flwr.supercore.task_identity import TaskIdentity

    # Message() reads the identity of the running task; outside a Flower run it is unset
    TaskIdentity.run_id, TaskIdentity.task_id, TaskIdentity.node_id = 1, 1, 1
    request = Message(
        content=RecordDict({"config": ConfigRecord({})}),
        dst_node_id=partition,
        message_type="train",
    )
    cfg = {"proto_stats": pickle.dumps(proto_stats)}
    if old_update is not None:
        cfg["old_proto_update"] = pickle.dumps(old_update)
    content = RecordDict(
        {
            "arrays": ArrayRecord({"w": torch.ones(2) * (partition + 1)}),
            "metrics": MetricRecord({"num-examples": 10, "train_loss": 1.0}),
            "config": ConfigRecord(cfg),
        }
    )
    return Message(content=content, reply_to=request)


def test_strategies_apply_old_updates():
    from src.server.FedAvgStrategy import FedAvgStrategy
    from src.server.FedProxStrategy import FedProxStrategy

    d = 4
    first = (
        torch.tensor([[2.0, 0, 0, 0], [0, 3.0, 0, 0]]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([0, 1]),
    )
    for cls, extra in ((FedAvgStrategy, {}), (FedProxStrategy, {"proximal_mu": 0.1})):
        strat = cls(embedding_dim=d, tau=10.0, old_tau=50.0, **extra)
        # round 1: classes 0 and 1 are born (no old update possible yet)
        strat.aggregate_train(1, [_make_reply(0, first)])
        base = strat.proto_aggregator.mu_global.clone()
        # round 2: a client saw only class 2 (new) and learned old prototypes of 0 and 1
        new_stats = (
            torch.tensor([[0, 0, 5.0, 0]]),
            torch.tensor([5.0]),
            torch.tensor([2]),
        )
        upd = (
            torch.tensor([0, 1]),
            torch.tensor([[1.0, 1.0, 0, 0], [0, 2.0, 2.0, 0]]),
            40,
        )
        strat.aggregate_train(2, [_make_reply(0, new_stats, upd)])
        mu = (
            strat.proto_aggregator.mu_global.clone()
        )  # snapshot: updated in place later
        alpha = 1 - math.exp(-40 / 50)
        assert torch.allclose(
            mu[0], (1 - alpha) * base[0] + alpha * upd[1][0], atol=1e-6
        )
        assert torch.allclose(
            mu[1], (1 - alpha) * base[1] + alpha * upd[1][1], atol=1e-6
        )
        assert bool(
            strat.proto_aggregator.initialized[2]
        )  # new class took the usual path
        # the broadcast bytes carry the updated old prototypes to the next round
        mu_sent, ids_sent = pickle.loads(strat._latest_proto_bytes)
        assert ids_sent.tolist() == [0, 1, 2]
        assert torch.allclose(mu_sent[0], mu[0])
        # a round whose clients only report an old update (no samples at all) still broadcasts
        strat.aggregate_train(
            3,
            [
                _make_reply(
                    0,
                    (
                        torch.zeros(0, d),
                        torch.zeros(0),
                        torch.tensor([], dtype=torch.long),
                    ),
                    (torch.tensor([0]), torch.ones(1, d), 25),
                )
            ],
        )
        mu_sent, _ = pickle.loads(strat._latest_proto_bytes)
        assert torch.allclose(mu_sent[0], strat.proto_aggregator.mu_global[0])
        assert not torch.allclose(mu_sent[0], mu[0])
    print("FedAvg / FedProx strategies fold old updates OK")


if __name__ == "__main__":
    test_aggregate_old_ema()
    test_aggregate_old_tau_controls_amount()
    test_aggregate_old_ignores_unknown_classes()
    test_old_update_does_not_touch_aggregate_path()
    test_collect_old_proto_updates()
    test_state_selects_old_classes_only()
    test_export_restores_norm_and_reports_drift()
    test_train_learns_old_prototypes()
    test_split_local_only_kd_does_not_move_prototypes()
    test_off_by_default_and_guards()
    test_client_server_round_trip()
    test_strategies_apply_old_updates()
    print("\nall old-prototype checks passed")
