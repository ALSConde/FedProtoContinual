"""Sanity checks for (1) FCLModel.parameter_report and (2) KD-teacher sync after incorporation.

Run from the project root:  python scripts/test_params_and_kd_sync.py
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from flwr.app import ConfigRecord

from src.client import Client_app as ca
from src.model.Models import FCLModel, count_parameters
from src.model.blocks.Adapter import Adapter, adapter_topology


class _Ctx:  # minimal stand-in for flwr's Context (only .state / .run_config are used)
    def __init__(self):
        self.state = {}
        self.run_config = {}


def test_parameter_report():
    m = FCLModel(input_dim=6, hidden_dim=64, d_hat_global=16, d_hat_local=8)
    r0 = m.parameter_report()
    assert r0["total"] == r0["shared"] + r0["local"]
    assert r0["incorporated_adapters"] == 0
    assert r0["total"] == sum(p.numel() for p in m.parameters())

    m.adapter_local.expand_width(4)
    r1 = m.parameter_report()
    assert r1["adapter_local"] > r0["adapter_local"], "width expansion must add params"
    assert r1["shared"] == r0["shared"], "local expansion must not touch shared params"

    m.adapter_local.expand_depth(None)
    r2 = m.parameter_report()
    assert r2["adapter_local"] > r1["adapter_local"], "depth expansion must add params"

    m.load_incorporated_topology([adapter_topology(m.adapter_local)])
    r3 = m.parameter_report()
    assert r3["incorporated_adapters"] == r2["adapter_local"]
    assert r3["shared"] == r2["shared"] + r3["incorporated_adapters"]

    off = FCLModel(input_dim=6, hidden_dim=64, use_local_adapter=False)
    assert off.parameter_report()["adapter_local"] == 0
    assert off.parameter_report()["alpha_gate"] == 0
    print("parameter_report OK:", r0["total"], "->", r3["total"])


def test_kd_teacher_sync():
    ctx = _Ctx()
    m = FCLModel(input_dim=6, hidden_dim=64)
    ca._snapshot_global_branch(
        ctx, m
    )  # step-boundary snapshot, no incorporated adapter
    x_feat = torch.randn(4, 64)

    fn0 = ca._load_kd_teacher_embed_fn(ctx, torch.device("cpu"))
    assert (
        len(
            torch.load(
                __import__("io").BytesIO(
                    ctx.state[ca._KD_TEACHER_SNAPSHOT_KEY]["blob"]
                ),
                weights_only=False,
            )["incorporated_adapters"]
        )
        == 0
    )

    # Incorporation accepted: model now holds a (non-trivial) incorporated adapter.
    cand = Adapter(64, 8)
    with torch.no_grad():
        for p in cand.parameters():
            p.normal_(0, 0.2)
    m.load_incorporated_topology([adapter_topology(cand)])
    m.incorporated_adapters[0].load_state_dict(cand.state_dict())

    ca._sync_kd_teacher_incorporated(ctx, m)
    bundle = torch.load(
        __import__("io").BytesIO(ctx.state[ca._KD_TEACHER_SNAPSHOT_KEY]["blob"]),
        weights_only=False,
    )
    assert (
        len(bundle["incorporated_adapters"]) == 1
    ), "teacher must hold the new adapter"
    # feature_extractor / adapter_global stay the step-boundary snapshot
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            bundle["adapter_global"].state_dict().values(),
            m.adapter_global.state_dict().values(),
        )
    )

    # Teacher output == student's shared branch output now (same fe/ag, same incorporated).
    m.eval()
    x = torch.randn(4, 6, 50)
    with torch.no_grad():
        _, h_shared = m.embed_both(x)
    fn1 = ca._load_kd_teacher_embed_fn(ctx, torch.device("cpu"))
    assert torch.allclose(
        fn1(x), h_shared, atol=1e-5
    ), "teacher must match student's shared branch"

    # Idempotent: no change in topology -> no rewrite.
    blob_before = ctx.state[ca._KD_TEACHER_SNAPSHOT_KEY]["blob"]
    ca._sync_kd_teacher_incorporated(ctx, m)
    assert ctx.state[ca._KD_TEACHER_SNAPSHOT_KEY]["blob"] == blob_before

    # Reverted: adapter removed -> teacher drops it too.
    m.load_incorporated_topology([])
    ca._sync_kd_teacher_incorporated(ctx, m)
    bundle = torch.load(
        __import__("io").BytesIO(ctx.state[ca._KD_TEACHER_SNAPSHOT_KEY]["blob"]),
        weights_only=False,
    )
    assert len(bundle["incorporated_adapters"]) == 0
    print("kd teacher sync OK (accept -> add, no-op when unchanged, revert -> drop)")


if __name__ == "__main__":
    test_parameter_report()
    test_kd_teacher_sync()
