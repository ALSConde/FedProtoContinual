"""
Tests backbones (CNN, LSTM, GRU, Transformer, ...) plugged into the
FedProtoContinual pipeline (global/local adapter + PrototypeClassifier).

Place this file in FedProtoContinual/scripts/ and run it from the repository root:

    python scripts/test_backbone.py --list-backbones
    python scripts/test_backbone.py --backbone gru                    # smoke + synthetic Class-IL
    python scripts/test_backbone.py --backbone cnn,lstm,gru,transformer   # compare all
    python scripts/test_backbone.py --backbone gru --scenario centralized   # without Class-IL
    python scripts/test_backbone.py --backbone gru --smoke-only
    python scripts/test_backbone.py --backbone my_pkg.my_mod:MyNet    # custom backbone
    python scripts/test_backbone.py --backbone gru --data utd --data-root ./storage/utd_mhad

Backbone contract
--------------------
    nn.Module with __init__(input_dim: int, hidden_dim: int, **kw)
    forward(x: Tensor[B, C, T]) -> Tensor[B, hidden_dim]

The output must have exactly `hidden_dim` features, as expected by the
global adapter, AlphaGate, and PrototypeClassifier.

What is tested
---------------
1. Smoke test (contract + integration with the project model):
   shape/finite output, gradient flow, eval determinism, KD teacher
   (frozen_global_embed_fn), get/set_global_arrays round-trip (federated
   serialization), and BatchNorm presence (problematic with FedAvg).
2. Centralized Class-IL simulation, mirroring a client + server:
   classifier expansion -> train_fn (CE + L_proto + KD) -> prototype
   statistics -> PrototypeAggregator (adaptive EMA) -> update_from_global.
   Reports accuracy, F1, BWT, and average forgetting.

Note: dynamic local-adapter expansion and the voting protocol are NOT
exercised here; the focus is the backbone + prototype classifier.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.client.ClientTask import train_fn  # noqa: E402
from src.model.Models import FCLModel  # noqa: E402
from src.model.Models import FeatureExtractor as RepoFeatureExtractor  # noqa: E402
from src.model.layers.PrototypeMemory import PrototypeMemory  # noqa: E402
from src.server.PrototypeAggregator import PrototypeAggregator  # noqa: E402


# =============================================================================
# Backbones  (B, C, T) -> (B, hidden_dim)
# Use GroupNorm/LayerNorm, never BatchNorm (BN buffers + FedAvg are problematic).
# =============================================================================
class CNN1DBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        chans = [input_dim, 32, 64, 128]
        layers: list[nn.Module] = []
        for i in range(3):
            layers += [
                nn.Conv1d(
                    chans[i],
                    chans[i + 1],
                    kernel_size=5,
                    padding=2,
                    stride=1 if i == 0 else 2,
                ),
                nn.GroupNorm(8, chans[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(chans[-1], hidden_dim)

    def forward(self, x):
        h = self.conv(x).mean(dim=-1)  # global average pooling over time
        return self.proj(self.drop(h))


class _RNNBackbone(nn.Module):
    rnn_cls: type[nn.RNNBase]

    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0
    ):
        super().__init__()
        self.rnn = self.rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        out, _ = self.rnn(x.permute(0, 2, 1))  # (B, T, C)
        return out.mean(dim=1)  # average over the sequence


class LSTMBackbone(_RNNBackbone):
    rnn_cls = nn.LSTM


class GRUBackbone(_RNNBackbone):
    rnn_cls = nn.GRU


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.0,
        max_len: int = 4096,
    ):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})."
            )
        self.embed = nn.Linear(input_dim, hidden_dim)
        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : hidden_dim // 2]
        self.register_buffer(
            "pe", pe, persistent=False
        )  # outside the federated state_dict
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x.permute(0, 2, 1))  # (B, T, D)
        h = h + self.pe[: h.size(1)]
        h = self.encoder(h)
        return self.norm(h.mean(dim=1))


def _repo_default(input_dim: int, hidden_dim: int, dropout: float = 0.0):
    return RepoFeatureExtractor(input_dim, hidden_dim)  # project CNN + LSTM


BACKBONES = {
    "cnn": CNN1DBackbone,
    "lstm": LSTMBackbone,
    "gru": GRUBackbone,
    "transformer": TransformerBackbone,
    "repo_default": _repo_default,
}


def build_backbone(
    spec: str, input_dim: int, hidden_dim: int, dropout: float
) -> nn.Module:
    """`spec` is a registry name or 'package.module:Class'."""
    if spec in BACKBONES:
        factory = BACKBONES[spec]
    elif ":" in spec:
        mod_name, cls_name = spec.split(":", 1)
        factory = getattr(importlib.import_module(mod_name), cls_name)
    else:
        raise ValueError(
            f"Unknown backbone '{spec}'. Use --list-backbones or 'module:Class'."
        )
    try:
        return factory(input_dim, hidden_dim, dropout=dropout)
    except TypeError:  # custom backbone without a `dropout` argument
        return factory(input_dim, hidden_dim)


def build_model(spec: str, args) -> FCLModel:
    model = FCLModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        d_hat_global=args.d_hat_global,
        d_hat_local=args.d_hat_local,
    )
    model.feature_extractor = build_backbone(
        spec, args.input_dim, args.hidden_dim, args.dropout
    )
    return model


# =============================================================================
# Data
# =============================================================================
def make_synthetic(num_classes, input_dim, window, n_train, n_test, seed):
    """Generate synthetic sinusoidal data with class-specific frequency/amplitude/offset."""
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.5, 6.0, size=(num_classes, input_dim))
    amp = rng.uniform(0.5, 2.0, size=(num_classes, input_dim))
    off = rng.normal(0, 0.3, size=(num_classes, input_dim))
    t = np.linspace(0, 2 * np.pi, window)[None, None, :]

    def gen(n_per_class):
        X, y = [], []
        for c in range(num_classes):
            phase = rng.uniform(0, 2 * np.pi, size=(n_per_class, input_dim, 1))
            sig = amp[c][None, :, None] * np.sin(freq[c][None, :, None] * t + phase)
            sig += off[c][None, :, None] + rng.normal(0, 0.3, size=sig.shape)
            X.append(sig.astype(np.float32))
            y.append(np.full(n_per_class, c))
        return (
            torch.from_numpy(np.concatenate(X)),
            torch.from_numpy(np.concatenate(y)).long(),
        )

    return (*gen(n_train), *gen(n_test))


def load_utd(root, window, stride, eval_subjects, max_classes):
    from src.utils.data.utd_mahd_dataset import (
        UTDMAHDInertial,
    )  # delayed import (scipy)

    ds = UTDMAHDInertial(root, window_size=window, stride=stride)
    X = (
        torch.from_numpy(np.stack([w for w, _, _ in ds.windows]))
        .permute(0, 2, 1)
        .contiguous()
    )
    y = torch.from_numpy(ds.labels).long()
    subj = torch.from_numpy(ds.subjects())
    test_mask = torch.isin(subj, torch.tensor(eval_subjects))
    keep = y < max_classes
    tr, te = (~test_mask) & keep, test_mask & keep
    mean = X[tr].mean(dim=(0, 2), keepdim=True)
    std = X[tr].std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    X = (X - mean) / std
    return X[tr], y[tr], X[te], y[te]


# =============================================================================
# Smoke test
# =============================================================================
def smoke_test(spec: str, args, device) -> bool:
    print(f"\n=== SMOKE TEST: {spec} ===")
    ok = True

    def check(cond: bool, msg: str, fatal: bool = True):
        nonlocal ok
        print(f"  [{'OK' if cond else ('FALHA' if fatal else 'AVISO')}] {msg}")
        if not cond and fatal:
            ok = False

    torch.manual_seed(args.seed)
    model = build_model(spec, args).to(device)
    B, C, T, D = 8, args.input_dim, args.window_size, args.hidden_dim
    x = torch.randn(B, C, T, device=device)

    n_bb = sum(p.numel() for p in model.feature_extractor.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  parameters: backbone={n_bb:,} | total model={n_all:,}")

    # 1. Shape contract
    model.eval()
    with torch.no_grad():
        feats = model.feature_extractor(x)
    check(
        feats.shape == (B, D),
        f"backbone: (B,C,T)={tuple(x.shape)} -> {tuple(feats.shape)} (expected {(B, D)})",
    )
    if feats.shape != (B, D):
        return False
    check(bool(torch.isfinite(feats).all()), "backbone output is finite")
    check(
        feats.std(dim=0).mean().item() > 1e-6,
        "output varies across samples (no trivial collapse)",
    )

    # 2. Prototypes + logits
    g = torch.Generator().manual_seed(0)
    n_cls = 3
    model.classifier.update_from_global(
        torch.randn(n_cls, D, generator=g), torch.arange(n_cls)
    )
    with torch.no_grad():
        logits = model(x)
    check(
        logits.shape == (B, n_cls),
        f"logits: {tuple(logits.shape)} (expected {(B, n_cls)})",
    )
    check(bool(torch.isfinite(logits).all()), "logits are finite")

    # 3. Gradient flow to the backbone
    model.train()
    y = torch.randint(0, n_cls, (B,), device=device)
    model.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    no_grad = [
        n
        for n, p in model.feature_extractor.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    bad = [
        n
        for n, p in model.feature_extractor.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    check(
        not no_grad,
        f"all backbone parameters receive gradients {no_grad[:3] if no_grad else ''}",
    )
    check(not bad, f"backbone gradients are finite {bad[:3] if bad else ''}")
    gnorm = (
        sum(
            p.grad.norm() ** 2
            for p in model.feature_extractor.parameters()
            if p.grad is not None
        )
        ** 0.5
    )
    check(float(gnorm) > 0, f"backbone gradient norm = {float(gnorm):.3e}")

    # 4. Eval determinism (catches forgotten active dropout)
    model.eval()
    with torch.no_grad():
        a, b = model.embed(x), model.embed(x)
    check(torch.allclose(a, b, atol=1e-6), "embed() is deterministic in eval")

    # 5. KD teacher == shared branch (in eval)
    teacher = model.frozen_global_embed_fn()
    with torch.no_grad():
        _, h_shared = model.embed_both(x)
        h_teacher = teacher(x)
    check(
        torch.allclose(h_shared, h_teacher, atol=1e-5),
        "frozen_global_embed_fn == shared branch",
    )

    # 6. Federated serialization round-trip
    state = {k: v.detach().cpu().clone() for k, v in model.get_global_arrays().items()}
    torch.manual_seed(args.seed + 1)  # different weights before loading
    fresh = build_model(spec, args).to(device)
    try:
        fresh.set_global_arrays(state)
        fresh.eval()
        with torch.no_grad():
            _, h_fresh = fresh.embed_both(x)
        check(
            torch.allclose(h_shared, h_fresh, atol=1e-5),
            "get/set_global_arrays reproduces the shared branch",
        )
    except Exception as e:  # noqa: BLE001
        check(False, f"get/set_global_arrays failed: {type(e).__name__}: {e}")
    non_float = [k for k, v in state.items() if not v.is_floating_point()]
    check(
        not non_float,
        f"federated state only contains float tensors {non_float[:3] if non_float else ''}",
        fatal=False,
    )

    # 7. BatchNorm
    bn = [
        n
        for n, m in model.feature_extractor.named_modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    check(
        not bn,
        f"no BatchNorm in backbone (BN statistics do not combine with FedAvg) {bn[:2] if bn else ''}",
        fatal=False,
    )

    # 8. Latency
    model.train()
    xb = torch.randn(args.batch_size, C, T, device=device)
    for _ in range(2):
        model.embed(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        model.embed(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(
        f"  forward latency (batch={args.batch_size}): {(time.perf_counter() - t0) / 10 * 1e3:.1f} ms"
    )

    print(f"  => {'PASSED' if ok else 'FAILED'}")
    return ok


# =============================================================================
# Class-IL simulation (one client + server)
# =============================================================================
@torch.no_grad()
def predict(model, X, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(
            model.classifier(model.embed(X[i : i + bs].to(device))).argmax(1).cpu()
        )
    return torch.cat(out)


def macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def prototype_diagnostics(model, seen):
    w = model.classifier.get_weights_normalized()[list(seen)]
    zero = int((model.classifier.prototypes[list(seen)].norm(dim=1) < 1e-8).sum())
    if len(seen) < 2:
        return {"zero_protos": zero, "mean_cos": float("nan")}
    sim = w @ w.T
    off = sim[~torch.eye(len(seen), dtype=torch.bool, device=sim.device)]
    return {"zero_protos": zero, "mean_cos": off.mean().item()}


@torch.no_grad()
def final_diagnostics(model, schedule, data, device, bs=256):
    """Separates 'the backbone forgot' from 'the prototypes/classifier became stale'.

    class_il      : standard accuracy (argmax over all classes).
    task_il       : argmax restricted to the classes of the true step (removes cross-step
                    interference; measures only separation within each step).
    oracle_proto  : RECALCULATED prototypes using the final backbone on the training data
                    of all classes (an upper bound on what the current classifier
                    could achieve if its prototypes were not stale).
    """
    Xtr, ytr, Xte, yte = data
    model.eval()

    def emb(X):
        return torch.cat(
            [model.embed(X[i : i + bs].to(device)).cpu() for i in range(0, len(X), bs)]
        )

    Hte = emb(Xte)
    logits = model.classifier(Hte.to(device)).cpu()
    class_il = (logits.argmax(1) == yte).float().mean().item()

    correct = 0
    for classes in schedule:
        m = torch.isin(yte, torch.tensor(classes))
        pred = torch.tensor(classes)[logits[m][:, classes].argmax(1)]
        correct += (pred == yte[m]).sum().item()
    task_il = correct / len(yte)

    Htr = torch.nn.functional.normalize(emb(Xtr), dim=1)
    n_cls = int(yte.max()) + 1
    P = torch.stack([Htr[ytr == c].mean(0) for c in range(n_cls)])
    sim = (
        torch.nn.functional.normalize(Hte, dim=1)
        @ torch.nn.functional.normalize(P, dim=1).T
    )
    oracle = (sim.argmax(1) == yte).float().mean().item()
    return {"class_il": class_il, "task_il": task_il, "oracle_proto": oracle}


def run_class_il(spec: str, args, data, device) -> dict:
    Xtr, ytr, Xte, yte = data
    num_classes = int(max(ytr.max(), yte.max())) + 1
    schedule = [
        list(range(s, min(s + args.classes_per_step, num_classes)))
        for s in range(0, num_classes, args.classes_per_step)
    ]
    n_steps = len(schedule)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = build_model(spec, args).to(device)
    aggregator = PrototypeAggregator(embedding_dim=args.hidden_dim, tau=args.tau)
    known: set[int] = set()
    R = np.full(
        (n_steps, n_steps), np.nan
    )  # R[i, j]: accuracy on step-j classes after training step i
    overall_acc, overall_f1 = [], []

    print(
        f"\n=== CLASS-IL: {spec} | {num_classes} classes in {n_steps} steps of {args.classes_per_step} ==="
    )
    t_start = time.perf_counter()

    for step, classes in enumerate(schedule):
        idx = torch.isin(ytr, torch.tensor(classes)).nonzero().squeeze(1)
        loader = DataLoader(
            TensorDataset(Xtr[idx], ytr[idx]), batch_size=args.batch_size, shuffle=True
        )

        # KD teacher: snapshot of the global branch at the start of each step > 0 (like the client)
        model.eval()  # deepcopy preserves mode -> teacher without dropout
        teacher = model.frozen_global_embed_fn() if step > 0 else None
        lambda_kd = args.lambda_kd if step > 0 else 0.0

        losses = []
        for _ in range(args.rounds_per_step):
            need = max(model.classifier.num_classes, max(classes) + 1)
            if need > model.classifier.num_classes:
                model.classifier._expand(
                    need
                )  # new rows are zero-initialized until the server aggregates

            memory = PrototypeMemory(
                args.hidden_dim, max(model.classifier.num_classes, 1), device
            )
            loss = train_fn(
                model,
                loader,
                memory,
                epochs=args.local_epochs,
                lr=args.lr,
                device=device,
                known_consolidated=known,
                lambda_proto=args.lambda_proto,
                lambda_kd=lambda_kd,
                kd_mode=args.kd_mode,
                kd_temperature=args.kd_temperature,
                frozen_embed_fn=teacher,
            )
            sum_h, counts, ids = memory.get_stats()
            memory.reset()
            aggregator.aggregate([(sum_h, counts, ids)])
            mu, all_ids = aggregator.get_prototypes_raw()
            model.classifier.update_from_global(mu.to(device), all_ids)
            known.update(int(c) for c in all_ids.tolist())
            losses.append(loss)

        # Evaluation over all seen classes
        seen = [c for s in schedule[: step + 1] for c in s]
        te_idx = torch.isin(yte, torch.tensor(seen)).nonzero().squeeze(1)
        pred = predict(model, Xte[te_idx], device)
        y_true = yte[te_idx]
        acc = (pred == y_true).float().mean().item()
        f1 = macro_f1(y_true, pred, seen)
        overall_acc.append(acc)
        overall_f1.append(f1)
        for j in range(step + 1):
            m = torch.isin(y_true, torch.tensor(schedule[j]))
            R[step, j] = (pred[m] == y_true[m]).float().mean().item()

        diag = prototype_diagnostics(model, seen)
        per_step = " ".join(f"{R[step, j]:.2f}" for j in range(step + 1))
        print(
            f"  step {step + 1}/{n_steps} | loss {losses[0]:.3f}->{losses[-1]:.3f} | acc {acc:.3f} | F1 {f1:.3f} "
            f"| acc per step [{per_step}] | mean prototype cosine {diag['mean_cos']:.2f}"
            + (
                f" | PROTOS ZERADOS: {diag['zero_protos']}"
                if diag["zero_protos"]
                else ""
            )
        )

    diag_final = final_diagnostics(model, schedule, data, device)
    T = n_steps
    if T > 1:
        bwt = float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))
        forgetting = float(
            np.mean([np.nanmax(R[j : T - 1, j]) - R[T - 1, j] for j in range(T - 1)])
        )
    else:
        bwt = forgetting = float("nan")

    result = {
        "backbone": spec,
        "final_acc": overall_acc[-1],
        "final_f1": overall_f1[-1],
        "avg_incremental_acc": float(np.mean(overall_acc)),
        "bwt": bwt,
        "avg_forgetting": forgetting,
        "params": sum(p.numel() for p in model.parameters()),
        "time_s": time.perf_counter() - t_start,
        "task_il_acc": diag_final["task_il"],
        "oracle_proto_acc": diag_final["oracle_proto"],
        "acc_matrix": R.tolist(),
    }
    print(
        f"  => final acc {result['final_acc']:.3f} | F1 {result['final_f1']:.3f} | mean incremental acc "
        f"{result['avg_incremental_acc']:.3f} | BWT {bwt:+.3f} | forgetting {forgetting:.3f} | {result['time_s']:.0f}s"
    )
    print(
        f"  final diagnostics: Class-IL {diag_final['class_il']:.3f} | Task-IL (argmax within the step) "
        f"{diag_final['task_il']:.3f} | recalculated prototypes {diag_final['oracle_proto']:.3f}"
    )
    return result


# =============================================================================
# Centralized scenario (without Class-IL): all classes at once
# =============================================================================
def run_linear_baseline(
    spec: str, args, data, device, num_classes: int, epochs: int
) -> float:
    """Same architecture (backbone + adapters + AlphaGate), but with a linear head
    trained with CE. It serves as a reference: if it also fails to learn, the
    problem is in the backbone/data, not the prototype classifier."""
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(args.seed)
    model = build_model(spec, args).to(device)
    head = nn.Linear(args.hidden_dim, num_classes).to(device)
    # classifier.scale is not used in the forward here (grad None -> Adam ignores it)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=args.lr
    )
    loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True
    )
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            nn.functional.cross_entropy(head(model.embed(x)), y).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat(
            [
                head(model.embed(Xte[i : i + 256].to(device))).argmax(1).cpu()
                for i in range(0, len(Xte), 256)
            ]
        )
    return (pred == yte).float().mean().item()


def run_centralized(spec: str, args, data, device) -> dict:
    Xtr, ytr, Xte, yte = data
    num_classes = int(max(ytr.max(), yte.max())) + 1
    total_epochs = args.rounds * args.local_epochs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = build_model(spec, args).to(device)
    aggregator = PrototypeAggregator(embedding_dim=args.hidden_dim, tau=args.tau)
    known: set[int] = set()
    loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True
    )
    classes = list(range(num_classes))

    # No old knowledge -> KD disabled (lambda_kd=0). The "null" teacher only
    # avoids deep-copying the model every round; its value is multiplied by zero.
    null_teacher = lambda x: torch.zeros(
        x.size(0), args.hidden_dim, device=x.device
    )  # noqa: E731

    print(
        f"\n=== CENTRALIZED: {spec} | {num_classes} classes at once | {args.rounds} rounds x {args.local_epochs} epoch(s) ==="
    )
    t_start = time.perf_counter()
    model.classifier._expand(
        num_classes
    )  # zero-initialized rows until the first aggregation round
    history = []

    for r in range(args.rounds):
        memory = PrototypeMemory(args.hidden_dim, num_classes, device)
        loss = train_fn(
            model,
            loader,
            memory,
            epochs=args.local_epochs,
            lr=args.lr,
            device=device,
            known_consolidated=known,
            lambda_proto=args.lambda_proto,
            lambda_kd=0.0,
            kd_mode=args.kd_mode,
            kd_temperature=args.kd_temperature,
            frozen_embed_fn=null_teacher,
        )
        sum_h, counts, ids = memory.get_stats()
        memory.reset()
        aggregator.aggregate([(sum_h, counts, ids)])
        mu, all_ids = aggregator.get_prototypes_raw()
        model.classifier.update_from_global(mu.to(device), all_ids)
        known.update(int(c) for c in all_ids.tolist())

        pred = predict(model, Xte, device)
        test_acc = (pred == yte).float().mean().item()
        history.append({"round": r + 1, "loss": loss, "test_acc": test_acc})
        if r == 0 or (r + 1) % args.log_every == 0 or r == args.rounds - 1:
            print(
                f"  round {r + 1:>3}/{args.rounds} | loss {loss:.3f} | test acc {test_acc:.3f}"
            )

    pred_te = predict(model, Xte, device)
    pred_tr = predict(model, Xtr, device)
    test_acc = (pred_te == yte).float().mean().item()
    train_acc = (pred_tr == ytr).float().mean().item()
    f1 = macro_f1(yte, pred_te, classes)
    best = max(history, key=lambda h: h["test_acc"])
    oracle = final_diagnostics(model, [classes], data, device)["oracle_proto"]
    diag = prototype_diagnostics(model, classes)

    baseline = None
    if not args.no_baseline:
        baseline = run_linear_baseline(
            spec, args, data, device, num_classes, total_epochs
        )

    chance = 1.0 / num_classes
    print(
        f"  => test acc {test_acc:.3f} | F1 {f1:.3f} | train acc {train_acc:.3f} | best test {best['test_acc']:.3f} (round {best['round']})"
    )
    print(
        f"     recalculated prototypes {oracle:.3f} | mean prototype cosine {diag['mean_cos']:.2f} | chance {chance:.3f}"
    )
    if baseline is not None:
        print(
            f"     linear baseline (same architecture, {total_epochs} epochs): {baseline:.3f}"
        )

    # Automatic interpretation
    if test_acc < 2 * chance and (baseline is None or baseline < 2 * chance):
        verdict = "DOES NOT LEARN: neither the prototype classifier nor the linear baseline exceeds ~2x chance -> review backbone/data/lr."
    elif baseline is not None and test_acc < baseline - 0.10:
        verdict = "PROTOTYPES BELOW BASELINE (>10 pp): the backbone learns, but the prototype pipeline (aggregation/EMA/losses) is losing performance."
    elif train_acc - test_acc > 0.25:
        verdict = "LEARNS, WITH OVERFITTING (train >> test): consider dropout, more data, or less capacity."
    else:
        verdict = "OK: the model learns in the centralized scenario."
    print(f"     interpretation: {verdict}")

    return {
        "backbone": spec,
        "scenario": "centralized",
        "test_acc": test_acc,
        "test_f1": f1,
        "train_acc": train_acc,
        "best_test_acc": best["test_acc"],
        "best_round": best["round"],
        "oracle_proto_acc": oracle,
        "linear_baseline_acc": baseline,
        "params": sum(p.numel() for p in model.parameters()),
        "time_s": time.perf_counter() - t_start,
        "history": history,
        "verdict": verdict,
    }


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--backbone",
        default="gru",
        help="name(s) separated by commas or 'module:Class'",
    )
    p.add_argument("--list-backbones", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--skip-smoke", action="store_true")
    # model
    p.add_argument("--input-dim", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--d-hat-global", type=int, default=16)
    p.add_argument("--d-hat-local", type=int, default=8)
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="see note about dropout in the KD teacher",
    )
    # data
    p.add_argument("--data", choices=["synthetic", "utd"], default="synthetic")
    p.add_argument("--data-root", default="./storage/utd_mhad")
    p.add_argument("--eval-subjects", default="6,7,8")
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="default: 12 (synthetic) / 27 (utd)",
    )
    p.add_argument(
        "--samples-per-class", type=int, default=120, help="training (synthetic)"
    )
    p.add_argument("--test-per-class", type=int, default=40, help="test (synthetic)")
    # training / Class-IL
    p.add_argument(
        "--scenario",
        choices=["class-il", "centralized"],
        default="class-il",
        help="class-il: incremental steps | centralized: all classes at once",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=30,
        help="training+aggregation rounds (scenario=centralized)",
    )
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the linear-head reference (centralized)",
    )
    p.add_argument("--classes-per-step", type=int, default=3)
    p.add_argument("--rounds-per-step", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-proto", type=float, default=1.0)
    p.add_argument("--lambda-kd", type=float, default=1.0)
    p.add_argument("--kd-mode", choices=["kl", "embedding_mse"], default="kl")
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--tau", type=float, default=15.0)
    # general
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_backbones:
        print("Available backbones:", ", ".join(BACKBONES), "| or 'module:Class'")
        return 0

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"device: {device} | seed: {args.seed}")
    specs = [s.strip() for s in args.backbone.split(",") if s.strip()]

    all_ok = True
    if not args.skip_smoke:
        for s in specs:
            all_ok &= smoke_test(s, args, device)
    if args.smoke_only:
        return 0 if all_ok else 1
    if not all_ok:
        print(
            "\nSmoke test failed; aborting the Class-IL simulation (use --skip-smoke to force)."
        )
        return 1

    if args.data == "synthetic":
        n_cls = args.num_classes or 12
        data = make_synthetic(
            n_cls,
            args.input_dim,
            args.window_size,
            args.samples_per_class,
            args.test_per_class,
            args.seed,
        )
    else:
        subjects = [int(s) for s in args.eval_subjects.split(",")]
        data = load_utd(
            args.data_root,
            args.window_size,
            args.stride,
            subjects,
            args.num_classes or 27,
        )
    print(
        f"data ({args.data}): training {tuple(data[0].shape)} | test {tuple(data[2].shape)}"
    )

    if args.scenario == "centralized":
        results = [run_centralized(s, args, data, device) for s in specs]
        if len(results) > 1:
            print("\n=== COMPARISON (centralized) ===")
            print(
                f"{'backbone':<14}{'params':>10}{'acc':>8}{'F1':>8}{'train':>8}{'linear':>8}{'oracle':>8}{'time(s)':>10}"
            )
            for r in sorted(results, key=lambda r: -r["test_acc"]):
                lin = (
                    "-"
                    if r["linear_baseline_acc"] is None
                    else f"{r['linear_baseline_acc']:.3f}"
                )
                print(
                    f"{r['backbone']:<14}{r['params']:>10,}{r['test_acc']:>8.3f}{r['test_f1']:>8.3f}"
                    f"{r['train_acc']:>8.3f}{lin:>8}{r['oracle_proto_acc']:>8.3f}{r['time_s']:>10.0f}"
                )
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(results, indent=2))
            print(f"results saved to {args.output_json}")
        return 0

    results = [run_class_il(s, args, data, device) for s in specs]

    if len(results) > 1:
        print("\n=== COMPARISON ===")
        print(
            f"{'backbone':<14}{'params':>10}{'acc':>8}{'F1':>8}{'acc_inc':>9}{'BWT':>8}{'forget':>8}{'taskIL':>8}{'oracle':>8}{'time(s)':>10}"
        )
        for r in sorted(results, key=lambda r: -r["final_acc"]):
            print(
                f"{r['backbone']:<14}{r['params']:>10,}{r['final_acc']:>8.3f}{r['final_f1']:>8.3f}"
                f"{r['avg_incremental_acc']:>9.3f}{r['bwt']:>+8.3f}{r['avg_forgetting']:>8.3f}{r['task_il_acc']:>8.3f}{r['oracle_proto_acc']:>8.3f}{r['time_s']:>10.0f}"
            )
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"results saved to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
