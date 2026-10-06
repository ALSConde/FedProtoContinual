"""Per-client partition statistics (samples and class entropy) for each Dirichlet alpha.

Runs without Flower: it calls the same ``load_data`` the clients call, so the
numbers describe exactly the data each client trains on. Partitions are deterministic
given (seed, partition_id, alpha) while ``dirichlet-mode`` is static, so one pass over
the alpha x seed grid is enough and no federated run is needed.

Usage (from the repo root):
    python scripts/partition_stats.py \
        --alphas 1000 1.0 0.5 0.3 0.1 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out outputs/partition_stats

Outputs:
    <out>/per_client.csv   one row per (alpha, seed, client)
    <out>/summary.csv      one row per alpha, averaged over seeds and clients
A very large alpha (>= 1000) is the "no retention" reference: the code has no switch
that disables retention, but keep fractions are ~1 for every class in that regime.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data.utd_mahd_dataset import (  # noqa: E402
    _get_cached_dataset,
    load_data,
    parse_int_list_config,
    resolve_available_subjects,
)


def entropy(counts: np.ndarray, num_classes: int) -> tuple[float, float]:
    """Shannon entropy in nats and normalised by ln(num_classes) (1 = uniform)."""
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0
    p = counts[counts > 0] / total
    h = float(-(p * np.log(p)).sum())
    return h, h / math.log(num_classes)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats between two class distributions."""
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "pyproject.toml"))
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "partition_stats"))
    ap.add_argument("--round", type=int, default=1, help="only matters if dirichlet-mode=dynamic")
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)["tool"]["flwr"]["app"]["config"]

    root = str(ROOT / cfg["data-root"]) if str(cfg["data-root"]).startswith(".") else str(cfg["data-root"])
    window, stride = int(cfg["window-size"]), int(cfg["stride"])
    held_out = parse_int_list_config(cfg.get("server-eval-subjects"))
    num_classes = int(cfg["num-classes-total"])
    dataset = _get_cached_dataset(root, window, stride)
    num_clients = len(resolve_available_subjects(dataset.subjects(), held_out))
    labels = dataset.labels

    rows = []
    for alpha in args.alphas:
        for seed in args.seeds:
            per_client_counts = []
            for pid in range(num_clients):
                # classes_per_step=None mirrors training-scenario="federated" (no schedule).
                train_loader, val_loader, _ = load_data(
                    pid,
                    num_clients,
                    root=root,
                    window_size=window,
                    stride=stride,
                    dirichlet_alpha=alpha,
                    batch_size=int(cfg["batch-size"]),
                    current_round=args.round,
                    classes_per_step=None,
                    num_classes_total=num_classes,
                    dirichlet_mode=str(cfg.get("dirichlet-mode", "static")).lower(),
                    held_out_subjects=held_out,
                    partition_mode=str(cfg.get("partition-mode", "subject")),
                    val_split=str(cfg.get("val-split", "recording")),
                    seed=seed,
                )
                tr = np.asarray(train_loader.dataset.indices, dtype=int)
                va = np.asarray(val_loader.dataset.indices, dtype=int)
                counts = np.bincount(labels[tr], minlength=num_classes).astype(float)
                per_client_counts.append(counts)
                h, h_norm = entropy(counts, num_classes)
                rows.append(
                    {
                        "alpha": alpha,
                        "seed": seed,
                        "client": pid,
                        "n_train": len(tr),
                        "n_val": len(va),
                        "n_classes": int((counts > 0).sum()),
                        "entropy_nats": h,
                        "entropy_norm": h_norm,
                    }
                )
            # divergence of each client from the federation-wide class distribution
            glob = np.sum(per_client_counts, axis=0)
            glob_p = glob / glob.sum() if glob.sum() > 0 else glob
            covered = int((glob > 0).sum())  # classes present in at least one client
            for pid, counts in enumerate(per_client_counts):
                p = counts / counts.sum() if counts.sum() > 0 else counts
                row = rows[-(num_clients - pid)]
                row["jsd_vs_global"] = jsd(p, glob_p)
                row["global_covered"] = covered

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["alpha", "seed", "client", "n_train", "n_val", "n_classes",
            "entropy_nats", "entropy_norm", "jsd_vs_global", "global_covered"]
    with open(out / "per_client.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    metrics = ["n_train", "n_val", "n_classes", "entropy_norm", "jsd_vs_global", "global_covered"]
    summary = []
    for alpha in args.alphas:
        sel = [r for r in rows if r["alpha"] == alpha]
        row = {"alpha": alpha}
        for m in metrics:
            vals = np.array([r[m] for r in sel], dtype=float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        # worst cases: clients that can break candidacy/voting or training
        row["n_train_min"] = min(r["n_train"] for r in sel)
        row["n_val_min"] = min(r["n_val"] for r in sel)
        row["n_classes_min"] = min(r["n_classes"] for r in sel)
        row["global_covered_min"] = min(r["global_covered"] for r in sel)
        summary.append(row)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"{'alpha':>8} {'n_train':>14} {'n_classes':>12} {'H_norm':>12} {'JSD':>12} "
          f"{'cover(min)':>10} {'min tr/val':>11}")
    for r in summary:
        print(
            f"{r['alpha']:>8g} "
            f"{r['n_train_mean']:>7.0f}±{r['n_train_std']:<6.0f} "
            f"{r['n_classes_mean']:>6.1f}±{r['n_classes_std']:<4.1f} "
            f"{r['entropy_norm_mean']:>6.3f}±{r['entropy_norm_std']:<4.3f} "
            f"{r['jsd_vs_global_mean']:>6.3f}±{r['jsd_vs_global_std']:<4.3f} "
            f"{r['global_covered_mean']:>5.1f}({r['global_covered_min']:>2d}) "
            f"{r['n_train_min']:>5d}/{r['n_val_min']:<4d}"
        )


if __name__ == "__main__":
    main()