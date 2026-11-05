"""Does the semantic-drift compensation move old prototypes towards where the old classes went?

Synthetic, small, deterministic. Per world: (1) train on classes 0-2 and take their class means as
the prototypes ("stale" once the model moves); (2) keep a copy of that model, train on classes 3-5
ONLY (no distillation), so the embedding drifts; (3) measure where the old classes really are now
(their class means under the trained model, using old data the method never sees); (4) shift the
stale prototypes with the drift computed from the NEW data only (compute_old_prototype_drift, alpha=1)
and compare with the true position. 'global' shifts every prototype by the mean drift of all new
samples (bandwidth -> infinity): it shows how much of the effect is just the common shift.

It is a sanity check of the mechanism on toy data, NOT evidence about UTD-MHAD: how far the drift of
new samples predicts the drift of old ones depends on the data and the model.
Run: python scripts/sdc_probe.py [--worlds 8 --sigmas 0.5 1 2]
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from test_old_only_drift import (  # reuse the toy model / data helpers
    DEVICE,
    PrototypeMemory,
    _loader,
    _model,
    _train,
    provisional_prototypes,
)
from src.client.ClientTask import _device_batches, compute_old_prototype_drift


@torch.no_grad()
def class_means(model, loader, classes):
    model.eval()
    sums, counts = {c: 0.0 for c in classes}, {c: 0 for c in classes}
    for x, y in _device_batches(loader, DEVICE):
        h = model.embed_both(x)[1]
        for c in classes:
            m = y == c
            if m.any():
                sums[c] = sums[c] + h[m].sum(0)
                counts[c] += int(m.sum())
    return torch.stack([sums[c] / counts[c] for c in classes])


def world(seed, epochs_new):
    torch.manual_seed(seed)
    model = _model(6)
    old_loader = _loader([0, 1, 2], per_class=16, seed=seed)
    provisional_prototypes(
        model, _loader(range(6), per_class=16, seed=seed), DEVICE, set()
    )
    _train(
        model,
        old_loader,
        epochs=20,
        kd_mode="kl",
        lambda_kd=0.0,
        known_consolidated=set(),
        lr=1e-3,
    )
    stale = class_means(model, old_loader, [0, 1, 2])
    model.classifier.prototypes[:3] = stale
    start = copy.deepcopy(model).eval()
    new_loader = _loader([3, 4, 5], per_class=16, seed=seed + 50)
    provisional_prototypes(model, new_loader, DEVICE, {0, 1, 2})
    _train(
        model,
        new_loader,
        epochs=epochs_new,
        kd_mode="kl",
        lambda_kd=0.0,
        known_consolidated={0, 1, 2},
        lr=1e-3,
    )
    true_now = class_means(model, old_loader, [0, 1, 2])
    return model, start, new_loader, stale, true_now


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--epochs-new", type=int, default=15)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = ap.parse_args()

    labels = ["stale"] + [f"sdc s={s:g}" for s in args.sigmas] + ["global shift"]
    dist = {k: [] for k in labels}
    cosv = {k: [] for k in labels}
    for seed in range(args.worlds):
        model, start, new_loader, stale, true_now = world(seed, args.epochs_new)
        cand = {"stale": stale}
        for s in args.sigmas:
            _, sd, mass = compute_old_prototype_drift(
                model, start, new_loader, DEVICE, {0, 1, 2}, s
            )
            cand[f"sdc s={s:g}"] = stale + sd / mass.clamp_min(1e-12).unsqueeze(1)
        _, sd, mass = compute_old_prototype_drift(
            model, start, new_loader, DEVICE, {0, 1, 2}, 1e6
        )
        cand["global shift"] = stale + sd / mass.unsqueeze(1)
        for k, p in cand.items():
            dist[k].append((p - true_now).norm(dim=1).mean().item())
            cosv[k].append(F.cosine_similarity(p, true_now, dim=1).mean().item())
    base = np.array(dist["stale"])
    print(
        f"{args.worlds} toy worlds; distance / cosine to the true new position of the old classes"
    )
    print(
        f"{'prototype':>14} | {'mean dist':>9} {'vs stale':>9} {'worlds better':>13} | {'mean cos':>8}"
    )
    for k in labels:
        d = np.array(dist[k])
        print(
            f"{k:>14} | {d.mean():9.3f} {100 * (d.mean() / base.mean() - 1):+8.1f}% "
            f"{int((d < base).sum()):>8}/{args.worlds} | {np.mean(cosv[k]):8.3f}"
        )


if __name__ == "__main__":
    main()
