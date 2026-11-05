"""How big is the PRD term next to the others, and how does it scale with K, T and B?

Synthetic probe (no data, no training): it builds near-collinear class prototypes (the regime
the loss-screen logs show: geo_proto_cos_mean 0.92-0.98), a teacher batch of embeddings of the
3 classes being learned, and a student = teacher + drift. For each loss term it prints the value
AND the norm of its gradient with respect to the student embeddings: the gradient, not the loss
value, tells which term steers the update (an Adam step rescales the total gradient but keeps
its direction, so a term that dominates the gradient dominates the direction).

Terms (the training objective with ce-local-classes, as in the screen):
  CE            cosine CE over the 3 local classes, scale s
  L_proto       2 (1 - cos(h, prototype of its class))
  KL(T2)        distillation_loss_kl (per-sample distribution over the K prototypes), x T^2
  PRD sum*T2    relation_distillation_loss as implemented (sum over the K prototypes, x T^2)
  PRD mean*T2   same, averaged over the K prototypes
  PRD sum       sum over the prototypes without the T^2 factor (the paper's formula)

It says nothing about the trained network: drift and geometry are assumptions you can change.
Run: python scripts/prd_magnitude_probe.py [--batch 32 --temperature 2 --scale 20 --dim 64]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.losses.Losses import distillation_loss_kl
from src.utils.losses.PrototypeLosses import relation_distillation_loss


def probe(K, drift, args, seeds=20):
    rows = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        d, b, s, t = args.dim, args.batch, args.scale, args.temperature
        shared = F.normalize(torch.randn(d), dim=0)  # common direction: the collapse
        centers = shared + torch.randn(27, d) * args.class_scale
        protos = F.normalize(centers, dim=1)[:K]
        local = [K - 3, K - 2, K - 1]  # the 3 classes of the current step
        y = torch.tensor(local)[torch.randint(0, 3, (b,))]
        h_t = centers[y] + torch.randn(b, d) * args.noise
        step = drift * torch.randn(b, d) * h_t.norm(dim=1, keepdim=True) / np.sqrt(d)
        h_s = (h_t + step).clone().requires_grad_(True)

        local_idx = torch.tensor([local.index(int(c)) for c in y])
        ce = F.cross_entropy(
            s * F.normalize(h_s, dim=1) @ F.normalize(protos[local], dim=1).t(),
            local_idx,
        )
        l_proto = (2 * (1 - (F.normalize(h_s, dim=1) * protos[y]).sum(1))).mean()
        kl = distillation_loss_kl(h_s, h_t, protos, t, s, s)
        prd = relation_distillation_loss(
            h_s, h_t, protos, temperature=t, scale_student=s
        )
        terms = {
            "CE": ce,
            "L_proto": l_proto,
            "KL(T2)": kl,
            "PRD sum*T2": prd,
            "PRD mean*T2": prd / K,
            "PRD sum": prd / t**2,
        }
        row = {}
        for name, loss in terms.items():
            (g,) = torch.autograd.grad(loss, h_s, retain_graph=True)
            row[name] = (loss.item(), g.norm().item())
        row["cos"] = F.cosine_similarity(h_s, h_t, dim=1).mean().item()
        pn = F.normalize(protos, dim=1)
        row["pcos"] = ((pn @ pn.t()).sum() - K).item() / (K * (K - 1))
        rows.append(row)
    names = [n for n in rows[0] if n not in ("cos", "pcos")]
    mean = {n: tuple(np.mean([r[n][i] for r in rows]) for i in (0, 1)) for n in names}
    mean["cos"] = float(np.mean([r["cos"] for r in rows]))
    mean["pcos"] = float(np.mean([r["pcos"] for r in rows]))
    return mean


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--scale", type=float, default=20.0, help="kd-logit-scale")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--class-scale", type=float, default=0.029, help="prototype spread")
    ap.add_argument("--noise", type=float, default=0.025, help="within-class spread")
    ap.add_argument("--drifts", type=float, nargs="+", default=[0.05, 0.15, 0.30])
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 6, 12, 24, 27])
    args = ap.parse_args()

    print(f"B={args.batch} s={args.scale} T={args.temperature} D={args.dim}")
    print("cells: loss / ||dloss/dh||_F, mean over 20 random worlds\n")
    for drift in args.drifts:
        print(f"student drift={drift}")
        names = ["CE", "L_proto", "KL(T2)", "PRD sum*T2", "PRD mean*T2", "PRD sum"]
        print(
            f"{'K':>3} {'cos(s,t)':>8} {'protoCos':>8} | "
            + " ".join(f"{n:>13}" for n in names)
            + " | grad PRD/CE"
        )
        for K in args.ks:
            m = probe(K, drift, args)
            cells = " ".join(f"{m[n][0]:6.3f}/{m[n][1]:5.2f}".rjust(13) for n in names)
            ratio = m["PRD sum*T2"][1] / max(m["CE"][1], 1e-12)
            print(f"{K:>3} {m['cos']:8.3f} {m['pcos']:8.3f} | {cells} | {ratio:8.1f}x")
        print()


if __name__ == "__main__":
    main()
