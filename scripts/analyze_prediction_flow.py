"""Summarise prediction_flow.csv of Class-IL runs (written by PredictionFlowMonitor).

For the test samples of OLD classes it reports where the global model sends them, at the
first round of a step (right after new classes appear), mid-step and at the step end:

  correct        predicted as the true class
  -> current     predicted as a class of the current step   (recency bias: new classes
                 capture old samples; the old representation may still be fine)
  -> other old   predicted as another old class            (old classes confused among
                 themselves: the representation itself degraded)

    python scripts/analyze_prediction_flow.py outputs/cil_subject_nodir_v4 --rounds-per-step 10
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in csv.DictReader(f)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--rounds-per-step", type=int, default=10)
    args = ap.parse_args()
    rps = args.rounds_per_step

    root = Path(args.results_dir)
    files = sorted(root.glob("*/seed_*/prediction_flow.csv"))
    if not files:
        print(f"no prediction_flow.csv under {root}/<profile>/seed_*/")
        return 1

    # profile -> moment -> list of (correct, to_current, to_other_old, pred_share, true_share)
    acc: dict = defaultdict(lambda: defaultdict(list))
    for path in files:
        profile = path.parent.parent.name
        for r in load(path):
            if not r["n_old"]:
                continue
            k = int(r["round"] - 1) % rps
            moment = "first round" if k == 0 else ("step end" if k == rps - 1 else ("mid-step" if k == rps // 2 else None))
            if moment is None:
                continue
            n = r["n_old"]
            acc[profile][moment].append(
                (r["old_correct"] / n, r["old_to_current"] / n, r["old_to_other_old"] / n,
                 r["pred_share_current"], r["true_share_current"])
            )

    print(f"{'profile':14s} {'moment':12s} {'correct':>8s} {'->current':>10s} {'->other old':>12s}   {'pred share cur':>14s} {'true share cur':>14s}")
    for profile in sorted(acc):
        for moment in ("first round", "mid-step", "step end"):
            rows = acc[profile].get(moment)
            if not rows:
                continue
            m = np.mean(np.array(rows, dtype=float), axis=0)
            print(f"{profile:14s} {moment:12s} {m[0]:8.3f} {m[1]:10.3f} {m[2]:12.3f}   {m[3]:14.3f} {m[4]:14.3f}")
    print("\nRead: a large '->current' with a small '->other old' means recency bias; a large"
          "\n'->other old' means the old classes are confused among themselves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
