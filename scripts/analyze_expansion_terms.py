"""Calibrate theta-exp / interference-floor from the per-round expansion diagnostics.

Reads the 'Aggregated MetricRecord' lines of run_log.txt files (written by the Flower server) that
carry g_cosine_old/new, g_entropy, interference, expansion_signal and their *_valid companions.
The aggregated value of a term is mean(v * valid) / mean(valid), i.e. the weighted mean over the
clients that could compute it, so each term is divided by its *_valid here.

    python scripts/analyze_expansion_terms.py outputs/cil_subject_nodir_v5/A4_full --rounds-per-step 10

It prints, by position k inside a step (k = 1 is the first round of a new step; step 1 skipped),
the mean of each term and the quantiles of the interference at k = 1, and for a few floors the
theta-exp that makes a given fraction of the step boundaries trigger:
    interference > floor + theta * (1 - floor)   <=>   signal > theta.
"""

import argparse
import ast
import re
from pathlib import Path

import numpy as np

TERMS = (
    "g_cosine_old",
    "g_cosine_new",
    "g_entropy",
    "interference",
    "expansion_signal",
)
LINE = re.compile(r"Aggregated MetricRecord: (\{.*\})")


def load_rounds(path: Path) -> list:
    rounds = []
    for match in LINE.finditer(path.read_text(errors="ignore")):
        try:
            record = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            continue
        if "interference_valid" in record:
            rounds.append(record)
    return rounds


def term(record: dict, name: str):
    valid = record.get(f"{name}_valid", 0.0)
    return record[name] / valid if valid > 1e-9 else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "run_dir",
        help="a run directory or a profile directory containing seed_*/run_log.txt",
    )
    ap.add_argument("--rounds-per-step", type=int, default=10)
    args = ap.parse_args()
    R = args.rounds_per_step

    root = Path(args.run_dir)
    logs = (
        [root / "run_log.txt"]
        if (root / "run_log.txt").exists()
        else sorted(root.glob("**/run_log.txt"))
    )
    by_k = {k: {t: [] for t in TERMS} for k in range(1, R + 1)}
    n_logs = 0
    for log in logs:
        rounds = load_rounds(log)
        if not rounds:
            continue
        n_logs += 1
        for i, rec in enumerate(rounds, start=1):
            if i <= R:
                continue
            k = (i - 1) % R + 1
            for t in TERMS:
                v = term(rec, t)
                if v is not None:
                    by_k[k][t].append(v)
    if n_logs == 0:
        print(
            "no run_log with the new expansion diagnostics found (needs interference_valid)."
        )
        return 1

    print(f"{n_logs} runs; mean of each term by position k inside a step (steps >= 2)")
    print(f"{'k':>3} " + " ".join(f"{t:>16s}" for t in TERMS))
    for k in range(1, R + 1):
        cells = [
            f"{np.mean(by_k[k][t]):16.4f}" if by_k[k][t] else f"{'-':>16s}"
            for t in TERMS
        ]
        print(f"{k:3d} " + " ".join(cells))

    interf = np.array(by_k[1]["interference"])
    if interf.size == 0:
        return 0
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    print(
        "\ninterference at k=1 (step boundaries): "
        + "  ".join(f"p{int(q * 100)}={np.quantile(interf, q):.3f}" for q in qs)
    )
    print(
        "\ntheta-exp that makes a fraction of the boundaries trigger (signal = interference mode):"
    )
    print(
        f"{'floor':>6} | "
        + " ".join(f"{int(f * 100):>3d}% trigger" for f in (0.25, 0.5, 0.75))
    )
    for floor in (0.0, 0.25, 0.5, 0.75):
        cells = []
        for frac in (0.25, 0.5, 0.75):
            thr = np.quantile(
                interf, 1.0 - frac
            )  # raw interference above which `frac` trigger
            theta = (thr - floor) / (1.0 - floor)
            cells.append(f"{theta:11.3f}" if theta > 0 else f"{'<=0 (always)':>11s}")
        print(f"{floor:6.2f} | " + " ".join(cells))
    print(
        "\nWith expansion-signal=interference or combined also set expansion-cooldown >= rounds-per-step:"
        "\nthe interference is a property of the prototypes and does not fall as the step progresses."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
