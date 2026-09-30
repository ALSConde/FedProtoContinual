"""Analyze a grid_search.py output directory: distributions, marginal means, and
correlations between the searched hyperparameters and the outcome metrics.

Three sources are collected and cross-referenced:

  1. leaderboard.csv        -- one row per trial, produced by grid_search.py.
                               Hyperparameter values + final-summary metrics
                               (avg_inc_acc, server_eval_acc, bwt, avg_forgetting,
                               worst_class_forgetting[_id], worst_class_bwt[_id], ...).
  2. trial_*/run_log.txt    -- per-round training/evaluation MetricRecords logged
                               by Flower (expansion_g, alpha_mean, contribution_ratio,
                               incorp_* counters, train_loss, server_eval_acc, ...).
  3. trial_*/forgetting_per_class.csv -- per-round, per-class n/acc/bwt/forgetting,
                               written by ForgettingMonitor (ServerEvaluation.py).

Which leaderboard.csv columns are "hyperparameters" is inferred automatically: any
column not in a known set of housekeeping/metric names is treated as a swept
hyperparameter, so this keeps working as new hyperparameters are added to future
grid searches without touching this script.

Usage
-----
    python scripts/analyze_grid_search.py
    python scripts/analyze_grid_search.py --grid-dir outputs/grid_search --metric avg_inc_acc
    python scripts/analyze_grid_search.py --top-n 10 --percentiles 50,75,90,95,99

Outputs (written under <grid-dir>/analysis/ by default)
---------------------------------------------------------
    round_metrics.csv    long-format per-round metrics for every trial/phase
    per_class.csv        concatenated forgetting_per_class.csv for every trial
    marginal_effects.csv mean/std/count of each metric, grouped by each hyperparameter
    correlations.csv     Spearman correlation of every hyperparameter vs every metric

Everything is also printed to the console as compact tables, so for a quick look
you don't need to open the CSVs at all.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover - scipy is already a project dependency
    spearmanr = None


# Columns in leaderboard.csv that are NOT swept hyperparameters. Extend this set
# (or pass --extra-metric-columns) if grid_search.py starts reporting new metrics.
NON_HYPERPARAM_COLUMNS = {
    "trial",
    "trial_dir",
    "status",
    "round",
    "avg_inc_acc",
    "server_eval_acc",
    "server_eval_loss",
    "bwt",
    "avg_forgetting",
    "worst_class_forgetting",
    "worst_class_forgetting_id",
    "worst_class_bwt",
    "worst_class_bwt_id",
    "error",
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ROUND_RE = re.compile(r"\[ROUND (\d+)/")
_METRIC_RECORD_RE = re.compile(r"MetricRecord: (\{.*\})")
_PHASE_MARKERS = {
    "aggregate_train": "train",
    "Global evaluation": "global",
    "aggregate_evaluate": "eval",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _coerce(value: str) -> Any:
    """Turn a leaderboard.csv / MetricRecord string into int/float/bool/str."""
    if value == "":
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except (TypeError, ValueError):
            continue
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def load_leaderboard(grid_dir: Path) -> list[dict[str, Any]]:
    path = grid_dir / "leaderboard.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Did you point --grid-dir at a grid_search.py output "
            "directory (the one containing leaderboard.csv and trial_XXXX/ folders)?"
        )
    with path.open(newline="") as f:
        rows = [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return rows


def parse_run_log(trial_dir: Path) -> list[dict[str, Any]]:
    """Extract one row per MetricRecord found in run_log.txt, tagged with round/phase."""
    log_path = trial_dir / "run_log.txt"
    if not log_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    round_no = 0
    phase: str | None = None
    with log_path.open(errors="ignore") as f:
        for raw_line in f:
            line = _ANSI_RE.sub("", raw_line).rstrip()

            m = _ROUND_RE.search(line)
            if m:
                round_no = int(m.group(1))
                continue

            for marker, name in _PHASE_MARKERS.items():
                if marker in line:
                    phase = name
                    break
            else:
                m = _METRIC_RECORD_RE.search(line)
                if m:
                    try:
                        record = ast.literal_eval(m.group(1))
                    except (ValueError, SyntaxError):
                        continue
                    if isinstance(record, dict):
                        rows.append(
                            {"trial": trial_dir.name, "round": round_no, "phase": phase, **record}
                        )
    return rows


def load_forgetting_per_class(trial_dir: Path) -> list[dict[str, Any]]:
    path = trial_dir / "forgetting_per_class.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = [{"trial": trial_dir.name, **{k: _coerce(v) for k, v in row.items()}} for row in csv.DictReader(f)]
    return rows


def collect(grid_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (leaderboard_rows, round_metric_rows, per_class_rows)."""
    leaderboard = load_leaderboard(grid_dir)
    round_metrics: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []

    for row in leaderboard:
        trial_dir = Path(row.get("trial_dir") or (grid_dir / str(row["trial"])))
        if not trial_dir.exists():
            # trial_dir in the CSV may be an absolute path from a different machine/
            # container; fall back to <grid_dir>/<trial name>.
            trial_dir = grid_dir / str(row["trial"])
        if not trial_dir.exists():
            print(f"warning: could not locate directory for trial '{row['trial']}', skipping logs", file=sys.stderr)
            continue
        round_metrics.extend(parse_run_log(trial_dir))
        per_class.extend(load_forgetting_per_class(trial_dir))

    return leaderboard, round_metrics, per_class


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def detect_hyperparameter_columns(
    leaderboard: list[dict[str, Any]], extra_metric_columns: Iterable[str] = ()
) -> list[str]:
    exclude = NON_HYPERPARAM_COLUMNS | set(extra_metric_columns)
    all_columns = {k for row in leaderboard for k in row}
    return sorted(all_columns - exclude)


def detect_metric_columns(
    leaderboard: list[dict[str, Any]], hyperparam_columns: Iterable[str]
) -> list[str]:
    exclude = set(hyperparam_columns) | {"trial", "trial_dir", "status", "error"}
    all_columns = {k for row in leaderboard for k in row}
    return sorted(all_columns - exclude)


def percentiles(values: list[float], pcts: list[float]) -> dict[float, float]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {p: float("nan") for p in pcts}
    n = len(values)
    out = {}
    for p in pcts:
        if n == 1:
            out[p] = values[0]
            continue
        rank = (p / 100) * (n - 1)
        lo, hi = int(rank), min(int(rank) + 1, n - 1)
        frac = rank - lo
        out[p] = values[lo] + (values[hi] - values[lo]) * frac
    return out


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        v = row.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
    return out


def signal_distributions(round_metrics: list[dict[str, Any]], signals: list[str]) -> dict[str, dict]:
    pcts = [50, 75, 90, 95, 98, 99, 100]
    report = {}
    for sig in signals:
        vals = numeric_values(round_metrics, sig)
        if not vals:
            continue
        report[sig] = {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            **{f"p{p}": v for p, v in percentiles(vals, pcts).items()},
        }
    return report


def marginal_effects(
    leaderboard: list[dict[str, Any]], hyperparam_columns: list[str], metric_columns: list[str]
) -> list[dict[str, Any]]:
    """For each (hyperparameter, metric) pair, group trials by hyperparameter value
    and report mean/std/count of the metric within each group."""
    out: list[dict[str, Any]] = []
    for hp in hyperparam_columns:
        values = sorted({row.get(hp) for row in leaderboard if row.get(hp) is not None}, key=str)
        if len(values) < 2:
            continue  # constant across all trials in this grid -- nothing to compare
        for metric in metric_columns:
            for val in values:
                group = [row for row in leaderboard if row.get(hp) == val]
                metric_vals = numeric_values(group, metric)
                if not metric_vals:
                    continue
                out.append(
                    {
                        "hyperparameter": hp,
                        "value": val,
                        "metric": metric,
                        "n": len(metric_vals),
                        "mean": statistics.fmean(metric_vals),
                        "std": statistics.pstdev(metric_vals) if len(metric_vals) > 1 else 0.0,
                    }
                )
    return out


def correlations(
    leaderboard: list[dict[str, Any]], hyperparam_columns: list[str], metric_columns: list[str]
) -> list[dict[str, Any]]:
    if spearmanr is None:
        print(
            "warning: scipy not available, skipping correlation computation "
            "(pip install scipy --break-system-packages)",
            file=sys.stderr,
        )
        return []

    out: list[dict[str, Any]] = []
    for hp in hyperparam_columns:
        hp_vals_all = [row.get(hp) for row in leaderboard]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in hp_vals_all if v is not None):
            continue  # non-numeric (e.g. a mode string) -- skip, can't rank sensibly
        if len({v for v in hp_vals_all if v is not None}) < 2:
            continue  # constant -- correlation undefined
        for metric in metric_columns:
            xs, ys = [], []
            for row in leaderboard:
                x, y = row.get(hp), row.get(metric)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(y, bool):
                    xs.append(float(x))
                    ys.append(float(y))
            if len(xs) < 3 or len({*xs}) < 2 or len({*ys}) < 2:
                continue
            rho, pval = spearmanr(xs, ys)
            out.append(
                {"hyperparameter": hp, "metric": metric, "spearman_rho": rho, "p_value": pval, "n": len(xs)}
            )
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list({k for row in rows for k in row})
    # keep a stable, readable column order: put common keys first
    priority = ["trial", "round", "phase", "hyperparameter", "value", "metric", "class_id"]
    fieldnames = [k for k in priority if k in fieldnames] + sorted(k for k in fieldnames if k not in priority)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, Any]], columns: list[str], title: str, max_rows: int | None = None) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("(no data)")
        return
    shown = rows[:max_rows] if max_rows else rows
    widths = {c: max(len(c), *(len(f"{row.get(c, '')}") for row in shown)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for row in shown:
        print("  ".join(f"{row.get(c, '')}".ljust(widths[c]) for c in columns))
    if max_rows and len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more rows in the CSV)")


def fmt(x: Any, nd: int = 4) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--grid-dir", type=Path, default=Path("outputs/grid_search"), help="grid_search.py output directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="where to write the CSV reports (default: <grid-dir>/analysis)"
    )
    parser.add_argument("--metric", default="avg_inc_acc", help="primary metric to rank trials / highlight by")
    parser.add_argument("--top-n", type=int, default=10, help="how many top trials to print")
    parser.add_argument(
        "--signals",
        default="expansion_g,alpha_mean,contribution_ratio",
        help="comma-separated per-round signal names (from run_log.txt) to report distributions for",
    )
    parser.add_argument(
        "--extra-metric-columns",
        default="",
        help="comma-separated leaderboard.csv columns to additionally treat as metrics, not hyperparameters",
    )
    args = parser.parse_args()

    grid_dir = args.grid_dir
    output_dir = args.output_dir or (grid_dir / "analysis")
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    extra_metric_columns = [s.strip() for s in args.extra_metric_columns.split(",") if s.strip()]

    leaderboard, round_metrics, per_class = collect(grid_dir)
    ok_trials = [r for r in leaderboard if r.get("status") != "error"]
    if len(ok_trials) < len(leaderboard):
        print(f"note: excluding {len(leaderboard) - len(ok_trials)} failed trial(s) (status == 'error') from analysis")

    hyperparam_columns = detect_hyperparameter_columns(ok_trials, extra_metric_columns)
    metric_columns = detect_metric_columns(ok_trials, hyperparam_columns)

    print(f"Loaded {len(leaderboard)} trials ({len(ok_trials)} usable), "
          f"{len(round_metrics)} per-round log rows, {len(per_class)} per-class rows.")
    print(f"Detected hyperparameters: {', '.join(hyperparam_columns) or '(none)'}")
    print(f"Detected metrics:         {', '.join(metric_columns) or '(none)'}")

    # -- Top trials -----------------------------------------------------------
    ranked = sorted(
        (r for r in ok_trials if isinstance(r.get(args.metric), (int, float))),
        key=lambda r: r[args.metric],
        reverse=True,
    )
    top_cols = ["trial", *hyperparam_columns, args.metric]
    print_table(
        [{**r, args.metric: fmt(r.get(args.metric))} for r in ranked],
        top_cols,
        f"Top {args.top_n} trials by {args.metric}",
        max_rows=args.top_n,
    )

    # -- Per-round signal distributions ----------------------------------------
    dist = signal_distributions(round_metrics, signals)
    dist_rows = [{"signal": sig, **{k: fmt(v) for k, v in stats.items()}} for sig, stats in dist.items()]
    print_table(
        dist_rows,
        ["signal", "n", "mean", "std", "min", "p50", "p75", "p90", "p95", "p98", "p99", "p100"],
        "Per-round signal distributions (pooled across all trials/rounds/clients)",
    )

    # -- Marginal effects -------------------------------------------------------
    marg = marginal_effects(ok_trials, hyperparam_columns, metric_columns)
    marg_primary = [row for row in marg if row["metric"] == args.metric]
    print_table(
        [{**row, "mean": fmt(row["mean"]), "std": fmt(row["std"])} for row in marg_primary],
        ["hyperparameter", "value", "n", "mean", "std"],
        f"Marginal effect of each hyperparameter on {args.metric}",
    )

    # -- Correlations -------------------------------------------------------
    corr = correlations(ok_trials, hyperparam_columns, metric_columns)
    corr_primary = sorted(
        (row for row in corr if row["metric"] == args.metric), key=lambda r: -abs(r["spearman_rho"])
    )
    print_table(
        [{**row, "spearman_rho": fmt(row["spearman_rho"]), "p_value": fmt(row["p_value"])} for row in corr_primary],
        ["hyperparameter", "spearman_rho", "p_value", "n"],
        f"Spearman correlation of each hyperparameter with {args.metric} (sorted by |rho|)",
    )
    if len(metric_columns) > 1:
        print(f"\n(Full hyperparameter x metric correlation matrix for all {len(metric_columns)} "
              f"metrics written to correlations.csv)")

    # -- Per-class summary (which classes are hardest, pooled across trials) ----
    if per_class:
        by_class: dict[Any, list[float]] = {}
        for row in per_class:
            cid = row.get("class_id")
            fgt = row.get("forgetting")
            if cid is not None and isinstance(fgt, (int, float)):
                by_class.setdefault(cid, []).append(float(fgt))
        class_rows = sorted(
            (
                {"class_id": cid, "n": len(vals), "mean_forgetting": statistics.fmean(vals)}
                for cid, vals in by_class.items()
            ),
            key=lambda r: -r["mean_forgetting"],
        )
        print_table(
            [{**r, "mean_forgetting": fmt(r["mean_forgetting"])} for r in class_rows],
            ["class_id", "n", "mean_forgetting"],
            "Per-class mean forgetting, pooled across all trials/rounds (hardest first)",
            max_rows=10,
        )

    # -- Write CSVs -------------------------------------------------------------
    write_csv(output_dir / "round_metrics.csv", round_metrics)
    write_csv(output_dir / "per_class.csv", per_class)
    write_csv(output_dir / "marginal_effects.csv", marg)
    write_csv(output_dir / "correlations.csv", corr)
    print(f"\nCSV reports written to {output_dir}/")


if __name__ == "__main__":
    main()