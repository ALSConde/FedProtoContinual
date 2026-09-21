"""Grid/random hyperparameter search for FedProtoContinual.

Launches multiple `flwr run .` executions, one for each hyperparameter
combination, with each execution writing its results (including the
per-class forgetting/BWT monitor) to an isolated subdirectory under
`--results-dir`. At the end, it reads the `summary.json` from each trial
(written by Server_app.py) and builds a leaderboard sorted by the
selected metric.

By default, it runs the "core" stage (the parameters that are always active,
regardless of whether an expansion/incorporation is ever triggered):
    local-epochs  (local training epochs per round)
    lambda-kd     (weight of the knowledge distillation term)
    tau           (confidence scale of the adaptive EMA of the global prototypes;
                   NOTE: prototype counts are accumulated on every epoch, so the
                   effective count per class is ~ local-epochs * n_samples and
                   tau interacts with local-epochs)

The "structural" parameters are searched in a second stage, with the core
parameters fixed at the best values of the first stage:
    theta-exp     (threshold on the saturation indicator g -> triggers expansion)
    theta-alpha   (threshold on the contribution ratio -> triggers candidacy)

Other hyperparameters commonly tuned in the project (all of which are
defined in [tool.flwr.app.config] in pyproject.toml and can be added via
--param): dirichlet-alpha, lambda-proto, lambda-kd, learning-rate,
candidacy-quorum, candidacy-patience, vote-margin, a-max,
incorporation-degrade-tolerance.

Examples
--------
Full grid with the three default parameters, 5 rounds per trial:
    python scripts/grid_search.py --num-server-rounds 5

Add/(override) grid axes manually:
    python scripts/grid_search.py \\
        --param theta-alpha=0.02,0.05,0.1 \\
        --param local-epochs=1,3 \\
        --param dirichlet-alpha=0.05,0.5

Random search with 15 combinations:
    python scripts/grid_search.py --strategy random --n-trials 15

Only preview which combinations would be executed, without running anything:
    python scripts/grid_search.py --dry-run

Rank by lower average forgetting instead of higher accuracy:
    python scripts/grid_search.py --metric avg_forgetting --minimize

Metrics available for --metric: every key in last_eval_metrics (server_eval_acc,
avg_forgetting, bwt, ...) plus `avg_inc_acc` -- the mean, over the incremental
steps, of the accuracy measured at the end of each step on all classes seen so
far (computed from forgetting_per_class.csv). It is much less noisy than the
last-round accuracy, so it is the default.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GRID: dict[str, list[Any]] = {
    "local-epochs": [3, 5, 7],
    "lambda-kd": [0.0, 0.5, 1.0, 2.0, 4.0],
    "tau": [15, 30, 60, 120, 250],
}

DEFAULT_METRIC = "avg_inc_acc"

# ------------------- #
# Parsing helpers
# ------------------- #


def _coerce_scaler(token: str) -> Any:
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    if token.lower() in ("true", "false"):
        return token.lower() == "true"
    return token


def _parse_param_arg(raw: str) -> tuple[str, list[Any]]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--param wait receive 'name=v1,v2...' received: {raw!r}"
        )
    name, values_raw = raw.split("=", 1)
    values = [_coerce_scaler(v) for v in values_raw.split(",")]
    return name.strip(), values


def _parse_fixed_arg(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--fixed wait receive 'name=value', received: {raw!r}"
        )
    name, value_raw = raw.split("=", 1)
    return name.strip(), _coerce_scaler(value_raw)


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return f"'{value}'"


def _build_run_config_string(overrides: dict[str, Any]) -> str:
    return " ".join(f"{k}={_format_toml_value(v)}" for k, v in overrides.items())


# ------------------- #
# Trial execution
# ------------------- #


def _summary_matches(summary_path: Path, overrides: dict[str, Any]) -> bool:
    """"""
    try:
        cfg = json.loads(summary_path.read_text()).get("run_config", {})
    except (json.JSONDecodeError, OSError):
        return False
    for key, wanted in overrides.items():
        if key == "output-dir":
            continue
        if key not in cfg:
            return False
        got = cfg[key]
        numeric = (int, float)
        if (
            isinstance(wanted, numeric)
            and not isinstance(wanted, bool)
            and isinstance(got, numeric)
            and not isinstance(got, bool)
        ):
            if not math.isclose(float(got), float(wanted), rel_tol=1e-9, abs_tol=1e-12):
                return False
        elif str(got).lower() != str(wanted).lower():
            return False
    return True


def _run_trial(
    trial_tag: str,
    params: dict[str, Any],
    fixed_overrides: dict[str, Any],
    results_dir: Path,
    force_rerun: bool,
    flwr_cmd: str,
) -> dict[str, Any]:
    trial_dir = results_dir / trial_tag
    summary_path = trial_dir / "summary.json"

    row: dict[str, Any] = {
        "trial": trial_tag,
        **fixed_overrides,
        **params,
        "trial_dir": str(trial_dir),
    }
    overrides = {**fixed_overrides, **params, "output-dir": str(trial_dir)}

    if summary_path.exists() and not force_rerun:
        if _summary_matches(summary_path, overrides):
            print(
                f"[{trial_tag}] summary.json already exists with the same config, "
                "skipping trial. (use --force-rerun to redo trials)"
            )
            row.update(_extract_metrics(summary_path))
            row["status"] = "cached"
            return row
        print(
            f"[{trial_tag}] summary.json exists but was produced with a different "
            "config (trial index reused by another combination); re-running."
        )

    trial_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("summary.json", "forgetting_per_class.csv"):
        (trial_dir / stale).unlink(missing_ok=True)
    run_config_str = _build_run_config_string(overrides)

    print(f'[{trial_tag}] flwr run . --run-config "{run_config_str}"')

    try:
        completed = subprocess.run(
            [flwr_cmd, "run", ".", "--run-config", run_config_str, "--stream"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            f"[{trial_tag}] ERROR: command '{flwr_cmd}' not found. "
            "Check the environment before running this script.",
            file=sys.stderr,
        )
        row["status"] = "error"
        row["error"] = f"'{flwr_cmd}' not found"
        return row

    log_path = trial_dir / "run_log.txt"
    log_path.write_text(completed.stdout + "\n" + completed.stderr)

    if completed.returncode != 0:
        print(
            f"[{trial_tag}] FAILED (exit code {completed.returncode}); check {log_path}"
        )
        row["status"] = "failed"
        row["error"] = f"exit code {completed.returncode} (see {log_path})"
        return row

    if not summary_path.exists():
        print(
            f"[{trial_tag}] finished without error, but summary.json was not generated; see {log_path}"
        )
        row["status"] = "failed"
        row["error"] = f"summary.json missing (see {log_path})"
        return row

    row.update(_extract_metrics(summary_path))
    row["status"] = "ok"
    return row


def _avg_incremental_accuracy(trial_dir: Path, run_config: dict[str, Any]) -> Any:
    """Mean over steps of the accuracy at the end of each step (all seen classes).

    Uses forgetting_per_class.csv (per-round, per-class accuracy and support), so
    the per-round accuracy is sum(n * acc) / sum(n) -- identical to server_eval_acc.
    """
    csv_path = trial_dir / "forgetting_per_class.csv"
    classes_per_step = run_config.get("classes-per-step")
    if not csv_path.exists() or not classes_per_step:
        return None
    rounds_per_step = int(run_config.get("rounds-per-step", 1))
    total_classes = int(run_config.get("num-classes-total", 27))
    n_steps = math.ceil(total_classes / classes_per_step)
    step_end_rounds = {rounds_per_step * (i + 1) for i in range(n_steps)}

    hits: dict[int, float] = {}
    support: dict[int, float] = {}
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            rnd = int(r["round"])
            if (
                rnd not in step_end_rounds
                or r["n"] in ("", None)
                or r["acc"] in ("", None)
            ):
                continue
            hits[rnd] = hits.get(rnd, 0.0) + float(r["n"]) * float(r["acc"])
            support[rnd] = support.get(rnd, 0.0) + float(r["n"])
    accs = [hits[k] / support[k] for k in sorted(support) if support[k] > 0]
    return sum(accs) / len(accs) if accs else None


def _extract_metrics(summary_path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"error": f"could not read {summary_path}: {exc}"}

    last_eval = summary.get("last_eval_metrics") or {}
    return {
        "avg_inc_acc": _avg_incremental_accuracy(
            summary_path.parent, summary.get("run_config") or {}
        ),
        "round": last_eval.get("round"),
        "server_eval_acc": last_eval.get("server_eval_acc"),
        "server_eval_loss": last_eval.get("server_eval_loss"),
        "bwt": last_eval.get("bwt"),
        "avg_forgetting": last_eval.get("avg_forgetting"),
        "worst_class_forgetting": last_eval.get("worst_class_forgetting"),
        "worst_class_forgetting_id": last_eval.get("worst_class_forgetting_id"),
        "worst_class_bwt": last_eval.get("worst_class_bwt"),
        "worst_class_bwt_id": last_eval.get("worst_class_bwt_id"),
    }


# ------------------- #
# Reporting
# ------------------- #


def _write_leaderboard_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    widths = {
        c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) if rows else len(c)
        for c in columns
    }
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


# ------------------- #
# Main
# ------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid/random search of hyperparams for FedProtoContinual.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="name=v1,v2...",
        help=(
            "Grid axis to vary, e.g., theta-alpha=0.02,0.05,0.1. Can be "
            "provided multiple times. If omitted, the default grid is used "
            f"({', '.join(DEFAULT_GRID.keys())})."
        ),
    )
    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        metavar="name=value",
        help="Fixed override applied to all trials, e.g., --fixed dirichlet-alpha=0.1",
    )
    parser.add_argument(
        "--num-server-rounds",
        type=int,
        default=None,
        help="Shortcut for --fixed num-server-rounds=N (useful for shortening the search).",
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "random"],
        default="grid",
        help="'grid' evaluates all combinations; 'random' samples --n-trials of them.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of combinations to sample when --strategy random.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for --strategy random."
    )
    parser.add_argument(
        "--results-dir",
        default="outputs/grid_search",
        help="Directory where each trial stores its results in a separate subdirectory.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=(
            "Metric used to rank the leaderboard (key in last_eval_metrics "
            "from summary.json), e.g., server_eval_acc, avg_forgetting, bwt."
        ),
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Sort by the lowest metric value (e.g., for avg_forgetting).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of rows to display in the final summary.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run trials even if a saved summary.json already exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the entire search on the first trial that fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the combinations that would be executed, without running anything.",
    )
    parser.add_argument(
        "--flwr-cmd",
        default="flwr",
        help="Path/name of the 'flwr' executable (default: 'flwr' from the active PATH).",
    )
    args = parser.parse_args()

    grid: dict[str, list[Any]] = dict(DEFAULT_GRID) if not args.param else {}
    for raw in args.param:
        name, values = _parse_param_arg(raw)
        grid[name] = values

    fixed_overrides: dict[str, Any] = {}
    for raw in args.fixed:
        name, value = _parse_fixed_arg(raw)
        fixed_overrides[name] = value
    if args.num_server_rounds is not None:
        fixed_overrides["num-server-rounds"] = args.num_server_rounds

    names = sorted(grid.keys())
    value_lists = [grid[n] for n in names]
    all_combos = list(itertools.product(*value_lists))

    if args.strategy == "random":
        if not args.n_trials:
            parser.error("--strategy random needs --n-trials")
        rng = random.Random(args.seed)
        if args.n_trials >= len(all_combos):
            print(
                f"--n-trials={args.n_trials} >= {len(all_combos)} possible combinations; "
                "running the full grid."
            )
            combos = all_combos
        else:
            combos = rng.sample(all_combos, args.n_trials)
    else:
        combos = all_combos

    print(f"Grid axes: {names}")
    print(f"Total trials: {len(combos)}")
    if fixed_overrides:
        print(f"Fixed overrides applied to all trials: {fixed_overrides}")

    results_dir = PROJECT_ROOT / args.results_dir

    trials = [
        (f"trial_{idx:04d}", dict(zip(names, combo)))
        for idx, combo in enumerate(combos)
    ]

    if args.dry_run:
        print("\n--dry-run: no executions will be performed.\n")
        for trial_tag, params in trials:
            overrides = {
                **fixed_overrides,
                **params,
                "output-dir": str(results_dir / trial_tag),
            }
            print(f'[{trial_tag}] --run-config "{_build_run_config_string(overrides)}"')
            return

    rows: list[dict[str, Any]] = []
    try:
        for trial_tag, params in trials:
            row = _run_trial(
                trial_tag,
                params,
                fixed_overrides,
                results_dir,
                args.force_rerun,
                args.flwr_cmd,
            )
            rows.append(row)
            if row["status"] in ("failed", "error") and args.stop_on_error:
                print("Aborting search (--stop-on-error).")
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user; saving partial results...")

    leaderboard_path = results_dir / "leaderboard.csv"
    _write_leaderboard_csv(rows, leaderboard_path)
    print(f"\nFull leaderboard saved to: {leaderboard_path}")

    ranked = [r for r in rows if isinstance(r.get(args.metric), (int, float))]
    ranked.sort(key=lambda r: r[args.metric], reverse=not args.minimize)
    failed = [r for r in rows if r["status"] in ("failed", "error")]

    if ranked:
        columns = [
            "trial",
            *names,
            args.metric,
            "avg_inc_acc",
            "server_eval_acc",
            "avg_forgetting",
            "bwt",
            "status",
        ]
        columns = list(dict.fromkeys(columns))
        print(
            f"\nTop {min(args.top_n, len(ranked))} by '{args.metric}' "
            f"({'lower' if args.minimize else 'higher'} is better):\n"
        )
        _print_table(ranked[: args.top_n], columns)
    else:
        print("\nNo trial successfully produced the requested metric.")

    if failed:
        print(f"\n{len(failed)} trial(s) failed:")
        for r in failed:
            print(f"  - {r['trial']}: {r.get('error', 'unknown reason')}")


if __name__ == "__main__":
    main()
