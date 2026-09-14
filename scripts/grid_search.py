"""Grid/random hyperparameter search for FedProtoContinual.

Launches multiple `flwr run .` executions, one for each hyperparameter
combination, with each execution writing its results (including the
per-class forgetting/BWT monitor) to an isolated subdirectory under
`--results-dir`. At the end, it reads the `summary.json` from each trial
(written by Server_app.py) and builds a leaderboard sorted by the
selected metric.

By default, it searches over the three requested hyperparameters:
    theta-alpha   (alpha_mean threshold for proposing incorporation)
    local-epochs  (local training epochs per round)
    theta-exp     (threshold for the saturation/expansion criterion)

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
"""

import argparse
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GRID: dict[str, list[Any]] = {
    "theta-alpha": [0.1, 0.2, 0.3, 0.4, 0.5],
    "local-epochs": [1, 3, 5, 7],
    "theta-exp": [0.3, 0.4, 0.5],
}

DEFAULT_METRIC = "server_eval_acc"

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

    row: dict[str, Any] = {"trial": trial_tag, **params, "trial_dir": str(trial_dir)}

    if summary_path.exists() and not force_rerun:
        print(
            f"[{trial_tag}] summary.json already exists, skipping trial. (uses --force-rerun to redo trials)"
        )
        row.update(_extract_metrics(summary_path))
        row["status"] = "cached"
        return row

    trial_dir.mkdir(parents=True, exist_ok=True)
    overrides = {**fixed_overrides, **params, "output-dir": str(trial_dir)}
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


def _extract_metrics(summary_path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"error": f"could not read {summary_path}: {exc}"}

    last_eval = summary.get("last_eval_metrics") or {}
    return {
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
