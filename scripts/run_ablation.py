"""Run the FedProtoContinual ablation ladder (plain federated) and report it.

Every experiment is just a set of run-config flags (src/utils/ablation.py), so this
script only maps named profiles to flags, runs each profile for several seeds through
grid_search._run_trial (same caching / logging / summary.json handling), and builds a
paired-by-seed comparison table.

Suites
------
ladder      A0..A4, cumulative:
              A0_protos  prototype classifier + global adapter (no local adapter, no KD)
              A1_local   + local adapter (personalization)
              A2_kd      + global-knowledge distillation
              A3_exp     + width/depth expansion of the local adapter
              A4_full    + candidacy/vote/incorporation of local adapters
loo         leave-one-out from A4_full (no KD / no expansion / no incorporation / no
            local adapter) -- the cumulative ladder confounds a component's effect with
            the order it is added in, leave-one-out does not.
algorithm   A4_full (FedAvg) vs FedProx (one ALG_fedprox_mu<mu> profile per --mu value).
central     CENTRAL (upper bound: ONE client holding the whole pool, partition-mode=pooled)
            and CENTRAL_UNION (like-for-like with the subject-mode ladder: ONE client holding
            the union of the subject clients' retained data, partition-mode=subject-union).
            Both: min-nodes=1, expected-nodes=1.
            Needs simulation.num-supernodes = 1 (see below), so it is NOT part of 'all'.
all         ladder + loo + algorithm.

Centralized baseline (suite 'central')
--------------------------------------
  1. federation.local.toml -> num-supernodes = 1, then python scripts/sync_federation_config.py
  2. python scripts/run_ablation.py --suite central --seeds 0-9 <same --fixed as the ladder>
  3. restore num-supernodes (e.g. 5) and sync again before the federated suites.
The run aborts at round 0 if the federation size is not 1 (expected-nodes guard), so a
forgotten sync can never produce a "centralized" run that is really federated.
For a like-for-like comparison run the federated suites with --fixed partition-mode=pooled
too: the union of the 5 Dirichlet clients is then exactly the pool the central client sees.
Rounds x local-epochs is the epoch budget: keep the same --fixed num-server-rounds /
local-epochs for CENTRAL and for the ladder.

Results live in a flat layout, <results-dir>/<profile>/seed_<k>/, so a profile that
belongs to several suites (A4_full) is run once and reused (grid_search's cache checks
that the saved run_config matches). To repeat a suite under another setting (e.g. the
ladder under FedProx), use a different --results-dir; the runner refuses to overwrite
runs produced with a different config unless --force-rerun is given.

Usage
-----
    python scripts/run_ablation.py --list
    python scripts/run_ablation.py --suite ladder --seeds 0-9 --dry-run
    python scripts/run_ablation.py --suite ladder --seeds 0-9 \
        --fixed local-epochs=5 --fixed tau=250 --num-server-rounds 90
    python scripts/run_ablation.py --suite algorithm --seeds 100-102 --mu 0.001,0.01,0.1 \
        --results-dir outputs/tuning
    python scripts/run_ablation.py --report outputs/ablation --suite ladder

Seeds: tune hyperparameters (and FedProx mu) on separate seeds (e.g. 100-102) and keep
0-9 for the final numbers. Pass tuned values to every profile with --fixed so that all
profiles share exactly the same hyperparameters.
"""

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ablation import resolve_ablation_flags


def _load_grid_search():
    spec = importlib.util.spec_from_file_location(
        "grid_search", PROJECT_ROOT / "scripts" / "grid_search.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["grid_search"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------- profiles -----------------------------------------

BASE_FLAGS: dict[str, Any] = {"training-scenario": "federated"}

_OFF = {
    "use-local-adapter": False,
    "enable-kd": False,
    "enable-expansion": False,
    "enable-incorporation": False,
}


def _on(**extra) -> dict[str, Any]:
    return {**_OFF, **extra}


LADDER: dict[str, dict[str, Any]] = {
    "A0_protos": _on(),
    "A1_local": _on(**{"use-local-adapter": True}),
    "A2_kd": _on(**{"use-local-adapter": True, "enable-kd": True}),
    "A3_exp": _on(
        **{"use-local-adapter": True, "enable-kd": True, "enable-expansion": True}
    ),
    "A4_full": _on(
        **{
            "use-local-adapter": True,
            "enable-kd": True,
            "enable-expansion": True,
            "enable-incorporation": True,
        }
    ),
}

FULL = LADDER["A4_full"]
LOO: dict[str, dict[str, Any]] = {
    "LOO_noKD": {**FULL, "enable-kd": False},
    "LOO_noExp": {**FULL, "enable-expansion": False},
    "LOO_noInc": {**FULL, "enable-incorporation": False},
    "LOO_noLocal": {
        **FULL,
        "use-local-adapter": False,
        "enable-expansion": False,
        "enable-incorporation": False,
    },
}


def algorithm_profiles(mus: list[float]) -> dict[str, dict[str, Any]]:
    profiles = {"A4_full": {**FULL, "fl-algorithm": "fedavg"}}
    for mu in mus:
        profiles[f"ALG_fedprox_mu{mu:g}"] = {
            **FULL,
            "fl-algorithm": "fedprox",
            "proximal-mu": mu,
        }
    return profiles


CENTRAL: dict[str, dict[str, Any]] = {
    # Upper bound: every window of every non-held-out subject, no class retention.
    "CENTRAL": {
        **LADDER["A0_protos"],
        "partition-mode": "pooled",
        "min-nodes": 1,
        "expected-nodes": 1,
    },
    # Like-for-like with the subject-mode ladder: the union of the data the 5 subject
    # clients hold (same per-subject class retention), seen by ONE client.
    "CENTRAL_UNION": {
        **LADDER["A0_protos"],
        "partition-mode": "subject-union",
        "min-nodes": 1,
        "expected-nodes": 1,
    },
}


def build_suite(name: str, mus: list[float]) -> dict[str, dict[str, Any]]:
    if name == "central":
        return dict(CENTRAL)
    if name == "ladder":
        return dict(LADDER)
    if name == "loo":
        return {"A4_full": LADDER["A4_full"], **LOO}
    if name == "algorithm":
        return algorithm_profiles(mus)
    if name == "all":
        return {**LADDER, **LOO, **algorithm_profiles(mus)}
    raise ValueError(f"unknown suite {name!r}")


def default_pairs(profiles: list[str]) -> list[tuple[str, str]]:
    """(treatment, reference) pairs: consecutive ladder steps, LOO vs full, prox vs avg."""
    pairs: list[tuple[str, str]] = []
    ladder_names = list(LADDER)
    for prev, cur in zip(ladder_names, ladder_names[1:]):
        if prev in profiles and cur in profiles:
            pairs.append((cur, prev))
    if "A4_full" in profiles:
        pairs += [(p, "A4_full") for p in profiles if p.startswith("LOO_")]
    if "A4_full" in profiles:
        pairs += [(p, "A4_full") for p in profiles if p.startswith("ALG_fedprox")]
    return pairs


# ------------------------------------------- helpers --------------------------------


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("no seeds given")
    return seeds


def parse_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


# -------------------------------- execution --------------------------------


def run_suite(args) -> None:
    gs = _load_grid_search()
    seeds = parse_seeds(args.seeds)
    mus = parse_floats(args.mu)
    profiles = build_suite(args.suite, mus)

    fixed: dict[str, Any] = dict(BASE_FLAGS)
    for raw in args.fixed:
        name, value = gs._parse_fixed_arg(raw)
        fixed[name] = value
    if args.num_server_rounds is not None:
        fixed["num-server-rounds"] = args.num_server_rounds

    if args.only:
        wanted = [p.strip() for p in args.only.split(",")]
        unknown = [p for p in wanted if p not in profiles]
        if unknown:
            raise SystemExit(
                f"--only: unknown profile(s) {unknown}; have {list(profiles)}"
            )
        profiles = {p: profiles[p] for p in wanted}

    # Fail fast on an invalid flag combination, before launching anything.
    for pname, flags in profiles.items():
        resolve_ablation_flags({**fixed, **flags})

    out_root = PROJECT_ROOT / args.results_dir
    print(
        f"Suite '{args.suite}': {len(profiles)} profiles x {len(seeds)} seeds "
        f"= {len(profiles) * len(seeds)} runs  ->  {out_root}"
    )
    if args.suite == "central":
        print(
            "NOTE: needs simulation.num-supernodes = 1 in the active Flower config "
            "(federation.local.toml + sync_federation_config.py); the server aborts "
            "at startup if the federation size differs."
        )
    if fixed:
        print(f"Fixed overrides: {fixed}")

    # Never silently overwrite finished runs that were produced with another config.
    if not args.force_rerun:
        conflicts = []
        for pname, flags in profiles.items():
            for seed in seeds:
                trial_dir = out_root / pname / f"seed_{seed}"
                summary = trial_dir / "summary.json"
                overrides = {
                    **fixed,
                    **flags,
                    "seed": seed,
                    "output-dir": str(trial_dir),
                }
                if summary.exists() and not gs._summary_matches(summary, overrides):
                    conflicts.append(f"{pname}/seed_{seed}")
        if conflicts:
            raise SystemExit(
                f"{len(conflicts)} existing run(s) in {out_root} were produced with a "
                f"different config (e.g. {conflicts[:3]}). Use another --results-dir, or "
                "--force-rerun to overwrite them."
            )

    rows: list[dict[str, Any]] = []
    for pname, flags in profiles.items():
        profile_fixed = {**fixed, **flags}
        print(f"\n=== {pname}: {resolve_ablation_flags(profile_fixed).describe()}")
        for seed in seeds:
            tag = f"seed_{seed}"
            params = {"seed": seed}
            if args.dry_run:
                overrides = {
                    **profile_fixed,
                    **params,
                    "output-dir": str(out_root / pname / tag),
                }
                print(
                    f'[{pname}/{tag}] --run-config "{gs._build_run_config_string(overrides)}"'
                )
                continue
            row = gs._run_trial(
                tag,
                params,
                profile_fixed,
                out_root / pname,
                args.force_rerun,
                args.flwr_cmd,
            )
            row = {
                "profile": pname,
                "seed": seed,
                **{
                    k: v
                    for k, v in row.items()
                    if k in _ROW_KEYS or k in ("status", "error")
                },
            }
            rows.append(row)
            if row["status"] in ("failed", "error") and args.stop_on_error:
                print("Aborting (--stop-on-error).")
                _write_results(rows, out_root)
                return

    if args.dry_run:
        print("\n--dry-run: nothing was executed.")
        return
    _write_results(rows, out_root)
    bad = [r for r in rows if r["status"] in ("failed", "error")]
    if bad:
        print(
            f"\n{len(bad)} run(s) failed: "
            + ", ".join(f"{r['profile']}/seed_{r['seed']}" for r in bad)
        )
    print(
        f"\nNext: python scripts/run_ablation.py --report {args.results_dir} --suite {args.suite}"
    )


_ROW_KEYS = {
    "round",
    "server_eval_acc",
    "server_eval_acc_tail",
    "server_eval_loss_tail",
    "client_eval_acc",
    "client_eval_acc_global",
    "client_eval_acc_tail",
    "client_eval_acc_global_tail",
    "personalization_gain_tail",
    "params_base_total",
    "params_shared",
    "params_incorporated",
    "client_params_total",
    "client_params_base",
    "client_params_expansion_overhead",
    "client_params_growth_ratio",
}


def _write_results(rows: list[dict[str, Any]], out_root: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "results.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Per-run results saved to {path}")


# -------------------------------- report ----------------------------------


def _collect(out_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """profile -> seed -> last_eval_metrics, read straight from the summary.json files."""
    data: dict[str, dict[int, dict[str, Any]]] = {}
    for summary in sorted(out_root.glob("*/seed_*/summary.json")):
        try:
            payload = json.loads(summary.read_text())
            seed = int(summary.parent.name.split("_", 1)[1])
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        metrics = dict(payload.get("last_eval_metrics") or {})
        data.setdefault(summary.parent.parent.name, {})[seed] = metrics
    return data


def _t_ci(diffs: list[float], conf: float = 0.95) -> tuple[float, float]:
    from scipy import stats

    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return mean, mean
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    half = stats.t.ppf((1 + conf) / 2, n - 1) * sd / math.sqrt(n)
    return mean - half, mean + half


def _holm(pvalues: list[Optional[float]]) -> list[Optional[float]]:
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    adjusted: list[Optional[float]] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _paired(a: dict[int, float], b: dict[int, float]) -> Optional[dict[str, Any]]:
    from scipy import stats

    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    x = [a[s] for s in seeds]
    y = [b[s] for s in seeds]
    diffs = [xi - yi for xi, yi in zip(x, y)]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    lo, hi = _t_ci(diffs)
    try:
        p_w = float(stats.wilcoxon(diffs).pvalue) if any(d != 0 for d in diffs) else 1.0
    except ValueError:
        p_w = None
    p_t = float(stats.ttest_rel(x, y).pvalue) if sd > 0 else (1.0 if mean == 0 else 0.0)
    return {
        "n": n,
        "diff": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "dz": (mean / sd) if sd > 0 else float("nan"),
        "p_wilcoxon": p_w,
        "p_ttest": p_t,
    }


def report(args) -> None:
    out_root = Path(args.report)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    data = _collect(out_root)
    if args.suite != "all":
        keep = set(build_suite(args.suite, parse_floats(args.mu)))
        data = {p: v for p, v in data.items() if p in keep or p.startswith("CENTRAL")}
    if not data:
        raise SystemExit(
            f"No <profile>/seed_*/summary.json found under {out_root} for suite '{args.suite}'"
        )

    metrics = [m.strip() for m in args.metrics.split(",")]
    profiles = list(data)
    order = [p for p in [*LADDER, *LOO] if p in profiles] + [
        p for p in profiles if p not in LADDER and p not in LOO
    ]

    if args.pairs:
        pairs = [tuple(p.split(":", 1)) for p in args.pairs.split(",")]
    else:
        pairs = default_pairs(order)

    csv_rows: list[dict[str, Any]] = []
    for metric in metrics:
        series = {
            p: {
                s: float(m[metric])
                for s, m in data[p].items()
                if isinstance(m.get(metric), (int, float))
            }
            for p in order
        }
        print(
            f"\n=== {metric}  (mean +- std over seeds, higher is better unless it is a loss)"
        )
        print(f"{'profile':<22}{'n':>3}  {'mean':>8}  {'std':>7}")
        for p in order:
            vals = list(series[p].values())
            if not vals:
                print(f"{p:<22}{0:>3}  {'-':>8}  {'-':>7}")
                continue
            mean = sum(vals) / len(vals)
            sd = (
                math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
                if len(vals) > 1
                else float("nan")
            )
            print(f"{p:<22}{len(vals):>3}  {mean:>8.4f}  {sd:>7.4f}")

        results = []
        for treat, ref in pairs:
            if treat not in series or ref not in series:
                continue
            stat = _paired(series[treat], series[ref])
            if stat is not None:
                results.append((treat, ref, stat))
        if not results:
            continue
        holm_w = _holm([r[2]["p_wilcoxon"] for r in results])
        holm_t = _holm([r[2]["p_ttest"] for r in results])
        print(
            f"\npaired by seed, treatment - reference (Holm over {len(results)} comparisons)"
        )
        print(
            f"{'treatment':<22}{'reference':<22}{'n':>3} {'diff':>8}  {'95% CI':>19}  {'d_z':>6}  {'p_wilc':>7}  {'p_holm':>7}  {'p_t':>7}  {'p_t_holm':>8}"
        )
        for (treat, ref, s), pw, pt in zip(results, holm_w, holm_t):
            fmt = lambda v: "   -   " if v is None else f"{v:7.4f}"
            print(
                f"{treat:<22}{ref:<22}{s['n']:>3} {s['diff']:>+8.4f}  [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  {s['dz']:>6.2f}  {fmt(s['p_wilcoxon'])}  {fmt(pw)}  {fmt(s['p_ttest'])}  {fmt(pt):>8}"
            )
            csv_rows.append(
                {
                    "metric": metric,
                    "treatment": treat,
                    "reference": ref,
                    **s,
                    "p_wilcoxon_holm": pw,
                    "p_ttest_holm": pt,
                }
            )
    if csv_rows:
        path = out_root / "paired_comparisons.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nSaved {path}")


# -------------------------------- main ----------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run / report the FedProtoContinual ablation (flags only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--list", action="store_true", help="List profiles and their flags, then exit."
    )
    p.add_argument(
        "--suite",
        choices=["ladder", "loo", "algorithm", "central", "all"],
        default="ladder",
        help="Profiles to run (or, with --report, to include). Default: ladder.",
    )
    p.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,1,2 (default 0-9).")
    p.add_argument(
        "--only", default=None, help="Comma-separated subset of profile names."
    )
    p.add_argument(
        "--mu",
        default="0.001,0.01,0.1",
        help="FedProx proximal-mu values for the 'algorithm' suite.",
    )
    p.add_argument(
        "--fixed",
        action="append",
        default=[],
        metavar="name=value",
        help="Override applied to every profile, e.g. --fixed local-epochs=5",
    )
    p.add_argument("--num-server-rounds", type=int, default=None)
    p.add_argument("--results-dir", default="outputs/ablation")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--flwr-cmd", default="flwr")
    p.add_argument(
        "--report",
        default=None,
        metavar="DIR",
        help="Skip running; build the paired comparison from DIR (e.g. outputs/ablation).",
    )
    p.add_argument(
        "--metrics",
        default="client_eval_acc_tail,server_eval_acc_tail,personalization_gain_tail",
        help="Metrics for --report (keys of last_eval_metrics in summary.json).",
    )
    p.add_argument(
        "--pairs",
        default=None,
        help="Custom comparisons for --report: 'treat:ref,treat2:ref2'.",
    )
    args = p.parse_args()

    if args.list:
        for pname, flags in build_suite("all", parse_floats(args.mu)).items():
            print(
                f"{pname:<24} {resolve_ablation_flags({**BASE_FLAGS, **flags}).describe()}"
            )
        return
    if args.report:
        report(args)
    else:
        run_suite(args)


if __name__ == "__main__":
    main()
