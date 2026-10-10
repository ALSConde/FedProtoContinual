import random
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
import numpy as np
import torch
from src.model.Models import FCLModel
from src.server.FedAvgStrategy import FedAvgStrategy
from src.server.FedProxStrategy import FedProxStrategy
from src.server.ServerEvaluation import ForgettingMonitor, evaluate_global_model
from src.utils.ablation import as_bool, resolve_ablation_flags
from src.utils.data.utd_mahd_dataset import (
    build_class_schedule,
    classes_seen_until_round,
    load_server_test_set,
    parse_int_list_config,
    resolve_classes_per_step,
    resolve_dirichlet_mode,
)

app = ServerApp()


def _tail_mean(history: list[dict], key: str, window: int):
    values = [h[key] for h in history[-window:] if key in h]
    return sum(values) / len(values) if values else None


def _final_window_metrics(
    server_history: list[dict], client_history: list[dict], window: int
) -> dict:
    """Last-round and last-`window`-rounds means of the federated metrics.

    server_eval_*  : global model on the held-out subjects (generalization to unseen users).
    client_eval_*  : accuracy on each client's own validation split, weighted by size;
                     client_eval_acc uses the personalized embedding, client_eval_acc_global
                     the shared one (so their difference is the personalization gain).
    """
    out: dict = {}
    tail_keys = {
        "server_eval_acc_tail": (server_history, "server_eval_acc"),
        "server_eval_loss_tail": (server_history, "server_eval_loss"),
        "client_eval_acc_tail": (client_history, "eval_acc"),
        "client_eval_acc_global_tail": (client_history, "eval_acc_global"),
    }
    for name, (hist, key) in tail_keys.items():
        value = _tail_mean(hist, key, window)
        if value is not None:
            out[name] = value
    if client_history:
        last = client_history[-1]
        out["client_eval_acc"] = last.get("eval_acc")
        out["client_eval_acc_global"] = last.get("eval_acc_global")
    if "client_eval_acc_tail" in out and "client_eval_acc_global_tail" in out:
        out["personalization_gain_tail"] = (
            out["client_eval_acc_tail"] - out["client_eval_acc_global_tail"]
        )
    return {k: v for k, v in out.items() if v is not None}


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_evaluate = context.run_config["fraction-evaluate"]
    batch_size = int(context.run_config["batch-size"])
    lr = context.run_config["learning-rate"]
    input_dim = int(context.run_config["input-dim"])
    hidden_dim = int(context.run_config["hidden-dim"])
    d_hat_global = int(context.run_config["d-hat-global"])
    d_hat_local = int(context.run_config["d-hat-local"])
    a_max = int(context.run_config.get("a-max", 3))
    incorp_flag = str(context.run_config.get("incorp_status", "false")).lower()
    seed = int(context.run_config.get("seed", 0))
    flags = resolve_ablation_flags(context.run_config)
    print(f"[ablation] {flags.describe()}")

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    scenario = str(context.run_config.get("training-scenario", "federated")).lower()
    class_scen = context.run_config.get("classes-per-step")
    if class_scen is not None:
        class_scen = int(class_scen)
    classes_per_step = resolve_classes_per_step(scenario, class_scen)
    dirichlet_mode = resolve_dirichlet_mode(
        scenario, str(context.run_config.get("dirichlet-mode", "static")).lower()
    )

    if classes_per_step is not None:
        rounds_per_step = int(context.run_config.get("rounds-per-step", 1))
        print(
            f"training-scenario='{scenario}': introducing "
            f"{classes_per_step} new classes every {rounds_per_step} round(s) "
            f"over {num_rounds} rounds. dirichlet-mode='{dirichlet_mode}' "
            "(forced static under class-incremental)."
        )
    else:
        print(
            f"training-scenario='{scenario}': standard non-IID "
            "federated learning, no class schedule applied. "
            f"dirichlet-mode='{dirichlet_mode}'."
        )

    global_model = FCLModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        d_hat_global=d_hat_global,
        d_hat_local=d_hat_local,
        a_max=a_max,
        use_local_adapter=flags.use_local_adapter,
    )
    base_report = global_model.parameter_report()
    print(
        f"[params] base model (no expansion): total={base_report['total']:,} "
        f"| shared={base_report['shared']:,} | local={base_report['local']:,}"
    )
    arrays = ArrayRecord(global_model.get_global_arrays())

    # Flower expects min_nodes >= 2, with 1 node we can force a centralized run by setting expected-nodes=1 (see below).
    min_nodes = int(context.run_config.get("min-nodes", 2))
    if min_nodes < 1:
        raise ValueError(f"min-nodes must be >= 1, got {min_nodes}.")

    # Optional guard: expected-nodes > 0 asserts the simulation really has that many
    # supernodes. Prevents a "centralized" run (1 client) from silently running with the
    # federation size left over in the global Flower config (or the opposite).
    expected_nodes = int(context.run_config.get("expected-nodes", 0))
    if expected_nodes > 0:
        n_nodes = len(list(grid.get_node_ids()))
        if n_nodes != expected_nodes:
            raise RuntimeError(
                f"expected-nodes={expected_nodes} but the federation has {n_nodes} "
                "node(s). Set simulation.num-supernodes in federation.local.toml and run "
                "scripts/sync_federation_config.py (or 'flwr federation simulation-config "
                "--num-supernodes=N') before launching."
            )
        print(f"[federation] {n_nodes} node(s), as expected")

    strategy_kwargs = dict(
        min_train_nodes=min_nodes,
        min_evaluate_nodes=min_nodes,
        min_available_nodes=min_nodes,
        embedding_dim=hidden_dim,
        tau=float(context.run_config.get("tau", 15.0)),
        fraction_evaluate=fraction_evaluate,
        a_max=a_max,
        candidacy_quorum=float(context.run_config.get("candidacy-quorum", 0.5)),
        incorporation_monitor_rounds=int(
            context.run_config.get("incorporation-monitor-rounds", 3)
        ),
        incorporation_degrade_tolerance=float(
            context.run_config.get("incorporation-degrade-tolerance", 0.05)
        ),
        incorporation_baseline_window=int(
            context.run_config.get("incorporation-baseline-window", 3)
        ),
        enable_incorporation=flags.enable_incorporation,
        incorporation_options=dict(
            vote_mode=str(context.run_config.get("vote-mode", "legacy"))
            .strip()
            .lower(),
            vote_loss_margin=float(context.run_config.get("vote-loss-margin", 0.01)),
            vote_z=float(context.run_config.get("vote-z", 1.0)),
            prune_enabled=as_bool(
                context.run_config.get("prune-enabled", True), "prune-enabled"
            ),
            prune_margin=float(context.run_config.get("prune-margin", 0.01)),
            prune_patience=int(context.run_config.get("prune-patience", 5)),
            loo_ema=float(context.run_config.get("loo-ema", 0.7)),
            shadow_rounds=int(context.run_config.get("shadow-rounds", 0)),
            shadow_window=int(context.run_config.get("shadow-window", 3)),
        ),
    )
    if flags.fl_algorithm == "fedprox":
        strategy = FedProxStrategy(proximal_mu=flags.proximal_mu, **strategy_kwargs)
    else:
        strategy = FedAvgStrategy(**strategy_kwargs)

    held_out_subjects = parse_int_list_config(
        context.run_config.get("server-eval-subjects")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = str(context.run_config.get("output-dir", "outputs/default_run"))
    forgetting_monitor = ForgettingMonitor(output_dir=output_dir)
    last_eval_metrics: dict = {}
    server_eval_history: list[dict] = []
    evaluate_fn = None

    if held_out_subjects:
        server_test_loader = load_server_test_set(
            root=str(context.run_config["data-root"]),
            held_out_subjects=held_out_subjects,
            window_size=int(context.run_config["window-size"]),
            stride=int(context.run_config["stride"]),
            batch_size=int(context.run_config.get("eval-batch-size", 32)),
        )
        print(
            "Centralized evaluation enabled on held-out subjects "
            f"{held_out_subjects} with ({len(server_test_loader.dataset)} samples)."
        )
        schedule = None
        rounds_per_step_for_eval = 1
        if classes_per_step is not None:
            num_classes_total = int(context.run_config["num-classes-total"])
            schedule = build_class_schedule(num_classes_total, classes_per_step)
            rounds_per_step_for_eval = int(context.run_config.get("rounds-per-step", 1))

        def evaluate_fn(current_round: int, eval_arrays: ArrayRecord):
            mu_all, ids_all = strategy.proto_aggregator.get_prototypes_raw()
            if len(ids_all) == 0:
                return None

            eval_model = FCLModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                d_hat_global=d_hat_global,
                d_hat_local=d_hat_local,
                a_max=a_max,
                use_local_adapter=flags.use_local_adapter,
            )

            eval_sd = eval_arrays.to_torch_state_dict()
            # A candidate on probation is not part of the deployed model.
            eval_sd = {
                k: v for k, v in eval_sd.items() if not k.startswith("shadow_adapter.")
            }
            n_topologies = len(strategy.incorporation.topologies)

            stale_prefixes = tuple(
                f"incorporated_adapter.{i}."
                for i in range(
                    n_topologies, n_topologies + strategy.incorporation.a_max
                )
            )
            dropped = [k for k in eval_sd.keys() if k.startswith(stale_prefixes)]
            if dropped:
                print(
                    f"[server_eval] round {current_round}: dropping {len(dropped)} "
                    "stale incorporated-adapter weight(s) not covered by the current "
                    "topology list (expected right after a same-round reversion)."
                )
                for k in dropped:
                    del eval_sd[k]

            eval_model.load_incorporated_topology(strategy.incorporation.topologies)
            eval_model.set_global_arrays(eval_sd)
            eval_report = eval_model.parameter_report()
            eval_model.classifier.update_from_global(mu_all, ids_all)
            eval_model.to(device)

            allowed_classes = None
            if schedule is not None:
                allowed_classes = classes_seen_until_round(
                    max(current_round, 1), rounds_per_step_for_eval, schedule
                )

            loss, acc, per_class_acc, per_class_n = evaluate_global_model(
                eval_model, server_test_loader, device, allowed_classes
            )

            metrics = {"server_eval_loss": loss, "server_eval_acc": acc}
            # Global (shared) model size, base vs. current: grows only through
            # incorporated adapters (local expansions never leave the client).
            metrics["params_shared"] = eval_report["shared"]
            metrics["params_shared_base"] = base_report["shared"]
            metrics["params_incorporated"] = eval_report["incorporated_adapters"]
            server_eval_history.append(
                {
                    "round": int(current_round),
                    "server_eval_acc": float(acc),
                    "server_eval_loss": float(loss),
                }
            )

            if schedule is not None:
                report = forgetting_monitor.update(
                    current_round, per_class_acc, per_class_n=per_class_n
                )
                if report["mean_bwt"] is not None:
                    metrics["bwt"] = report["mean_bwt"]
                if report["mean_forgetting"] is not None:
                    metrics["avg_forgetting"] = report["mean_forgetting"]
                if report["worst_class_forgetting"] is not None:
                    metrics["worst_class_forgetting"] = report["worst_class_forgetting"]
                    metrics["worst_class_forgetting_id"] = report[
                        "worst_class_forgetting_id"
                    ]
                if report["worst_class_bwt"] is not None:
                    metrics["worst_class_bwt"] = report["worst_class_bwt"]
                    metrics["worst_class_bwt_id"] = report["worst_class_bwt_id"]
                metrics["num_classes_seen"] = int(len(allowed_classes))

            metrics.update(strategy.incorporation.metrics_snapshot())
            last_eval_metrics.clear()
            last_eval_metrics.update(metrics)
            last_eval_metrics["round"] = current_round
            return MetricRecord(metrics)

    else:
        print("server-side evaluation disable (no held-out subjects specified).")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {"lr": lr, "batch-size": batch_size, "incorp_status": incorp_flag}
        ),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    print(f"Training completed.")
    torch.save(result.arrays.to_torch_state_dict(), "./final_global_model.pt")

    # Final-window metrics: the mean over the last `report-window` evaluated rounds is
    # far less noisy than the single last round, so it is what the ablation compares.
    window = int(context.run_config.get("report-window", 5))
    client_eval_history = list(getattr(strategy, "client_eval_history", []))
    final_metrics = _final_window_metrics(
        server_eval_history, client_eval_history, window
    )
    last_eval_metrics.update(final_metrics)

    # Parameter counters: base architecture vs. final (what the expansions cost).
    client_param_history = list(getattr(strategy, "client_param_history", []))
    last_eval_metrics["params_base_total"] = base_report["total"]
    if client_param_history:
        for key, value in client_param_history[-1].items():
            if key != "round":
                last_eval_metrics[f"client_{key}"] = value
        base = client_param_history[-1].get("params_base")
        total = client_param_history[-1].get("params_total")
        if base and total is not None:
            last_eval_metrics["client_params_growth_ratio"] = total / base
            print(
                "[params] final (mean over clients, last round): "
                f"base={base:,.0f} | total={total:,.0f} | "
                f"overhead={total - base:,.0f} ({100.0 * (total - base) / base:.2f}%)"
            )

    forgetting_monitor.save_run_summary(
        num_server_rounds=num_rounds,
        server_eval_enabled=bool(held_out_subjects),
        last_eval_metrics=last_eval_metrics,
        run_config={k: v for k, v in context.run_config.items()},
        ablation=flags.as_dict(),
        report_window=window,
        server_eval_history=server_eval_history,
        client_eval_history=client_eval_history,
        client_param_history=client_param_history,
    )
