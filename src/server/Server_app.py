import random

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
import numpy as np
import torch
from src.model.Models import FCLModel
from src.server.FedAvgStrategy import FedAvgStrategy
from src.server.FedProxStrategy import FedProxStrategy
from src.server.ServerEvaluation import ForgettingMonitor, evaluate_global_model
from src.utils.data.utd_mahd_dataset import (
    build_class_schedule,
    classes_seen_until_round,
    load_server_test_set,
    parse_int_list_config,
    resolve_classes_per_step,
    resolve_dirichlet_mode,
)

app = ServerApp()


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
    )
    arrays = ArrayRecord(global_model.get_global_arrays())

    strategy = FedProxStrategy(
        embedding_dim=hidden_dim,
        tau=15,
        fraction_evaluate=fraction_evaluate,
        proximal_mu=0.01,
        a_max=a_max,
        candidacy_quorum=float(context.run_config.get("candidacy-quorum", 0.5)),
        incorporation_monitor_rounds=int(
            context.run_config.get("incorporation-monitor-rounds", 3)
        ),
        incorporation_degrade_tolerance=float(
            context.run_config.get("incorporation-degrade-tolerance", 0.02)
        ),
    )

    held_out_subjects = parse_int_list_config(
        context.run_config.get("server-eval-subjects")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = str(context.run_config.get("output-dir", "outputs/default_run"))
    forgetting_monitor = ForgettingMonitor(output_dir=output_dir)
    last_eval_metrics: dict = {}
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
            )

            eval_model.load_incorporated_topology(strategy.incorporation.topologies)
            eval_model.set_global_arrays(eval_arrays.to_torch_state_dict())
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

    forgetting_monitor.save_run_summary(
        num_server_rounds=num_rounds,
        server_eval_enabled=bool(held_out_subjects),
        last_eval_metrics=last_eval_metrics,
        run_config={k: v for k, v in context.run_config.items()},
    )
