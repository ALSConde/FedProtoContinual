from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import MetricsRecord
from flwr.serverapp import Grid, ServerApp
from matplotlib.pylab import dirichlet
import torch
from src.model.Models import FCLModel
from src.server.FedAvgStrategy import FedAvgStrategy
from src.server.ServerEvaluation import ContinualMetricsTracker, evaluate_global_model
from src.utils.utd_mahd_dataset import (
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
    lr = context.run_config["learning-rate"]
    input_dim = int(context.run_config["input-dim"])
    hidden_dim = int(context.run_config["hidden-dim"])
    d_hat_global = int(context.run_config["d-hat-global"])

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
    )
    arrays = ArrayRecord(global_model.get_global_arrays())

    strategy = FedAvgStrategy(
        embedding_dim=hidden_dim,
        fraction_evaluate=fraction_evaluate,
    )

    held_out_subjects = parse_int_list_config(
        context.run_config.get("server-eval-subjects")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_tracker = ContinualMetricsTracker()
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
            )
            eval_model.set_global_arrays(eval_arrays.to_torch_state_dict())
            eval_model.classifier.update_from_global(mu_all, ids_all)
            eval_model.to(device)

            allowed_classes = None
            if schedule is not None:
                allowed_classes = classes_seen_until_round(
                    max(current_round, 1), rounds_per_step_for_eval, schedule
                )

            loss, acc, per_class_acc = evaluate_global_model(
                eval_model, server_test_loader, device, allowed_classes
            )

            metrics = {"server_eval_loss": loss, "server_eval_acc": acc}

            if schedule is not None:
                metrics_tracker.update(current_round, per_class_acc)
                bwt = metrics_tracker.backward_transfer(current_round)
                forgetting = metrics_tracker.average_forgetting(current_round)
                if bwt is not None:
                    metrics["bwt"] = bwt
                if forgetting is not None:
                    metrics["avg_forgetting"] = forgetting
                metrics["num_classes_seen"] = int(len(allowed_classes))
            return MetricsRecord(metrics)

    else:
        print("server-side evaluation disable (no held-out subjects specified).")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    print(f"Training completed.")
    torch.save(result.arrays.to_torch_state_dict(), "./final_global_model.pt")
