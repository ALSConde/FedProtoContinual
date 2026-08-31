from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from matplotlib.pylab import dirichlet
import torch
from src.model.Models import FCLModel
from src.server.FedAvgStrategy import FedAvgStrategy
from src.utils.utd_mahd_dataset import (
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
        hidden_dim=int(context.run_config["hidden-dim"]),
        d_hat_global=int(context.run_config["d-hat-global"]),
    )
    arrays = ArrayRecord(global_model.get_global_arrays())

    strategy = FedAvgStrategy(
        embedding_dim=int(context.run_config["hidden-dim"]),
        fraction_evaluate=fraction_evaluate,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    print(f"Training completed.")
    torch.save(result.arrays.to_torch_state_dict(), "./final_global_model.pt")
