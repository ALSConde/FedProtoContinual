from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
import torch
from src.model.Models import FCLModel
from src.server.FedAvgStrategy import FedAvgStrategy
from src.utils.utd_mahd_dataset import INPUT_CHANNELS

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_evaluate = context.run_config["fraction-evaluate"]
    lr = context.run_config["learning-rate"]
    input_dim = int(context.run_config["input-dim"])

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
