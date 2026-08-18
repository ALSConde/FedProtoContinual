from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from src.server.FedAvgStrategy import FedAvgStrategy

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = context.run_config["num-server-rounds"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    lr = context.run_config["learning-rate"]

    global_model = some_model()
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
