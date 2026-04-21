from flwr.app import Context
from flwr.serverapp import Grid, ServerApp
from server.HybridFedAVG import HybridStrategy
from server.PrototypeAggregator import PrototypeAggregator
from server.Server import Server

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    print("Starting server...")
    prototype_aggregator = PrototypeAggregator(
        embedding_dim=512,
        num_classes=10,
        tau=50.0,
    )
    strategy = HybridStrategy(
        prototype_aggregator=prototype_aggregator,
        fraction_train=float(context.run_config.get("fraction_train", 1.0)),
        fraction_evaluate=float(context.run_config.get("fraction_evaluate", 1.0)),
        min_train_nodes=int(context.run_config.get("min_train_nodes", 1)),
        min_evaluate_nodes=int(context.run_config.get("min_evaluate_nodes", 1)),
        min_available_nodes=int(context.run_config.get("min_available_nodes", 1)),
    )
    server = Server(grid=grid, context=context, strategy=strategy)
    server.execute()
    print("Server finished execution.")