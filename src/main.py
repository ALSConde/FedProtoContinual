from flwr.app import Context
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp import Grid, ServerApp
from .server.Server import Server

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    print("Starting server...")
    strategy = FedAvg(fraction_evaluate=context.run_config["fraction_evaluate"])
    server = Server(grid=grid, context=context, strategy=strategy)
    server.execute()
    print("Server finished execution.")