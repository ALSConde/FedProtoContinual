from flwr.server.strategy import FedAvg
import torch

from server.PrototypeAggregator import PrototypeAggregator


class HybridStrategy(FedAvg):
    def __init__(self, prototype_aggregator: PrototypeAggregator, **kwargs):
        super().__init__(**kwargs)
        self.proto_agg = prototype_aggregator

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        client_stats = []

        for _, fit_res in results:
            m = fit_res.metrics

            sum_h = torch.tensor(m["proto_sum_h"])
            counts = torch.tensor(m["proto_counts"])
            class_ids = torch.tensor(m["proto_class_ids"])

            client_stats.append((sum_h, counts, class_ids))

        mu_global, updated_ids = self.proto_agg.aggregate(client_stats)

    def configure_fit(self, server_round, parameters, client_manager):
        config = {}

        wc, class_ids = self.proto_agg.get_prototypes_normalized()

        config["prototypes"] = wc.numpy()
        config["proto_ids"] = class_ids.numpy()

        return super().configure_fit(server_round, parameters, client_manager, config)
