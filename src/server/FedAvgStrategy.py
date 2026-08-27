import pickle
from typing import Iterable, Optional
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from src.server.PrototypeAggregator import PrototypeAggregator


class FedAvgStrategy(FedAvg):
    def __init__(self, embedding_dim: int, tau: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.proto_aggregator = PrototypeAggregator(
            embedding_dim=embedding_dim, tau=tau
        )
        self._latest_proto_bytes: Optional[bytes] = None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self._latest_proto_bytes is not None:
            config["global_prototypes"] = self._latest_proto_bytes
        return super().configure_train(server_round, arrays, config, grid)
    
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self._latest_proto_bytes is not None:
            config["global_prototypes"] = self._latest_proto_bytes
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        replies = list(replies)

        arrays, metrics = super().aggregate_train(server_round, replies)

        client_stats = []
        for reply in replies:
            if not reply.has_content():
                continue
            config_record = reply.content.get("config")
            if config_record is None or "proto_stats" not in config_record:
                continue
            sum_h, counts, class_ids = pickle.loads(config_record["proto_stats"])
            if len(class_ids) > 0:
                client_stats.append((sum_h, counts, class_ids))

        if client_stats:
            self.proto_aggregator.aggregate(client_stats)
            mu_all, ids_all = self.proto_aggregator.get_prototypes_raw()
            if len(ids_all) > 0:
                self._latest_proto_bytes = pickle.dumps((mu_all, ids_all))

        return arrays, metrics
