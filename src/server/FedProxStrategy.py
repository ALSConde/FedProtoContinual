import pickle
from typing import Iterable, Optional
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedProx
from src.server.AdapterIncorporation import AdapterIncorporationState
from src.server.PrototypeAggregator import PrototypeAggregator


class FedProxStrategy(FedProx):
    def __init__(
        self,
        embedding_dim: int,
        tau: float = 100.0,
        a_max: int = 3,
        candidacy_quorum: float = 0.5,
        incorporation_monitor_rounds: int = 3,
        incorporation_degrade_tolerance: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.proto_aggregator = PrototypeAggregator(
            embedding_dim=embedding_dim, tau=tau
        )
        self._latest_proto_bytes: Optional[bytes] = None
        self.incorporation = AdapterIncorporationState(
            a_max=a_max,
            quorum=candidacy_quorum,
            monitor_rounds=incorporation_monitor_rounds,
            degrade_tolerance=incorporation_degrade_tolerance,
        )

        self.incorp_flag = "false"

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        config["server_round"] = server_round
        if self._latest_proto_bytes is not None:
            config["global_prototypes"] = self._latest_proto_bytes
        if config["incorp_status"] is not None:
            self.incorp_flag = str(config["incorp_status"]).lower()
        else:
            self.incorp_flag = "false"
        arrays = self.incorporation.on_configure_train(arrays, config)
        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        config["server_round"] = server_round
        if self._latest_proto_bytes is not None:
            config["global_prototypes"] = self._latest_proto_bytes
        self.incorporation.on_configure_evaluate(arrays, config)
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

        self.incorporation.on_aggregate_train(replies)

        if metrics is not None and self.incorp_flag != "false":
            for k, v in self.incorporation.metrics_snapshot().items():
                metrics[k] = v

        return arrays, metrics

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> Optional[MetricRecord]:
        replies = list(replies)
        metrics = super().aggregate_evaluate(server_round, replies)
        self.incorporation.on_aggregate_evaluate(replies, metrics)
        return metrics
