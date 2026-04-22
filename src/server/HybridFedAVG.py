from __future__ import annotations

from collections.abc import Iterable

import torch
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from flwr.common import Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy.fedavg import FedAvg

from .PrototypeAggregator import PrototypeAggregator


class HybridStrategy(FedAvg):
    def __init__(self, prototype_aggregator: PrototypeAggregator, **kwargs):
        super().__init__(**kwargs)
        self.proto_agg = prototype_aggregator
        self._last_train_state: dict[str, torch.Tensor] | None = None

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        self._last_train_state = {
            key: value.clone() for key, value in arrays.to_torch_state_dict().items()
        }
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        arrays, metrics = super().aggregate_train(server_round, replies)

        if arrays is None:
            return arrays, metrics

        merged_state = {
            key: value.clone() for key, value in arrays.to_torch_state_dict().items()
        }

        if self._last_train_state is not None:
            baseline_state = {
                key: value.clone() for key, value in self._last_train_state.items()
            }
            for key, value in merged_state.items():
                if key.startswith("shared_encoder.") or key.startswith(
                    "global_features."
                ):
                    baseline_state[key] = value
        else:
            baseline_state = merged_state

        client_stats = self._collect_client_prototype_stats(replies)
        if client_stats:
            mu_global, updated_ids = self.proto_agg.aggregate(client_stats)
            if len(updated_ids) > 0:
                prototype_key = "classifier.prototypes"
                if prototype_key not in baseline_state:
                    baseline_state[prototype_key] = merged_state[prototype_key].clone()

                prototypes = baseline_state[prototype_key]
                if prototypes.size(0) < mu_global.size(0):
                    extra = mu_global.size(0) - prototypes.size(0)
                    padding = torch.zeros(
                        extra,
                        prototypes.size(1),
                        dtype=prototypes.dtype,
                        device=prototypes.device,
                    )
                    prototypes = torch.cat([prototypes, padding], dim=0)

                prototypes[updated_ids] = mu_global[updated_ids].to(
                    device=prototypes.device, dtype=prototypes.dtype
                )
                baseline_state[prototype_key] = prototypes

        arrays = ArrayRecord(baseline_state)
        return arrays, metrics

    def _collect_client_prototype_stats(
        self, replies: Iterable[Message]
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        client_stats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for reply in replies:
            stats = self._extract_proto_stats(reply)
            if stats is not None:
                client_stats.append(stats)

        return client_stats

    def _extract_proto_stats(
        self, reply: Message
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        content = reply.content

        for array_record in content.array_records.values():
            state_dict = array_record.to_torch_state_dict()
            required_keys = {"proto_sum_h", "proto_counts", "proto_class_ids"}
            if required_keys.issubset(state_dict.keys()):
                return (
                    state_dict["proto_sum_h"].detach().cpu(),
                    state_dict["proto_counts"].detach().cpu(),
                    state_dict["proto_class_ids"].detach().cpu().long(),
                )

        for metric_record in content.metric_records.values():
            required_keys = {"proto_sum_h", "proto_counts", "proto_class_ids"}
            if required_keys.issubset(metric_record.keys()):
                return (
                    torch.tensor(metric_record["proto_sum_h"]),
                    torch.tensor(metric_record["proto_counts"]),
                    torch.tensor(metric_record["proto_class_ids"]).long(),
                )

        return None
