import pickle
from typing import Iterable, Optional
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
import torch
import io
from src.model.blocks.Adapter import Adapter, adapter_topology


class AdapterIncorporationState:
    def __init__(
        self,
        a_max: int = 3,
        quorum: float = 0.5,
        monitor_rounds: int = 3,
        degrade_tolerance: float = 0.02,
    ) -> None:
        self.a_max = a_max
        self.quorum = quorum
        self.monitor_rounds = monitor_rounds
        self.degrade_tolerance = degrade_tolerance

        self.topologies: list[dict] = []

        self.pending_candidate: Optional[dict] = None
        self._pending_full_arrays_override: Optional[dict] = None
        self._post_vote_signal: Optional[dict] = None
        self._broadcast_signal: Optional[str] = None

        self._latest_arrays_sd: Optional[dict] = None
        self._checkpoint_before_incorp: Optional[dict] = None
        self._reversion_watch: Optional[dict] = None
        self._last_known_acc: Optional[float] = None

        self.total_candidacies_proposed: int = 0
        self.total_accepted: int = 0
        self.total_rejected: int = 0
        self.total_reverted: int = 0
        self.total_confirmed: int = 0
        self._last_vote_favorable_fraction: Optional[float] = None

    def metrics_snapshot(self) -> dict:
        snapshot = {
            "incorp_active_adapters": len(self.topologies),
            "incorp_candidacies_proposed": self.total_candidacies_proposed,
            "incorp_accepted": self.total_accepted,
            "incorp_rejected": self.total_rejected,
            "incorp_reverted": self.total_reverted,
            "incorp_confirmed": self.total_confirmed,
            "incorp_candidacy_pending": int(self.pending_candidate is not None),
            "incorp_monitoring": int(self._reversion_watch is not None),
        }
        if self._last_vote_favorable_fraction is not None:
            snapshot["incorp_last_vote_favorable_fraction"] = (
                self._last_vote_favorable_fraction
            )
        return snapshot

    def on_configure_train(
        self, arrays: ArrayRecord, config: ConfigRecord
    ) -> ArrayRecord:
        config["incorporated_topologies"] = pickle.dumps(self.topologies)
        config["candidacy_locked"] = (
            self.pending_candidate is not None or self._reversion_watch is not None
        )

        outcome = self._post_vote_signal
        self._post_vote_signal = None
        config["candidate_outcome_partition_id"] = (
            outcome["partition_id"] if outcome is not None else -1
        )
        config["candidate_outcome_status"] = (
            outcome["status"] if outcome is not None else "none"
        )

        broadcast = self._broadcast_signal
        self._broadcast_signal = None
        config["last_incorporation_reverted"] = broadcast == "reverted"
        config["last_incorporation_confirmed"] = broadcast == "confirmed"

        if self._pending_full_arrays_override is not None:
            arrays = ArrayRecord(self._pending_full_arrays_override)
            self._pending_full_arrays_override = None

        return arrays

    def on_aggregate_train(self, replies: Iterable[Message]) -> None:
        if self.pending_candidate is not None or self._reversion_watch is not None:
            return
        for reply in replies:
            if not reply.has_content():
                continue
            cfg = reply.content.get("config")
            if cfg is not None and cfg.get("propose_candidate", False):
                self.pending_candidate = {
                    "partition_id": int(cfg["candidate_partition_id"]),
                    "adapter_bytes": cfg["candidate_adapter"],
                }
                self.total_candidacies_proposed += 1
                break

    def on_configure_evaluate(self, arrays: ArrayRecord, config: ConfigRecord) -> None:
        self._latest_arrays_sd = {
            k: v.clone() for k, v in arrays.to_torch_state_dict().items()
        }

        config["incorporated_topologies"] = pickle.dumps(self.topologies)
        if self.pending_candidate is not None:
            config["vote_round"] = True
            config["candidate_adapter"] = self.pending_candidate["adapter_bytes"]
            config["candidate_partition_id"] = self.pending_candidate["partition_id"]
        else:
            config["vote_round"] = False
            config["candidate_partition_id"] = -1

    def on_aggregate_evaluate(
        self,
        replies: Iterable[Message],
        aggregated_metrics: Optional[MetricRecord],
    ) -> None:
        replies = list(replies)

        if self.pending_candidate is not None:
            self._tally_votes(replies)
            return

        if aggregated_metrics is not None and "eval_acc" in aggregated_metrics:
            acc = float(aggregated_metrics["eval_acc"])
            if self._reversion_watch is not None:
                self._check_reversion(acc)
            else:
                self._last_known_acc = acc

    def _reject_candidate(self, candidate: dict) -> None:
        self.total_rejected += 1
        self._post_vote_signal = {
            "partition_id": candidate["partition_id"],
            "status": "reverted",
        }
        self._broadcast_signal = "reverted"

    def _tally_votes(self, replies: list[Message]) -> None:
        candidate = self.pending_candidate
        self.pending_candidate = None

        if candidate is None:
            return

        votes = []
        for reply in replies:
            if not reply.has_content():
                continue
            metrics = reply.content.get("metrics")
            if (
                metrics is None
                or "vote" not in metrics
                or "partition_id" not in metrics
            ):
                continue
            if int(metrics["partition_id"]) == candidate["partition_id"]:
                continue
            votes.append(float(metrics["vote"]))

            if not votes:
                self._last_vote_favorable_fraction = None
                self._reject_candidate(candidate)
                return

        favorable_fraction = sum(votes) / len(votes)
        self._last_vote_favorable_fraction = favorable_fraction
        if favorable_fraction >= self.quorum:
            self._accept_candidate(candidate)
        else:
            self._reject_candidate(candidate)

    def _accept_candidate(self, candidate: dict) -> None:
        if len(self.topologies) >= self.a_max:
            # TODO: implement something like MoFe, to compare the new candidate
            # against the others incorporated topologies, and replace the worst one
            # if the new candidate is better.
            self._reject_candidate(candidate)
            return

        adapter: Adapter = torch.load(
            io.BytesIO(candidate["adapter_bytes"]), weights_only=False
        )
        idx = len(self.topologies)

        base_sd = self._latest_arrays_sd or {}
        self._checkpoint_before_incorp = {
            "topologies": list(self.topologies),
            "arrays_sd": {k: v.clone() for k, v in base_sd.items()},
            "accepted_partition_id": candidate["partition_id"],
        }

        self.topologies.append(adapter_topology(adapter))
        prefix = f"incorporated_adapters.{idx}."
        new_full_sd = dict(base_sd)
        for k, v in adapter.state_dict().items():
            new_full_sd[f"{prefix}{k}"] = v.clone()

        self.total_accepted += 1
        self._pending_full_arrays_override = new_full_sd
        self._reversion_watch = {"rounds_elapsed": 0}
        self._post_vote_signal = {
            "partition_id": candidate["partition_id"],
            "status": "accepted",
        }

    def _check_reversion(self, acc: float) -> None:
        watch = self._reversion_watch
        if watch is None:
            return
        watch["rounds_elapsed"] += 1
        baseline = self._last_known_acc

        if baseline is not None and acc < baseline - self.degrade_tolerance:
            self._revert_last_incorporation()
        elif watch["rounds_elapsed"] >= self.monitor_rounds:
            accepted_pid = (
                self._checkpoint_before_incorp["accepted_partition_id"]
                if self._checkpoint_before_incorp is not None
                else None
            )
            self._reversion_watch = None
            self._checkpoint_before_incorp = None
            self._broadcast_signal = "confirmed"
            self.total_confirmed += 1
            if accepted_pid is not None:
                self._post_vote_signal = {
                    "partition_id": accepted_pid,
                    "status": "confirmed",
                }

        self._last_known_acc = acc

    def _revert_last_incorporation(self) -> None:
        checkpoint = self._checkpoint_before_incorp
        self._reversion_watch = None
        self._checkpoint_before_incorp = None
        if checkpoint is None:
            return
        self.total_reverted += 1
        self.topologies = checkpoint["topologies"]
        self._pending_full_arrays_override = checkpoint["arrays_sd"]
        self._broadcast_signal = "reverted"
        self._post_vote_signal = {
            "partition_id": checkpoint["accepted_partition_id"],
            "status": "reverted",
        }
