import pickle
from typing import Iterable, Optional
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
import torch
import io
from src.model.blocks.Adapter import (
    Adapter,
    adapter_topology,
    build_adapter_from_topology,
)

_INC_PREFIX = "incorporated_adapter."
_SHADOW_PREFIX = "shadow_adapter."


def _drop_slot(sd: dict, slot: int) -> dict:
    """Remove incorporated adapter `slot` from a flat state dict and shift the
    indices of the following adapters down by one (keys keep their order)."""
    out: dict = {}
    for k, v in sd.items():
        if not k.startswith(_INC_PREFIX):
            out[k] = v
            continue
        idx_str, sub = k[len(_INC_PREFIX) :].split(".", 1)
        idx = int(idx_str)
        if idx == slot:
            continue
        if idx > slot:
            idx -= 1
        out[f"{_INC_PREFIX}{idx}.{sub}"] = v
    return out


def _summarize(payloads: list, name: str, margin: float) -> Optional[dict]:
    """Pool the paired per-sample loss differences of configuration `name` against
    the current model (positive = `name` has lower loss), across voting clients.

    mean : sample-weighted mean loss gain
    se   : standard error of that mean (paired, sample level; optimistic because
           samples of one client are correlated)
    fav  : fraction of clients whose own mean gain is > 0
    viol : MoFe hinge, mean of max(0, margin - gain) -- penalizes configurations
           where the candidate does not reduce the task loss by `margin`
    """
    n_tot, d_sum, d_sq, fav, n_cli, viol = 0, 0.0, 0.0, 0, 0, 0.0
    for p in payloads:
        st = p["stats"].get(name)
        n = int(p["n"])
        if st is None or n <= 0:
            continue
        n_tot += n
        d_sum += st["d_sum"]
        d_sq += st["d_sq"]
        n_cli += 1
        mean_k = st["d_sum"] / n
        fav += int(mean_k > 0.0)
        viol += n * max(0.0, margin - mean_k)
    if n_tot == 0 or n_cli == 0:
        return None
    mean = d_sum / n_tot
    var = max(d_sq / n_tot - mean * mean, 0.0)
    return {
        "mean": mean,
        "se": (var / n_tot) ** 0.5,
        "fav": fav / n_cli,
        "n": n_tot,
        "viol": viol / n_tot,
    }


class AdapterIncorporationState:
    def __init__(
        self,
        a_max: int = 3,
        quorum: float = 0.5,
        monitor_rounds: int = 3,
        degrade_tolerance: float = 0.02,
        enabled: bool = True,
        baseline_window: int = 3,
        vote_mode: str = "legacy",
        vote_loss_margin: float = 0.01,
        vote_z: float = 1.0,
        prune_enabled: bool = True,
        prune_margin: float = 0.01,
        prune_patience: int = 5,
        loo_ema: float = 0.7,
        shadow_rounds: int = 0,
        shadow_window: int = 3,
    ) -> None:
        # degrade_tolerance is relative: the fraction of the pre-incorporation accuracy
        # that may be lost before reverting (0.05 -> revert if acc < 0.95 * baseline).
        self.enabled = enabled
        self.a_max = a_max
        self.quorum = quorum
        self.monitor_rounds = monitor_rounds
        self.degrade_tolerance = degrade_tolerance
        # Pre-incorporation baseline = mean of the last `baseline_window` client-eval
        # accuracies (including the vote round itself), to damp round-to-round noise.
        self.baseline_window = max(1, int(baseline_window))

        if vote_mode not in ("legacy", "mofe"):
            raise ValueError(
                f"Unknown vote_mode '{vote_mode}' (use 'legacy' or 'mofe')."
            )
        self.vote_mode = vote_mode
        # MoFe vote: the candidate must lower the clients' validation loss of the
        # shared embedding by `vote_loss_margin` (loss units) and by `vote_z`
        # standard errors, and a `quorum` fraction of voters must individually gain.
        self.vote_loss_margin = float(vote_loss_margin)
        self.vote_z = float(vote_z)
        # Leave-one-out pruning: an adapter whose removal lowers the validation
        # loss by more than `prune_margin` (EMA, `loo_ema`) for `prune_patience`
        # consecutive evaluations is removed.
        self.prune_enabled = bool(prune_enabled)
        self.prune_margin = float(prune_margin)
        self.prune_patience = max(1, int(prune_patience))
        self.loo_ema = float(loo_ema)
        self._loo_value: list[float] = []
        self._loo_streak: list[int] = []
        # Probation: a proposed candidate is first co-trained by everybody
        # as a shadow adapter for `shadow_rounds` rounds (0 = vote right away); the
        # MoFe decision pools the last `shadow_window` evaluation rounds.
        if shadow_rounds > 0 and vote_mode != "mofe":
            raise ValueError("shadow_rounds > 0 requires vote_mode='mofe'.")
        self.shadow_rounds = max(0, int(shadow_rounds))
        self.shadow_window = max(1, int(shadow_window))
        self._shadow: Optional[dict] = None
        self._strip_shadow_keys: bool = False
        self._shadow_last_gain: Optional[float] = None
        self._pending_replace: Optional[dict] = None
        self._pending_prune: Optional[int] = None
        self._deferred_restore: Optional[dict] = None
        self.total_replaced: int = 0
        self.total_pruned: int = 0
        self._last_vote_delta: Optional[float] = None
        self._last_vote_se: Optional[float] = None
        self._last_vote_violation: Optional[float] = None

        self.topologies: list[dict] = []

        self.pending_candidate: Optional[dict] = None
        self._pending_full_arrays_override: Optional[dict] = None
        self._post_vote_signal: Optional[dict] = None
        self._broadcast_signal: Optional[str] = None

        self._latest_arrays_sd: Optional[dict] = None
        self._checkpoint_before_incorp: Optional[dict] = None
        self._reversion_watch: Optional[dict] = None
        self._acc_history: list[float] = []

        self.total_candidacies_proposed: int = 0
        self.total_accepted: int = 0
        self.total_rejected: int = 0
        self.total_reverted: int = 0
        self.total_confirmed: int = 0
        self._last_vote_favorable_fraction: Optional[float] = None

    def _record_acc(self, acc: float) -> None:
        self._acc_history.append(acc)
        del self._acc_history[: -self.baseline_window]

    def _baseline(self) -> Optional[float]:
        if not self._acc_history:
            return None
        return sum(self._acc_history) / len(self._acc_history)

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
        if self.vote_mode == "mofe":
            snapshot["incorp_shadow_active"] = int(self._shadow is not None)
            snapshot["incorp_shadow_round"] = (
                self._shadow["rounds_done"] if self._shadow is not None else 0
            )
            if self._shadow_last_gain is not None:
                snapshot["incorp_shadow_loss_gain"] = self._shadow_last_gain
            snapshot["incorp_replaced"] = self.total_replaced
            snapshot["incorp_pruned"] = self.total_pruned
            if self._last_vote_delta is not None:
                snapshot["incorp_last_vote_loss_gain"] = self._last_vote_delta
                snapshot["incorp_last_vote_loss_gain_se"] = self._last_vote_se
                snapshot["incorp_last_vote_mofe_violation"] = self._last_vote_violation
        return snapshot

    def on_configure_train(
        self, arrays: ArrayRecord, config: ConfigRecord
    ) -> ArrayRecord:
        arrays = self._drop_shadow_arrays(arrays)
        arrays = self._apply_deferred_changes(arrays)
        arrays = self._inject_shadow(arrays)
        config["incorporated_topologies"] = pickle.dumps(self.topologies)
        config["shadow_topology"] = pickle.dumps(self._active_shadow_topology())
        config["candidacy_locked"] = (
            self.pending_candidate is not None
            or self._reversion_watch is not None
            or self._shadow is not None
        )

        outcome = self._post_vote_signal
        self._post_vote_signal = None
        config["candidate_outcome_partition_id"] = (
            outcome["partition_id"] if outcome is not None else -1
        )
        config["incorporation_outcome_status"] = (
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
        if not self.enabled:
            return
        if (
            self.pending_candidate is not None
            or self._reversion_watch is not None
            or self._shadow is not None
        ):
            return
        for reply in replies:
            if not reply.has_content():
                continue
            cfg = reply.content.get("config")
            if cfg is not None and cfg.get("propose_candidate", False):
                if self.shadow_rounds > 0:
                    self._start_shadow(
                        int(cfg["candidate_partition_id"]), cfg["candidate_adapter"]
                    )
                    self.total_candidacies_proposed += 1
                    break
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
        config["shadow_topology"] = pickle.dumps(self._active_shadow_topology())
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

        acc = None
        if aggregated_metrics is not None and "eval_acc" in aggregated_metrics:
            acc = float(aggregated_metrics["eval_acc"])

        if self.pending_candidate is not None:
            # Vote rounds report the regular eval_acc, measured on the model without
            # the candidate: it is part of the pre-incorporation baseline. Record it
            # before tallying so that an acceptance freezes the window including it.
            if acc is not None:
                self._record_acc(acc)
            if self.vote_mode == "mofe":
                self._tally_mofe(replies)
            else:
                self._tally_votes(replies)
            return

        if self._shadow is not None:
            # Probation round: eval_acc is the deployed (shadow-free) model, i.e. the
            # pre-incorporation baseline of a later acceptance.
            if acc is not None:
                self._record_acc(acc)
            self._shadow_step(replies)
            return

        if acc is not None:
            if self._reversion_watch is not None:
                self._check_reversion(acc)
            else:
                self._record_acc(acc)

        if self.vote_mode == "mofe":
            self._update_loo(replies)

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

    def _accept_candidate(
        self, candidate: dict, adapter: Optional[Adapter] = None
    ) -> None:
        if len(self.topologies) >= self.a_max:
            if self.vote_mode == "mofe":
                raise RuntimeError(
                    "_accept_candidate called with a full incorporated set in "
                    "vote_mode='mofe'; replacement must go through _stage_replacement."
                )
            self._reject_candidate(candidate)
            return

        if adapter is None:
            adapter = torch.load(
                io.BytesIO(candidate["adapter_bytes"]),
                map_location="cpu",
                weights_only=False,
            )
        idx = len(self.topologies)

        base_sd = {
            k: v
            for k, v in (self._latest_arrays_sd or {}).items()
            if not k.startswith(_SHADOW_PREFIX)
        }
        self._checkpoint_before_incorp = {
            "topologies": list(self.topologies),
            "arrays_sd": {k: v.clone() for k, v in base_sd.items()},
            "accepted_partition_id": candidate["partition_id"],
        }

        self.topologies.append(adapter_topology(adapter))
        prefix = f"incorporated_adapter.{idx}."
        new_full_sd = dict(base_sd)
        for k, v in adapter.state_dict().items():
            new_full_sd[f"{prefix}{k}"] = v.clone()

        self.total_accepted += 1
        self._pending_full_arrays_override = new_full_sd
        self._reversion_watch = {
            "rounds_elapsed": 0,
            "baseline_acc": self._baseline(),
        }
        self._post_vote_signal = {
            "partition_id": candidate["partition_id"],
            "status": "accepted",
        }

    def _check_reversion(self, acc: float) -> None:
        watch = self._reversion_watch
        if watch is None:
            return
        watch["rounds_elapsed"] += 1
        # Fixed during the whole monitoring window: accuracy of the model right
        # before the incorporation.
        baseline = watch.get("baseline_acc")
        threshold = (
            (1.0 - self.degrade_tolerance) * baseline if baseline is not None else None
        )

        if threshold is not None and acc < threshold:
            print(
                f"[incorporation reverted: acc={acc:.4f} < "
                f"(1 - {self.degrade_tolerance}) * baseline={baseline:.4f} "
                f"= {threshold:.4f}]"
            )
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

        if self._reversion_watch is None:
            # Watch ended (confirmed or reverted): the model changed.
            # Start a fresh baseline window from this round.
            self._acc_history = [acc]

    def _revert_last_incorporation(self) -> None:
        checkpoint = self._checkpoint_before_incorp
        self._reversion_watch = None
        self._checkpoint_before_incorp = None
        if checkpoint is None:
            return
        self.total_reverted += 1
        if checkpoint.get("kind") == "replace":
            # Indices shift on replacement, so the restore must wait for the next
            # configure_train: this round's evaluate_fn still pairs the current
            # topologies with this round's (current-indexing) arrays.
            self._deferred_restore = checkpoint
        else:
            self.topologies = checkpoint["topologies"]
            self._pending_full_arrays_override = checkpoint["arrays_sd"]
        self._broadcast_signal = "reverted"
        self._post_vote_signal = {
            "partition_id": checkpoint["accepted_partition_id"],
            "status": "reverted",
        }

    # ------------------------- MoFe vote -----------------------------------------
    @staticmethod
    def _parse_payloads(replies: list) -> list:
        payloads = []
        for reply in replies:
            if not reply.has_content():
                continue
            cfg = reply.content.get("config")
            if cfg is None or "mofe_stats" not in cfg:
                continue
            payloads.append(pickle.loads(cfg["mofe_stats"]))
        return payloads

    def _passes(self, summary: Optional[dict]) -> bool:
        if summary is None:
            return False
        return (
            summary["mean"] > self.vote_loss_margin
            and summary["mean"] >= self.vote_z * summary["se"]
            and summary["fav"] >= self.quorum
        )

    def _tally_mofe(self, replies: list) -> None:
        candidate = self.pending_candidate
        self.pending_candidate = None
        if candidate is None:
            return

        payloads = [
            p
            for p in self._parse_payloads(replies)
            if p.get("kind") == "vote"
            and int(p["partition_id"]) != candidate["partition_id"]
            and int(p.get("k", -1)) == len(self.topologies)
        ]
        full = len(self.topologies) >= self.a_max

        if not full:
            summary = _summarize(payloads, "plus", self.vote_loss_margin)
            slot = None
        else:
            summary, slot = None, None
            for j in range(len(self.topologies)):
                cand = _summarize(payloads, f"rep_{j}", self.vote_loss_margin)
                if cand is not None and (
                    summary is None or cand["mean"] > summary["mean"]
                ):
                    summary, slot = cand, j

        if summary is None:
            self._last_vote_favorable_fraction = None
            self._last_vote_delta = self._last_vote_se = None
            self._last_vote_violation = None
            self._reject_candidate(candidate)
            return

        self._last_vote_favorable_fraction = summary["fav"]
        self._last_vote_delta = summary["mean"]
        self._last_vote_se = summary["se"]
        self._last_vote_violation = summary["viol"]

        if not self._passes(summary):
            self._reject_candidate(candidate)
        elif slot is None:
            self._accept_candidate(candidate)
        else:
            self._stage_replacement(candidate, slot)

    def _stage_replacement(
        self, candidate: dict, slot: int, adapter: Optional[Adapter] = None
    ) -> None:
        if adapter is None:
            adapter = torch.load(
                io.BytesIO(candidate["adapter_bytes"]),
                map_location="cpu",
                weights_only=False,
            )
        self._pending_replace = {
            "slot": slot,
            "adapter": adapter,
            "partition_id": candidate["partition_id"],
        }
        self.total_accepted += 1
        self._reversion_watch = {
            "rounds_elapsed": 0,
            "baseline_acc": self._baseline(),
        }
        self._post_vote_signal = {
            "partition_id": candidate["partition_id"],
            "status": "accepted",
        }

    # ----------------------- leave-one-out pruning --------------------------------
    def _reset_loo(self) -> None:
        self._loo_value = []
        self._loo_streak = []

    def _update_loo(self, replies: list) -> None:
        k = len(self.topologies)
        if k == 0:
            self._reset_loo()
            return
        if not self.prune_enabled:
            return
        payloads = [
            p
            for p in self._parse_payloads(replies)
            if p.get("kind") == "loo" and int(p.get("k", -1)) == k
        ]
        if not payloads:
            return
        if len(self._loo_value) != k:
            self._loo_value = (self._loo_value + [0.0] * k)[:k]
            self._loo_streak = (self._loo_streak + [0] * k)[:k]
        for j in range(k):
            summary = _summarize(payloads, f"drop_{j}", self.prune_margin)
            if summary is None:
                continue
            # positive = validation loss gets lower without adapter j (j is harmful)
            self._loo_value[j] = (
                self.loo_ema * self._loo_value[j]
                + (1.0 - self.loo_ema) * summary["mean"]
            )
            if self._loo_value[j] > self.prune_margin:
                self._loo_streak[j] += 1
            else:
                self._loo_streak[j] = 0

        busy = (
            self.pending_candidate is not None
            or self._reversion_watch is not None
            or self._pending_full_arrays_override is not None
            or self._pending_replace is not None
            or self._pending_prune is not None
            or self._deferred_restore is not None
        )
        if busy:
            return
        due = [j for j in range(k) if self._loo_streak[j] >= self.prune_patience]
        if due:
            self._pending_prune = max(due, key=lambda j: self._loo_value[j])

    # ----------------- deferred (index-shifting) changes --------------------------
    def _apply_deferred_changes(self, arrays: ArrayRecord) -> ArrayRecord:
        if self._deferred_restore is not None:
            checkpoint = self._deferred_restore
            self._deferred_restore = None
            self.topologies = list(checkpoint["topologies"])
            self._reset_loo()
            return ArrayRecord(checkpoint["arrays_sd"])

        if self._pending_replace is not None:
            pending = self._pending_replace
            self._pending_replace = None
            sd = {k: v.clone() for k, v in arrays.to_torch_state_dict().items()}
            self._checkpoint_before_incorp = {
                "topologies": list(self.topologies),
                "arrays_sd": {k: v.clone() for k, v in sd.items()},
                "accepted_partition_id": pending["partition_id"],
                "kind": "replace",
            }
            new_sd = _drop_slot(sd, pending["slot"])
            self.topologies.pop(pending["slot"])
            self.topologies.append(adapter_topology(pending["adapter"]))
            idx = len(self.topologies) - 1
            for k, v in pending["adapter"].state_dict().items():
                new_sd[f"{_INC_PREFIX}{idx}.{k}"] = v.clone()
            self.total_replaced += 1
            self._reset_loo()
            return ArrayRecord(new_sd)

        if self._pending_prune is not None:
            slot = self._pending_prune
            self._pending_prune = None
            if slot < len(self.topologies):
                sd = arrays.to_torch_state_dict()
                new_sd = _drop_slot(dict(sd), slot)
                self.topologies.pop(slot)
                self.total_pruned += 1
                self._reset_loo()
                print(f"[incorporation pruned: adapter slot {slot} removed (LOO)]")
                return ArrayRecord(new_sd)
        return arrays

    # ----------------------------- probation -------------------------------------
    def _start_shadow(self, partition_id: int, adapter_bytes: bytes) -> None:
        adapter: Adapter = torch.load(
            io.BytesIO(adapter_bytes), map_location="cpu", weights_only=False
        )
        self._shadow = {
            "partition_id": partition_id,
            "topology": adapter_topology(adapter),
            "init_sd": {k: v.clone() for k, v in adapter.state_dict().items()},
            "inject": True,  # weights go into the global arrays at the next configure_train
            "active": False,  # clients only get the topology once the weights are there
            "rounds_done": 0,
            "summaries": [],
        }

    def _active_shadow_topology(self) -> Optional[dict]:
        if self._shadow is not None and self._shadow["active"]:
            return self._shadow["topology"]
        return None

    def _inject_shadow(self, arrays: ArrayRecord) -> ArrayRecord:
        sh = self._shadow
        if sh is None or not sh["inject"]:
            return arrays
        sd = {k: v.clone() for k, v in arrays.to_torch_state_dict().items()}
        for k, v in sh["init_sd"].items():
            sd[f"{_SHADOW_PREFIX}0.{k}"] = v.clone()
        sh["inject"] = False
        sh["active"] = True
        return ArrayRecord(sd)

    def _drop_shadow_arrays(self, arrays: ArrayRecord) -> ArrayRecord:
        """Remove the (finished) shadow adapter's weights from the global arrays."""
        if not self._strip_shadow_keys or self._shadow is not None:
            return arrays
        self._strip_shadow_keys = False
        sd = arrays.to_torch_state_dict()
        if not any(k.startswith(_SHADOW_PREFIX) for k in sd):
            return arrays
        return ArrayRecord(
            {k: v for k, v in sd.items() if not k.startswith(_SHADOW_PREFIX)}
        )

    def _shadow_step(self, replies: list) -> None:
        sh = self._shadow
        if sh is None or not sh["active"]:
            return
        payloads = [
            p
            for p in self._parse_payloads(replies)
            if p.get("kind") == "vote"
            and int(p["partition_id"]) != sh["partition_id"]
            and int(p.get("k", -1)) == len(self.topologies)
        ]
        full = len(self.topologies) >= self.a_max
        names = [f"rep_{j}" for j in range(len(self.topologies))] if full else ["plus"]
        round_summary = {}
        for name in names:
            summary = _summarize(payloads, name, self.vote_loss_margin)
            if summary is not None:
                round_summary[name] = summary
        sh["summaries"].append(round_summary)
        sh["rounds_done"] += 1
        if round_summary:
            self._shadow_last_gain = max(s["mean"] for s in round_summary.values())
        if sh["rounds_done"] >= self.shadow_rounds:
            self._finish_shadow()

    def _finish_shadow(self) -> None:
        sh = self._shadow
        self._shadow = None
        self._strip_shadow_keys = True
        window = sh["summaries"][-self.shadow_window :]
        candidate = {"partition_id": sh["partition_id"], "adapter_bytes": None}

        best_name, best = None, None
        names = set().union(*[w.keys() for w in window]) if window else set()
        for name in sorted(names):
            per_round = [w[name] for w in window if name in w]
            if len(per_round) < len(window):
                continue  # need every round of the window to trust the pooled estimate
            m = len(per_round)
            agg = {
                "mean": sum(s["mean"] for s in per_round) / m,
                "se": (sum(s["se"] ** 2 for s in per_round) / m) ** 0.5,
                "fav": sum(s["fav"] for s in per_round) / m,
                "viol": sum(s["viol"] for s in per_round) / m,
                "n": per_round[-1]["n"],
            }
            if best is None or agg["mean"] > best["mean"]:
                best_name, best = name, agg

        if best is None:
            self._last_vote_favorable_fraction = None
            self._last_vote_delta = self._last_vote_se = self._last_vote_violation = (
                None
            )
            self._reject_candidate(candidate)
            return

        self._last_vote_favorable_fraction = best["fav"]
        self._last_vote_delta = best["mean"]
        self._last_vote_se = best["se"]
        self._last_vote_violation = best["viol"]
        self._shadow_last_gain = best["mean"]

        if not self._passes(best):
            self._reject_candidate(candidate)
            return

        # The module that gets incorporated is the co-trained one (aggregated weights
        # at the last evaluation), not the proposer's original.
        prefix = f"{_SHADOW_PREFIX}0."
        trained = {
            k[len(prefix) :]: v.clone()
            for k, v in (self._latest_arrays_sd or {}).items()
            if k.startswith(prefix)
        }
        if not trained:
            raise RuntimeError(
                "Probation finished but the aggregated shadow-adapter weights are missing "
                "from the latest global arrays."
            )
        adapter = build_adapter_from_topology(sh["topology"])
        adapter.load_state_dict(trained)
        if best_name == "plus":
            self._accept_candidate(candidate, adapter)
        else:
            self._stage_replacement(candidate, int(best_name.split("_")[1]), adapter)
