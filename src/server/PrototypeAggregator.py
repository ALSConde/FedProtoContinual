import pickle
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


def collect_old_proto_updates(
    replies: Iterable,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Old-prototype updates (class_ids, learned prototypes, num_samples) that the clients
    attached to their train reply (key 'old_proto_update' of the 'config' record, written
    by the client when the PRD prototypes are learnable). Replies without it are skipped.
    """
    updates = []
    for reply in replies:
        if not reply.has_content():
            continue
        config_record = reply.content.get("config")
        if config_record is None or "old_proto_update" not in config_record:
            continue
        ids, protos, n = pickle.loads(config_record["old_proto_update"])
        if len(ids) > 0:
            updates.append((ids, protos, int(n)))
    return updates


class PrototypeAggregator:
    """
    Federated prototype aggregator — runs on the Flower server.

    Receives sufficient statistics from all participating clients:
        sum_h_c^k  → sum of embeddings for class c from client k
        n_c^k      → count of samples for class c from client k

    Computes the new global prototypes as a weighted average:
        μc_new = Σ_k(sum_h_c^k) / Σ_k(n_c^k)

    Updates the global state via adaptive EMA with a rate based on
    the total number of observed samples for each class:
        α_c = 1 - exp(-total_c / τ)

    Behavior:
        Cold start  (new class):      μc_global ← μc_new  (α_c = 1.0)
        Few samples (total_c << τ):  μc_global almost unchanged
        Many samples (total_c >> τ):  μc_global ← μc_new  (dominant update)

    the τ (tau) controls the scale of confidence — should be calibrated as
    a fraction of the expected number of samples per class per aggregated round.
    As a general rule: τ ≈ (total_samples_per_round / num_classes).

    OLD classes (no samples this round) are updated by `aggregate_old`, from the prototypes
    the clients learned through the PRD term (see ClientTask.OldPrototypeState): same EMA
    form, with the evidence N_k = number of samples that drove the distillation and its own
    scale τ_old.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int = 0,
        tau: float = 100.0,
        old_tau: Optional[float] = None,
    ):
        """
        Args:
            embedding_dim:  dimension of the embeddings h
            num_classes:    initial number of classes (can be 0 for pure growth)
            tau:            scale of confidence for the adaptive EMA
                            τ ≈ expected number of samples per class per aggregated round
            old_tau:        scale of the EMA of `aggregate_old` (default: `tau`). Its
                            evidence counts the client samples used as proxy for the old
                            classes, which is much larger than a per-class count, so a
                            larger value than `tau` damps the drift of old prototypes
        """
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.tau = tau
        self.old_tau = float(tau if old_tau is None else old_tau)
        if self.old_tau <= 0:
            raise ValueError(f"old_tau must be > 0 (got {self.old_tau}).")

        self.mu_global: torch.Tensor = torch.zeros(num_classes, embedding_dim)

        self.initialized: torch.Tensor = torch.zeros(num_classes, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Main aggregation
    # ------------------------------------------------------------------
    def aggregate(
        self,
        client_stats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates statistics from all clients and updates μc_global.

        Args:
            client_stats: list of (sum_h, counts, class_ids) per client
                          — direct output of PrototypeMemory.get_stats()
                sum_h:     (num_seen_classes_k, embedding_dim)
                counts:    (num_seen_classes_k,)
                class_ids: (num_seen_classes_k,) — global class indices

        Returns:
            mu_global:  (num_classes, embedding_dim) prototypes updated but not normalized
            class_ids:  (num_updated_classes,) indices of the classes that were updated
                        useful for the server to send only the delta to the clients
        """
        max_class_id = (
            max(int(ids.max().item()) for _, _, ids in client_stats if len(ids) > 0) + 1
        )

        if max_class_id > self.num_classes:
            self._expand(max_class_id)

        sum_h_agg = torch.zeros(self.num_classes, self.embedding_dim)
        total_n = torch.zeros(self.num_classes, dtype=torch.float)

        for sum_h, counts, class_ids in client_stats:
            sum_h_agg[class_ids] += sum_h.float()
            total_n[class_ids] += counts.float()

        updated_mask = total_n > 0
        updated_ids = updated_mask.nonzero(as_tuple=False).squeeze(1)

        if len(updated_ids) == 0:
            return self.mu_global.clone(), updated_ids

        mu_new = sum_h_agg[updated_ids] / total_n[updated_ids].unsqueeze(1)

        for i, c in enumerate(updated_ids):
            c = c.item()
            if not self.initialized[c]:
                # Cold start: initialize with μc_new directly
                self.mu_global[c] = mu_new[i]
                self.initialized[c] = True
            else:
                # EMA adaptive: α_c = 1 - exp(-total_c / τ)
                total_c = total_n[c].detach().item()
                tau = torch.tensor(self.tau, dtype=torch.float)
                alpha_c = 1.0 - torch.exp(-total_c / tau).item()
                self.mu_global[c] = (1.0 - alpha_c) * self.mu_global[
                    c
                ] + alpha_c * mu_new[i]

        return self.mu_global.clone(), updated_ids

    # ------------------------------------------------------------------
    #
    # ------------------------------------------------------------------

    def get_prototypes_normalized(
        self,
        class_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns Wc = μc / ‖μc‖ for the specified class_ids or all initialized classes.

        Args:
            class_ids: if provided, returns only the specified classes
                       if None, returns all initialized classes

        Returns:
            wc:        (num_classes_out, embedding_dim) normalized prototypes
            class_ids: (num_classes_out,) indices of the returned classes
        """
        if class_ids is None:
            class_ids = self.initialized.nonzero(as_tuple=False).squeeze(1)

        wc = F.normalize(self.mu_global[class_ids], dim=1)
        return wc, class_ids

    def get_prototypes_raw(
        self,
        class_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns μc_global not normalized.
        Useful when the client needs to continue accumulating EMA locally.
        """
        if class_ids is None:
            class_ids = self.initialized.nonzero(as_tuple=False).squeeze(1)

        return self.mu_global[class_ids].clone(), class_ids

    def aggregate_old(
        self,
        old_updates: list[tuple[torch.Tensor, torch.Tensor, int]],
    ) -> torch.Tensor:
        """EMA update of OLD classes, which `aggregate` cannot touch (no samples).

        Args:
            old_updates: per client (class_ids (K,), prototypes (K, d), num_samples):
                         the prototypes the client learned for classes absent from its data,
                         in the raw space of `mu_global` (the client restores the norm),
                         and the number of local training samples that drove the PRD term.

        For each class k reported by at least one client:
            p̄_k = Σ_c n_c p̃_{k,c} / Σ_c n_c         (weighted by the client evidence)
            N_k = Σ_c n_c
            α_k = 1 - exp(-N_k / τ_old)
            μ_k ← (1 - α_k) μ_k + α_k p̄_k
        Classes the server has never initialised (or beyond the current class count) are
        ignored: there is nothing to follow. Call it after `aggregate`, so a class that is new
        for some clients and old for others gets both updates in that order.

        Returns the ids of the updated classes.
        """
        sum_p = torch.zeros(self.num_classes, self.embedding_dim)
        total_n = torch.zeros(self.num_classes)
        for ids, protos, n in old_updates:
            ids = torch.as_tensor(ids, dtype=torch.long)
            protos = torch.as_tensor(protos, dtype=torch.float32)
            if len(ids) == 0 or n <= 0:
                continue
            in_range = ids < self.num_classes
            ids, protos = ids[in_range], protos[in_range]
            known = self.initialized[ids]
            ids, protos = ids[known], protos[known]
            sum_p[ids] += float(n) * protos
            total_n[ids] += float(n)
        updated = (total_n > 0).nonzero(as_tuple=False).squeeze(1)
        if len(updated) == 0:
            return updated
        p_bar = sum_p[updated] / total_n[updated].unsqueeze(1)
        alpha = (1.0 - torch.exp(-total_n[updated] / self.old_tau)).unsqueeze(1)
        self.mu_global[updated] = (1.0 - alpha) * self.mu_global[
            updated
        ] + alpha * p_bar
        return updated

    # ------------------------------------------------------------------
    # Growth management
    # ------------------------------------------------------------------

    def _expand(self, new_num_classes: int) -> None:
        """
        Expands the global state to accommodate new classes.
        """
        extra = new_num_classes - self.num_classes

        padding_mu = torch.zeros(extra, self.embedding_dim)
        padding_init = torch.zeros(extra, dtype=torch.bool)

        self.mu_global = torch.cat([self.mu_global, padding_mu], dim=0)
        self.initialized = torch.cat([self.initialized, padding_init], dim=0)
        self.num_classes = new_num_classes

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def num_initialized_classes(self) -> int:
        """Number of classes that have already gone through cold start."""
        return int(self.initialized.sum().item())

    def __repr__(self) -> str:
        return (
            f"PrototypeAggregator("
            f"num_classes={self.num_classes}, "
            f"initialized={self.num_initialized_classes}, "
            f"embedding_dim={self.embedding_dim}, "
            f"tau={self.tau})"
        )
