import torch
import torch.nn.functional as F


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
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int = 0,
        tau: float = 100.0,
    ):
        """
        Args:
            embedding_dim:  dimension of the embeddings h
            num_classes:    initial number of classes (can be 0 for pure growth)
            tau:            scale of confidence for the adaptive EMA
                            τ ≈ expected number of samples per class per aggregated round
        """
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.tau = tau

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
