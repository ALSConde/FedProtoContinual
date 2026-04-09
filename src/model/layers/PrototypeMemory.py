import torch


class PrototypeMemory:
    """
    Local memory for accumulating class prototypes during training on the client side.

    Does not store samples — only sufficient statistics for federated aggregation
    on the server:
        sum_h_c  → sum of embeddings h per class c
        n_c      → count of samples per class c

    These statistics allow the server to compute:
        μc = sum_h_c / n_c  (mean of embeddings per class)

    The state is reset at each federated round via reset().
    There is no persistence on disk — everything is in RAM/VRAM.

    Typical usage in the training loop:
        # during training iteration
        memory.update(features, labels)

        # after_training (before sending to server)
        sum_h, counts, class_ids = memory.get_stats()
        memory.reset()
    """

    def __init__(self, embedding_dim: int, num_classes: int, device: torch.device):
        """
        Args:
            embedding_dim:   dimension of the embeddings h
            num_classes:    maximum number of classes known up to the moment
                            can grow via expand() when new classes arrive
            device:         device where the tensors will be kept
        """
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.device = device

        self._sum_h  = torch.zeros(num_classes, embedding_dim, device=device)
        self._counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Accumulation during training
    # ------------------------------------------------------------------

    def update(self, h: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Accumulates embeddings and counts for each class present in the batch.

        Must be called after the forward pass, before the backward pass.
        The embeddings are accumulated without normalization — normalization
        occurs on the server after aggregation.

        Args:
            h:      final embeddings of the model, shape (batch_size, embedding_dim)
                    must be detached from the graph: h.detach()
            labels: labels of the batch, shape (batch_size,), values in [0, num_classes)
        """
        h = h.detach().to(self.device)
        labels = labels.to(self.device)

        max_label = int(labels.max().item()) + 1
        if max_label > self.num_classes:
            self.expand(max_label)

        for c in labels.unique():
            mask = labels == c
            self._sum_h[c]  += h[mask].sum(dim=0)
            self._counts[c] += mask.sum()

    # ------------------------------------------------------------------
    # Aggregation and reset
    # ------------------------------------------------------------------

    def get_stats(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the accumulated statistics only for classes seen in this round.

        Returns:
            sum_h:     shape (num_seen_classes, embedding_dim)
                       sum of embeddings per seen class
            counts:    shape (num_seen_classes,)
                       number of samples per seen class
            class_ids: shape (num_seen_classes,)
                       indices of the seen classes — necessary for the server
                       to know which prototypes to update

        Example of usage in the Flower client:
            sum_h, counts, class_ids = memory.get_stats()
            # serialize and send as custom metrics to the server
        """
        seen_mask  = self._counts > 0
        class_ids  = seen_mask.nonzero(as_tuple=False).squeeze(1)
        sum_h      = self._sum_h[class_ids]
        counts     = self._counts[class_ids]

        return sum_h.cpu(), counts.cpu(), class_ids.cpu()

    def reset(self) -> None:
        """
        Resets the accumulators for the next federated round.
        Must be called in after_training, after get_stats().
        """
        self._sum_h.zero_()
        self._counts.zero_()

    # ------------------------------------------------------------------
    # Incremental class growth
    # ------------------------------------------------------------------

    def expand(self, new_num_classes: int) -> None:
        """
        Expands the buffers to accommodate new classes.
        Called automatically by update() when necessary.

        Args:
            new_num_classes: new total number of classes (must be > current num_classes)
        """
        if new_num_classes <= self.num_classes:
            return

        extra = new_num_classes - self.num_classes

        padding_h = torch.zeros(extra, self.embedding_dim, device=self.device)
        padding_n = torch.zeros(extra, dtype=torch.long,   device=self.device)

        self._sum_h  = torch.cat([self._sum_h,  padding_h], dim=0)
        self._counts = torch.cat([self._counts, padding_n], dim=0)
        self.num_classes = new_num_classes

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def local_prototypes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes μc_local = sum_h_c / n_c for seen classes.
        Useful for the local classifier before receiving updates from the server.

        Returns:
            mu_local:  shape (num_seen_classes, embedding_dim)
            class_ids: shape (num_seen_classes,)
        """
        seen_mask  = self._counts > 0
        class_ids  = seen_mask.nonzero(as_tuple=False).squeeze(1)
        counts     = self._counts[class_ids].float().unsqueeze(1)
        mu_local   = self._sum_h[class_ids] / counts

        return mu_local, class_ids

    def __repr__(self) -> str:
        seen = int((self._counts > 0).sum().item())
        return (
            f"PrototypeMemory("
            f"num_classes={self.num_classes}, "
            f"seen_this_round={seen}, "
            f"embedding_dim={self.embedding_dim})"
        )