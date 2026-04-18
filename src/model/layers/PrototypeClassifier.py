import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeClassifier(nn.Module):
    """
    Classifier based on prototypes with cosine similarity.

    Weights Wc are normalized prototypes — not trainable by gradient.
    They are updated analytically from the server's federated aggregation through update_from_global().

    Suports incremental class growth across federated rounds:
    new classes are dynamically added without reinitializing existing ones.

    Forward:
        logits_c = h · Wc  where h and Wc are L2-normalized
        equivalent to: logits = F.linear(F.normalize(h), F.normalize(prototypes))

    Attributes:
        num_classes     : number of knownledged classes
        embedding_dim   : dimension of the input features (h)
        prototypes      : buffer (num_classes, embedding_dim) with μc without normalization
                          for compatibility with EMA updates.
    """

    def __init__(self, embedding_dim: int, scale_init: float = 20.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = 0

        self.scale = nn.Parameter(
            torch.tensor(scale_init)
        )  # Learnable scale for sharper softmax distribution, as in cosine classifiers

        self.register_buffer(
            "prototypes",
            torch.empty(0, embedding_dim),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute logits with cossine similarity between h and the prototypes.

        Args:
            h: final features of the model, shape (batch_size, embedding_dim)

        Returns:
            logits: shape (batch_size, num_classes)
                    values in [-1, 1], scaled by self.scale for sharper softmax distribution
        """
        if self.num_classes == 0:
            raise RuntimeError(
                "PrototypeClassifier does not have initialized classes. "
                "Call update_from_global() before forward."
            )

        h_norm = F.normalize(h, dim=1)  # (B, D)
        w_norm = F.normalize(self.prototypes, dim=1)  # (C, D)
        return F.linear(h_norm, w_norm) * self.scale  # (B, C)

    # ------------------------------------------------------------------
    # Update via federated aggregation
    # ------------------------------------------------------------------

    def update_from_global(
        self,
        mu_global: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> None:
        """
        Updates the prototypes with the values aggregated by the server.

        Supports incremental class growth: if class_ids contains indices beyond
        the current num_classes, the prototypes buffer is expanded.

        Args:
            mu_global:  prototypes aggregated, shape (len(class_ids), embedding_dim)
                        values not normalized — normalization occurs in the forward pass
            class_ids:  1D tensor with the indices of the classes being updated
                        e.g., tensor([0, 3, 7]) to update only these classes
        """
        max_class_id = int(class_ids.max().item()) + 1

        if max_class_id > self.num_classes:
            self._expand(max_class_id)

        self.prototypes[class_ids] = mu_global.to(self.prototypes.device)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _expand(self, new_num_classes: int) -> None:
        """
        Expand the prototypes buffer to accommodate new classes.
        Existing prototypes are preserved; new ones are initialized with zeros
        and will be overwritten by the cold start on the first server update.
        """
        extra = new_num_classes - self.num_classes
        padding = torch.zeros(
            extra,
            self.embedding_dim,
            device=self.prototypes.device,
            dtype=self.prototypes.dtype,
        )
        self.prototypes = torch.cat([self.prototypes, padding], dim=0)
        self.num_classes = new_num_classes

    def get_weights_normalized(self) -> torch.Tensor:
        """
        Returns Wc = μc / ‖μc‖ — the effective weights of the classifier.
        Useful for inspection and sending to the server if necessary.
        """
        return F.normalize(self.prototypes, dim=1)

    def __repr__(self) -> str:
        return (
            f"PrototypeClassifier("
            f"num_classes={self.num_classes}, "
            f"embedding_dim={self.embedding_dim})"
        )
