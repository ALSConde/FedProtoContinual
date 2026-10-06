import torch


class WDStats:
    """Per-layer activation statistics for the width-expansion criterion.

    Activations are accumulated as a running sum of per-batch means that stays on the
    layer's device, so recording them never forces a GPU->CPU synchronisation. The
    mean over records is identical to stacking all per-batch means and averaging them.
    """

    def __init__(self):
        self.history = {"activations_mean": 0.0}
        self._sum = None  
        self._n = 0 

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sum"] = None
        state["_n"] = 0
        return state

    def record_activations(self, activations: torch.Tensor):
        batch_mean = activations.detach().mean(dim=0)
        if (
            self._sum is None
            or self._sum.shape != batch_mean.shape
            or self._sum.device != batch_mean.device
        ):
            self._sum = torch.zeros_like(batch_mean)
            self._n = 0
        self._sum = self._sum + batch_mean
        self._n += 1

    def _mean_vector(self) -> torch.Tensor:
        return self._sum / self._n

    def get_activation_mean(self) -> float:
        if self._n == 0:
            return 0.0
        return self._mean_vector().mean().item()

    def get_indexes_of_inactive_neurons(self, threshold: float):
        if self._n == 0:
            return 0.0
        inactive = (self._mean_vector() <= threshold).nonzero(as_tuple=True)[0]
        return inactive.tolist()

    def reset(self):
        self.update_history()
        self._sum = None
        self._n = 0

    def update_history(self):
        previous = self.history.get("activations_mean", 0.0)
        self.history = {
            "activations_mean": (previous + self.get_activation_mean()) / 2
        }

    def usage_ratio(self, threshold: float = 0.05) -> float:
        if self._n == 0:
            return 0.0
        mean_activations = self._mean_vector()
        active = int((mean_activations > threshold).sum().item())
        return active / mean_activations.size(0)