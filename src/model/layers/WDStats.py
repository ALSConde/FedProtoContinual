import torch


class WDStats:
    def __init__(self):
        self.history = {"activations_mean": 0.0}
        self.activations = []

    def __getstate__(self):
        state = self.__dict__.copy()
        state["activations"] = []
        return state

    def _stacked_mean(self) -> torch.Tensor:
        return torch.stack([a.detach().cpu() for a in self.activations]).mean(dim=0)

    def get_activation_mean(self) -> float:
        if len(self.activations) == 0:
            return 0.0
        mean_activations = self._stacked_mean()

        return mean_activations.mean().item()

    def get_indexes_of_inactive_neurons(self, threshold: float):
        if len(self.activations) == 0:
            return 0.0
        mean_activations = self._stacked_mean()
        inactive_neurons = (mean_activations <= threshold).nonzero(as_tuple=True)[0]
        return inactive_neurons.tolist()

    def record_activations(self, activations: torch.Tensor):
        self.activations.append(activations.detach().cpu().mean(dim=0))

    def reset(self):
        self.update_history()
        self.activations = []

    def update_history(self):
        previous = self.history.get("activations_mean", 0.0)
        self.history = {"activations_mean": (previous + self.get_activation_mean()) / 2}

    def usage_ratio(self, threshold: float = 0.05) -> float:
        if len(self.activations) == 0:
            return 0.0
        mean_activations = self._stacked_mean()
        active_neurons = (mean_activations > threshold).nonzero(as_tuple=True)[0]
        return len(active_neurons) / mean_activations.size(0)
