import torch


class WDStats:
    def __init__(self):
        self.history = {"activations_mean": 0.0}
        self.activations = []

    def get_activation_mean(self) -> float:
        if len(self.activations) == 0:
            return 0.0
        activations_tensor = torch.stack(self.activations)
        mean_activations = torch.mean(activations_tensor, dim=0)

        return mean_activations.mean().item()

    def get_indexes_of_inactive_neurons(self, threshold: float):
        if len(self.activations) == 0:
            return 0.0
        activations_tensor = torch.stack(self.activations)
        mean_activations = torch.mean(activations_tensor, dim=0)
        inactive_neurons = (mean_activations <= threshold).nonzero(as_tuple=True)[0]
        return inactive_neurons.tolist()

    def record_activations(self, activations: torch.Tensor):
        self.activations.append(activations.detach().cpu().mean(dim=0))

    def reset(self):
        self.update_history()
        self.activations = []

    def update_history(self):
        self.history = {
            "activations_mean": (
                self.history["activations_mean"] if "activations_mean" in self.history else 0.0
                + self.get_activation_mean()
            )
            / 2
        }

    def usage_ratio(self, threshold: float = 0.05) -> float:
        if len(self.activations) == 0:
            return 0.0
        activations_tensor = torch.stack(self.activations)
        mean_activations = torch.mean(activations_tensor, dim=0)
        active_neurons = (mean_activations > threshold).nonzero(as_tuple=True)[0]
        return len(active_neurons) / mean_activations.size(0)
