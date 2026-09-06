from typing import Optional


class CandidacyCriterion:
    def __init__(
        self,
        theta_alpha: float = 0.32,
        patience: int = 2,
        min_local_acc: Optional[float] = None,
        cooldown_rounds: int = 5,
    ) -> None:
        self.theta_alpha = theta_alpha
        self.patience = patience
        self.min_local_acc = min_local_acc
        self.cooldown_rounds = cooldown_rounds
        self._rounds_above = 0
        self._cooldown = 0

    def step(self, alpha_mean: float, local_acc: Optional[float] = None) -> bool:
        if self._cooldown > 0:
            self._cooldown -= 1
            self._rounds_above = 0
            return False

        meets_alpha = alpha_mean >= self.theta_alpha
        meets_acc = self.min_local_acc is None or (
            local_acc is not None and local_acc >= self.min_local_acc
        )

        if meets_alpha and meets_acc:
            self._rounds_above += 1
        else:
            self._rounds_above = 0

        return self._rounds_above >= self.patience

    def notify_outcome(self, incorporated: bool) -> None:
        self._rounds_above = 0
        if not incorporated:
            self._cooldown = self.cooldown_rounds

    def state(self) -> tuple[int, int]:
        return self._rounds_above, self._cooldown

    def load_state(self, rounds_above: int, cooldown: int) -> None:
        self._rounds_above = rounds_above
        self._cooldown = cooldown
