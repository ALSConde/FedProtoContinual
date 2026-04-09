import copy

import torch
import torch.nn.functional as F

from core.Plugins import Plugins


class LwFPlugin(Plugins):
    def __init__(
        self,
        beta: float = 1.0,
        temperature: float = 2.0,
        distill_on: str = "features",  # "features" or "logits"
    ):
        """
        Args:
            beta:         weight of the distillation loss relative to the task loss.
            temperature:  temperature of the KD (smooths the distributions).
            distill_on:   where to apply the distillation.
                          "features" → KL on the normalized embeddings.
                          "logits"   → KL on the cosine similarities with the prototypes.
        """
        super().__init__()
        self.beta = beta
        self.temperature = temperature
        self.distill_on = distill_on
        self.teacher = None

    def before_training(self, strategy) -> None:
        if strategy.model is None:
            return

        self.teacher = copy.deepcopy(strategy.model)
        self.teacher.eval()

        for param in self.teacher.parameters():
            param.requires_grad_(False)

        device = getattr(strategy, "device", torch.device("cpu"))
        self.teacher.to(device)

    def before_backward(self, strategy) -> None:

        if self.teacher is None:
            return

        x = strategy.mb_x

        with torch.no_grad():
            teacher_out = self._teacher_forward(strategy, x)

        student_out = self._student_forward(strategy, x)

        distill_loss = self._distillation_loss(student_out, teacher_out)
        strategy.loss = strategy.loss + self.beta * distill_loss

    def after_training(self, strategy) -> None:
        self.teacher = None

    def _teacher_forward(self, strategy, x: torch.Tensor) -> torch.Tensor:
        if hasattr(strategy.model, "global_forward"):
            return strategy.model.global_forward(x)
        return self.teacher(x) # type: ignore

    def _student_forward(self, strategy, x: torch.Tensor) -> torch.Tensor:
        if self.distill_on == "features" and hasattr(
            strategy.model, "extract_features"
        ):
            return strategy.model.extract_features(x)
        return strategy.mb_output

    # ------------------------------------------------------------------
    # Distillation loss
    # ------------------------------------------------------------------

    def _distillation_loss(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
    ) -> torch.Tensor:
        T = self.temperature

        if self.distill_on == "features":
            # Normalize the features to get cosine similarities
            s = F.normalize(student_out, dim=1)
            t = F.normalize(teacher_out, dim=1)
        else:
            s = student_out
            t = teacher_out

        q = F.softmax(t / T, dim=1)  # target distribution (teacher)
        log_p = F.log_softmax(s / T, dim=1)  # log-distribution of the student

        return F.kl_div(log_p, q, reduction="batchmean") * (T**2)
