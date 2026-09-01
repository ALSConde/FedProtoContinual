from typing import Optional
import torch
import torch.nn.functional as F
from src.model.Models import FCLModel


class ContinualMetricsTracker:
    def __init__(self) -> None:
        self._history: dict[int, dict[int, float]] = {}
        self.first_round: dict[int, int] = {}

    def update(self, current_round: int, per_class_acc: dict[int, float]) -> None:
        for class_id, acc in per_class_acc.items():
            self._history.setdefault(class_id, {})[current_round] = acc
            self.first_round.setdefault(class_id, current_round)

    def backward_transfer(self, current_round: int) -> Optional[float]:
        diffs = []
        for class_id, history in self._history.items():
            first_round = self.first_round[class_id]
            if first_round >= current_round or current_round not in history:
                continue
            diffs.append(history[current_round] - history[first_round])
        return sum(diffs) / len(diffs) if diffs else None

    def average_forgetting(self, current_round: int) -> Optional[float]:
        drops = []
        for class_id, history in self._history.items():
            first_round = self.first_round[class_id]
            if first_round >= current_round or current_round not in history:
                continue
            past = [history[r] for r in history if first_round <= r < current_round]
            if not past:
                continue
            drops.append(max(0.0, max(past) - history[current_round]))
        return sum(drops) / len(drops) if drops else None


@torch.no_grad()
def _global_embed(model: FCLModel, x: torch.Tensor) -> torch.Tensor:
    feats = model.feature_extractor(x)
    h = model.adapter_global(feats)
    return model.prototype_projection(h)


@torch.no_grad()
def evaluate_global_model(
    model: FCLModel,
    test_loader: DataLoader,
    device: torch.device,
    allowed_classes: Optional[set] = None,
) -> tuple[float, float, dict[int, float]]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}

    for x, y in test_loader:
        if allowed_classes is not None:
            mask = torch.tensor(
                [int(label) in allowed_classes for label in y], dtype=torch.bool
            )
            if not mask.any():
                continue
            x, y = x[mask], y[mask]

        x, y = x.to(device), y.to(device)
        h = _global_embed(model, x)

        valid_mask = y < model.classifier.num_classes
        if not valid_mask.any():
            continue
        y_valid, h_valid = y[valid_mask], h[valid_mask]

        logits = model.classifier(h_valid)
        loss = F.cross_entropy(logits, y_valid, reduction="sum")
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        total_n += len(y_valid)
        total_correct += (preds == y_valid).sum().item()

        for c in y_valid.unique():
            c_int = int(c.item())
            c_mask = y_valid == c
            class_total[c_int] = class_total.get(c_int, 0) + int(c_mask.sum().item())
            class_correct[c_int] = class_correct.get(c_int, 0) + int(
                (preds[c_mask] == c).sum().item()
            )

    if total_n == 0:
        return 0.0, 0.0, {}

    avg_loss = total_loss / total_n
    overall_acc = total_correct / total_n
    per_class_acc = {c: class_correct.get(c, 0) / class_total[c] for c in class_total}
    return avg_loss, overall_acc, per_class_acc
