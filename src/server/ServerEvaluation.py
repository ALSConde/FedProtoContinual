import csv
import json
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from src.model.Models import FCLModel
from torch.utils.data import DataLoader
from src.model.blocks.Adapter import Adapter


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

    def per_class_backward_transfer(self, current_round: int) -> dict[int, float]:
        result: dict[int, float] = {}
        for class_id, history in self._history.items():
            first_round = self.first_round[class_id]
            if first_round >= current_round or current_round not in history:
                continue
            result[class_id] = history[current_round] - history[first_round]
        return result

    def per_class_forgetting(self, current_round: int) -> dict[int, float]:
        result: dict[int, float] = {}
        for class_id, history in self._history.items():
            first_round = self.first_round[class_id]
            if first_round >= current_round or current_round not in history:
                continue
            past = [history[r] for r in history if first_round <= r < current_round]
            if not past:
                continue
            result[class_id] = max(0.0, max(past) - history[current_round])
        return result

    def seen_classes(self) -> list[int]:
        return sorted(self._history.keys())


class ForgettingMonitor:
    _FIELDNAMES = ["round", "class_id", "acc", "bwt", "forgetting"]

    def __init__(
        self, output_dir: str, tracker: Optional[ContinualMetricsTracker] = None
    ) -> None:
        self.tracker = tracker if tracker is not None else ContinualMetricsTracker()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "forgetting_per_class.csv"
        self._header_written = (
            self.csv_path.exists() and self.csv_path.stat().st_size > 0
        )
        self.last_summary: dict = {}

    def update(self, current_round: int, per_class_acc: dict[int, float]) -> dict:
        self.tracker.update(current_round, per_class_acc)

        per_class_bwt = self.tracker.per_class_backward_transfer(current_round)
        per_class_forgetting = self.tracker.per_class_forgetting(current_round)

        rows = [
            {
                "round": current_round,
                "class_id": class_id,
                "acc": per_class_acc.get(class_id),
                "bwt": per_class_bwt.get(class_id, ""),
                "forgetting": per_class_forgetting.get(class_id, ""),
            }
            for class_id in sorted(per_class_acc)
        ]
        self._append_csv(rows)

        bwt_values = list(per_class_bwt.values())
        forgetting_values = list(per_class_forgetting.values())
        worst_forgetting = max(
            per_class_forgetting.items(), key=lambda kv: kv[1], default=(None, None)
        )
        worst_bwt = min(
            per_class_bwt.items(), key=lambda kv: kv[1], default=(None, None)
        )

        summary = {
            "round": current_round,
            "mean_bwt": sum(bwt_values) / len(bwt_values) if bwt_values else None,
            "mean_forgetting": (
                sum(forgetting_values) / len(forgetting_values)
                if forgetting_values
                else None
            ),
            "worst_class_forgetting_id": worst_forgetting[0],
            "worst_class_forgetting": worst_forgetting[1],
            "worst_class_bwt_id": worst_bwt[0],
            "worst_class_bwt": worst_bwt[1],
            "per_class_bwt": per_class_bwt,
            "per_class_forgetting": per_class_forgetting,
        }
        self.last_summary = summary
        return summary

    def _append_csv(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(rows)

    def save_run_summary(self, path: Optional[str] = None, **extra) -> dict:
        payload = {"final_report": self.last_summary, **extra}
        out_path = (
            Path(path) if path is not None else (self.output_dir / "summary.json")
        )
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        return payload


@torch.no_grad()
def _global_embed(model: FCLModel, x: torch.Tensor) -> torch.Tensor:
    feats = model.feature_extractor(x)
    h = model.adapter_global(feats)
    if len(model.incorporated_adapters) > 0:
        h += sum(
            a.forward_delta(h)
            for a in model.incorporated_adapters
            if isinstance(a, Adapter)
        )
    return h


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
