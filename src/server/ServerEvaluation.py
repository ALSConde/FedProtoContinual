import csv
import json
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
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
    _FIELDNAMES = ["round", "class_id", "n", "acc", "bwt", "forgetting"]

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

    def update(
        self,
        current_round: int,
        per_class_acc: dict[int, float],
        per_class_n: Optional[dict[int, int]] = None,
    ) -> dict:
        self.tracker.update(current_round, per_class_acc)
        per_class_n = per_class_n or {}

        per_class_bwt = self.tracker.per_class_backward_transfer(current_round)
        per_class_forgetting = self.tracker.per_class_forgetting(current_round)

        rows = [
            {
                "round": current_round,
                "class_id": class_id,
                "n": per_class_n.get(class_id, ""),
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


class PredictionFlowMonitor:
    """Where do the test samples of old classes go?

    For every evaluation it splits the global model's predictions into
      old samples (true class from an earlier step) -> correct / predicted as a class of
      the CURRENT step (recency bias) / predicted as another old class (the old classes
      got confused among themselves, i.e. the representation degraded);
      current-step samples -> correct / predicted as an old class / other current class;
    plus the share of all predictions that land on current-step classes versus the share
    of samples that truly belong to them. Written to prediction_flow.csv; the full
    confusion matrices of the first and last round of each step go to
    confusion_matrices.npz (key round_<r>, rows = true, columns = predicted).
    """

    _FIELDNAMES = [
        "round",
        "step",
        "n_old",
        "old_correct",
        "old_to_current",
        "old_to_other_old",
        "n_cur",
        "cur_correct",
        "cur_to_old",
        "cur_to_other_cur",
        "pred_share_current",
        "true_share_current",
    ]

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "prediction_flow.csv"
        self.npz_path = self.output_dir / "confusion_matrices.npz"
        self._header_written = (
            self.csv_path.exists() and self.csv_path.stat().st_size > 0
        )
        self._matrices: dict[str, np.ndarray] = {}

    @staticmethod
    def summarize(
        confusion: torch.Tensor,
        current_classes: Iterable[int],
        seen_classes: Iterable[int],
    ) -> dict:
        size = confusion.shape[0]
        cur = sorted(c for c in set(current_classes) if c < size)
        old = sorted(c for c in set(seen_classes) - set(current_classes) if c < size)
        conf = confusion.long()
        total = int(conf.sum())

        def _block(rows, cols):
            if not rows or not cols:
                return 0
            return int(conf[rows][:, cols].sum())

        n_old = int(conf[old].sum()) if old else 0
        old_correct = int(sum(conf[c, c] for c in old))
        old_to_cur = _block(old, cur)
        n_cur = int(conf[cur].sum()) if cur else 0
        cur_correct = int(sum(conf[c, c] for c in cur))
        cur_to_old = _block(cur, old)
        return {
            "n_old": n_old,
            "old_correct": old_correct,
            "old_to_current": old_to_cur,
            "old_to_other_old": n_old - old_correct - old_to_cur,
            "n_cur": n_cur,
            "cur_correct": cur_correct,
            "cur_to_old": cur_to_old,
            "cur_to_other_cur": n_cur - cur_correct - cur_to_old,
            "pred_share_current": (
                (_block(list(range(size)), cur) / total) if total else None
            ),
            "true_share_current": (n_cur / total) if total else None,
        }

    def update(
        self,
        current_round: int,
        confusion: torch.Tensor,
        current_classes: Iterable[int],
        seen_classes: Iterable[int],
        step: int,
        save_matrix: bool = False,
    ) -> dict:
        row = self.summarize(confusion, current_classes, seen_classes)
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow({"round": current_round, "step": step, **row})
        if save_matrix:
            self._matrices[f"round_{current_round}"] = confusion.numpy()
            np.savez_compressed(self.npz_path, **self._matrices)

        def _frac(num, den):
            return (row[num] / row[den]) if row[den] else None

        metrics = {
            "flow_old_correct": _frac("old_correct", "n_old"),
            "flow_old_to_current": _frac("old_to_current", "n_old"),
            "flow_old_to_other_old": _frac("old_to_other_old", "n_old"),
            "flow_cur_to_old": _frac("cur_to_old", "n_cur"),
            "flow_pred_share_current": row["pred_share_current"],
        }
        return {k: v for k, v in metrics.items() if v is not None}


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
    chunk_size: int = 512,
    return_confusion: bool = False,
):
    """Same results as the previous per-batch version, with no per-class Python loop and
    a single host sync at the end (it used ~50 syncs per batch).

    Returns (loss, acc, per_class_acc, per_class_n). With return_confusion=True a fifth
    element is appended: a CPU LongTensor [C, C] of counts (row = true class, column =
    predicted class, C = number of classes of the classifier), built from the same
    predictions, so its diagonal reproduces per_class_acc exactly."""
    model.eval()
    num_classes = model.classifier.num_classes
    empty = (0.0, 0.0, {}, {}, None) if return_confusion else (0.0, 0.0, {}, {})
    if num_classes == 0:
        return empty

    xs, ys = [], []
    for x, y in test_loader:  # CPU tensors: filtering here costs no GPU sync
        if allowed_classes is not None:
            keep = torch.isin(y, torch.tensor(sorted(allowed_classes), dtype=y.dtype))
            if not keep.any():
                continue
            x, y = x[keep], y[keep]
        xs.append(x)
        ys.append(y)
    if not xs:
        return empty
    X, Y = torch.cat(xs).to(device), torch.cat(ys).to(device)

    loss_sum = torch.zeros((), device=device)
    n_total = torch.zeros((), device=device)
    n_correct = torch.zeros((), device=device)
    class_total = torch.zeros(num_classes, device=device)
    class_correct = torch.zeros(num_classes, device=device)
    confusion = torch.zeros(num_classes * num_classes, device=device)

    for i in range(0, X.shape[0], chunk_size):
        x, y = X[i : i + chunk_size], Y[i : i + chunk_size]
        h = _global_embed(model, x)
        valid = (y < num_classes).to(
            h.dtype
        )  # classes the classifier does not know yet are skipped
        y_safe = y.clamp(max=num_classes - 1)
        logits = model.classifier(h)
        ce = F.cross_entropy(logits, y_safe, reduction="none")
        pred = logits.argmax(dim=1)
        hit = (pred == y_safe).to(h.dtype) * valid
        confusion.index_add_(0, y_safe * num_classes + pred, valid)
        loss_sum = loss_sum + (ce * valid).sum()
        n_total = n_total + valid.sum()
        n_correct = n_correct + hit.sum()
        class_total.index_add_(0, y_safe, valid)
        class_correct.index_add_(0, y_safe, hit)

    total_n = int(n_total.item())
    if total_n == 0:
        return empty
    ct, cc = class_total.cpu(), class_correct.cpu()
    seen = ct.nonzero(as_tuple=True)[0].tolist()
    per_class_acc = {c: float(cc[c]) / float(ct[c]) for c in seen}
    per_class_n = {c: int(ct[c]) for c in seen}
    result = (
        float(loss_sum) / total_n,
        float(n_correct) / total_n,
        per_class_acc,
        per_class_n,
    )
    if return_confusion:
        conf = confusion.view(num_classes, num_classes).round().long().cpu()
        return (*result, conf)
    return result
