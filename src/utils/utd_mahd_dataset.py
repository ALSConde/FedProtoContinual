from functools import lru_cache
import re
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, Subset
import torch

NUM_ACTIONS = 27
NUM_SUBJECTS = 8
NUM_TRAILS = 4
INPUT_CHANNELS = 6
SAMPLING_RATE_HZ = 50

_INERTIAL_FILENAME_RE = re.compile(
    r"a(?P<action>\d{1,2})_s(?P<subject>\d{1,2})_t(?P<trial>\d{1,2})_inertial",
    re.IGNORECASE,
)


def _parse_meta_from_path(path: Path) -> Optional[dict]:
    m = _INERTIAL_FILENAME_RE.search(path.stem)
    if not m:
        return None
    return {
        "action": int(m.group("action")),
        "subject": int(m.group("subject")),
        "trial": int(m.group("trial")),
    }


def _load_inertial_sequence(path: Path) -> np.ndarray:
    mat = loadmat(path)
    key = next(k for k in mat.keys() if not k.startswith("__"))
    arr = np.asarray(mat[key], dtype=np.float32)

    if (
        arr.ndim == 2
        and arr.shape[0] == INPUT_CHANNELS
        and arr.shape[1] != INPUT_CHANNELS
    ):
        arr = arr.T
    return arr


class UTDMAHDInertial(Dataset):
    def __init__(self, root: str, window_size: int = 60, stride: int = 30):
        self.root = Path(root)
        self.window_size = window_size
        self.stride = stride

        files = [
            f
            for f in self.root.rglob("*_inertial.mat")
            if _parse_meta_from_path(f) is not None
        ]

        if not files:
            raise ValueError(
                f"No valid inertial files found in {root}. "
                "Make sure you have downloaded and extracted the UTD-MHAD dataset in the specified directory."
                "And the file names follow the expected pattern: a<action>_s<subject>_t<trial>_inertial.mat"
            )

        failed = 0
        self.windows: list[tuple[np.ndarray, int, int]] = []
        for f in files:
            meta = _parse_meta_from_path(f)
            try:
                seq = _load_inertial_sequence(f)
            except Exception as e:
                print(f"Failed to load {f}: {e}")
                failed += 1
                continue

            if (
                seq.ndim != 2
                or seq.shape[1] != INPUT_CHANNELS
                or seq.shape[0] < window_size
            ):
                print(f"Invalid sequence shape for {f}: {seq.shape}")
                failed += 1
                continue

            if meta is not None:
                action, subject = meta["action"] - 1, meta["subject"]
                for start in range(0, seq.shape[0] - window_size + 1, stride):
                    window = seq[start : start + window_size]
                    self.windows.append((window, action, subject))

        if failed > 0:
            print(
                f"Warning: {failed} files failed to load or were invalid and were skipped."
            )

        if len(self.windows) == 0:
            raise RuntimeError(
                "No valid windows were extracted from the dataset. Please check the dataset files and ensure they are in the correct format."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window, label, _ = self.windows[idx]
        x = torch.from_numpy(window.T).contiguous()  # Shape: (C, T)
        return x, label

    @property
    def labels(self) -> np.ndarray:
        return np.array([l for _, l, _ in self.windows])

    def subjects(self) -> np.ndarray:
        return np.array([s for _, _, s in self.windows])


def dirichlet_partition(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int = 0
) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    num_classes = int(labels.max()) + 1
    client_indices: list[list[int]] = [[] for _ in range(num_classes)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cut_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for client_id, split in enumerate(np.split(idx_c, cut_points)):
            client_indices[client_id].extend(split.tolist())

    return [np.array(indices) for indices in client_indices]


@lru_cache(maxsize=4)
def _get_cached_dataset(root: str, window_size: int, stride: int) -> UTDMAHDInertial:
    return UTDMAHDInertial(root, window_size, stride)


def load_data(
    partition_id: int,
    num_partitions: int,
    root: str,
    window_size: int = 60,
    stride: int = 30,
    dirichlet_alpha: float = 0.5,
    val_fraction: float = 0.2,
    batch_size: int = 16,
    seed: int = 0,
):
    dataset = _get_cached_dataset(root, window_size, stride)
    partitions = dirichlet_partition(
        dataset.labels, num_partitions, dirichlet_alpha, seed
    )
    client_indices = partitions[partition_id]
    rng = np.random.RandomState(seed)
    rng.shuffle(client_indices)
    n_val = int(val_fraction * len(client_indices))
    val_idx, train_idx = client_indices[:n_val], client_indices[n_val:]

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

    client_classes = sorted(set(dataset.labels[client_indices].tolist()))
    return train_loader, val_loader, client_classes
