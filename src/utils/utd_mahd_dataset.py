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

SCENARIO_FEDERATED = "federated"
SCENARIO_CLASS_INCREMENTAL = "class-incremental"
VALID_TRAINING_SCENARIOS = (SCENARIO_FEDERATED, SCENARIO_CLASS_INCREMENTAL)

DIRICHLET_STATIC = "static"
DIRICHLET_DYNAMIC = "dynamic"
VALID_DIRICHLET_MODES = (DIRICHLET_STATIC, DIRICHLET_DYNAMIC)

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


def resolve_classes_per_step(
    scenario: str, classes_per_step: Optional[int]
) -> Optional[int]:
    if scenario not in VALID_TRAINING_SCENARIOS:
        raise ValueError(
            f"Unknown training scenario '{scenario}'."
            f"Expected one of: {VALID_TRAINING_SCENARIOS}"
        )

    if scenario == SCENARIO_FEDERATED:
        return None

    if not classes_per_step:
        raise ValueError(
            "training-scenario is 'class-incremental', but classes-per-step "
            "was not set (or is 0) in pyproject.toml."
        )

    return int(classes_per_step)


def resolve_dirichlet_mode(scenario: str, dirichlet_mode: str) -> str:
    if dirichlet_mode not in VALID_DIRICHLET_MODES:
        raise ValueError(
            f"Unknown dirichlet-mode '{dirichlet_mode}'. "
            f"Expected one of {VALID_DIRICHLET_MODES}"
        )

    if scenario == SCENARIO_CLASS_INCREMENTAL:
        return DIRICHLET_STATIC

    return dirichlet_mode


def parse_int_list_config(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def build_class_schedule(
    num_classes_total: int, classes_per_step: int
) -> list[list[int]]:
    if classes_per_step <= 0:
        raise ValueError("classes_per_step must be a positive integer.")
    return [
        list(range(start, min(start + classes_per_step, num_classes_total)))
        for start in range(0, num_classes_total, classes_per_step)
    ]


def classes_seen_until_round(
    current_round: int, rounds_per_step: int, schedule: list[list[int]]
) -> set[int]:
    if rounds_per_step <= 0:
        raise ValueError("rounds_per_step must be a positive integer.")
    step_idx = (max(current_round, 1) - 1) // rounds_per_step
    step_idx = min(step_idx, len(schedule) - 1)

    seen: set[int] = set()
    for step_classes in schedule[: step_idx + 1]:
        seen.update(step_classes)
    return seen


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
    current_round: int = 1,
    classes_per_step: Optional[int] = None,
    rounds_per_step: int = 1,
    num_classes_total: Optional[int] = None,
    dirichlet_mode: str = DIRICHLET_STATIC,
    held_out_subjects: Optional[list[int]] = None,
):

    if dirichlet_mode not in VALID_DIRICHLET_MODES:
        raise ValueError(
            f"Unknown dirichlet-mode '{dirichlet_mode}'. "
            f"Expected one of {VALID_DIRICHLET_MODES}."
        )

    dataset = _get_cached_dataset(root, window_size, stride)

    if held_out_subjects:
        pool_indices = np.where(~np.isin(dataset.subjects(), held_out_subjects))[0]
    else:
        pool_indices = np.arange(len(dataset))
    pool_labels = dataset.labels[pool_indices]

    partition_seed = (
        seed + current_round if dirichlet_mode == DIRICHLET_DYNAMIC else seed
    )

    partitions = dirichlet_partition(
        pool_labels, num_partitions, dirichlet_alpha, partition_seed
    )
    client_indices = pool_indices[partitions[partition_id]]

    if classes_per_step is not None:
        total_classes = (
            num_classes_total
            if num_classes_total is not None
            else int(dataset.labels.max()) + 1
        )
        schedule = build_class_schedule(total_classes, classes_per_step)
        allowed_classes = classes_seen_until_round(
            current_round, rounds_per_step, schedule
        )
        labels_for_client = dataset.labels[client_indices]
        mask = np.isin(labels_for_client, list(allowed_classes))
        client_indices = client_indices[mask]

        if len(client_indices) == 0:
            empty_loader = DataLoader(Subset(dataset, []), batch_size=batch_size)
            return empty_loader, empty_loader, []

    rng = np.random.RandomState(partition_seed)
    rng.shuffle(client_indices)
    n_val = int(val_fraction * len(client_indices))
    val_idx, train_idx = client_indices[:n_val], client_indices[n_val:]

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

    client_classes = sorted(set(dataset.labels[client_indices].tolist()))
    return train_loader, val_loader, client_classes


def load_server_test_set(
    root: str,
    held_out_subjects: list[int],
    window_size: int = 60,
    stride: int = 30,
    batch_size: int = 32,
) -> DataLoader:
    if not held_out_subjects:
        raise ValueError(
            "held_out_subjects must be a non-empty list of subjects ids "
            "(see the 'server-eval-subjects' key in pyproject.toml)."
        )
    dataset = _get_cached_dataset(root, window_size, stride)
    test_indices = np.where(np.isin(dataset.subjects(), held_out_subjects))[0]
    if len(test_indices) == 0:
        raise ValueError(
            f"No samples found for held_out_subjects={held_out_subjects}. "
            "Check the 'server-eval-subjects' key against the dataset's "
            "actual subjects ids."
        )
    return DataLoader(Subset(dataset, test_indices), batch_size=batch_size)
