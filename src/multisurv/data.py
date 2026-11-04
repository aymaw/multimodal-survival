"""All features precede time zero. No patient or externally sourced data."""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

BINS = 12


@dataclass
class Cohort:
    tabular: torch.Tensor
    image: torch.Tensor
    sequence: torch.Tensor
    available: torch.Tensor
    time: torch.Tensor
    event: torch.Tensor

    def subset(self, indices):
        return Cohort(**{k: v[indices] for k, v in vars(self).items()})

    def fingerprint(self):
        h = hashlib.sha256()
        for value in vars(self).values():
            h.update(value.numpy().tobytes())
        return h.hexdigest()


def generate(n=1200, seed=41):
    if n < 30:
        raise ValueError("Use at least 30 subjects")
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    tab = rng.normal(scale=0.35, size=(n, 8))
    tab[:, :2] += z[:, :2]
    grid = np.linspace(-1, 1, 12)
    x, y = np.meshgrid(grid, grid)
    pattern = np.exp(-4 * (x * x + y * y))
    image = z[:, 2, None, None] * pattern + rng.normal(scale=0.15, size=(n, 12, 12))
    seq = rng.normal(scale=0.25, size=(n, 6, 4))
    seq[:, :, 0] += z[:, 3, None] * np.linspace(0.1, 1, 6)
    seq[:, :, 1] += z[:, 1, None]
    risk = 0.7 * z[:, 0] - 0.4 * z[:, 1] + 0.9 * z[:, 2] + 0.8 * z[:, 3] + 0.3 * z[:, 0] * z[:, 2]
    hazards = 1 / (1 + np.exp(-(-3 + 0.16 * np.arange(BINS)[None, :] + risk[:, None])))
    occurred = rng.random((n, BINS)) < hazards
    event_time = np.where(occurred.any(1), occurred.argmax(1), BINS)
    # Censoring is at END of a bin. An event at that bin precedes censoring.
    censor = rng.integers(3, BINS + 4, n).clip(max=BINS - 1)
    event = (event_time <= censor) & (event_time < BINS)
    time = np.minimum(event_time, censor)
    available = rng.random((n, 3)) > 0.12
    available[:, 0] = True  # a stable anchor; image/sequence may be absent
    return Cohort(
        torch.tensor(tab, dtype=torch.float32),
        torch.tensor(image[:, None], dtype=torch.float32),
        torch.tensor(seq, dtype=torch.float32),
        torch.tensor(available),
        torch.tensor(time),
        torch.tensor(event),
    )


def split(cohort, seed=17):
    idx = np.random.default_rng(seed).permutation(len(cohort.time))
    a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
    groups = {"train": idx[:a], "validation": idx[a:b], "test": idx[b:]}
    # Fit continuous feature normalization on TRAIN ONLY; image has known fixed scale.
    mean = cohort.tabular[groups["train"]].mean(0)
    std = cohort.tabular[groups["train"]].std(0).clamp_min(1e-6)
    normalized = Cohort(**vars(cohort))
    normalized.tabular = (cohort.tabular - mean) / std
    return {k: normalized.subset(v) for k, v in groups.items()}, {
        "indices": {k: v.tolist() for k, v in groups.items()},
        "tabular_mean": mean.tolist(),
        "tabular_std": std.tolist(),
    }
