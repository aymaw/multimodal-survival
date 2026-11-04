import torch
from torch import nn
from torch.nn import functional as F

from .data import BINS


class FusionSurvival(nn.Module):
    """CNN + GRU + tabular MLP, gated fusion with explicit availability masks."""

    def __init__(self, mode="fusion", width=24):
        super().__init__()
        if mode not in ("fusion", "tabular", "image", "sequence"):
            raise ValueError(mode)
        self.mode = mode
        self.tabular = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, width))
        self.image = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 12 * 12, width),
            nn.ReLU(),
        )
        self.sequence = nn.GRU(4, width, batch_first=True)
        self.gate = nn.Linear(width, 1)
        self.head = nn.Sequential(nn.Linear(width + 3, width), nn.ReLU(), nn.Linear(width, BINS))

    def forward(self, cohort):
        available = cohort.available.clone()
        if self.mode != "fusion":
            keep = ("tabular", "image", "sequence").index(self.mode)
            available[:, [i for i in range(3) if i != keep]] = False
        if self.training and self.mode == "fusion":
            dropped = torch.rand_like(available.float()) > 0.15
            dropped[:, 0] = True
            available &= dropped
        # Mask BEFORE encoding: multiplying a missing NaN embedding by zero is still NaN.
        inputs = [
            torch.where(available[:, i].reshape(-1, *([1] * (x.ndim - 1))), x, 0.0)
            for i, x in enumerate((cohort.tabular, cohort.image, cohort.sequence))
        ]
        if any(not torch.isfinite(x).all() for x in inputs):
            raise ValueError("Observed modality inputs must be finite")
        embeddings = torch.stack(
            (self.tabular(inputs[0]), self.image(inputs[1]), self.sequence(inputs[2])[1][-1]),
            dim=1,
        )
        # All-missing ablation rows receive zero embeddings and explicit absent flags.
        scores = self.gate(embeddings).squeeze(-1).masked_fill(~available, -1e4)
        weights = scores.softmax(1) * available
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        fused = (embeddings * weights[..., None]).sum(1)
        return self.head(torch.cat((fused, available.float()), dim=1))


def survival(logits):
    return torch.exp(F.logsigmoid(-logits).cumsum(1))


def discrete_nll(logits, time, event):
    """Observed event bin: h(t); preceding bins: 1-h. Censored: survive THROUGH t."""
    bins = torch.arange(logits.shape[1], device=logits.device)[None, :]
    survive = (bins < time[:, None]) | ((bins == time[:, None]) & ~event[:, None])
    fail = (bins == time[:, None]) & event[:, None]
    return -(F.logsigmoid(-logits) * survive + F.logsigmoid(logits) * fail).sum(1).mean()
