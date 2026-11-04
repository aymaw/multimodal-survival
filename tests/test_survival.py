import math

import numpy as np
import pytest
import torch

from multisurv.data import generate, split
from multisurv.metrics import censor_survival, concordance, ipcw_brier
from multisurv.model import FusionSurvival, discrete_nll, survival


def test_censor_event_likelihood_and_gradient():
    x = torch.zeros((2, 3), requires_grad=True)
    loss = discrete_nll(x, torch.tensor([1, 2]), torch.tensor([True, False]))
    assert loss.item() == pytest.approx(2.5 * math.log(2))
    loss.backward()
    assert x.grad[0].tolist() == [0.25, -0.25, 0.0]
    assert x.grad[1].tolist() == [0.25, 0.25, 0.25]


def test_large_logits_and_monotone_curves():
    logits = torch.tensor([[1000.0, -1000.0, 0.0]], requires_grad=True)
    loss = discrete_nll(logits, torch.tensor([2]), torch.tensor([True]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    s = survival(torch.randn(10, 12))
    assert ((s[:, 1:] <= s[:, :-1]) & (s[:, 1:] >= 0)).all()


def test_disjoint_reproducible_split():
    cohort = generate(100)
    assert cohort.fingerprint() == generate(100).fingerprint()
    parts, m = split(cohort)
    a, b, c = map(set, m["indices"].values())
    assert not (a & b or b & c or a & c)
    assert len(a | b | c) == 100
    assert torch.allclose(parts["train"].tabular.mean(0), torch.zeros(8), atol=1e-6)


def test_cindex_orientation_ties_and_empty():
    assert concordance([1, 2, 3], [1, 1, 0], [3, 2, 1]) == 1
    assert concordance([1, 2, 3], [1, 1, 0], [1, 2, 3]) == 0
    assert concordance([1, 2, 3], [1, 1, 0], [1, 1, 1]) == 0.5
    assert concordance([1, 1], [0, 0], [0, 1]) is None


def test_brier_no_censoring_matches_squared_error():
    result = ipcw_brier([1, 2, 3], [1, 1, 1], [1, 2, 3], [1, 1, 1], [0.1, 0.3, 0.8], 1)
    assert result == pytest.approx(np.mean([0.01, 0.49, 0.04]))


def test_censoring_end_of_bin_and_support():
    assert censor_survival([1, 1, 1, 3], [1, 1, 0, 0], 1) == 0.5
    assert censor_survival([1, 1, 1, 3], [1, 1, 0, 0], 1, left=True) == 1
    assert ipcw_brier([1, 3], [0, 0], [1], [0], [0.8], 1) == pytest.approx(0.04)
    assert ipcw_brier([1, 1], [0, 0], [3], [0], [0.8], 2) is None


def test_missing_modality_cannot_change_prediction():
    data = generate(40)
    data.available[:, 1] = False
    model = FusionSurvival().eval()
    before = model(data).detach()
    data.image[:] = 1e4
    assert torch.allclose(before, model(data), atol=1e-6)


def test_missing_nan_modality_is_ignored_before_encoding():
    data = generate(40)
    data.available[:, 1:] = False
    model = FusionSurvival().eval()
    before = model(data).detach()
    data.image[:] = float("nan")
    data.sequence[:] = float("nan")
    assert torch.allclose(before, model(data))
    model.train()
    model(data).sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_observed_nan_is_rejected():
    data = generate(30)
    data.tabular[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        FusionSurvival().eval()(data)


@pytest.mark.parametrize("mode", ["tabular", "image", "sequence", "fusion"])
def test_all_missing_is_finite(mode):
    data = generate(30)
    data.available[:] = False
    assert torch.isfinite(FusionSurvival(mode).eval()(data)).all()
