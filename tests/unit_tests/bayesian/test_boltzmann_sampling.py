"""Tests for GP proposal Boltzmann candidate sampling."""

import torch

from lordcapulet.functions.proposal_modes.gaussian_process import _stable_boltzmann_sample


def test_stable_boltzmann_sampling_handles_large_eta_float32():
    """Large eta should not overflow candidate weights before sampling."""

    def acq_func(X):
        return torch.linspace(
            -10.0,
            10.0,
            X.shape[0],
            dtype=torch.float32,
            device=X.device,
        )

    candidates = torch.arange(10, dtype=torch.float32).unsqueeze(-1)

    sampled = _stable_boltzmann_sample(
        acq_func=acq_func,
        X=candidates,
        num_samples=3,
        eta=1000.0,
        replacement=False,
    )

    assert sampled.shape == (3, 1)
    assert torch.isfinite(sampled).all()
    assert set(sampled.squeeze(-1).tolist()).issubset(set(candidates.squeeze(-1).tolist()))


def test_stable_boltzmann_score_floor_prevents_starvation():
    """Score floor keeps tail candidates sample-able so replacement=False
    can draw num_samples distinct candidates even at collapsing eta.

    Without the exp(-15) floor a large eta underflows all-but-the-top weights
    to 0, leaving fewer nonzero-weight candidates than requested and forcing
    multinomial(..., replacement=False) to fail or repeat.
    """

    n_candidates = 100
    n_samples = 50

    def acq_func(X):
        # Wide, monotone spread -> after standardize*eta the bottom candidates
        # would underflow to weight 0 without the floor.
        return torch.linspace(
            0.0,
            100.0,
            X.shape[0],
            dtype=torch.float32,
            device=X.device,
        )

    candidates = torch.arange(n_candidates, dtype=torch.float32).unsqueeze(-1)

    sampled = _stable_boltzmann_sample(
        acq_func=acq_func,
        X=candidates,
        num_samples=n_samples,
        eta=1000.0,
        replacement=False,
    )

    values = sampled.squeeze(-1).tolist()
    assert sampled.shape == (n_samples, 1)
    assert len(set(values)) == n_samples  # all distinct -> no starvation
    assert set(values).issubset(set(candidates.squeeze(-1).tolist()))
