"""
Bandit acquisition utilities — numpy only, no PyTorch.

Provides LCB acquisition function and Boltzmann (softmax) sampling
for bandit-based proposal modes.
"""

import numpy as np
from typing import Optional


# ==============================================================================
# Lower Confidence Bound (LCB)
# ==============================================================================

def lcb(mu: np.ndarray, sigma: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Lower Confidence Bound for minimization.

    LCB(x) = mu(x) - beta * sigma(x)

    Lower LCB values are better (low predicted energy + high uncertainty
    penalised less aggressively, encouraging exploration).

    Args:
        mu: Predicted mean values, shape (n_samples,).
        sigma: Predicted standard deviation, shape (n_samples,).
        beta: Exploration-exploitation tradeoff.
              beta=0 → pure exploitation (minimise mu only).
              Larger beta → more exploration (favours high sigma).

    Returns:
        LCB values, shape (n_samples,).
    """
    return mu - beta * sigma


def lcb_acquisition(
    mu: np.ndarray,
    sigma: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Acquisition scores for Boltzmann sampling.
    
    Returns negative LCB so that higher scores = better candidates
    (suitable for boltzmann_sample which favours high values).
    
    score = -(mu - beta * sigma) = -mu + beta * sigma
    """
    return -lcb(mu, sigma, beta)


# ==============================================================================
# Boltzmann (softmax) sampling
# ==============================================================================

def boltzmann_sample(
    acq_values: np.ndarray,
    eta: float = 30.0,
    num_samples: int = 10,
    replacement: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample indices from a Boltzmann distribution over acquisition values.

    P(i) ∝ exp(eta * acq_values[i])

    Higher eta → lower temperature → more exploitation (favours high acq).
    Lower eta → higher temperature → more exploration (flatter distribution).

    Args:
        acq_values: 1D array of acquisition scores (higher = better).
        eta: Inverse temperature parameter (default 30).
        num_samples: Number of indices to sample.
        replacement: Whether to sample with replacement.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        1D array of sampled indices (integers).

    Example:
        >>> scores = np.array([0.1, 0.5, 0.3, 0.9])
        >>> indices = boltzmann_sample(scores, eta=10, num_samples=2)
        >>> indices  # e.g., array([3, 1]) — favours high scores
    """
    if rng is None:
        rng = np.random.default_rng()

    # Shift for numerical stability: exp(eta * (x - max)) avoids overflow
    shifted = acq_values - np.max(acq_values)
    weights = np.exp(eta * shifted)
    probs = weights / np.sum(weights)

    indices = rng.choice(
        len(acq_values),
        size=num_samples,
        replace=replacement,
        p=probs,
    )
    return indices
