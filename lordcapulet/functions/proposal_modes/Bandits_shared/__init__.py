"""
Shared utilities for bandit-based proposal modes (no PyTorch dependency).

Provides:
- Boltzmann sampling
- LCB acquisition function
- Feature matrix construction helpers
"""

from .acquisition import lcb, lcb_acquisition, boltzmann_sample

__all__ = [
    'lcb',
    'lcb_acquisition',
    'boltzmann_sample',
]
