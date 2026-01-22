"""
Proposal modes for generating occupation matrix constraints.

This module contains different modes for proposing new occupation matrices
for DFT+U calculations.
"""

from .random_mode import propose_random_constraints
from .random_so_n_mode import propose_random_so_n_constraints
from .gaussian_process import propose_gaussian_process_constraints

__all__ = [
    'propose_random_constraints',
    'propose_random_so_n_constraints', 
    'propose_gaussian_process_constraints'
]
