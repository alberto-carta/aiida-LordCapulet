"""
Proposal modes for generating occupation matrix constraints.

This module contains different modes for proposing new occupation matrices
for DFT+U calculations.
"""

from .random_mode import propose_random_constraints

__all__ = ['propose_random_constraints']
