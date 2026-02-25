"""
Simulator module for DFT+U occupation matrix discovery process.

This module provides tools to simulate the iterative search process
for discovering occupation matrix configurations using a ground truth
DataBank as an oracle.
"""

from .search_simulator import SearchSimulator

__all__ = ['SearchSimulator']
