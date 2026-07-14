"""
Data structures for LordCapulet.

This module provides core data structures for handling occupation matrices
and collections of calculation data.
"""

from .occupation_matrix import (
    OccupationMatrixData,
    clip_occupation_numbers,
    extract_occupations_from_calc,
    filter_atoms_by_species,
    compute_occupation_distance,
)

from .databank import DataBank
from .template_library import (
    AtomicTemplateLibrary,
    atomic_template_distance,
    normalize_specie,
    occupation_fingerprint,
)

__all__ = [
    'OccupationMatrixData',
    'clip_occupation_numbers',
    'extract_occupations_from_calc',
    'filter_atoms_by_species',
    'compute_occupation_distance',
    'DataBank',
    'AtomicTemplateLibrary',
    'atomic_template_distance',
    'normalize_specie',
    'occupation_fingerprint',
]
