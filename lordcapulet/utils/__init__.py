"""
Utility modules for LordCapulet.
"""

# Import utility functions when created
# from .structure_utils import tag_and_list_atoms
# from .magnetism_utils import create_magnetic_configs

from lordcapulet.data_structures.occupation_matrix import (
    OccupationMatrixData,
    extract_occupations_from_calc,
    filter_atoms_by_species
)

from lordcapulet.utils.preprocessing.submission import (
    tag_and_list_atoms,
    get_default_manifolds,
    get_dimensions,
    prepare_hubbard_corr_info,
    prepare_hubbard_structure,
)

__all__ = [
    'OccupationMatrixData',
    'extract_occupations_from_calc',
    'filter_atoms_by_species',
    'tag_and_list_atoms',
    'get_default_manifolds',
    'get_dimensions',
    'prepare_hubbard_corr_info',
    'prepare_hubbard_structure',
]
