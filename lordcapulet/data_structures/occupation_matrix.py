#!/usr/bin/env python3
"""
Unified occupation matrix data structure for LordCapulet.

This module provides a unified way to handle occupation matrices throughout LordCapulet,
abstracting away the differences between various AiiDA-QE API versions and internal formats.
"""

import json
import numpy as np
from typing import Dict, List, Any, Union, Optional
from aiida.orm import JsonableData, load_node





class OccupationMatrixData:
    """
    Unified occupation matrix data structure.
    
    This class provides a consistent interface for occupation matrices across LordCapulet,
    supporting conversion from various formats and ensuring JSON serializability.
    
    Structure:
    {
        'Atom_1': {
            'specie': 'Fe1',
            'shell': '3d',
            'occupation_matrix': {
                'up': [[...], [...], ...],    # 2D list (JSON serializable)
                'down': [[...], [...], ...]
            }
        },
        'Atom_2': {
            ...
        }
    }
    """
    
    def __init__(self, data: Dict[str, Any] = None):
        """Initialize OccupationMatrixData with optional initial data."""
        self._data = data if data is not None else {}
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get the internal data dictionary."""
        return self._data
    
    def as_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary representation."""
        return self._data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OccupationMatrixData':
        """Create instance from dictionary."""
        return cls(data)
    
    @classmethod
    def from_aiida_qe_occupations(cls, occupations_list: List[Dict[str, Any]]) -> 'OccupationMatrixData':
        """
        Create instance from AiiDA-QE get_occupations() output.
        
        Args:
            occupations_list: Output from calc.tools.get_occupations()
                Format: [{'atom_label': 1, 'atom_specie': 'Fe1', 'shell': '3d',
                         'occupation_matrix': {'up': array(...), 'down': array(...)}}]
        
        Returns:
            OccupationMatrixData instance
        """
        data = {}

        # conventions in AiiDA-QE keep changing
        # so here we keep an internal structure
        
        for atom_data in occupations_list:
            atom_label = atom_data['atom_index']
            atom_specie = atom_data['kind_name']
            shell = atom_data['manifold']
            occ_matrix = atom_data['occupations']


            # TODO: Huge red flag that comes from quantum espresso logic
            # the occupation matrix is always as big as the biggest shell
            # among the inputs, this means that if i have a d and a p shell
            # then the p-atom will have also a 5x5 matrix for its p shell
            # if I have a d and an f shell, the d-atom will have also a 7x7 matrix for its d shell
            # this needs to be handled explicitly in the future
            
            # Convert numpy arrays to lists for JSON serialization
            up_matrix = occ_matrix['up'].tolist() if hasattr(occ_matrix['up'], 'tolist') else occ_matrix['up']
            down_matrix = occ_matrix['down'].tolist() if hasattr(occ_matrix['down'], 'tolist') else occ_matrix['down']
            
            # Reshape 1D arrays to 2D matrices if needed
            # Assuming square matrices, so if we have N^2 elements, reshape to (N, N)
            if isinstance(up_matrix, list) and len(up_matrix) > 0:
                if not isinstance(up_matrix[0], list):
                    # It's a 1D list, need to reshape
                    n = int(len(up_matrix) ** 0.5)
                    if n * n == len(up_matrix):
                        up_matrix = [up_matrix[i*n:(i+1)*n] for i in range(n)]
                        down_matrix = [down_matrix[i*n:(i+1)*n] for i in range(n)]
            
            data[atom_label] = {
                'specie': atom_specie,
                'shell': shell,
                'occupation_matrix': {
                    'up': up_matrix,
                    'down': down_matrix
                }
            }
        
        return cls(data)


    # the legacy stuff must be eventually removed 
    @classmethod
    def from_legacy_dict(cls, legacy_data: Dict[str, Any]) -> 'OccupationMatrixData':
        """
        Create instance from legacy internal format.
        
        Args:
            legacy_data: Legacy format like {'1': {'occupation_matrix': {'up': ..., 'down': ...}}}
                         or {'1': {'specie': 'Fe', 'spin_data': {'up': {'occupation_matrix': ...}}}}
        
        Returns:
            OccupationMatrixData instance
        """
        data = {}
        
        for atom_key, atom_info in legacy_data.items():
            # Handle different legacy formats
            if 'spin_data' in atom_info:
                # Format: {'1': {'specie': 'Fe', 'spin_data': {'up': {'occupation_matrix': ...}}}}
                atom_label = atom_key
                specie = atom_info.get('specie', 'Unknown')
                up_matrix = atom_info['spin_data']['up']['occupation_matrix']
                down_matrix = atom_info['spin_data']['down']['occupation_matrix']
            elif 'occupation_matrix' in atom_info:
                # Format: {'1': {'occupation_matrix': {'up': ..., 'down': ...}}}
                atom_label = atom_key
                specie = atom_info.get('specie', 'Unknown')
                up_matrix = atom_info['occupation_matrix']['up']
                down_matrix = atom_info['occupation_matrix']['down']
            else:
                raise ValueError(f"Unrecognized legacy format for atom {atom_key}")
            
            # Ensure matrices are lists
            if hasattr(up_matrix, 'tolist'):
                up_matrix = up_matrix.tolist()
            if hasattr(down_matrix, 'tolist'):
                down_matrix = down_matrix.tolist()
            
            data[atom_label] = {
                'specie': specie,
                'shell': 'UNKNOWN',  # Default assumption, can be updated if available
                'occupation_matrix': {
                    'up': up_matrix,
                    'down': down_matrix
                }
            }
        
        return cls(data)
    
    @classmethod
    def from_constrained_matrix_format(cls, matrix_data: Dict[str, Any], atom_species: List[str] = None) -> 'OccupationMatrixData':
        """
        Create instance from ConstrainedPW input matrix format.
        
        Args:
            matrix_data: Format like {"matrix": [atom][spin][orbital][orbital]}
            atom_species: List of atom species for labeling (optional)
        
        Returns:
            OccupationMatrixData instance
        """
        data = {}
        matrix = matrix_data['matrix']
        
        for atom_idx, atom_matrix in enumerate(matrix):
            atom_label = atom_idx + 1
            specie = atom_species[atom_idx] if atom_species and atom_idx < len(atom_species) else 'Unknown'
            
            up_matrix = atom_matrix[0]  # First spin channel
            down_matrix = atom_matrix[1] if len(atom_matrix) > 1 else atom_matrix[0]  # Second spin channel or copy
            
            data[atom_label] = {
                'specie': specie,
                'shell': 'UNKNOWN',  # Default assumption
                'occupation_matrix': {
                    'up': up_matrix,
                    'down': down_matrix
                }
            }
        
        return cls(data)
    
    def to_constrained_matrix_format(self) -> Dict[str, Any]:
        """
        Convert to ConstrainedPW input matrix format.
        
        Returns:
            Dictionary with format {"matrix": [atom][spin][orbital][orbital]}
        """
        matrix = []
        
        # Sort atoms by their numeric part for consistent ordering
        sorted_atoms = sorted(self._data.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        
        for atom_label in sorted_atoms:
            atom_data = self._data[atom_label]
            up_matrix = atom_data['occupation_matrix']['up']
            down_matrix = atom_data['occupation_matrix']['down']
            
            matrix.append([up_matrix, down_matrix])
        
        return {'matrix': matrix}
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy internal format.
        
        Returns:
            Dictionary in legacy format
        """
        legacy_data = {}
        
        for atom_label, atom_data in self._data.items():
            # Extract numeric part from atom_label (e.g., "Atom_1" -> "1")
            atom_key = atom_label.split('_')[1] if '_' in atom_label else atom_label
            
            legacy_data[atom_key] = {
                'specie': atom_data['specie'],
                'spin_data': {
                    'up': {
                        'occupation_matrix': atom_data['occupation_matrix']['up']
                    },
                    'down': {
                        'occupation_matrix': atom_data['occupation_matrix']['down']
                    }
                }
            }
        
        return legacy_data
    
    def get_atom_labels(self) -> List[str]:
        """Get list of atom labels."""
        return list(self._data.keys())
    
    def get_atom_species(self) -> List[str]:
        """Get list of atom species."""
        return [self._data[atom]['specie'] for atom in self._data.keys()]
    
    def get_occupation_matrix(self, atom_label: str, spin: str) -> List[List[float]]:
        """
        Get occupation matrix for specific atom and spin.
        
        Args:
            atom_label: Atom label (e.g., 'Atom_1')
            spin: Spin channel ('up' or 'down')
        
        Returns:
            2D list representing the occupation matrix
        """
        return self._data[atom_label]['occupation_matrix'][spin]
    
    def get_occupation_matrix_as_numpy(self, atom_label: str, spin: str) -> np.ndarray:
        """
        Get occupation matrix as numpy array.
        
        Args:
            atom_label: Atom label (e.g., 'Atom_1')
            spin: Spin channel ('up' or 'down')
        
        Returns:
            numpy array representing the occupation matrix
        """
        return np.array(self._data[atom_label]['occupation_matrix'][spin])
    
    def set_occupation_matrix(self, atom_label: str, spin: str, matrix: Union[List[List[float]], np.ndarray]):
        """
        Set occupation matrix for specific atom and spin.
        
        Args:
            atom_label: Atom label (e.g., 'Atom_1')
            spin: Spin channel ('up' or 'down')
            matrix: 2D array or list representing the occupation matrix
        """
        if hasattr(matrix, 'tolist'):
            matrix = matrix.tolist()
        
        if atom_label not in self._data:
            self._data[atom_label] = {
                'specie': 'Unknown',
                'shell': '3d',
                'occupation_matrix': {'up': None, 'down': None}
            }
        
        self._data[atom_label]['occupation_matrix'][spin] = matrix
    
    def get_trace(self, atom_label: str, spin: str) -> float:
        """
        Get trace of occupation matrix (sum of diagonal elements).
        
        Args:
            atom_label: Atom label (e.g., 'Atom_1')
            spin: Spin channel ('up' or 'down')
        
        Returns:
            Trace of occupation matrix
        """
        matrix = self.get_occupation_matrix_as_numpy(atom_label, spin)
        return float(np.trace(matrix))
    
    def get_electron_number(self, atom_label: str) -> float:
        """
        Get total number of electrons for an atom (trace_up + trace_down).
        
        Args:
            atom_label: Atom label (e.g., 'Atom_1')
        
        Returns:
            Total electron number
        """
        return self.get_trace(atom_label, 'up') + self.get_trace(atom_label, 'down')
    
    def get_magnetic_moment(self, atom_label: str) -> float:
        """
        Get magnetic moment for an atom (trace_up - trace_down).
        
        Args:
            atom_label: Atom label (e.g., 'Atom_1')
        
        Returns:
            Magnetic moment
        """
        return self.get_trace(atom_label, 'up') - self.get_trace(atom_label, 'down')
    
    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self._data)
    
    def __iter__(self):
        """Iterate over atom labels."""
        return iter(self._data)
    
    def __getitem__(self, atom_label: str) -> Dict[str, Any]:
        """Get atom data by label."""
        return self._data[atom_label]
    
    def __str__(self) -> str:
        """String representation."""
        return f"OccupationMatrixData({len(self._data)} atoms: {list(self._data.keys())})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()




# should this be removed? We do not really use it
class OccupationMatrixAiidaData(JsonableData):
    """
    AiiDA JsonableData wrapper for OccupationMatrixData.
    
    This allows storing OccupationMatrixData in the AiiDA database
    """
    
    def __init__(self, obj: OccupationMatrixData = None, **kwargs):
        if obj is None:
            obj = OccupationMatrixData()
        super().__init__(obj, **kwargs)
    
    @property
    def occupation_data(self) -> OccupationMatrixData:
        """Get the wrapped OccupationMatrixData object."""
        return self.obj


# Utility functions for backward compatibility and easy migration

def extract_occupations_from_calc(calc_node) -> OccupationMatrixData:
    """
    Extract occupations from any calculation node, handling API changes automatically.
    
    This function provides a single point where AiiDA-QE API calls are made,
    making it easy to adapt to future API changes.
    
    Args:
        calc_node: AiiDA calculation node (PW or ConstrainedPW)
    
    Returns:
        OccupationMatrixData instance
    """
    try:
        # Try the new API
        occupations_list = calc_node.tools.get_occupations()
        return OccupationMatrixData.from_aiida_qe_occupations(occupations_list)
    
    except AttributeError:
        # Fallback for older API versions
        try:
            occupations_dict = calc_node.tools.get_occupations_dict()
            return OccupationMatrixData.from_legacy_dict(occupations_dict)
        except AttributeError:
            # Last resort: try to get from output_atomic_occupations
            if 'output_atomic_occupations' in calc_node.outputs:
                legacy_dict = calc_node.outputs.output_atomic_occupations.get_dict()
                return OccupationMatrixData.from_legacy_dict(legacy_dict)
            else:
                raise ValueError(f"Could not extract occupations from calculation {calc_node.pk}")
    
    except Exception as e:
        raise ValueError(f"Error extracting occupations from calculation {calc_node.pk}: {e}")


def filter_atoms_by_species(occupation_data: OccupationMatrixData, 
                           target_species: List[str]) -> OccupationMatrixData:
    """
    Filter occupation data to only include atoms of specified species.
    
    Args:
        occupation_data: Input occupation data
        target_species: List of species to keep (e.g., ['Fe', 'Ni'])
    
    Returns:
        Filtered OccupationMatrixData instance
    """
    filtered_data = {}
    
    for atom_label, atom_info in occupation_data.data.items():
        if atom_info['specie'] in target_species:
            filtered_data[atom_label] = atom_info
    
    return OccupationMatrixData(filtered_data)


def compute_occupation_distance(occ_data1: OccupationMatrixData,
                                occ_data2: OccupationMatrixData,
                                atom_label: Optional[str] = None,
                                spins: Optional[List[str]] = None) -> float:
    """
    Compute Euclidean distance between two occupation matrix datasets.
    
    The distance is computed as the Frobenius norm of the difference between
    occupation matrices, optionally for specific atoms and spin channels.
    
    Args:
        occ_data1: First OccupationMatrixData object
        occ_data2: Second OccupationMatrixData object
        atom_label: If specified, compute distance only for this atom.
                   If None, compute total distance across all atoms.
        spins: List of spin channels to include (default: ['up', 'down'])
    
    Returns:
        Euclidean distance (float)
    
    Raises:
        ValueError: If atom sets don't match or matrices have incompatible dimensions
    
    Examples:
        >>> # Distance between all atoms
        >>> dist = compute_occupation_distance(occ1, occ2)
        
        >>> # Distance for specific atom
        >>> dist = compute_occupation_distance(occ1, occ2, atom_label='Atom_1')
        
        >>> # Distance for up-spin only
        >>> dist = compute_occupation_distance(occ1, occ2, spins=['up'])
    """
    if spins is None:
        spins = ['up', 'down']
    
    # Determine which atoms to compare
    if atom_label is not None:
        # Single atom case
        if atom_label not in occ_data1.data:
            raise ValueError(f"Atom {atom_label} not found in first occupation data")
        if atom_label not in occ_data2.data:
            raise ValueError(f"Atom {atom_label} not found in second occupation data")
        atom_labels = [atom_label]
    else:
        # All atoms case - check that atom sets match
        atoms1 = set(occ_data1.get_atom_labels())
        atoms2 = set(occ_data2.get_atom_labels())
        if atoms1 != atoms2:
            raise ValueError(f"Atom labels don't match: {atoms1} vs {atoms2}")
        atom_labels = sorted(atoms1, key=lambda x: (int(x.split('_')[1]) if '_' in x else 0))
    
    # Compute total squared distance
    total_squared_dist = 0.0
    
    for atom in atom_labels:
        for spin in spins:
            # Get matrices as numpy arrays
            matrix1 = occ_data1.get_occupation_matrix_as_numpy(atom, spin)
            matrix2 = occ_data2.get_occupation_matrix_as_numpy(atom, spin)
            
            # Check dimensions match
            if matrix1.shape != matrix2.shape:
                raise ValueError(
                    f"Matrix dimensions don't match for atom {atom}, spin {spin}: "
                    f"{matrix1.shape} vs {matrix2.shape}"
                )
            
            # Compute squared Frobenius norm of difference
            diff = matrix1 - matrix2
            total_squared_dist += np.sum(diff ** 2)
    
    # Return Euclidean distance (square root of sum of squared differences)
    return float(np.sqrt(total_squared_dist))