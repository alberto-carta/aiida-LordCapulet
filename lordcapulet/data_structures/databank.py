#!/usr/bin/env python3
"""
DataBank: Efficient storage and PyTorch conversion for OccupationMatrixData collections.

This module provides a functional, immutable container for multiple calculations,
optimized for machine learning workflows with PyTorch.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from copy import deepcopy

try:
    import torch
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH = False
    torch = None

from .occupation_matrix import OccupationMatrixData, compute_occupation_distance


class DataBank:
    """
    Immutable container for multiple OccupationMatrixData objects with metadata.
    
    Design principles:
    - Functional: operations return new DataBank instances
    - Single source of truth: all data in _records list
    - Lazy computation: flattening cached only when needed
    - PyTorch-ready: efficient tensor conversion
    
    Each record contains:
        - pk: Calculation primary key
        - energy: Total energy (eV)
        - energy_uncertainty: Energy uncertainty (TODO: temporary, currently 0.0)
        - converged: Boolean convergence status
        - occ_data: OccupationMatrixData object
        - metadata: Optional additional fields (hubbard_energy, etc.)
    """
    
    def __init__(self, records: Optional[List[Dict[str, Any]]] = None,
                 include_electron_number: bool = False,
                 include_moment: bool = False):
        """
        Initialize DataBank with calculation records.
        
        Args:
            records: List of dicts with keys: 'pk', 'energy', 'converged', 'occ_data', 'metadata'
            include_electron_number: If True, compute and store electron numbers per atom in each record
            include_moment: If True, compute and store magnetic moments per atom in each record
        """
        self._records = records if records is not None else []
        
        # Compute and store electron numbers and moments if requested
        if include_electron_number or include_moment:
            for record in self._records:
                occ_data = record['occ_data']
                atom_labels = occ_data.get_atom_labels()
                
                if include_electron_number:
                    record['electron_numbers'] = {
                        atom: occ_data.get_electron_number(atom) for atom in atom_labels
                    }
                
                if include_moment:
                    record['moments'] = {
                        atom: occ_data.get_magnetic_moment(atom) for atom in atom_labels
                    }
        
        # Cache for flattened data (lazy computed)
        self._cache = None  # Will be dict when computed
    
    # ============================================================================
    # Factory methods - Loading data
    # ============================================================================


    # Needs implementation from a list of 
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path], only_converged: bool = True,
                  include_electron_number: bool = False,
                  include_moment: bool = False) -> 'DataBank':
        """
        Load DataBank from JSON file (output of gather_workchain_data).
        
        Args:
            json_path: Path to JSON file
            only_converged: If True, only load converged calculations (default: True)
            include_electron_number: If True, compute and store electron numbers per atom
            include_moment: If True, compute and store magnetic moments per atom
            
        Returns:
            DataBank instance with loaded calculations
        """
        json_path = Path(json_path)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        records = []
        calculations = data.get('calculations', {})
        
        for pk_str, calc_data in calculations.items():
            # Extract basic fields
            pk = int(pk_str)
            converged = calc_data.get('converged', False)
            
            # Skip non-converged if requested
            if only_converged and not converged:
                continue
            
            # Extract energy
            output_params = calc_data.get('output_parameters', {})
            energy = output_params.get('energy', None) if output_params else None
            if energy is None:
                energy = output_params.get('energy_eV', None) if output_params else None
            
            # Extract occupation matrices
            occ_matrices = calc_data.get('occupation_matrices')
            if occ_matrices is None or not isinstance(occ_matrices, dict):
                continue  # Skip calculations without occupation data
            
            # Convert to OccupationMatrixData
            occ_data = OccupationMatrixData.from_dict(occ_matrices)
            
            # Store metadata
            metadata = {
                'hubbard_energy': output_params.get('energy_hubbard', None) if output_params else None,
                'process_type': calc_data.get('process_type'),
                'source': calc_data.get('calculation_source'),
            }
            
            # TODO: energy_uncertainty is temporary placeholder, should be computed/extracted
            energy_uncertainty = 0.0
            
            records.append({
                'pk': pk,
                'energy': energy,
                'energy_uncertainty': energy_uncertainty,
                'converged': converged,
                'occ_data': occ_data,
                'metadata': metadata
            })
        
        return cls(records, include_electron_number=include_electron_number,
                  include_moment=include_moment)
    
    @classmethod
    def from_workchain(cls, workchain_pk: int, only_converged: bool = True, **kwargs) -> 'DataBank':
        """
        Temporary placeholder for loading from workchain PK.
        """
        # raise implementation error for now
        raise NotImplementedError("from_workchain is not yet implemented. Use from_json instead.")
    
    @classmethod
    def from_calculation_pks(cls, calc_pks: List[int]) -> 'DataBank':
        """
        Load DataBank from list of calculation PKs (future implementation).
        
        Args:
            calc_pks: List of calculation primary keys
            
        Returns:
            DataBank instance
        """
        from aiida.orm import load_node
        from lordcapulet.data_structures.occupation_matrix import extract_occupations_from_calc
        
        records = []
        
        for pk in calc_pks:
            calc = load_node(pk)
            
            # Extract energy
            if 'output_parameters' in calc.outputs:
                params = calc.outputs.output_parameters.get_dict()
                energy = params.get('energy', params.get('energy_eV'))
            else:
                energy = None
            
            # Extract occupation data
            try:
                occ_data = extract_occupations_from_calc(calc)
            except Exception:
                continue  # Skip if can't extract
            
            # TODO: energy_uncertainty is temporary placeholder
            energy_uncertainty = 0.0
            
            records.append({
                'pk': pk,
                'energy': energy,
                'energy_uncertainty': energy_uncertainty,
                'converged': calc.exit_status == 0,
                'occ_data': occ_data,
                'metadata': {}
            })
        
        return cls(records)
    
    @classmethod
    def from_matrices(cls, 
                      occ_matrices: List[OccupationMatrixData],
                      energies: List[float],
                      pks: Optional[List[int]] = None,
                      converged: Optional[List[bool]] = None,
                      energy_uncertainties: Optional[List[float]] = None,
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      include_electron_number: bool = False,
                      include_moment: bool = False) -> 'DataBank':
        """
        Create DataBank from lists of occupation matrices and energies.
        
        This is useful when you have occupation matrices from sources other than
        AiiDA calculations (e.g., from proposal functions, external calculations).
        
        Args:
            occ_matrices: List of OccupationMatrixData objects
            energies: List of energies (eV) for each calculation
            pks: Optional list of PKs. If None, auto-generate sequential PKs starting from 0
            converged: Optional list of convergence status. If None, all assumed converged (True)
            energy_uncertainties: Optional list of energy uncertainties. If None, all set to 0.0
            metadata: Optional list of metadata dicts. If None, empty dicts used
            include_electron_number: If True, compute and store electron numbers per atom
            include_moment: If True, compute and store magnetic moments per atom
            
        Returns:
            DataBank instance
            
        Raises:
            ValueError: If lists have inconsistent lengths
            
        Examples:
            >>> # Create from matrices and energies only
            >>> databank = DataBank.from_matrices(occ_matrices, energies)
            
            >>> # Create with custom PKs and convergence status
            >>> databank = DataBank.from_matrices(
            ...     occ_matrices, energies, 
            ...     pks=[1000, 1001, 1002],
            ...     converged=[True, True, False]
            ... )
        """
        n = len(occ_matrices)
        
        # Validate input lengths
        if len(energies) != n:
            raise ValueError(f"Length mismatch: {n} matrices but {len(energies)} energies")
        
        if pks is not None and len(pks) != n:
            raise ValueError(f"Length mismatch: {n} matrices but {len(pks)} PKs")
        
        if converged is not None and len(converged) != n:
            raise ValueError(f"Length mismatch: {n} matrices but {len(converged)} convergence flags")
        
        if energy_uncertainties is not None and len(energy_uncertainties) != n:
            raise ValueError(f"Length mismatch: {n} matrices but {len(energy_uncertainties)} uncertainties")
        
        if metadata is not None and len(metadata) != n:
            raise ValueError(f"Length mismatch: {n} matrices but {len(metadata)} metadata entries")
        
        # Set defaults
        if pks is None:
            pks = list(range(n))
        
        if converged is None:
            converged = [True] * n
        
        if energy_uncertainties is None:
            energy_uncertainties = [0.0] * n
        
        if metadata is None:
            metadata = [{}] * n
        
        # Build records
        records = []
        for i in range(n):
            record = {
                'pk': pks[i],
                'energy': energies[i],
                'energy_uncertainty': energy_uncertainties[i],
                'converged': converged[i],
                'occ_data': occ_matrices[i],
                'metadata': metadata[i]
            }
            records.append(record)
        
        return cls(records, include_electron_number=include_electron_number,
                   include_moment=include_moment)
    
    # ============================================================================
    # Basic operations - Immutable
    # ============================================================================
    
    def __len__(self) -> int:
        """Return number of calculations."""
        return len(self._records)
    
    def __getitem__(self, idx: Union[int, slice, List[int], np.ndarray]) -> 'DataBank':
        """
        Get subset of DataBank by index/slice/array.
        
        Args:
            idx: Integer index, slice, or array of indices
            
        Returns:
            New DataBank with selected records
        """
        if isinstance(idx, int):
            # Single index - return DataBank with one record
            return DataBank([self._records[idx]])
        elif isinstance(idx, slice):
            # Slice - return DataBank with sliced records
            return DataBank(self._records[idx])
        elif isinstance(idx, (list, np.ndarray)):
            # Array of indices - return DataBank with selected records
            selected_records = [self._records[i] for i in idx]
            return DataBank(selected_records)
        else:
            raise TypeError(f"Indices must be int, slice, list, or ndarray, got {type(idx)}")
    
    def __repr__(self) -> str:
        """String representation."""
        n_converged = sum(1 for r in self._records if r['converged'])
        return f"DataBank({len(self)} calculations, {n_converged} converged)"
    
    # ============================================================================
    # Properties - Extract from records
    # ============================================================================
    
    @property
    def energies(self) -> np.ndarray:
        """Extract energies as numpy array."""
        return np.array([r['energy'] for r in self._records])
    
    @property
    def pks(self) -> np.ndarray:
        """Extract PKs as numpy array."""
        return np.array([r['pk'] for r in self._records])
    
    @property
    def converged(self) -> np.ndarray:
        """Extract convergence status as boolean array."""
        return np.array([r['converged'] for r in self._records])
    
    @property
    def energy_uncertainties(self) -> np.ndarray:
        """Extract energy uncertainties as numpy array (TODO: temporary, currently all zeros)."""
        return np.array([r.get('energy_uncertainty', 0.0) for r in self._records])
    
    @property
    def atom_ids(self) -> List[str]:
        """Get list of unique atom IDs across all calculations."""
        if len(self._records) == 0:
            return []
        
        # Get atom IDs from first record (assuming consistent structure)
        return self._records[0]['occ_data'].get_atom_labels()
    
    @property
    def n_orbitals_dict(self) -> Dict[str, int]:
        """Get dictionary mapping atom IDs to number of orbitals."""
        if len(self._records) == 0:
            return {}
        
        return {atom_id: self.get_n_orbitals(atom_id) for atom_id in self.atom_ids}
    
    def get_n_orbitals(self, atom_id: Union[str, int]) -> int:
        """
        Get number of orbitals for a given atom.
        
        Args:
            atom_id: Atom label (e.g., 'Atom_1') or integer index into atom_ids list
            
        Returns:
            Number of orbitals
        """
        if len(self._records) == 0:
            raise ValueError("DataBank is empty")
        
        # Handle integer index
        if isinstance(atom_id, int):
            atom_labels = self.atom_ids
            if atom_id >= len(atom_labels):
                raise IndexError(f"Atom index {atom_id} out of range (0-{len(atom_labels)-1})")
            atom_id = atom_labels[atom_id]
        
        matrix = self._records[0]['occ_data'].get_occupation_matrix(atom_id, 'up')
        return len(matrix)
    
    def get_trace(self, spin: str, atom_id: Optional[str] = None, calc_index: Optional[int] = None) -> Union[float, np.ndarray, Dict[str, np.ndarray]]:
        """
        Get trace of occupation matrices.
        
        Args:
            spin: Spin channel ('up' or 'down')
            atom_id: Atom label (e.g., 'Atom_1'). If None, returns dict with all atoms.
            calc_index: Calculation index. If specified, returns value for that calculation only.
            
        Returns:
            If calc_index specified: float (single atom) or dict (all atoms)
            If atom_id specified: numpy array of trace values (one per calculation)
            If atom_id is None: dict mapping atom_id -> numpy array
        """
        if len(self._records) == 0:
            return {} if atom_id is None else np.array([])
        
        # Single calculation case
        if calc_index is not None:
            if atom_id is None:
                return {atom: self._records[calc_index]['occ_data'].get_trace(atom, spin) 
                       for atom in self.atom_ids}
            else:
                return self._records[calc_index]['occ_data'].get_trace(atom_id, spin)
        
        # All calculations case
        if atom_id is None:
            # Return dict with all atoms
            result = {}
            for atom in self.atom_ids:
                traces = np.array([r['occ_data'].get_trace(atom, spin) for r in self._records])
                result[atom] = traces
            return result
        else:
            # Return array for specific atom
            return np.array([r['occ_data'].get_trace(atom_id, spin) for r in self._records])
    
    def get_electron_number(self, atom_id: Optional[str] = None, calc_index: Optional[int] = None) -> Union[float, np.ndarray, Dict[str, Union[float, np.ndarray]]]:
        """
        Get total electron number (trace_up + trace_down).
        
        Args:
            atom_id: Atom label (e.g., 'Atom_1'). If None, returns dict with all atoms.
            calc_index: Calculation index. If specified, returns value for that calculation only.
            
        Returns:
            If calc_index specified: float (single atom) or dict (all atoms)
            If atom_id specified: numpy array of electron numbers (one per calculation)
            If atom_id is None: dict mapping atom_id -> numpy array
        """
        if len(self._records) == 0:
            return {} if atom_id is None else np.array([])
        
        # Single calculation case
        if calc_index is not None:
            record = self._records[calc_index]
            # Check if precomputed
            if 'electron_numbers' in record:
                if atom_id is None:
                    return record['electron_numbers']
                else:
                    return record['electron_numbers'][atom_id]
            else:
                # Compute on the fly
                if atom_id is None:
                    return {atom: record['occ_data'].get_electron_number(atom) 
                           for atom in self.atom_ids}
                else:
                    return record['occ_data'].get_electron_number(atom_id)
        
        # All calculations case
        if atom_id is None:
            # Return dict with all atoms
            result = {}
            for atom in self.atom_ids:
                electrons = np.array([r['occ_data'].get_electron_number(atom) for r in self._records])
                result[atom] = electrons
            return result
        else:
            # Return array for specific atom
            return np.array([r['occ_data'].get_electron_number(atom_id) for r in self._records])
    
    def get_magnetic_moment(self, atom_id: Optional[str] = None, calc_index: Optional[int] = None) -> Union[float, np.ndarray, Dict[str, Union[float, np.ndarray]]]:
        """
        Get magnetic moment (trace_up - trace_down).
        
        Args:
            atom_id: Atom label (e.g., 'Atom_1'). If None, returns dict with all atoms.
            calc_index: Calculation index. If specified, returns value for that calculation only.
            
        Returns:
            If calc_index specified: float (single atom) or dict (all atoms)
            If atom_id specified: numpy array of magnetic moments (one per calculation)
            If atom_id is None: dict mapping atom_id -> numpy array
        """
        if len(self._records) == 0:
            return {} if atom_id is None else np.array([])
        
        # Single calculation case
        if calc_index is not None:
            record = self._records[calc_index]
            # Check if precomputed
            if 'moments' in record:
                if atom_id is None:
                    return record['moments']
                else:
                    return record['moments'][atom_id]
            else:
                # Compute on the fly
                if atom_id is None:
                    return {atom: record['occ_data'].get_magnetic_moment(atom) 
                           for atom in self.atom_ids}
                else:
                    return record['occ_data'].get_magnetic_moment(atom_id)
        
        # All calculations case
        if atom_id is None:
            # Return dict with all atoms
            result = {}
            for atom in self.atom_ids:
                moments = np.array([r['occ_data'].get_magnetic_moment(atom) for r in self._records])
                result[atom] = moments
            return result
        else:
            # Return array for specific atom
            return np.array([r['occ_data'].get_magnetic_moment(atom_id) for r in self._records])
    
    # ============================================================================
    # Filtering operations - Return new DataBank
    # ============================================================================
    
    def filter_converged(self, converged: bool = True) -> 'DataBank':
        """
        Filter by convergence status.
        
        Args:
            converged: If True, keep converged; if False, keep non-converged
            
        Returns:
            New DataBank with filtered records
        """
        filtered = [r for r in self._records if r['converged'] == converged]
        return DataBank(filtered)
    
    def filter_energy_range(self, min_energy: Optional[float] = None, 
                           max_energy: Optional[float] = None) -> 'DataBank':
        """
        Filter by energy range.
        
        Args:
            min_energy: Minimum energy (inclusive), None for no lower bound
            max_energy: Maximum energy (inclusive), None for no upper bound
            
        Returns:
            New DataBank with filtered records
        """
        filtered = []
        for r in self._records:
            energy = r['energy']
            if energy is None:
                continue
            if min_energy is not None and energy < min_energy:
                continue
            if max_energy is not None and energy > max_energy:
                continue
            filtered.append(r)
        
        return DataBank(filtered)
    
    def filter_atoms(self, atom_ids: List[str]) -> 'DataBank':
        """
        Filter to only include calculations with specific atoms.
        
        Note: This doesn't modify the occupation data, just filters
        calculations that have all the requested atoms.
        
        Args:
            atom_ids: List of atom labels to require
            
        Returns:
            New DataBank with filtered records
        """
        filtered = []
        for r in self._records:
            calc_atoms = r['occ_data'].get_atom_labels()
            if all(atom_id in calc_atoms for atom_id in atom_ids):
                filtered.append(r)
        
        return DataBank(filtered)
    
    # ============================================================================
    # Sorting operations - Return new DataBank
    # ============================================================================
    
    def sort_by_energy(self, ascending: bool = True) -> 'DataBank':
        """
        Sort by energy.
        
        Args:
            ascending: If True, sort low to high; if False, high to low
            
        Returns:
            New DataBank with sorted records
        """
        sorted_records = sorted(self._records, 
                              key=lambda r: r['energy'] if r['energy'] is not None else float('inf'),
                              reverse=not ascending)
        return DataBank(sorted_records)
    
    def sort_by_pk(self, ascending: bool = True) -> 'DataBank':
        """Sort by PK."""
        sorted_records = sorted(self._records, 
                              key=lambda r: r['pk'],
                              reverse=not ascending)
        return DataBank(sorted_records)
    
    # ============================================================================
    # Modification operations - Return new DataBank
    # ============================================================================
    
    def append(self, other: Union['DataBank', Dict[str, Any]]) -> 'DataBank':
        """
        Append records from another DataBank or single record.
        
        Args:
            other: DataBank instance or single record dict
            
        Returns:
            New DataBank with appended records
        """
        new_records = self._records.copy()
        
        if isinstance(other, DataBank):
            new_records.extend(other._records)
        elif isinstance(other, dict):
            # Validate record has required keys
            required = {'pk', 'energy', 'energy_uncertainty', 'converged', 'occ_data'}
            if not required.issubset(other.keys()):
                raise ValueError(f"Record must contain keys: {required}")
            new_records.append(other)
        else:
            raise TypeError("Can only append DataBank or dict")
        
        return DataBank(new_records)
    
    def remove(self, indices: Union[int, List[int], np.ndarray]) -> 'DataBank':
        """
        Remove records by index.
        
        Args:
            indices: Single index or array of indices to remove
            
        Returns:
            New DataBank with records removed
        """
        if isinstance(indices, int):
            indices = [indices]
        
        indices_set = set(indices)
        new_records = [r for i, r in enumerate(self._records) if i not in indices_set]
        
        return DataBank(new_records)
    
    def remove_by_pk(self, pks: Union[int, List[int]]) -> 'DataBank':
        """
        Remove records by PK.
        
        Args:
            pks: Single PK or list of PKs to remove
            
        Returns:
            New DataBank with records removed
        """
        if isinstance(pks, (int, np.integer)):
            pks = [pks]
        
        pks_set = set(int(pk) for pk in pks)  # Convert to int to handle numpy types
        new_records = [r for r in self._records if r['pk'] not in pks_set]
        
        return DataBank(new_records)
    
    # ============================================================================
    # PyTorch conversion - Core functionality
    # ============================================================================
    
    def _build_flat_index_map(self, atom_ids: List[str], spins: List[str]) -> Dict[str, Any]:
        """
        Build mapping from (atom, spin, i, j) to flat index for upper-triangular elements.
        
        Args:
            atom_ids: List of atom labels to include
            spins: List of spin channels ('up', 'down')
            
        Returns:
            Dict with 'forward_map', 'reverse_map', 'size', 'atom_ids', 'spins'
        """
        forward_map = {}  # (atom, spin, i, j) -> flat_index
        reverse_map = []  # [flat_index] -> (atom, spin, i, j)
        diagonal_elements = {} # (atom, spin) -> list of diagonal flat indices
        off_diagonal_elements = {} # (atom, spin) -> list of off-diagonal flat indices
        
        idx = 0
        
        for atom in sorted(atom_ids):
            n_orb = self.get_n_orbitals(atom)
            
            for spin in spins:
                # Upper triangular: i <= j
                for i in range(n_orb):
                    for j in range(i, n_orb):
                        forward_map[(atom, spin, i, j)] = idx
                        reverse_map.append((atom, spin, i, j))
                        if i == j:
                            diagonal_elements.setdefault((atom, spin), []).append(idx)
                        else:
                            off_diagonal_elements.setdefault((atom, spin), []).append(idx)
                        idx += 1
        
        return {
            'forward_map': forward_map,
            'reverse_map': reverse_map,
            'diagonal_elements': diagonal_elements,
            'off_diagonal_elements': off_diagonal_elements,
            'size': idx,
            'atom_ids': atom_ids,
            'spins': spins
        }
    
    def _flatten_single_record(self, record: Dict[str, Any], index_map: Dict[str, Any]) -> np.ndarray:
        """
        Flatten a single calculation record to upper-triangular vector.
        
        Args:
            record: Calculation record
            index_map: Index mapping from _build_flat_index_map
            
        Returns:
            1D numpy array of flattened matrix elements
        """
        vec = np.zeros(index_map['size'], dtype=float)
        occ_data = record['occ_data']
        
        for flat_idx, (atom, spin, i, j) in enumerate(index_map['reverse_map']):
            try:
                matrix = occ_data.get_occupation_matrix(atom, spin)
                vec[flat_idx] = matrix[i][j]
            except (KeyError, IndexError):
                vec[flat_idx] = 0.0  # Missing data
        
        return vec
    
    def _compute_flattened_cache(self, atom_ids: Optional[List[str]] = None,
                                spins: List[str] = ['up', 'down']) -> Dict[str, Any]:
        """
        Compute and cache flattened representation.
        
        Args:
            atom_ids: Atom labels to include (None = all)
            spins: Spin channels to include
            
        Returns:
            Cache dict with flattened data and metadata
        """
        if atom_ids is None:
            atom_ids = self.atom_ids
        
        # Build index mapping
        index_map = self._build_flat_index_map(atom_ids, spins)
        
        # Flatten all records
        n_records = len(self._records)
        flattened = np.zeros((n_records, index_map['size']), dtype=float)
        
        for i, record in enumerate(self._records):
            flattened[i] = self._flatten_single_record(record, index_map)
        
        return {
            'flattened_matrices': flattened,
            'index_map': index_map,
            'atom_ids': atom_ids,
            'spins': spins
        }
    
    def to_numpy(self, atom_ids: Optional[List[str]] = None,
                 spins: List[str] = ['up', 'down'],
                 include_energies: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert to numpy arrays.
        
        Args:
            atom_ids: Atom labels to include (None = all)
            spins: Spin channels to include
            include_energies: If True, return (matrices, energies) tuple
            
        Returns:
            Flattened matrices array, or (matrices, energies) if include_energies=True
        """
        # Determine actual atom_ids to use
        actual_atom_ids = atom_ids if atom_ids is not None else self.atom_ids
        
        # Compute cache if needed or if parameters changed
        if (self._cache is None or 
            self._cache.get('atom_ids') != actual_atom_ids or
            self._cache.get('spins') != spins):
            self._cache = self._compute_flattened_cache(atom_ids, spins)
        
        matrices = self._cache['flattened_matrices']
        
        if include_energies:
            return matrices, self.energies
        return matrices
    
    def to_pytorch(self, atom_ids: Optional[List[str]] = None,
                   spins: List[str] = ['up', 'down'],
                   include_energies: bool = False,
                   device: str = 'cpu') -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert to PyTorch tensors.
        
        Args:
            atom_ids: Atom labels to include (None = all)
            spins: Spin channels to include
            include_energies: If True, return (matrices, energies) tuple
            device: PyTorch device ('cpu', 'cuda', etc.)
            
        Returns:
            Matrices tensor, or (matrices, energies) if include_energies=True
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        
        if include_energies:
            matrices, energies = self.to_numpy(atom_ids, spins, include_energies=True)
            matrices_tensor = torch.tensor(matrices, dtype=torch.float32, device=device)
            energies_tensor = torch.tensor(energies, dtype=torch.float32, device=device)
            return matrices_tensor, energies_tensor
        else:
            matrices = self.to_numpy(atom_ids, spins, include_energies=False)
            return torch.tensor(matrices, dtype=torch.float32, device=device)
    
    def from_numpy(self, matrices: np.ndarray, 
                   atom_ids: Optional[List[str]] = None,
                   spins: List[str] = ['up', 'down']) -> List[OccupationMatrixData]:
        """
        Reconstruct OccupationMatrixData from flattened numpy array.
        
        Args:
            matrices: Flattened matrices array (n_samples, n_features)
            atom_ids: Atom labels used in flattening
            spins: Spin channels used in flattening
            
        Returns:
            List of OccupationMatrixData objects
        """
        if atom_ids is None:
            atom_ids = self.atom_ids
        
        # Build index map
        index_map = self._build_flat_index_map(atom_ids, spins)
        
        # Handle single vector or batch
        if matrices.ndim == 1:
            matrices = matrices.reshape(1, -1)
        
        results = []
        for vec in matrices:
            # Initialize occupation data structure
            occ_dict = {}
            
            for atom in atom_ids:
                n_orb = self.get_n_orbitals(atom)
                try:
                    specie = self._records[0]['occ_data'][atom]['specie']
                except:
                    specie = 'unknown'
                
                try:
                    shell = self._records[0]['occ_data'][atom]['shell']
                except:
                    shell = 'unknown'
                
                occ_dict[atom] = {
                    'specie': specie,
                    'shell': shell,
                    'occupation_matrix': {
                        'up': [[0.0] * n_orb for _ in range(n_orb)],
                        'down': [[0.0] * n_orb for _ in range(n_orb)]
                    }
                }
            
            # Fill in values from flattened vector
            for flat_idx, (atom, spin, i, j) in enumerate(index_map['reverse_map']):
                value = float(vec[flat_idx])
                occ_dict[atom]['occupation_matrix'][spin][i][j] = value
                # Symmetric matrix - fill both triangles
                if i != j:
                    occ_dict[atom]['occupation_matrix'][spin][j][i] = value
            
            results.append(OccupationMatrixData(occ_dict))
        
        return results
    
    def from_pytorch(self, matrices: 'torch.Tensor',
                    atom_ids: Optional[List[str]] = None,
                    spins: List[str] = ['up', 'down']) -> List[OccupationMatrixData]:
        """
        Reconstruct OccupationMatrixData from PyTorch tensor.
        
        Args:
            matrices: Flattened matrices tensor (n_samples, n_features)
            atom_ids: Atom labels used in flattening
            spins: Spin channels used in flattening
            
        Returns:
            List of OccupationMatrixData objects
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed")
        
        # Convert to numpy and use numpy method
        matrices_np = matrices.cpu().numpy()
        return self.from_numpy(matrices_np, atom_ids, spins)
    
    def to_numpy_single_matrix(self, occ_data: OccupationMatrixData,
                               atom_ids: Optional[List[str]] = None,
                               spins: List[str] = ['up', 'down']) -> np.ndarray:
        """
        Convert a single OccupationMatrixData to flattened numpy array using DataBank's index map.
        
        This method ensures consistency with the DataBank's flattening scheme by using
        the same index mapping.
        
        Args:
            occ_data: OccupationMatrixData object to flatten
            atom_ids: Atom labels to include (None = use DataBank's atoms)
            spins: Spin channels to include
            
        Returns:
            1D numpy array of flattened matrix elements
            
        Raises:
            ValueError: If occ_data is incompatible with DataBank structure
            
        Examples:
            >>> # Flatten a single occupation matrix
            >>> vec = databank.to_numpy_single_matrix(occ_data)
            
            >>> # Flatten with specific atoms
            >>> vec = databank.to_numpy_single_matrix(occ_data, atom_ids=['Atom_1'])
        """
        if len(self._records) == 0:
            raise ValueError("DataBank is empty - cannot determine structure")
        
        # Use DataBank's atoms if not specified
        if atom_ids is None:
            atom_ids = self.atom_ids
        
        # Validate atom compatibility
        occ_atoms = occ_data.get_atom_labels()
        for atom in atom_ids:
            if atom not in occ_atoms:
                raise ValueError(f"Atom '{atom}' not found in input OccupationMatrixData. "
                               f"Available atoms: {occ_atoms}")
        
        # Validate orbital count compatibility
        for atom in atom_ids:
            expected_n_orb = self.get_n_orbitals(atom)
            occ_matrix = occ_data.get_occupation_matrix(atom, spins[0])
            actual_n_orb = len(occ_matrix)
            
            if actual_n_orb != expected_n_orb:
                raise ValueError(f"Incompatible number of orbitals for atom '{atom}': "
                               f"expected {expected_n_orb}, got {actual_n_orb}")
        
        # Build index map
        index_map = self._build_flat_index_map(atom_ids, spins)
        
        # Flatten the single OccupationMatrixData
        vec = np.zeros(index_map['size'], dtype=float)
        
        for flat_idx, (atom, spin, i, j) in enumerate(index_map['reverse_map']):
            try:
                matrix = occ_data.get_occupation_matrix(atom, spin)
                vec[flat_idx] = matrix[i][j]
            except (KeyError, IndexError):
                vec[flat_idx] = 0.0  # Missing data
        
        return vec
    
    def to_pytorch_single_matrix(self, occ_data: OccupationMatrixData,
                                 atom_ids: Optional[List[str]] = None,
                                 spins: List[str] = ['up', 'down'],
                                 device: str = 'cpu') -> 'torch.Tensor':
        """
        Convert a single OccupationMatrixData to flattened PyTorch tensor using DataBank's index map.
        
        This method ensures consistency with the DataBank's flattening scheme by using
        the same index mapping.
        
        Args:
            occ_data: OccupationMatrixData object to flatten
            atom_ids: Atom labels to include (None = use DataBank's atoms)
            spins: Spin channels to include
            device: PyTorch device ('cpu', 'cuda', etc.)
            
        Returns:
            1D PyTorch tensor of flattened matrix elements
            
        Raises:
            ValueError: If occ_data is incompatible with DataBank structure
            ImportError: If PyTorch is not installed
            
        Examples:
            >>> # Flatten to PyTorch tensor
            >>> tensor = databank.to_pytorch_single_matrix(occ_data)
            
            >>> # Flatten to GPU tensor
            >>> tensor = databank.to_pytorch_single_matrix(occ_data, device='cuda')
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        
        # Use numpy method and convert to tensor
        vec = self.to_numpy_single_matrix(occ_data, atom_ids, spins)
        return torch.tensor(vec, dtype=torch.float32, device=device)
    
    # ============================================================================
    # Utility methods
    # ============================================================================
    
    def get_record(self, idx: int) -> Dict[str, Any]:
        """Get a single record by index."""
        return self._records[idx]
    
    def get_occ_data(self, idx: int) -> OccupationMatrixData:
        """Get OccupationMatrixData for a single calculation."""
        return self._records[idx]['occ_data']
    
    def get_forward_index_map(self) -> Dict[str, Any]:
        """Get forward index map from cached data."""
        if self._cache is None:
            self.to_numpy()  # Compute cache
        return self._cache['index_map']['forward_map']
    
    def get_reverse_index_map(self) -> List[Tuple[str, str, int, int]]:
        """Get reverse index map from cached data."""
        if self._cache is None:
            self.to_numpy()  # Compute cache
        return self._cache['index_map']['reverse_map']
    
    def compute_distances(self, 
                         reference: OccupationMatrixData,
                         atom_label: Optional[str] = None,
                         spins: Optional[List[str]] = None,
                         calc_id: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Compute Euclidean distance from all calculations to a reference occupation matrix.
        
        Args:
            reference: Reference OccupationMatrixData to compare against
            atom_label: If specified, compute distance only for this atom.
                       If None, compute total distance across all atoms.
            spins: List of spin channels to include (default: ['up', 'down'])
            calc_id: If specified, compute distance only for this single calculation.
                    If None, compute distances for all calculations.
        
        Returns:
            If calc_id specified: float (distance for that calculation)
            If calc_id is None: numpy array of distances (one per calculation)
        
        Examples:
            >>> # Distance from all calculations to a reference
            >>> distances = databank.compute_distances(reference_occ)
            
            >>> # Distance for specific calculation
            >>> dist = databank.compute_distances(reference_occ, calc_id=0)
            
            >>> # Distance for specific atom only
            >>> distances = databank.compute_distances(reference_occ, atom_label='Atom_1')
        """
        if len(self._records) == 0:
            return np.array([]) if calc_id is None else 0.0
        
        # Single calculation case
        if calc_id is not None:
            if calc_id < 0 or calc_id >= len(self._records):
                raise IndexError(f"calc_id {calc_id} out of range (0-{len(self._records)-1})")
            
            calc_occ = self._records[calc_id]['occ_data']
            return compute_occupation_distance(calc_occ, reference, 
                                              atom_label=atom_label, 
                                              spins=spins)
        
        # All calculations case
        distances = np.zeros(len(self._records))
        for i, record in enumerate(self._records):
            calc_occ = record['occ_data']
            distances[i] = compute_occupation_distance(calc_occ, reference,
                                                       atom_label=atom_label,
                                                       spins=spins)
        
        return distances
    
    def summary(self) -> str:
        """Get summary statistics."""
        if len(self._records) == 0:
            return "Empty DataBank"
        
        n_total = len(self._records)
        n_converged = sum(1 for r in self._records if r['converged'])
        
        energies = self.energies
        valid_energies = energies[~np.isnan(energies)]
        
        summary = [
            f"DataBank Summary:",
            f"  Total calculations: {n_total}",
            f"  Converged: {n_converged} ({100*n_converged/n_total:.1f}%)",
            f"  Energy range: {valid_energies.min():.4f} to {valid_energies.max():.4f} eV",
            f"  Atoms: {', '.join(self.atom_ids)}",
        ]
        
        return '\n'.join(summary)
    
    def as_dict(self) -> List[Dict[str, Any]]:
        """
        Export records as list of dictionaries.
        
        Note: OccupationMatrixData objects are converted to dicts via as_dict().
        
        Returns:
            List of record dictionaries with occupation data as dicts
        """
        result = []
        for record in self._records:
            record_copy = record.copy()
            record_copy['occ_data'] = record['occ_data'].as_dict()
            result.append(record_copy)
        return result
    
    def to_dataframe(self):
        """
        Convert to pandas DataFrame with flattened occupation matrices.
        
        If electron numbers or moments were computed at initialization (via include_electron_number
        or include_moment flags), they will be included as columns automatically.
        
        Returns:
            pandas DataFrame with columns: pk, energy, energy_uncertainty, converged,
            plus flattened occupation matrix elements as separate columns,
            plus electron_number_<atom> and moment_<atom> columns if they were precomputed
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")
        
        if len(self._records) == 0:
            return pd.DataFrame()
        
        # Build base dataframe with metadata
        base_data = {
            'pk': self.pks,
            'energy': self.energies,
            'energy_uncertainty': self.energy_uncertainties,
            'converged': self.converged,
        }
        
        # Check if electron numbers are present in records
        if 'electron_numbers' in self._records[0]:
            for atom in self.atom_ids:
                base_data[f'electron_number_{atom}'] = [
                    r['electron_numbers'][atom] for r in self._records
                ]
        
        # Check if moments are present in records
        if 'moments' in self._records[0]:
            for atom in self.atom_ids:
                base_data[f'moment_{atom}'] = [
                    r['moments'][atom] for r in self._records
                ]
        
        # Add flattened matrices
        matrices = self.to_numpy()
        
        # Get column names from index map
        if self._cache is None:
            self.to_numpy()  # Compute cache
        
        index_map = self._cache['index_map']
        for flat_idx, (atom, spin, i, j) in enumerate(index_map['reverse_map']):
            col_name = f"{atom}_{spin}_occ_{i}_{j}"
            base_data[col_name] = matrices[:, flat_idx]
        
        return pd.DataFrame(base_data)
