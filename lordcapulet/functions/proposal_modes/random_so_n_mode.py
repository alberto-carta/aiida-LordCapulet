"""
Random SO(N) mode for generating occupation matrix proposals.

This module implements SO(N) decomposition-based generation of occupation matrices 
for DFT+U calculations using random Euler angles in the Lie algebra basis.
"""

import numpy as np
from typing import List, Dict, Any

from lordcapulet.utils.so_n_decomposition import (
    get_so_n_lie_basis, 
    euler_angles_to_rotation,
    canonicalize_angles
)
from lordcapulet.data_structures import OccupationMatrixData
from .shared_functionality import calculate_average_traces, create_random_diagonal_matrices

def propose_random_so_n_constraints(occ_matr_list, natoms, N, debug=False, **kwargs) -> List[OccupationMatrixData]:
    """
    Generate N random occupation matrix proposals using SO(N) decomposition.
    
    Strategy:
    1. Calculate target electron counts (traces) from existing data or kwargs (reuses random_mode logic)
    2. For each proposal:
       - For each atom: create diagonal matrices with 1s and 0s for occupied/unoccupied states
       - Generate random Euler angles in the SO(norb) Lie algebra
       - Apply SO(N) rotation to the diagonal matrices using matrix exponential
       - Preserve specie and shell metadata from input matrices
    
    This maintains consistency with the original random mode by using the same
    1s/0s diagonal matrix generation and electron count logic.
    
    :param occ_matr_list: List of OccupationMatrixData objects for reference
    :param natoms: Number of atoms in the system
    :param N: Number of proposals to generate
    :param debug: Whether to print debug information
    :param kwargs: Additional parameters:
        - 'target_traces': List of target electron counts per atom (if not provided, calculated from data)
        - 'randomize_oxidation': Whether to add random variation to electron counts (default: True)
    
    Note: Always uses full angle range [-π, π] with canonicalization enabled for optimal coverage.
    
    :return: List of N OccupationMatrixData objects (proposals)
    """
    
    if debug:
        print(f"Generating {N} SO(N)-based occupation matrices for {natoms} atoms")
    
    proposals = []

    # Extract metadata from first reference matrix for reuse
    first_occ_data = occ_matr_list[0]
    atom_labels = first_occ_data.get_atom_labels()
    atom_species = {label: first_occ_data[label]['specie'] for label in atom_labels}
    atom_shells = {label: first_occ_data[label]['shell'] for label in atom_labels}

    # STEP 1: Determine target electron counts (traces) for each atom
    if 'target_traces' not in kwargs:
        # Calculate average traces from existing occupation matrices (reuse from random_mode)
        average_traces = calculate_average_traces(occ_matr_list, natoms, debug)
    else:
        average_traces = np.array(kwargs['target_traces'])

    # Get additional parameters
    randomize_oxidation = kwargs.get('randomize_oxidation', True)

    if debug:
        print(f"Target electron counts per atom: {average_traces}")
        print(f"Using full angle range: [-π, π] with canonicalization")
        print(f"Randomize oxidation: {randomize_oxidation}")

    # STEP 2: Get matrix dimensions and generate SO(N) basis
    dim = len(first_occ_data.get_occupation_matrix(atom_labels[0], 'up'))
    generators = get_so_n_lie_basis(dim)
    
    if debug:
        print(f"Orbital dimension: {dim}, SO({dim}) has {len(generators)} generators")

    # STEP 3: Generate N random proposals
    for iteration in range(N):
        if debug:
            print(f"  Generating SO(N) proposal {iteration + 1}/{N}")
        
        # Create new OccupationMatrixData for this proposal
        proposal_data = {}
        
        for iatom, atom_label in enumerate(atom_labels):
            if debug:
                print(f"    {atom_label}: dim = {dim}x{dim}")
            
            # STEP 3a: Determine target electron count for this atom (reuse from random_mode)
            target_oxidation = int(round(average_traces[iatom]))
            if randomize_oxidation:
                # Add small random variation (-1, 0, or +1)
                target_oxidation += np.random.randint(-1, 2)
            
            if debug:
                print(f"      Target electrons = {target_oxidation}")
            
            # STEP 3b: Create random diagonal matrices with 1s and 0s (reuse from random_mode)
            target_matrix_np = create_random_diagonal_matrices(dim, target_oxidation)
            
            # STEP 3c: Generate random Euler angles
            euler_angles = np.random.uniform(-np.pi, np.pi, len(generators))
            
            # STEP 3d: Always canonicalize angles to principal branch
            euler_angles = canonicalize_angles(euler_angles, generators)
            
            # STEP 3e: Apply SO(N) rotation to both spin channels
            rotated_matrices = np.zeros_like(target_matrix_np)
            for ispin in range(2):  # up and down
                # Generate rotation matrix from Euler angles
                R = euler_angles_to_rotation(euler_angles, generators)
                
                # Apply rotation: R * diagonal_matrix * R^T
                rotated_matrices[ispin] = R @ target_matrix_np[ispin] @ R.T
            
            if debug:
                print(f"      Applied SO({dim}) rotation with {len(euler_angles)} parameters")
            
            # For collinear calculations, matrices should be real
            # Store in unified format with preserved metadata
            proposal_data[atom_label] = {
                'specie': atom_species[atom_label],
                'shell': atom_shells[atom_label],
                'occupation_matrix': {
                    'up': rotated_matrices[0].real.tolist(),
                    'down': rotated_matrices[1].real.tolist()
                }
            }

        # Create OccupationMatrixData from proposal
        proposal = OccupationMatrixData(proposal_data)
        proposals.append(proposal)
    
    if debug:
        print(f"Successfully generated {len(proposals)} SO(N)-based proposals")
    
    return proposals


