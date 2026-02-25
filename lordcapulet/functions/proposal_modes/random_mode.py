#%%
"""
Random mode for generating occupation matrix proposals.

This module implements random generation of occupation matrices for DFT+U calculations.
"""

import numpy as np
from typing import List, Dict, Any

from lordcapulet.data_structures import OccupationMatrixData
from lordcapulet.functions.proposal_modes.shared_functionality import (
    calculate_average_traces,
    create_random_diagonal_matrices,
    apply_random_rotation
)


def propose_random_constraints(occ_matr_list, natoms, N, debug=False, randomize_oxidation=True, **kwargs) -> List[OccupationMatrixData]:
    """
    Generate N random occupation matrix proposals.
    
    Strategy:
    1. Calculate target electron counts (traces) from existing data or kwargs
    2. For each proposal:
       - For each atom: create diagonal occupation matrices with target electron count
       - Optionally randomize the electron count slightly
       - Apply random rotations to break symmetry
       - Preserve specie and shell metadata from input matrices
    
    :param occ_matr_list: List of OccupationMatrixData objects for reference
    :param natoms: Number of atoms in the system
    :param N: Number of proposals to generate
    :param debug: Whether to print debug information
    :param randomize_oxidation: Whether to add random variation to electron counts
    :param kwargs: Additional parameters:
        - 'target_traces': List of target electron counts per atom (if not provided, calculated from data)
    
    :return: List of N OccupationMatrixData objects (proposals)
    """
    
    if debug:
        print(f"Generating {N} random occupation matrices for {natoms} atoms")
    
    proposals = []

    # Extract metadata from first reference matrix for reuse
    first_occ_data = occ_matr_list[0]
    atom_labels = first_occ_data.get_atom_labels()
    atom_species = {label: first_occ_data[label]['specie'] for label in atom_labels}
    atom_shells = {label: first_occ_data[label]['shell'] for label in atom_labels}

    # STEP 1: Determine target electron counts (traces) for each atom
    if 'target_traces' not in kwargs:
        # Calculate average traces from existing occupation matrices
        average_traces = calculate_average_traces(occ_matr_list, natoms, debug)
    else:
        average_traces = np.array(kwargs['target_traces'])

    if debug:
        print(f"Target electron counts per atom: {average_traces}")

    # STEP 2: Generate N random proposals
    for iteration in range(N):
        if debug:
            print(f"  Generating proposal {iteration + 1}/{N}")
        
        # Create new OccupationMatrixData for this proposal
        proposal_data = {}
        
        for iatom, atom_label in enumerate(atom_labels):
            # Get matrix dimensions from reference data
            dim = len(first_occ_data.get_occupation_matrix(atom_label, 'up'))
            
            # STEP 2a: Determine target electron count for this atom
            target_oxidation = int(round(average_traces[iatom]))
            if randomize_oxidation:
                # Add small random variation (-1, 0, or +1)
                target_oxidation += np.random.randint(-1, 2)  # randint is exclusive of upper bound
            
            if debug:
                print(f"    {atom_label}: target electrons = {target_oxidation}, matrix size = {dim}x{dim}")
            
            # STEP 2b: Create random diagonal occupation matrices
            # This generates a random multiplet configuration
            # with the specified number of electrons
            target_matrix_np = create_random_diagonal_matrices(dim, target_oxidation)
            
            # STEP 2c: Apply random rotation to break symmetry
            target_matrix_np = apply_random_rotation(target_matrix_np)
            
            # For collinear calculations, matrices should be real
            # Store in unified format with preserved metadata
            proposal_data[atom_label] = {
                'specie': atom_species[atom_label],
                'shell': atom_shells[atom_label],
                'occupation_matrix': {
                    'up': target_matrix_np[0].real.tolist(),
                    'down': target_matrix_np[1].real.tolist()
                }
            }

        # Create OccupationMatrixData from proposal
        proposal = OccupationMatrixData(proposal_data)
        proposals.append(proposal)
    
    if debug:
        print(f"Successfully generated {len(proposals)} random proposals")
    
    return proposals




#%%
# from aiida.orm import load_node
# from aiida.orm import List, Float, Int, Str, Dict, Bool
# import aiida
# from numpy.linalg import eig

# aiida.load_profile()


# occ_matrices = []
# pk_list = load_node(5431)
# for pk in pk_list.get_list():
#     node = load_node(pk)
#     if node.__class__.__name__ == "Dict":
#         occupation_matrix = node.get_dict()
#         occ_matrices.append(occupation_matrix)

# # np.trace(occ_matrices[0]['1']['spin_data']['up']['occupation_matrix'])

# a = propose_random_constraints(occ_matrices, natoms=2, N=10, debug=True, target_traces=[5, 5], randomize_oxidation=False)

# with np.printoptions(precision=3, suppress=True):
#    print(np.array(a[3]['matrix'][0][0]).real)
# #    take sum of eigenvalues
#    print(np.sum(np.linalg.eigvals(np.array(a[3]['matrix'][0]).real)) + np.sum(np.array(a[3]['matrix'][0]).real))
# # %%
