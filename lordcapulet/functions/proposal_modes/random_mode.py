#%%
"""
Random mode for generating occupation matrix proposals.

This module implements random generation of occupation matrices for DFT+U calculations.
"""

import numpy as np
from typing import List, Dict, Any

from lordcapulet.utils.rotation_matrices import rotate_QE_matrix
from scipy.stats import uniform_direction # random direction generator


def propose_random_constraints(occ_matr_list, natoms, N, debug=False, randomize_oxidation=True, **kwargs) -> list[dict[str, Any]]:
    """
    Generate N random occupation matrix proposals.
    
    Strategy:
    1. Calculate target electron counts (traces) from existing data or kwargs
    2. For each proposal:
       - For each atom: create diagonal occupation matrices with target electron count
       - Optionally randomize the electron count slightly
       - Apply random rotations to break symmetry
    
    :param occ_matr_list: List of existing occupation matrix dictionaries for reference
    :param natoms: Number of atoms in the system
    :param N: Number of proposals to generate
    :param debug: Whether to print debug information
    :param randomize_oxidation: Whether to add random variation to electron counts
    :param kwargs: Additional parameters:
        - 'target_traces': List of target electron counts per atom (if not provided, calculated from data)
    
    :return: List of N dictionaries containing the random occupation matrices
    """
    
    if debug:
        print(f"Generating {N} random occupation matrices for {natoms} atoms")
    
    proposals = []

    # STEP 1: Determine target electron counts (traces) for each atom
    if 'target_traces' not in kwargs:
        # Calculate average traces from existing occupation matrices
        average_traces = _calculate_average_traces(occ_matr_list, natoms, debug)
    else:
        average_traces = np.array(kwargs['target_traces'])

    if debug:
        print(f"Target electron counts per atom: {average_traces}")


    # STEP 2: Generate N random proposals
    for iteration in range(N):
        if debug:
            print(f"  Generating proposal {iteration + 1}/{N}")
        
        # Create occupation matrices for all atoms
        occ_mat_list_per_atom = []
        
        for iatom in range(natoms):
            # Get matrix dimensions from reference data
            dim = len(occ_matr_list[0][f'{iatom+1}']['spin_data']['up']['occupation_matrix'])
            
            # STEP 2a: Determine target electron count for this atom
            target_oxidation = int(round(average_traces[iatom]))
            if randomize_oxidation:
                # Add small random variation (-1, 0, or +1)
                target_oxidation += np.random.randint(-1, 2)  # randint is exclusive of upper bound
            
            if debug:
                print(f"    Atom {iatom+1}: target electrons = {target_oxidation}, matrix size = {dim}x{dim}")
            
            # STEP 2b: Create random diagonal occupation matrices
            # This generates a random multiplet configuration
            # with the specified number of electrons
            target_matrix_np = _create_random_diagonal_matrices(dim, target_oxidation)
            
            # STEP 2c: Apply random rotation to break symmetry
            target_matrix_np = _apply_random_rotation(target_matrix_np)
            
            # for collinear calculations, matrices should be real, we enforce this
            occ_mat_list_per_atom.append(target_matrix_np.real.tolist())

        # Create proposal dictionary
        proposal = {'matrix': occ_mat_list_per_atom}
        proposals.append(proposal)
    
    if debug:
        print(f"Successfully generated {len(proposals)} random proposals")
    
    return proposals


def _calculate_average_traces(occ_matr_list, natoms, debug=False):
    """
    Calculate average electron counts (traces) from existing occupation matrices.
    
    :param occ_matr_list: List of occupation matrix dictionaries
    :param natoms: Number of atoms
    :param debug: Whether to print debug info
    :return: Array of average traces per atom
    """
    average_traces = np.zeros(natoms)
    
    for iatom in range(natoms):
        total_trace = 0
        for occ_mat_card in occ_matr_list:
            up_matrix = occ_mat_card[f'{iatom+1}']['spin_data']['up']['occupation_matrix']
            down_matrix = occ_mat_card[f'{iatom+1}']['spin_data']['down']['occupation_matrix']
            total_trace += np.trace(up_matrix) + np.trace(down_matrix)
        
        average_traces[iatom] = total_trace / len(occ_matr_list)
    
    if debug:
        print(f"  Calculated average traces: {average_traces}")
    
    return average_traces


def _create_random_diagonal_matrices(dim, target_electrons):
    """
    Create random diagonal occupation matrices with specified electron count.
    
    Strategy: 
    1. Create a list of 1s and 0s representing occupied/unoccupied states
    2. Randomly shuffle and distribute between up/down spins
    3. Create diagonal matrices
    
    :param dim: Orbital dimension (matrix size will be dim x dim)
    :param target_electrons: Total number of electrons to distribute
    :return: Array of shape (2, dim, dim) for [up, down] spin matrices
    """
    target_matrix_np = np.zeros((2, dim, dim), dtype=complex)
    
    # Create list of occupied (1) and unoccupied (0) states
    # Total available states = 2 * dim (up and down for each orbital)
    max_electrons = 2 * dim
    actual_electrons = min(target_electrons, max_electrons)  # Can't exceed total states
    
    diagonal_elements = [1] * actual_electrons + [0] * (max_electrons - actual_electrons)
    np.random.shuffle(diagonal_elements)
    
    # Split randomly between up and down spins
    up_elements = diagonal_elements[:dim]
    down_elements = diagonal_elements[dim:]
    
    # Create diagonal matrices
    target_matrix_np[0] = np.diag(up_elements)
    target_matrix_np[1] = np.diag(down_elements)
    
    return target_matrix_np


def _apply_random_rotation(matrices):
    """
    Apply random rotations to occupation matrices to break symmetry.
    
    :param matrices: Array of shape (2, dim, dim) for [up, down] matrices
    :return: Rotated matrices
    """
    # Generate random rotation parameters
    angle = np.random.uniform(0, 2 * np.pi)
    direction = uniform_direction.rvs(3)
    
    # Apply rotation to both spin channels
    rotated_matrices = matrices.copy()
    rotated_matrices[0] = rotate_QE_matrix(matrices[0], angle, direction)
    rotated_matrices[1] = rotate_QE_matrix(matrices[1], angle, direction)
    
    return rotated_matrices


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
