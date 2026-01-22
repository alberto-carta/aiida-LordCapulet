"""
Functions shared across different proposal modes.
"""

import numpy as np
from typing import List, Dict, Any


from scipy.stats import uniform_direction # random direction generator
from lordcapulet.utils.rotation_matrices import rotate_QE_matrix
from scipy.stats import uniform_direction, special_ortho_group

#

def calculate_average_traces(occ_matr_list, natoms, debug=False):
    """
    Calculate average electron counts (traces) from existing occupation matrices.
    
    :param occ_matr_list: List of OccupationMatrixData objects
    :param natoms: Number of atoms
    :param debug: Whether to print debug info
    :return: Array of average traces per atom
    """
    average_traces = np.zeros(natoms)
    
    # Get atom labels from first matrix
    atom_labels = occ_matr_list[0].get_atom_labels()
    
    for iatom, atom_label in enumerate(atom_labels):
        total_trace = 0
        for occ_data in occ_matr_list:
            up_matrix = np.array(occ_data.get_occupation_matrix(atom_label, 'up'))
            down_matrix = np.array(occ_data.get_occupation_matrix(atom_label, 'down'))
            total_trace += np.trace(up_matrix) + np.trace(down_matrix)
        
        average_traces[iatom] = total_trace / len(occ_matr_list)
    
    if debug:
        print(f"  Calculated average traces: {average_traces}")
    
    return average_traces


def create_random_diagonal_matrices(dim, target_electrons):
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


def apply_random_rotation(matrices):
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

def create_patchwork_guess(databank,
                        x_train,
                        atoms,
                        device,
                        apply_rotation=False,
                        rotation_type = 'SO(3)',
                        rotation_prob=1):
    """
    Create a diverse initial guess by mixing atoms from different training points.

    This should be reworked to only use a training databank,
    It has absolute garbage performance right now and has no reason to do so
    
    Strategy:
    1. For each atom, randomly select a training point
    2. With 50% probability, flip spins (swap up/down matrices)
    3. If atoms are of the same species, optionally permute them
    4. Optionally apply random rotations to break symmetry
    
    Args:
        databank: DataBank with atom information
        x_train: Training data tensor [N, features]
        atoms: List of atom IDs
        device: torch device
        apply_rotation: Whether to apply random rotations to each atom's matrices
    
    Returns:
        Patchwork guess tensor [1, 1, features]
    """
    import random
    from copy import deepcopy
    
    # Step 1: Create patchwork by selecting random training points for each atom
    # Start with a copy of the first training point's OccupationMatrixData
    patchwork_occ = databank.from_pytorch(x_train[0].unsqueeze(0), atom_ids=atoms, spins=['up', 'down'])[0]
    
    # Deep copy the internal data to avoid modifying the original
    patchwork_data = deepcopy(patchwork_occ.data)
    
    source_indices = {}  # Track which training point each atom came from

    if rotation_type not in ['SO(3)', 'SO(N)', 'Mixed']:
        raise ValueError(f"Unknown rotation type: {rotation_type}")
    
    if rotation_type == 'Mixed':
        rotation_type = random.choice(['SO(3)', 'SO(N)'])

    
    for atom_id in atoms:
        # Randomly select a training point for this atom
        source_idx = random.randint(0, len(x_train) - 1)
        source_indices[atom_id] = source_idx
        
        # Extract occupation matrices from this training point
        source_occ = databank.from_pytorch(x_train[source_idx].unsqueeze(0), atom_ids=atoms, spins=['up', 'down'])[0]
        up_mat = source_occ.get_occupation_matrix(atom_id, 'up')
        down_mat = source_occ.get_occupation_matrix(atom_id, 'down')
        
        # Step 2: With 50% probability, flip spins
        if random.random() < 0.5:
            up_mat, down_mat = down_mat, up_mat  # Swap
        
        # Update the patchwork data for this atom
        patchwork_data[atom_id]['occupation_matrix']['up'] = up_mat
        patchwork_data[atom_id]['occupation_matrix']['down'] = down_mat

        if apply_rotation and (random.random() < rotation_prob):
            if rotation_type == 'SO(3)':
                # Generate random rotation parameters
                angle = np.random.uniform(0, 2 * np.pi)
                direction = uniform_direction.rvs(3)
                
                # Apply rotation
                up_mat_rotated = rotate_QE_matrix(np.array(up_mat), angle, direction)
                down_mat_rotated = rotate_QE_matrix(np.array(down_mat), angle, direction)
                
                # Update the data (store real parts for collinear calculations)
                patchwork_data[atom_id]['occupation_matrix']['up'] = up_mat_rotated.real.tolist()
                patchwork_data[atom_id]['occupation_matrix']['down'] = down_mat_rotated.real.tolist()
            elif rotation_type == 'SO(N)':
                # Use random SO(N) matrix for rotation
                R = special_ortho_group.rvs(len(up_mat))
                up_mat_np = np.array(up_mat)
                down_mat_np = np.array(down_mat)
                up_mat_rotated = R @ up_mat_np @ R.T
                down_mat_rotated = R @ down_mat_np @ R.T
                patchwork_data[atom_id]['occupation_matrix']['up'] = up_mat_rotated.real.tolist()
                patchwork_data[atom_id]['occupation_matrix']['down'] = down_mat_rotated.real.tolist()
    
    # Step 3: If atoms are same species, optionally permute
    # Group atoms by species (assuming atom IDs like "Fe1", "Fe2" have same species)
    species_groups = {}
    for atom_id in atoms:
        # Get species from the patchwork data
        species = patchwork_data[atom_id]['specie']
        if species not in species_groups:
            species_groups[species] = []
        species_groups[species].append(atom_id)
    
    # For each species with multiple atoms, decide whether to permute
    for species, atom_group in species_groups.items():
        if len(atom_group) > 1 and random.random() < 0.5:
            # Permute matrices among atoms of same species
            shuffled_group = atom_group.copy()
            random.shuffle(shuffled_group)
            
            # Create temporary storage of occupation matrices
            temp_matrices = {}
            for atom in atom_group:
                temp_matrices[atom] = {
                    'up': patchwork_data[atom]['occupation_matrix']['up'],
                    'down': patchwork_data[atom]['occupation_matrix']['down']
                }
            
            # Apply permutation
            for orig_atom, new_atom in zip(atom_group, shuffled_group):
                patchwork_data[orig_atom]['occupation_matrix'] = temp_matrices[new_atom]
    
    # Create new OccupationMatrixData from the modified data
    from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
    patchwork_occ_final = OccupationMatrixData(patchwork_data)
    
        
    # Recreate OccupationMatrixData with rotated matrices
    patchwork_occ_final = OccupationMatrixData(patchwork_data)
    
    # Convert to tensor using the new method
    patchwork_tensor = databank.to_pytorch_single_matrix(
        patchwork_occ_final, 
        atom_ids=atoms, 
        spins=['up', 'down'], 
        device=device
    )
    
    # Reshape to [1, 1, features] for optimize_acqf
    return patchwork_tensor.unsqueeze(0).unsqueeze(0)