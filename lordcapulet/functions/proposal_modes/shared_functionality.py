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

def create_patchwork_guess(
    databank,
    x_train,
    atoms,
    device=None,
    apply_rotation=False,
    rotation_type='SO(3)',
    rotation_prob=1.0,
    rng=None,
    use_torch=False,
):
    """
    Create a diverse candidate occupation matrix by mixing atom blocks
    from different training points, with optional spin flips, atom permutations,
    and random rotations.

    Works with both numpy arrays (default) and PyTorch tensors.

    Strategy:
    1. For each atom, randomly select a training point
    2. With 50% probability, flip spins (swap up/down matrices)
    3. If atoms are of the same species, optionally permute them
    4. Optionally apply random rotations to break symmetry

    Args:
        databank: DataBank instance.
        x_train: Training data (numpy [N, features] or torch [N, features]).
        atoms: List of atom IDs.
        device: torch device (only used if use_torch=True).
        apply_rotation: Whether to apply random rotations.
        rotation_type: 'SO(3)', 'SO(N)', or 'Mixed'.
        rotation_prob: Probability of applying rotation to each atom.
        rng: Optional np.random.Generator for reproducibility.
        use_torch: If True, x_train is a torch tensor; default False (numpy).

    Returns:
        numpy array [n_features] if use_torch=False,
        torch tensor [1, 1, n_features] if use_torch=True.
    """
    from copy import deepcopy

    if rng is None:
        rng = np.random.default_rng()

    if use_torch:
        x_train_np = x_train.cpu().numpy()
    else:
        x_train_np = x_train

    n_train = x_train_np.shape[0]

    # --- Build patchwork ---
    # Start with a copy of the first training point
    patchwork_occ = databank.from_numpy(x_train_np[0:1], atom_ids=atoms, spins=['up', 'down'])[0]
    patchwork_data = deepcopy(patchwork_occ.data)

    if rotation_type not in ['SO(3)', 'SO(N)', 'Mixed']:
        raise ValueError(f"Unknown rotation type: {rotation_type}")
    if rotation_type == 'Mixed':
        rotation_type = 'SO(3)' if rng.random() < 0.5 else 'SO(N)'

    for atom_id in atoms:
        source_idx = rng.integers(0, n_train)
        source_occ = databank.from_numpy(
            x_train_np[source_idx:source_idx + 1], atom_ids=atoms, spins=['up', 'down']
        )[0]
        up_mat = source_occ.get_occupation_matrix(atom_id, 'up')
        down_mat = source_occ.get_occupation_matrix(atom_id, 'down')

        # 50% chance to flip spins
        if rng.random() < 0.5:
            up_mat, down_mat = down_mat, up_mat

        patchwork_data[atom_id]['occupation_matrix']['up'] = up_mat
        patchwork_data[atom_id]['occupation_matrix']['down'] = down_mat

        if apply_rotation and rng.random() < rotation_prob:
            if rotation_type == 'SO(3)':
                angle = rng.uniform(0, 2 * np.pi)
                direction = uniform_direction.rvs(3, random_state=rng)
                up_mat_rot = rotate_QE_matrix(np.array(up_mat), angle, direction)
                down_mat_rot = rotate_QE_matrix(np.array(down_mat), angle, direction)
            elif rotation_type == 'SO(N)':
                R = special_ortho_group.rvs(len(up_mat), random_state=rng)
                up_mat_np = np.array(up_mat)
                down_mat_np = np.array(down_mat)
                up_mat_rot = R @ up_mat_np @ R.T
                down_mat_rot = R @ down_mat_np @ R.T
            patchwork_data[atom_id]['occupation_matrix']['up'] = up_mat_rot.real.tolist()
            patchwork_data[atom_id]['occupation_matrix']['down'] = down_mat_rot.real.tolist()

    # --- Permute atoms of same species ---
    species_groups = {}
    for a in atoms:
        species = patchwork_data[a]['specie']
        species_groups.setdefault(species, []).append(a)

    for species, group in species_groups.items():
        if len(group) > 1 and rng.random() < 0.5:
            shuffled = group.copy()
            rng.shuffle(shuffled)
            temp = {a: {
                'up': patchwork_data[a]['occupation_matrix']['up'],
                'down': patchwork_data[a]['occupation_matrix']['down'],
            } for a in group}
            for orig, new in zip(group, shuffled):
                patchwork_data[orig]['occupation_matrix'] = temp[new]

    # --- Finalize ---
    from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
    patchwork_occ_final = OccupationMatrixData(patchwork_data)

    if use_torch:
        vec = databank.to_pytorch_single_matrix(
            patchwork_occ_final, atom_ids=atoms, spins=['up', 'down'], device=device
        )
        return vec.unsqueeze(0).unsqueeze(0)
    else:
        return databank.to_numpy_single_matrix(
            patchwork_occ_final, atom_ids=atoms, spins=['up', 'down'],
        )
