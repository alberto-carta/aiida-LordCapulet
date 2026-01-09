"""
Acquisition function optimization utilities.

This module provides functions to optimize acquisition functions to find
the next candidate point for evaluation. Supports multiple optimization
methods with easy extensibility for new methods (e.g., particle swarm).
"""

import torch
import torch.optim as optim
from botorch.optim.initializers import gen_batch_initial_conditions
import numpy as np
from lordcapulet.utils.rotation_matrices import rotate_QE_matrix
from scipy.stats import uniform_direction, special_ortho_group


def optimize_acquisition(acqf, bounds, optimization_config, initial_guess=None):
    """
    Optimize the acquisition function to find the next candidate.
    
    This is the main entry point for acquisition optimization. It supports
    multiple methods and can be easily extended for new optimizers.
    
    Args:
        acqf: The acquisition function to optimize
        bounds: Tensor of shape [2, num_features] with [lower_bounds, upper_bounds]
        optimization_config: Dictionary with optimization parameters
        initial_guess: Optional initial guess [1, 1, num_features]
        
    Returns:
        tuple: (candidate, acqf_value)
            - candidate: Best point found [1, num_features]
            - acqf_value: Acquisition function value at candidate
            
    Example optimization_config:
        {
            "method": "adam",  # or "lbfgs", "particle_swarm", etc.
            "num_restarts": 20,
            "raw_samples": 1048,
            "num_steps": 150,
            "learning_rate": 0.01,
            "use_best_train": False,
            "use_random_train": True,
        }
    """
    method = optimization_config.get("method", "adam")
    
    if method == "adam":
        return _optimize_adam(acqf, bounds, optimization_config, initial_guess)
    elif method == "lbfgs":
        return _optimize_lbfgs(acqf, bounds, optimization_config, initial_guess)
    # Extensible: add more methods here
    # elif method == "particle_swarm":
    #     return _optimize_particle_swarm(acqf, bounds, optimization_config, initial_guess)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def _optimize_adam(acqf, bounds, config, initial_guess=None):
    """
    Optimize acquisition function using Adam with bound projection.
    
    Args:
        acqf: The acquisition function
        bounds: Tensor [2, num_features]
        config: Optimization configuration
        initial_guess: Optional initial guess [1, 1, num_features]
        
    Returns:
        tuple: (candidate, acqf_value)
    """
    num_restarts = config.get("num_restarts", 20)
    raw_samples = config.get("raw_samples", 1048)
    num_steps = config.get("num_steps", 150)
    learning_rate = config.get("learning_rate", 0.01)
    
    # Generate random initial conditions
    random_initial_conditions = gen_batch_initial_conditions(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples
    )
    
    # Combine with initial guess if provided
    if initial_guess is not None:
        all_initial_conditions = torch.cat(
            [initial_guess, random_initial_conditions],
            dim=0
        )
    else:
        all_initial_conditions = random_initial_conditions
    
    # Set up the tensor to be optimized
    candidates = all_initial_conditions.clone().detach().requires_grad_(True)
    
    # Instantiate the Adam optimizer
    optimizer = optim.Adam([candidates], lr=learning_rate)
    
    # Optimization loop
    for i in range(num_steps):
        optimizer.zero_grad()
        
        # Get acquisition function values for all restart candidates
        acq_values = acqf(candidates)
        
        # We want to MAXIMIZE acq_values, so we MINIMIZE its negative
        loss = -acq_values.sum()
        
        # Backpropagate
        loss.backward()
        
        # Take an optimization step
        optimizer.step()
        
        # Project candidates back into the bounds
        with torch.no_grad():
            candidates.data = torch.max(
                torch.min(candidates.data, bounds[1]), 
                bounds[0]
            )
    
    # Get the best candidate found across all restarts
    with torch.no_grad():
        final_acq_values = acqf(candidates)
    
    best_idx = final_acq_values.argmax()
    
    # Select the best candidate
    candidate = candidates[best_idx].unsqueeze(0)  # Shape [1, 1, num_features]
    acqf_value = final_acq_values[best_idx].unsqueeze(0)  # Shape [1]
    
    # Squeeze to match standard output format
    candidate = candidate.squeeze(0)  # Shape [1, num_features]
    
    return candidate, acqf_value


def _optimize_lbfgs(acqf, bounds, config, initial_guess=None):
    """
    Optimize acquisition function using L-BFGS-B.
    
    Note: This is a placeholder for future implementation.
    L-BFGS-B can be more efficient than Adam for smooth landscapes.
    
    Args:
        acqf: The acquisition function
        bounds: Tensor [2, num_features]
        config: Optimization configuration
        initial_guess: Optional initial guess [1, 1, num_features]
        
    Returns:
        tuple: (candidate, acqf_value)
    """
    raise NotImplementedError("L-BFGS-B optimization not yet implemented. Use 'adam' for now.")


# Placeholder for future particle swarm implementation
# def _optimize_particle_swarm(acqf, bounds, config, initial_guess=None):
#     """
#     Optimize acquisition function using Particle Swarm Optimization.
#     
#     This is a placeholder for future implementation.
#     PSO can be useful for highly multimodal acquisition functions.
#     
#     Args:
#         acqf: The acquisition function
#         bounds: Tensor [2, num_features]
#         config: Optimization configuration (should include PSO-specific params)
#         initial_guess: Optional initial guess
#         
#     Returns:
#         tuple: (candidate, acqf_value)
#     """
#     raise NotImplementedError("Particle swarm optimization not yet implemented.")


def create_bounds_tensor(databank, atom_ids, device):
    """
    Create bounds tensor for optimization based on occupation matrix constraints.
    
    Diagonal elements are bounded [0, 1], off-diagonal elements [-0.5, 0.5].
    
    Args:
        databank: DataBank instance
        atom_ids: List of atom IDs to include
        device: Device to use (CPU or CUDA)
        
    Returns:
        Tensor of shape [2, num_features] with [lower_bounds, upper_bounds]
    """
    # Build index map using private method
    index_map = databank._build_flat_index_map(atom_ids, spins=['up', 'down'])
    
    bounds = []
    for (atom, spin, i, j) in index_map['reverse_map']:
        if i == j:
            # Diagonal: [0, 1]
            bounds.append((0.0, 1.0))
        else:
            # Off-diagonal: [-0.5, 0.5]
            bounds.append((-0.5, 0.5))
    
    bounds = torch.tensor(bounds, device=device)
    # Reshape to [2, num_features]
    bounds = torch.stack((bounds[:, 0], bounds[:, 1]), dim=0)
    
    return bounds

## initialization routines


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
    
    # Step 4: Optionally apply random rotation to each atom
    # Rotating the matrices gives a huge overhead if a lot of candidates are created
    # since it is not done in pytorch, but passes through numpy
    # if apply_rotation and (random.random() < rotation_prob):
    #     for atom_id in atoms:
            # # Generate random rotation parameters
            # angle = np.random.uniform(0, 2 * np.pi)
            # direction = uniform_direction.rvs(3)
            
            # # Get matrices and convert to numpy
            # up_mat = np.array(patchwork_occ_final.get_occupation_matrix(atom_id, 'up'))
            # down_mat = np.array(patchwork_occ_final.get_occupation_matrix(atom_id, 'down'))
            # # print(direction)
            
            # # Apply rotation
            # up_mat_rotated = rotate_QE_matrix(up_mat, angle, direction)
            # down_mat_rotated = rotate_QE_matrix(down_mat, angle, direction)
            
            # # Update the data (store real parts for collinear calculations)
            # patchwork_data[atom_id]['occupation_matrix']['up'] = up_mat_rotated.real.tolist()
            # patchwork_data[atom_id]['occupation_matrix']['down'] = down_mat_rotated.real.tolist()

            # Alternative: use random SO(3) matrix for rotation
            # R = special_ortho_group.rvs(3)
            # # Get matrices and convert to numpy
            # up_mat = np.array(patchwork_occ_final.get_occupation_matrix(atom_id, 'up'))
            # down_mat = np.array(patchwork_occ_final.get_occupation_matrix(atom_id, 'down'))
            # # Apply rotation
            # up_mat_rotated = R @ up_mat @ R.T
            # down_mat_rotated = R @ down_mat @ R.T

            # # Update the data (store real parts for collinear calculations)
            # patchwork_data[atom_id]['occupation_matrix']['up'] = up_mat_rotated.real.tolist()
            # patchwork_data[atom_id]['occupation_matrix']['down'] = down_mat_rotated.real.tolist()

        
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
