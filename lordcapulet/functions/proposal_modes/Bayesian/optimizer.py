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



