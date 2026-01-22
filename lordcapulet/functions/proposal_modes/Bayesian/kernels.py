"""
Custom kernel functions for occupation matrix GP models.

This module provides:
- SpinFlipInvariantKernel: Wrapper that enforces spin-flip symmetry
- Flexible kernel construction utilities for building complex kernels
- Helper functions for extracting atom-specific indices
"""

import torch
from gpytorch.kernels import (
    Kernel, MaternKernel, LinearKernel, PolynomialKernel,
    ScaleKernel, AdditiveKernel, ProductKernel
)
from itertools import combinations
from .priors import get_botorch_lengthscale_prior, get_outputscale_prior


def get_spin_indices(databank, atom_id):
    """
    Extract indices for up and down spin channels for a specific atom.
    
    Args:
        databank: DataBank instance
        atom_id: Atom identifier
        
    Returns:
        tuple: (up_indices, down_indices) as torch.Tensors
    """
    # Get the reverse index map from databank
    index_map = databank._build_flat_index_map([atom_id], spins=['up', 'down'])
    
    up_indices = []
    down_indices = []
    for idx, (atom, spin, i, j) in enumerate(index_map['reverse_map']):
        if atom == atom_id:
            if spin == 'up':
                up_indices.append(idx)
            elif spin == 'down':
                down_indices.append(idx)
    return torch.tensor(up_indices, dtype=torch.long), torch.tensor(down_indices, dtype=torch.long)


def get_all_indices_for_atom(databank, atom_id):
    """
    Get all indices that involve a given atom_id (both spins).
    
    Args:
        databank: DataBank instance
        atom_id: Atom identifier
        
    Returns:
        torch.Tensor of indices
    """
    # Get the reverse index map for all atoms
    all_atom_ids = databank.atom_ids if hasattr(databank, 'atom_ids') else [atom_id]
    index_map = databank._build_flat_index_map(all_atom_ids, spins=['up', 'down'])
    
    indices = []
    for idx, (atom, spin, i, j) in enumerate(index_map['reverse_map']):
        if atom == atom_id:
            indices.append(idx)
    return torch.tensor(indices, dtype=torch.long)


class SpinFlipInvariantKernel(Kernel):
    """
    A wrapper kernel that enforces spin-flip invariance.
    
    Given a base kernel k(x1, x2), this computes:
        K_inv(x1, x2) = 0.25 * [k(x1, x2) + k(x1_f, x2) + k(x1, x2_f) + k(x1_f, x2_f)]
    
    where x_f denotes the spin-flipped version of x.
    
    This ensures the kernel is invariant to simultaneous spin flips,
    which is a physical symmetry for many magnetic systems.
    
    Args:
        base_kernel: The underlying kernel to wrap
        up_indices: Indices corresponding to spin-up components
        down_indices: Indices corresponding to spin-down components
    """
    
    is_stationary = False
    
    def __init__(self, base_kernel: Kernel, up_indices: torch.Tensor, down_indices: torch.Tensor, **kwargs):
        _has_lengthscale = base_kernel.has_lengthscale
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.has_lengthscale = _has_lengthscale
        
        if not isinstance(up_indices, torch.Tensor):
            up_indices = torch.tensor(up_indices, dtype=torch.long)
        if not isinstance(down_indices, torch.Tensor):
            down_indices = torch.tensor(down_indices, dtype=torch.long)
            
        self.register_buffer("up_indices", up_indices)
        self.register_buffer("down_indices", down_indices)
        
        if len(self.up_indices) != len(self.down_indices):
            raise ValueError("up_indices and down_indices must have the same length")

    def _swap_spins(self, X: torch.Tensor) -> torch.Tensor:
        """Swap up and down spin components."""
        X_flipped = X.clone()
        X_flipped[..., self.up_indices] = X[..., self.down_indices]
        X_flipped[..., self.down_indices] = X[..., self.up_indices]
        return X_flipped

    def forward(self, x1, x2, diag=False, **params):
        """Compute the spin-flip invariant kernel."""
        # Create the four versions of the inputs
        x1_flipped = self._swap_spins(x1)
        x2_flipped = self._swap_spins(x2)
        
        # Calculate the four covariance matrices
        # Call the kernel as a function, not .forward()
        k_1_2 = self.base_kernel(x1, x2, diag=diag, **params)
        k_1f_2 = self.base_kernel(x1_flipped, x2, diag=diag, **params)
        k_1_2f = self.base_kernel(x1, x2_flipped, diag=diag, **params)
        k_1f_2f = self.base_kernel(x1_flipped, x2_flipped, diag=diag, **params)
        
        # Return the average
        return 0.25 * (k_1_2 + k_1f_2 + k_1_2f + k_1f_2f)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)


def build_kernel(databank, atom_ids, kernel_config):
    """
    Flexible kernel constructor that builds complex kernels based on configuration.
    
    This function creates an additive kernel combining:
    - Local kernels (Matern, Linear, Polynomial) for each atom
    - Non-local kernels (products of kernels) for atom pairs
    - Optional spin-flip invariance
    
    Args:
        databank: DataBank instance with flat_index_rev attribute
        atom_ids: List of atom IDs to include
        kernel_config: Dictionary specifying which kernels to include and their priors
        
    Returns:
        AdditiveKernel combining all requested kernel components
        
    Example kernel_config:
        {
            "local": {
                "matern": {"enabled": True, "nu": 2.5, "outputscale_prior": {"mean": 2.0, "std": 0.1}},
                "linear": {"enabled": True, "outputscale_prior": {"mean": 0.3, "std": 0.05}},
                "polynomial": {"enabled": True, "power": 2, "outputscale_prior": {"mean": 0.3, "std": 0.05}},
            },
            "nonlocal": {
                "heisenberg": {"enabled": False, "outputscale_prior": {"mean": 0.2, "std": 0.05}},
                "kugel_khomskii": {"enabled": False, "outputscale_prior": {"mean": 0.2, "std": 0.05}},
                "residual": {"enabled": True, "outputscale_prior": {"mean": 0.2, "std": 0.05}},
            },
            "spin_flip_invariant": True,
        }
    """
    ker_list = []
    
    # --- Build LOCAL Kernels (ANOVA-1) ---
    local_config = kernel_config.get("local", {})
    spin_flip_enabled = kernel_config.get("spin_flip_invariant", True)
    
    for atom_id in atom_ids:
        up_idcs, down_idcs = get_spin_indices(databank, atom_id)
        atom_all_idcs = get_all_indices_for_atom(databank, atom_id)
        num_local_features = len(atom_all_idcs)
        
        if num_local_features == 0:
            continue
        
        # Matern kernel
        if local_config.get("matern", {}).get("enabled", False):
            matern_cfg = local_config["matern"]
            ls_prior, ls_constraint = get_botorch_lengthscale_prior(num_local_features)
            
            matern_base = MaternKernel(
                nu=matern_cfg.get("nu", 2.5),
                active_dims=atom_all_idcs,
                ard_num_dims=num_local_features,
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint
            )
            
            if spin_flip_enabled:
                matern_base = SpinFlipInvariantKernel(matern_base, up_idcs, down_idcs)
            
            outputscale_prior = get_outputscale_prior(**matern_cfg["outputscale_prior"])
            ker_list.append(ScaleKernel(matern_base, outputscale_prior=outputscale_prior))
        
        # Linear kernel
        if local_config.get("linear", {}).get("enabled", False):
            linear_cfg = local_config["linear"]
            
            linear_base = LinearKernel(
                active_dims=atom_all_idcs,
                ard_num_dims=num_local_features,
                variance_prior=None
            )
            
            if spin_flip_enabled:
                linear_base = SpinFlipInvariantKernel(linear_base, up_idcs, down_idcs)
            
            outputscale_prior = get_outputscale_prior(**linear_cfg["outputscale_prior"])
            ker_list.append(ScaleKernel(linear_base, outputscale_prior=outputscale_prior))
        
        # Polynomial kernel
        if local_config.get("polynomial", {}).get("enabled", False):
            poly_cfg = local_config["polynomial"]
            
            poly_base = PolynomialKernel(
                power=poly_cfg.get("power", 2),
                ard_num_dims=num_local_features,
                active_dims=atom_all_idcs,
            )
            
            if spin_flip_enabled:
                poly_base = SpinFlipInvariantKernel(poly_base, up_idcs, down_idcs)
            
            outputscale_prior = get_outputscale_prior(**poly_cfg["outputscale_prior"])
            ker_list.append(ScaleKernel(poly_base, outputscale_prior=outputscale_prior))
    
    # --- Build NON-LOCAL Kernels (ANOVA-2) ---
    nonlocal_config = kernel_config.get("nonlocal", {})
    
    for atom_id_1, atom_id_2 in combinations(atom_ids, 2):
        atom_1_all_idcs = get_all_indices_for_atom(databank, atom_id_1)
        atom_2_all_idcs = get_all_indices_for_atom(databank, atom_id_2)
        num_local_1 = len(atom_1_all_idcs)
        num_local_2 = len(atom_2_all_idcs)

        if num_local_1 == 0 or num_local_2 == 0:
            continue

        # Define base kernels for atom 1
        ls_prior_1, ls_constraint_1 = get_botorch_lengthscale_prior(num_local_1)
        mat_1 = MaternKernel(
            nu=2.5, 
            active_dims=atom_1_all_idcs, 
            ard_num_dims=num_local_1, 
            lengthscale_prior=ls_prior_1, 
            lengthscale_constraint=ls_constraint_1
        )
        lin_1 = LinearKernel(active_dims=atom_1_all_idcs)
        poly_1 = PolynomialKernel(power=2, active_dims=atom_1_all_idcs, ard_num_dims=num_local_1)

        # Define base kernels for atom 2
        ls_prior_2, ls_constraint_2 = get_botorch_lengthscale_prior(num_local_2)
        mat_2 = MaternKernel(
            nu=2.5, 
            active_dims=atom_2_all_idcs, 
            ard_num_dims=num_local_2,
            lengthscale_prior=ls_prior_2, 
            lengthscale_constraint=ls_constraint_2
        )
        lin_2 = LinearKernel(active_dims=atom_2_all_idcs)
        poly_2 = PolynomialKernel(power=2, active_dims=atom_2_all_idcs, ard_num_dims=num_local_2)

        # Heisenberg term (Linear x Linear)
        if nonlocal_config.get("heisenberg", {}).get("enabled", False):
            heisenberg_cfg = nonlocal_config["heisenberg"]
            outputscale_prior = get_outputscale_prior(**heisenberg_cfg["outputscale_prior"])
            heisenberg_kernel = ScaleKernel(
                ProductKernel(lin_1, lin_2),
                outputscale_prior=outputscale_prior
            )
            ker_list.append(heisenberg_kernel)

        # Kugel-Khomskii term (Poly x Poly)
        if nonlocal_config.get("kugel_khomskii", {}).get("enabled", False):
            kk_cfg = nonlocal_config["kugel_khomskii"]
            outputscale_prior = get_outputscale_prior(**kk_cfg["outputscale_prior"])
            kk_kernel = ScaleKernel(
                ProductKernel(poly_1, poly_2),
                outputscale_prior=outputscale_prior
            )
            ker_list.append(kk_kernel)

        # Residual term (Matern x Matern)
        if nonlocal_config.get("residual", {}).get("enabled", False):
            residual_cfg = nonlocal_config["residual"]
            outputscale_prior = get_outputscale_prior(**residual_cfg["outputscale_prior"])
            residual_kernel = ScaleKernel(
                ProductKernel(mat_1, mat_2),
                outputscale_prior=outputscale_prior
            )
            ker_list.append(residual_kernel)
    
    # Build the final kernel
    if len(ker_list) == 0:
        raise ValueError("No kernels were enabled in the configuration")
    
    return AdditiveKernel(*ker_list)
