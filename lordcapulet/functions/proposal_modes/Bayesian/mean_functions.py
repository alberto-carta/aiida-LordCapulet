"""
Custom mean functions with physics-based priors.

This module implements mean functions that encode physical knowledge
about the energy landscape of occupation matrices, including:
- Magnetization contributions (Heisenberg-like J*M^2 terms)
- Hubbard U corrections (orbital correlation penalties)
"""

import torch
from gpytorch.means import Mean
from gpytorch.priors import NormalPrior
from gpytorch.constraints import Interval


class VectorizedPhysicsMean(Mean):
    """
    A fully-vectorized, atom-agnostic custom mean function.
    
    It accepts a list of atom IDs and pre-computes index maps
    for all of them. The forward pass sums the physics contributions
    from all specified atoms.
    
    Implements:
        Total_Mean = sum_i [ -(J/4) * M_i^2 + J_lin * |M_i| ]
                   + sum_i,spin [ U * tr(n_i_spin * (1 - n_i_spin)) ]
               
    where the sum 'i' is over all atoms in `atom_ids`.
    
    Args:
        databank: DataBank instance with flat_index_rev attribute
        atom_ids: List of atom IDs to include in the mean function
        J_prior_mean: Mean of the prior on J (default: 0.5)
        J_prior_std: Standard deviation of the prior on J (default: 0.2)
        J_lin_prior_mean: Mean of the prior on J_lin (default: 0.1)
        J_lin_prior_std: Standard deviation of the prior on J_lin (default: 0.05)
        U_prior_mean: Mean of the prior on U (default: 4.5 eV)
        U_prior_std: Standard deviation of the prior on U (default: 1.0 eV)
    """
    
    def __init__(self, databank, atom_ids: list,
                 J_prior_mean: float = 0.5,
                 J_prior_std: float = 0.2,
                #  J_lin_prior_mean: float = 0.1,
                #  J_lin_prior_std: float = 0.05,
                 U_prior_mean: float = 4.5,
                 U_prior_std: float = 1.0,
                 constant_mean: float = 0.0):
        super().__init__()
        
        # Store atom list
        self.atom_ids = atom_ids
        
        # Register learnable parameters initialized at prior means (ensure float dtype)
        self.register_parameter('J', torch.nn.Parameter(torch.tensor(J_prior_mean, dtype=torch.float32)))
        # self.register_parameter('J_lin', torch.nn.Parameter(torch.tensor(J_lin_prior_mean, dtype=torch.float32)))
        self.register_parameter('U', torch.nn.Parameter(torch.tensor(U_prior_mean, dtype=torch.float32)))
        self.register_parameter('constant', torch.nn.Parameter(torch.tensor(constant_mean, dtype=torch.float32)))
        
        # Add normal priors on all parameters
        self.register_prior(
            'J_prior',
            NormalPrior(J_prior_mean, J_prior_std),
            lambda m: m.J,
            lambda m, v: m._set_J(v)
        )
        # self.register_prior(
        #     'J_lin_prior',
        #     NormalPrior(J_lin_prior_mean, J_lin_prior_std),
        #     lambda m: m.J_lin,
        #     lambda m, v: m._set_J_lin(v)
        # )
        self.register_prior(
            'U_prior',
            NormalPrior(U_prior_mean, U_prior_std),
            lambda m: m.U,
            lambda m, v: m._set_U(v)
        )
        
        # Add constraints to keep parameters in reasonable ranges
        self.register_constraint('J', Interval(0.0, 2.5))
        # self.register_constraint('J_lin', Interval(-0.5, 0.5))
        self.register_constraint('U', Interval(0.0, 15.0))

        # Pre-computation: Build index maps
        idx_map = {}
        spins = ['up', 'down']
        for atom in self.atom_ids:
            for spin in spins:
                idx_map[(atom, spin)] = {'diag': [], 'off_diag': []}

        # Build index map from DataBank
        index_map = databank._build_flat_index_map(self.atom_ids, spins)
        
        # This is the only slow loop, and it only runs once
        for idx, (atom, spin, i, j) in enumerate(index_map['reverse_map']):
            if atom not in self.atom_ids:
                continue
            
            key = (atom, spin)
            if i == j:
                idx_map[key]['diag'].append(idx)
            else:
                # We assume the reverse_map only contains
                # one entry for each (i, j) pair (upper triangle)
                idx_map[key]['off_diag'].append(idx)
        
        # Dynamically register index tensors as buffers
        for atom in self.atom_ids:
            for spin in spins:
                diag_indices = idx_map[(atom, spin)]['diag']
                off_diag_indices = idx_map[(atom, spin)]['off_diag']
                
                # Register buffer for diagonal indices
                self.register_buffer(
                    f'diag_idx_{atom}_{spin}', 
                    torch.tensor(diag_indices, dtype=torch.long)
                )
                # Register buffer for off-diagonal indices
                self.register_buffer(
                    f'off_diag_idx_{atom}_{spin}',
                    torch.tensor(off_diag_indices, dtype=torch.long)
                )

    def _compute_tr_and_tr_n_sq(self, X_batch, diag_idx, off_diag_idx):
        """
        Vectorized helper to compute trace(n) and trace(n@n).
        
        Args:
            X_batch: Batch of flattened occupation matrices [batch_size, num_features]
            diag_idx: Indices of diagonal elements
            off_diag_idx: Indices of off-diagonal elements
            
        Returns:
            tuple: (trace(n), trace(n@n)) for each batch element
        """
        # Handle empty matrices
        if len(diag_idx) == 0 and len(off_diag_idx) == 0:
            zeros = torch.zeros(X_batch.shape[0], device=X_batch.device, dtype=X_batch.dtype)
            return zeros, zeros
            
        diag_elems = X_batch.index_select(-1, diag_idx)
        off_diag_elems = X_batch.index_select(-1, off_diag_idx)

        tr_n = diag_elems.sum(dim=-1)
        
        sum_sq_diag = (diag_elems**2).sum(dim=-1)
        sum_sq_off_diag = (off_diag_elems**2).sum(dim=-1)
        tr_n_sq = sum_sq_diag + 2.0 * sum_sq_off_diag

        return tr_n, tr_n_sq

    def forward(self, X_batch):
        """
        Fully-vectorized forward pass.
        Loops over the (small) list of atoms, not the (large) batch.
        
        Args:
            X_batch: Batch of flattened occupation matrices
            
        Returns:
            Mean function values for each batch element
        """
        original_shape = X_batch.shape[:-1]
        if X_batch.ndim > 2:
            X_batch = X_batch.reshape(-1, X_batch.shape[-1])
        
        batch_size = X_batch.shape[0]
        
        # Initialize total contributions
        total_M_contrib = torch.zeros(batch_size, device=X_batch.device, dtype=X_batch.dtype)
        total_U_contrib = torch.zeros(batch_size, device=X_batch.device, dtype=X_batch.dtype)
        
        # Loop over all atoms
        for atom in self.atom_ids:
            # Retrieve this atom's buffers by name
            diag_idx_up = getattr(self, f'diag_idx_{atom}_up')
            off_diag_idx_up = getattr(self, f'off_diag_idx_{atom}_up')
            diag_idx_down = getattr(self, f'diag_idx_{atom}_down')
            off_diag_idx_down = getattr(self, f'off_diag_idx_{atom}_down')
            
            # Compute traces
            tr_up, tr_n_sq_up = self._compute_tr_and_tr_n_sq(X_batch, diag_idx_up, off_diag_idx_up)
            tr_down, tr_n_sq_down = self._compute_tr_and_tr_n_sq(X_batch, diag_idx_down, off_diag_idx_down)

            # Part 1: Magnetization contribution (M)
            M_atom = tr_up - tr_down
            total_M_contrib += -(self.J / 4.0) * M_atom.pow(2) #+ self.J_lin * M_atom.abs()
            
            # Part 2: Hubbard U contribution
            # U/2 * tr(n * (1-n)) = U/2 * (tr(n) - tr(n^2))
            contrib_U_up = self.U/2 * (tr_up - tr_n_sq_up)
            contrib_U_down = self.U/2 * (tr_down - tr_n_sq_down)
            total_U_contrib += contrib_U_up + contrib_U_down
        
        # Total mean (constant handles overall energy scale)
        final_mean = self.constant + total_M_contrib + total_U_contrib
        
        # Reshape to original batch shape
        return final_mean.reshape(original_shape)
    
    def _set_J(self, value):
        """Helper method for prior registration."""
        self.J.data.copy_(value)
    
    # def _set_J_lin(self, value):
    #     """Helper method for prior registration."""
    #     self.J_lin.data.copy_(value)
    
    def _set_U(self, value):
        """Helper method for prior registration."""
        self.U.data.copy_(value)


class MagnetizationMean(Mean):
    """
    Custom mean function implementing only the magnetization-based trend:
    Total_Mean = sum_i [ -(J/4) * M_i^2 + J_lin * |M_i| ]
    
    where M_i = trace(up_i) - trace(down_i)
    
    Args:
        databank: DataBank instance with unflatten_vector_to_matrices method
        atom_ids: List of atom IDs to include
        J_init: Initial value for J parameter
        J_lin_init: Initial value for J_lin parameter
    """
    
    def __init__(self, databank, atom_ids: list, J_init: float = 1.0, J_lin_init: float = 0.1):
        super().__init__()
        self.databank = databank
        self.atom_ids = atom_ids
        
        self.J = torch.nn.Parameter(torch.tensor(J_init))
        self.J_lin = torch.nn.Parameter(torch.tensor(J_lin_init))

    def _compute_mean(self, x_entry):
        """Helper function to process a single 1D x_entry."""
        try:
            total_contrib = torch.tensor(0.0, device=x_entry.device, dtype=x_entry.dtype)
            
            for atom_id in self.atom_ids:
                mats = self.databank.unflatten_vector_to_matrices(x_entry, arraytype='pytorch')[atom_id]
                M = torch.trace(mats['up']) - torch.trace(mats['down'])
                contrib = -(self.J / 4.0) * M.pow(2) + self.J_lin * M.abs()
                total_contrib += contrib
            
            return total_contrib
        
        except Exception as e:
            print(f"Warning in MagnetizationMean calculation: {e}")
            return torch.tensor(0.0, device=x_entry.device, dtype=x_entry.dtype)

    def forward(self, X_batch):
        """Forward pass for batch of inputs."""
        batch_shape = X_batch.shape[:-1]
        n_features = X_batch.shape[-1]
        X_flat = X_batch.reshape(-1, n_features)
        mean_values = [self._compute_mean(x) for x in X_flat]
        return torch.stack(mean_values).reshape(batch_shape)


class HubbardUMean(Mean):
    """
    Custom mean function for only the Hubbard U term:
    sum_i,spin [ U * tr(n_i_spin * (1 - n_i_spin)) ]
    
    Args:
        databank: DataBank instance with unflatten_vector_to_matrices method
        atom_ids: List of atom IDs to include
        U_init: Initial value for U parameter
    """
    
    def __init__(self, databank, atom_ids: list, U_init: float = 1.0):
        super().__init__()
        self.databank = databank
        self.atom_ids = atom_ids
        self.U = torch.nn.Parameter(torch.tensor(U_init))

    def _compute_U_contrib(self, n):
        """Compute U * tr(n * (1-n)) for a single matrix."""
        if n.shape[0] == 0: 
            return torch.tensor(0.0, device=n.device, dtype=n.dtype)
        I = torch.eye(n.shape[0], device=n.device, dtype=n.dtype)
        n_times_I_minus_n = n @ (I - n)
        return self.U / 2 * torch.trace(n_times_I_minus_n)

    def _compute_mean(self, x_entry):
        """Helper function to process a single 1D x_entry."""
        try:
            total_contrib = torch.tensor(0.0, device=x_entry.device, dtype=x_entry.dtype)
            
            for atom_id in self.atom_ids:
                mats = self.databank.unflatten_vector_to_matrices(x_entry, arraytype='pytorch')[atom_id]
                contrib_up = self._compute_U_contrib(mats['up'])
                contrib_down = self._compute_U_contrib(mats['down'])
                total_contrib += contrib_up + contrib_down
            
            return total_contrib
            
        except Exception:
            return torch.tensor(0.0, device=x_entry.device, dtype=x_entry.dtype)

    def forward(self, X_batch):
        """Forward pass for batch of inputs."""
        batch_shape = X_batch.shape[:-1]
        n_features = X_batch.shape[-1]
        X_flat = X_batch.reshape(-1, n_features)
        mean_values = [self._compute_mean(x) for x in X_flat]
        return torch.stack(mean_values).reshape(batch_shape)


class SumMean(Mean):
    """
    A Mean that returns the elementwise sum of two child means.
    
    Both child means must accept the same input x shape and return 
    compatible output shapes.
    
    Args:
        mean1: First mean function
        mean2: Second mean function
    """
    
    def __init__(self, mean1: Mean, mean2: Mean):
        super().__init__()
        self.mean1 = mean1
        self.mean2 = mean2

    def forward(self, x):
        """Elementwise sum of the two mean outputs."""
        return self.mean1(x) + self.mean2(x)
