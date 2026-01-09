"""
Custom acquisition functions with physics-based constraints.

This module provides acquisition functions that incorporate physical
constraints on occupation matrices, such as:
- Trace constraints (target electron counts)
- Principal minor constraints (positive semi-definite condition via 2x2 minors)
"""

import torch
import numpy as np
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from collections import defaultdict


def prepare_eigenvalue_indices(forward_map):
    """
    Parses the forward_map from the DataBank to create batch indices for efficient
    reconstruction of symmetric matrices.

    Importantly this batches together all matrices contained in a single flat feature vector
    and not batches of different feature vectors.

    Returns a dictionary where keys are:
    - Matrix dimension N (e.g., 5 for d-orbitals)
    Values are dictionaries with:
    - "B": Tensor of batch indices
    - "R": Tensor of row indices
    - "C": Tensor of column indices
    - "Src": Tensor of source indices in the flat feature vector
    - "batch_shape": Tuple indicating the shape of the batch (num_matrices, N, N)

    # The indices have this logic, simple example for 2x2 matrices:
    # B: [0, 0, 0, 0, 1, 1, 1, 1]
    # R: [0, 0, 1, 1, 0, 0, 1, 1]
    # C: [0, 1, 0, 1, 0, 1, 0, 1]
    # Src: [0, 1, 1, 2, 3, 4, 4, 5]
    
    This means:
    - For batch 0, element (0,0) comes from flat index 0
    - For batch 0, element (0,1) comes from flat index 1
    - For batch 0, element (1,0) also comes from flat index 1 (symmetric)
    - For batch 0, element (1,1) comes from flat index 2
    - For batch 1, element (0,0) comes from flat index 3
    - etc.

    """
    # 1. Group the flat map by (atom, spin) blocks
    blocks = defaultdict(list)
    for (atom, spin, row, col), flat_idx in forward_map.items():
        blocks[(atom, spin)].append((row, col, flat_idx))

    # 2. Group blocks by their matrix dimension (N)
    #    We assume the max index found + 1 is the dimension.
    batches_by_size = defaultdict(list)
    
    for (atom, spin), entries in blocks.items():
        # Find dimension N for this block (e.g., 5 for d-orbitals)
        max_idx = max(max(r, c) for r, c, _ in entries)
        N = max_idx + 1
        batches_by_size[N].append(entries)

    # 3. Create PyTorch indexing tensors for each size group
    prepared_batches = {}
    
    for N, block_list in batches_by_size.items():
        num_matrices = len(block_list)
        
        # Lists to hold the coordinate data
        b_indices = [] # Batch dimension
        r_indices = [] # Row dimension
        c_indices = [] # Col dimension
        src_indices = [] # Source index in the flat feature vector
        
        for batch_idx, entries in enumerate(block_list):
            for (row, col, flat_idx) in entries:
                # Add (i, j)
                b_indices.append(batch_idx)
                r_indices.append(row)
                c_indices.append(col)
                src_indices.append(flat_idx)
                
                # If off-diagonal, also add (j, i) pointing to same flat_idx
                if row != col:
                    b_indices.append(batch_idx)
                    r_indices.append(col) # Swapped
                    c_indices.append(row) # Swapped
                    src_indices.append(flat_idx)
                    
        # Convert to tensors for fast indexing later
        prepared_batches[N] = {
            "B": torch.tensor(b_indices, dtype=torch.long),
            "R": torch.tensor(r_indices, dtype=torch.long),
            "C": torch.tensor(c_indices, dtype=torch.long),
            "Src": torch.tensor(src_indices, dtype=torch.long),
            "batch_shape": (num_matrices, N, N)
        }
        
    return prepared_batches


def compute_eigenvalue_preference_batch(X_batch, prepared_batches, k=2000.0):
    """
    Computes eigenvalue preference for a BATCH of flat feature vectors.
    Fully vectorized - processes all samples and all matrices simultaneously.
    
    Args:
        X_batch: Batch of flattened occupation matrices [batch_size, num_features]
        prepared_batches: Index mapping from prepare_eigenvalue_indices
        k: Stiffness parameter for sigmoid penalty
        
    Returns:
        Tensor of preference scores [batch_size]
    """
    device = X_batch.device
    dtype = X_batch.dtype
    batch_size = X_batch.shape[0]
    
    # Initialize preferences for each sample
    total_pref = torch.ones(batch_size, device=device, dtype=dtype)
    
    # Iterate over matrix size groups (e.g., 5x5 for d-orbitals, 7x7 for f-orbitals)
    for N, indices in prepared_batches.items():
        num_matrices_per_sample = indices["batch_shape"][0]
        
        # Move indices to correct device
        B_idx = indices["B"].to(device)
        R_idx = indices["R"].to(device)
        C_idx = indices["C"].to(device)
        Src_idx = indices["Src"].to(device)
        
        # 1. Reconstruct all matrices for all samples at once
        # Shape: [batch_size, num_matrices_per_sample, N, N]
        batch_matrices = torch.zeros(
            (batch_size, num_matrices_per_sample, N, N),
            device=device,
            dtype=dtype
        )
        
        # Fill matrices for each batch element
        # We need to iterate over assignments but vectorize across batch dimension
        # Extract indices as Python integers to avoid tensor indexing issues
        for idx in range(len(Src_idx)):
            b = B_idx[idx].item()
            r = R_idx[idx].item()
            c = C_idx[idx].item()
            src = Src_idx[idx].item()
            batch_matrices[:, b, r, c] = X_batch[:, src]
        
        # 2. Reshape for batch eigenvalue computation
        # Merge batch and matrix dimensions: [batch_size * num_matrices_per_sample, N, N]
        flat_matrices = batch_matrices.reshape(-1, N, N)
        
        # 3. Compute eigenvalues for all matrices at once
        # Returns: [batch_size * num_matrices_per_sample, N]
        evals = torch.linalg.eigvalsh(flat_matrices)
        
        # 4. Apply constraints
        min_evals = evals[:, 0]   # Smallest eigenvalue
        max_evals = evals[:, -1]  # Largest eigenvalue
        
        pref_positive = torch.sigmoid(k * min_evals)
        pref_bounded = torch.sigmoid(k * (1.0 - max_evals))
        
        # Combined preference for each matrix
        matrix_prefs = pref_positive * pref_bounded  # [batch_size * num_matrices_per_sample]
        
        # 5. Reshape back and aggregate per sample
        # [batch_size, num_matrices_per_sample]
        matrix_prefs = matrix_prefs.reshape(batch_size, num_matrices_per_sample)
        
        # Product across all matrices for this size group, per sample
        sample_prefs = torch.prod(matrix_prefs, dim=1)  # [batch_size]
        
        # Multiply into total preference
        total_pref *= sample_prefs
    
    return total_pref


def compute_minor_preference_offdiag_only(matrix, k=20.0):
    """
    Enforces sylvester's criterion (at least for 2x2 principal minors)
    Computes a preference score (0-1) for a matrix based *only* on its
    2x2 principal minors, using a smooth sigmoid penalty.
    
    This enforces: p_ii*p_jj - p_ij^2 >= 0 (for all i < j)
    
    It assumes the diagonal (p_ii) is already handled by the optimizer's 
    bounds (e.g., [0, 1]).
    
    Args:
        matrix: The input matrix (assumed symmetric)
        k: A "stiffness" parameter. Higher k = steeper penalty (default: 20.0)
        
    Returns:
        Preference score from 0 to 1
    """
    # Get the diagonal elements, p_ii
    diag = torch.diag(matrix)
    
    # Create a matrix of p_ii * p_jj products
    diag_outer = torch.outer(diag, diag)
    
    # Create a matrix of p_ij^2
    p_ij_squared = matrix**2
    
    # violation_matrix[i, j] = p_ii*p_jj - p_ij^2 and (1-p_ii)
    violation_matrix = diag_outer - p_ij_squared
    
    # We only care about the upper triangle (where i < j)
    n = matrix.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1, device=matrix.device)
    
    # Get the violations for just the i < j pairs
    off_diag_violations = violation_matrix[triu_indices[0], triu_indices[1]]
    
    # Calculate preference for all off-diagonal pairs
    pref_off_diag = torch.sigmoid(k * off_diag_violations)
    
    # Combine scores. All must be good.
    total_pref_off_diag = torch.prod(pref_off_diag)
    
    return total_pref_off_diag




def compute_trace_for_atom(X_batch, forward_map, atom_id, spin):
    """
    Efficiently compute trace for a specific atom and spin channel across entire batch.
    
    Args:
        X_batch: Batch of flattened matrices [batch_size, num_features]
        forward_map: Forward index map from databank
        atom_id: Atom identifier
        spin: Spin channel ('up' or 'down')
        
    Returns:
        Tensor of traces [batch_size]
    """
    # Find all diagonal elements for this (atom, spin) pair
    diagonal_indices = []
    
    for (atom, s, i, j), flat_idx in forward_map.items():
        if atom == atom_id and s == spin and i == j:
            diagonal_indices.append(flat_idx)
    
    if not diagonal_indices:
        return torch.zeros(X_batch.shape[0], device=X_batch.device, dtype=X_batch.dtype)
    
    # Extract diagonal elements and sum them (vectorized across batch)
    # X_batch[:, diagonal_indices] gives [batch_size, num_diagonal_elements]
    diagonal_elements = X_batch[:, diagonal_indices]
    traces = diagonal_elements.sum(dim=1)  # [batch_size]
    
    return traces


def compute_trace_preference(trace_vals, target, sigma, supergaussian_index=1):
    """
    Compute Gaussian preference for trace values.
    
    Works for both single values and batches of trace values.
    This creates a soft constraint that the trace should be near the target value.
    
    Args:
        trace_vals: Tensor of trace values (can be scalar or [batch_size])
        target: Target trace value (scalar)
        sigma: Width of Gaussian (scalar, smaller = tighter constraint)
        
    Returns:
        Preference scores from 0 to 1 (1 = trace exactly at target)
    """
    return torch.exp(-((trace_vals - target)**2 / (2 * sigma**2))**supergaussian_index)


def compute_total_preference_fast(X_batch,
                                databank,
                                atom_ids,
                                trace_target,
                                trace_sigma,
                                use_eigenvalue_preference=False,
                                eig_k=2000.0,
                                supergaussian_index=1):
    """
    Calculates preference scores for a BATCH of X vectors using vectorized operations.
    
    Args:
        X_batch: Batch of flattened occupation matrices [batch_size, num_features]
        databank: DataBank instance
        atom_ids: List of atom IDs to constrain
        trace_target: Target trace value per atom (electron count). 
                     Can be a single float (same for all atoms) or list of floats (one per atom)
        trace_sigma: Width of trace preference Gaussian per atom.
                    Can be a single float (same for all atoms) or list of floats (one per atom)
        use_eigenvalue_preference: If True, apply eigenvalue constraints (0 < λ < 1)
        eig_k: Stiffness parameter for eigenvalue preference (default: 2000.0)
        
    Returns:
        Tensor of preference scores [batch_size], one per batch element
        
    Example:
        >>> # Same target for all atoms
        >>> prefs = compute_total_preference_fast(X, db, ['Atom_1', 'Atom_2'], 8.0, 0.5)
        
        >>> # Different targets per atom
        >>> prefs = compute_total_preference_fast(X, db, ['Fe', 'O'], [6.0, 6.5], [0.3, 0.4])
    """
    device = X_batch.device
    dtype = X_batch.dtype
    batch_size = X_batch.shape[0]
    
    # Handle scalar vs list inputs for trace_target and trace_sigma
    if isinstance(trace_target, (int, float)):
        trace_targets = [float(trace_target)] * len(atom_ids)
    else:
        trace_targets = list(trace_target)
        assert len(trace_targets) == len(atom_ids), \
            f"trace_target list length ({len(trace_targets)}) must match atom_ids length ({len(atom_ids)})"
    
    if isinstance(trace_sigma, (int, float)):
        trace_sigmas = [float(trace_sigma)] * len(atom_ids)
    else:
        trace_sigmas = list(trace_sigma)
        assert len(trace_sigmas) == len(atom_ids), \
            f"trace_sigma list length ({len(trace_sigmas)}) must match atom_ids length ({len(atom_ids)})"
    
    # Get the forward map from databank (computed once)
    forward_map = databank.get_forward_index_map()
    
    # Initialize preference scores for the batch
    total_pref_trace = torch.ones(batch_size, device=device, dtype=dtype)
    total_pref_eigenvalue = torch.ones(batch_size, device=device, dtype=dtype)
    
    # 1. Compute trace preferences (vectorized across batch)
    for atom_idx, atom_id in enumerate(atom_ids):
        # Get traces for up and down spin (vectorized for entire batch)
        trace_up = compute_trace_for_atom(X_batch, forward_map, atom_id, 'up')
        trace_down = compute_trace_for_atom(X_batch, forward_map, atom_id, 'down')
        
        # Total trace for this atom across all batch elements
        total_trace = trace_up + trace_down  # [batch_size]
        
        # Compute trace preference for this atom (vectorized)
        target = trace_targets[atom_idx]
        sigma = trace_sigmas[atom_idx]
        pref_trace = compute_trace_preference(total_trace, target, sigma, supergaussian_index=supergaussian_index)  # [batch_size]
        
        total_pref_trace *= pref_trace
    
    # 2. Compute eigenvalue preferences if requested (fully vectorized)
    if use_eigenvalue_preference:
        # Prepare batch indices once for all samples
        prepared_batches = prepare_eigenvalue_indices(forward_map)
        
        # Process entire batch at once 
        total_pref_eigenvalue = compute_eigenvalue_preference_batch(X_batch, prepared_batches, k=eig_k)
    
    # 3. Combine preferences
    return total_pref_trace * total_pref_eigenvalue


class AnalyticCustomPreference(AnalyticAcquisitionFunction):
    """
    Multiplies a base acquisition function (e.g., LCB) by a custom
    preference score (from 0 to 1) calculated from X.
    
    This allows incorporating physics constraints into the acquisition
    function, guiding the optimizer toward physically valid regions.
    
    Args:
        model: The GP model
        base_acqf: The base acquisition function (e.g., UpperConfidenceBound)
        compute_preference_func: Function that computes preference scores
                                 Should accept X_batch and return scores
    """
    
    def __init__(self, model, base_acqf, compute_preference_func):
        super().__init__(model=model)
        self.base_acqf = base_acqf
        self.compute_pref = compute_preference_func

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """
        Compute the constrained acquisition function value.
        
        Args:
            X: Input tensor [batch_size, q, num_features] or [batch_size, 1, num_features]
            
        Returns:
            Acquisition values multiplied by preference scores
        """
        # X has shape [batch_size, q, num_features] where q is number of candidates
        # For batch optimization (q>1), we need to handle each candidate
        
        # 1. Get energy score from base acquisition function
        energy_score = self.base_acqf(X)
        
        # 2. Get preference scores for each candidate in the batch
        # Reshape X to [batch_size * q, num_features] for preference computation
        batch_size, q, num_features = X.shape
        X_flat = X.reshape(-1, num_features)  # [batch_size * q, num_features]
        
        pref_scores_flat = self.compute_pref(X_flat)  # [batch_size * q]
        
        # For q>1 batch optimization, we need the minimum preference across all q candidates
        # because a batch is only as good as its worst element
        pref_scores = pref_scores_flat.reshape(batch_size, q)  # [batch_size, q]
        pref_score = pref_scores.min(dim=-1, keepdim=True)[0]  # [batch_size, 1]
        
        # 3. Combine: The preference score "gates" the energy score
        return energy_score * pref_score.squeeze(-1)


class BatchedAcqFunc:
    """
    Wraps an acquisition function to evaluate inputs in mini-batches.
    This prevents OOM errors when processing large candidate pools.
    """
    def __init__(self, acq_func, batch_size):
        self.acq_func = acq_func
        self.batch_size = batch_size

    def __call__(self, X):
        # X usually has shape (num_candidates, batch_shape, q, d)
        # BoltzmannSampling moves candidates to the 0-th dimension.
        results = []
        
        for i in range(0, X.shape[0], self.batch_size):
            batch_X = X[i : i + self.batch_size]
            results.append(self.acq_func(batch_X))
        
        return torch.cat(results)
    
    # Forward attribute lookups to the original acq_func (e.g. for .model)
    def __getattr__(self, name):
        return getattr(self.acq_func, name)