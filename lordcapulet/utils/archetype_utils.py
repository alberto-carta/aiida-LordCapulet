import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import numpy as np
import pandas as pd

from scipy.optimize import nnls, lsq_linear
from scipy.linalg import eigh # Use eigh for symmetric matrices
from scipy.optimize import minimize # Import minimize
from sklearn.linear_model import Lasso



def project_psd(X):
    """
    Projects a matrix X onto the cone of positive semi-definite (PSD) matrices.
    
    1. Symmetrizes the matrix.
    2. Performs eigendecomposition.
    3. Clips negative eigenvalues to zero.
    4. Reconstructs the matrix.
    """
    # 1. Symmetrize
    X_sym = 0.5 * (X + X.T)
    
    # 2. Eigendecomposition
    # eigh is for symmetric/hermitian matrices and is more efficient
    eigvals, eigvecs = eigh(X_sym)
    
    # 3. Clip negative eigenvalues
    eigvals_clipped = np.maximum(eigvals, 0)
    
    # 4. Reconstruct the PSD matrix
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

# NOTE: use ArchetypeFitter.fit_psd_factorization and ArchetypeFitter.decompose_matrices
# directly. Wrappers have been intentionally removed to keep the API explicit.


class ArchetypeFitter:
    """
    Archetype fitting operations.
    """
    def __init__(self):
        # Core arrays - now atom-specific dictionaries
        self.archetypes = {}  # Dict[atom_id, np.ndarray] with shape (k, n, n) per atom
        self.weights_train = {}  # Dict[atom_id, np.ndarray] with shape (N_train, k) per atom - training subset weights
        self.weights_test = {}  # Dict[atom_id, np.ndarray] with shape (N_test, k) per atom - test subset weights
        self.weights_all = {}  # Dict[atom_id, np.ndarray] with shape (N_all, k) per atom - entire dataset weights
        self.matrices_all = {}  # Dict[atom_id, np.ndarray] with shape (N_all, n, n) per atom - entire dataset
        self.matrices_train = {}  # Dict[atom_id, np.ndarray] with shape (N_train, n, n) per atom
        self.matrices_test = {}  # Dict[atom_id, np.ndarray] with shape (N_test, n, n) per atom
        
        # Training metadata - now per atom
        self.k_values = {}  # Dict[atom_id, int] - number of archetypes per atom
        self.target_trace = {}  # Dict[atom_id, float] - target trace per atom
        self.n_orbitals = {}  # Dict[atom_id, int] - matrix dimension per atom
        
        # Training results - now per atom
        self.train_error = {}  # Dict[atom_id, float] - final training error per atom
        self.test_error = {}  # Dict[atom_id, float] - validation error per atom
        self.train_indices = {}  # Dict[atom_id, np.ndarray] - indices used for training per atom
        self.test_indices = {}  # Dict[atom_id, np.ndarray] - indices used for testing per atom
        self.convergence_iteration = {}  # Dict[atom_id, int] - convergence iteration per atom
        
        # Per-atom results (for multi-atom training)
        self.atom_results = {}  # Dict[atom_id, dict] storing per-atom fitting results
        
        # Training parameters - global for now, could be made per-atom if needed
        self.fit_params = {}  # Store parameters used during fitting

        self.retraining_threshold_factor = 3.0  # Default retraining threshold factor

    
    def fit_psd_factorization(self, C_data: np.ndarray, atom_id: int, k: int, target_trace: float = 1.0, 
                             n_iter: int = 1000, tol: float = 1e-5, use_lasso: bool = False, 
                             lasso_alpha: float = 0.01, starting_guess_archetypes: Union[str, np.ndarray] = 'random',
                             verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Performs Non-negative PSD Factorization with trace-normalized archetypes.
        
        Args:
            C_data: Dataset of shape (N, n, n)
            atom_id: ID of the atom being fitted
            k: Number of archetypes for this atom
            target_trace: The desired trace for each archetype
            n_iter: Maximum iterations
            tol: Convergence tolerance
            use_lasso: Whether to use Lasso regularization
            lasso_alpha: Lasso regularization parameter
            starting_guess_archetypes: 'random' or array of shape (k, n, n)
            verbose: Whether to print progress
            
        Returns:
            Tuple of (weights, archetypes, final_error, convergence_iteration)
        """
        # Store fitting parameters (global for now)
        self.fit_params = {
            'k': k,
            'target_trace': target_trace,
            'n_iter': n_iter,
            'tol': tol,
            'use_lasso': use_lasso,
            'lasso_alpha': lasso_alpha,
            'verbose': verbose
        }
        
        # Store metadata per atom
        self.k_values[atom_id] = k
        self.target_trace[atom_id] = target_trace
        self.n_orbitals[atom_id] = C_data.shape[1]
        N, n, _ = C_data.shape
        
        # Initialize archetypes
        if isinstance(starting_guess_archetypes, str) and starting_guess_archetypes == 'random':
            idx = np.random.choice(N, k, replace=False)
            A = C_data[idx].copy()
        else:
            if not isinstance(starting_guess_archetypes, np.ndarray):
                raise ValueError("starting_guess_archetypes must be 'random' or numpy array")
            if starting_guess_archetypes.shape != (k, n, n):
                raise ValueError(f"starting_guess_archetypes shape must be ({k}, {n}, {n})")
            A = starting_guess_archetypes.copy()
            if verbose:
                print("Using provided archetypes as initialization")
        
        # Normalize initial archetypes
        for j in range(k):
            tr = np.trace(A[j])
            if tr > 1e-9:
                A[j] = A[j] * (target_trace / tr)
                
        W = np.zeros((N, k))
        C_vec = C_data.reshape(N, -1)
        
        prev_error = np.inf
        convergence_iter = n_iter  # Default to max iterations if no convergence
        
        if verbose:
            print(f"Starting optimization for k={k} with trace={target_trace}...")
        
        for i in range(n_iter):
            # Update W (weights)
            A_vec_T = A.reshape(k, -1).T
            
            for j in range(N):
                y = C_vec[j]
                W[j, :], _ = nnls(A_vec_T, y)
                
            # Update A (archetypes)
            if not use_lasso:
                A_vec_unc, _, _, _ = np.linalg.lstsq(W, C_vec, rcond=None)
            else:
                lasso_model = Lasso(alpha=lasso_alpha, fit_intercept=False, max_iter=1000)
                lasso_model.fit(W, C_vec)
                A_vec_unc = lasso_model.coef_.T
            
            A_unc = A_vec_unc.reshape(k, n, n)
            
            # Project and normalize each archetype
            for j in range(k):
                A_psd = project_psd(A_unc[j, :, :])
                current_trace = np.trace(A_psd)
                
                if current_trace > 1e-9:
                    A[j, :, :] = A_psd * (target_trace / current_trace)
                else:
                    A[j, :, :] = A_psd
                    
            # Check convergence
            current_error = np.linalg.norm(C_vec - W @ A.reshape(k, -1), 'fro')
            
            if (prev_error - current_error) < tol:
                if verbose:
                    print(f"Converged at iteration {i+1} with error {current_error:.4f}")
                convergence_iter = i + 1
                break
                
            prev_error = current_error
            
            if verbose and (i+1) % 10 == 0:
                current_error_per_matrix = current_error / N
                print(f"Iteration {i+1}/{n_iter}, error per matrix: {current_error_per_matrix:.4f}")
        
        # Store results in instance per atom
        self.archetypes[atom_id] = A
        self.weights_train[atom_id] = W
        self.train_error[atom_id] = current_error
        self.convergence_iteration[atom_id] = convergence_iter
                
        return W, A, current_error, convergence_iter
    
    def decompose_matrices(self, C_data_new: np.ndarray, atom_id: Optional[int] = None,
                          A_archetypes: Optional[np.ndarray] = None, 
                          mode: str = 'non-negative', max_weight: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Decompose matrices using fixed archetypes.
        
        Args:
            C_data_new: New matrices to analyze, shape (N_test, n, n)
            atom_id: ID of the atom (required if A_archetypes is None)
            A_archetypes: Fixed archetypes from training, shape (k, n, n). If None, uses self.archetypes[atom_id]
            mode: 'non-negative' or 'capped'
            max_weight: Upper bound for 'capped' mode
            
        Returns:
            Tuple of (weights, fit_error)
        """
        # Use stored archetypes if none provided
        if A_archetypes is None:
            if atom_id is None:
                raise ValueError("Must provide either atom_id or A_archetypes")
            if atom_id not in self.archetypes:
                raise ValueError(f"No archetypes available for atom {atom_id}. Run fit_psd_factorization first.")
            A_archetypes = self.archetypes[atom_id]
            
        N_test, n, _ = C_data_new.shape
        k = A_archetypes.shape[0]

        C_vec_new = C_data_new.reshape(N_test, -1)
        A_vec_T = A_archetypes.reshape(k, -1).T
        
        W_new = np.zeros((N_test, k))

        for j in range(N_test):
            y = C_vec_new[j]
            
            if mode == 'non-negative':
                w, _ = nnls(A_vec_T, y)
            elif mode == 'capped':
                res = lsq_linear(A_vec_T, y, bounds=(0, max_weight))
                w = res.x
            else:
                raise ValueError(f"Unknown mode: '{mode}'")
            
            W_new[j, :] = w
        
        fit_error = np.linalg.norm(C_vec_new - W_new @ A_archetypes.reshape(k, -1), 'fro')
        
        # Store test results per atom if atom_id is provided
        # (Note: we don't always store here since this method is used for various decompositions)
        
        return W_new, fit_error
    
    @staticmethod
    def train_test_split(matrices: np.ndarray, train_fraction: float = 0.6, 
                        random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split matrices into training and test sets.
        
        Returns:
            Tuple of (train_matrices, test_matrices, train_indices, test_indices)
        """
        np.random.seed(random_seed)
        shuffled_indices = np.random.permutation(len(matrices))
        split_idx = int(len(matrices) * train_fraction)
        
        train_indices = shuffled_indices[:split_idx]
        test_indices = shuffled_indices[split_idx:]
        
        train_matrices = matrices[train_indices]
        test_matrices = matrices[test_indices]
        
        return train_matrices, test_matrices, train_indices, test_indices
    
    def fit_with_validation(self, matrices: np.ndarray, atom_id: int, k: int, train_fraction: float = 0.6,
                           store_complete_decomposition: bool = True, **fit_kwargs) -> dict:
        """
        Fit archetypes with automatic train/test validation.
        
        Args:
            matrices: Dataset matrices to fit
            atom_id: ID of the atom being fitted
            k: Number of archetypes
            train_fraction: Fraction for training (rest for testing)
            store_complete_decomposition: If True (default), decomposes and stores entire dataset.
                                        If False, only stores training/test results for efficiency.
            **fit_kwargs: Additional arguments for fit_psd_factorization
        
        Returns:
            Dictionary containing training/validation results and optionally complete decomposition
        """
        train_matrices, test_matrices, train_idx, test_idx = self.train_test_split(
            matrices, train_fraction
        )
        
        print(f"Training on {len(train_matrices)} matrices, testing on {len(test_matrices)} for atom {atom_id}")
        
        # Store indices per atom
        self.train_indices[atom_id] = train_idx
        self.test_indices[atom_id] = test_idx
        
        # Store matrices per atom
        self.matrices_all[atom_id] = matrices      # Store entire dataset
        self.matrices_train[atom_id] = train_matrices  # Store training subset
        if len(test_matrices) > 0:
            self.matrices_test[atom_id] = test_matrices    # Store test subset
        
        # Fit on training data
        W_train, A_archetypes, train_error, conv_iter = self.fit_psd_factorization(
            train_matrices, atom_id, k, **fit_kwargs
        )
        
        # Test on validation set
        test_error = None
        W_test = None
        if len(test_matrices) > 0:
            W_test, test_error = self.decompose_matrices(
                test_matrices, atom_id, A_archetypes, mode='non-negative'
            )
            # Store test weights
            self.weights_test[atom_id] = W_test
        
        result = {
            'archetypes': A_archetypes,
            'weights_train': W_train,  # Training subset weights
            'weights_test': W_test,    # Test subset weights  
            'train_error': train_error,
            'test_error': test_error,
            'train_indices': train_idx,
            'test_indices': test_idx,
            'convergence_iteration': conv_iter
        }
        
        # Optionally store complete decomposition (default behavior for backward compatibility)
        if store_complete_decomposition:
            decomp_result = self.store_complete_decomposition(matrices, atom_id, A_archetypes)
            result.update(decomp_result)
        
        return result
    
    def store_complete_decomposition(self, matrices: np.ndarray, atom_id: int, 
                                   A_archetypes: Optional[np.ndarray] = None) -> dict:
        """
        Decompose and store the complete dataset for an atom using specified or stored archetypes.
        
        Args:
            matrices: Complete dataset matrices to decompose
            atom_id: ID of the atom
            A_archetypes: Archetypes to use. If None, uses stored archetypes for this atom
            
        Returns:
            Dictionary with weights and error for the complete dataset
        """
        if A_archetypes is None:
            if atom_id not in self.archetypes:
                raise ValueError(f"No archetypes available for atom {atom_id}")
            A_archetypes = self.archetypes[atom_id]
        
        print(f"Decomposing entire dataset ({len(matrices)} matrices) for atom {atom_id}")
        W_all, all_error = self.decompose_matrices(matrices, A_archetypes=A_archetypes, mode='non-negative')
        
        # Store the complete decomposition
        self.weights_all[atom_id] = W_all
        
        # Store matrices if not already stored
        if atom_id not in self.matrices_all:
            self.matrices_all[atom_id] = matrices
        
        return {
            'weights_all': W_all,
            'all_error': all_error
        }
    
    def fit_and_store_all(self, matrices: np.ndarray, atom_id: int, k: int, train_fraction: float = 0.6,
                          **fit_kwargs) -> dict:
        """
        Convenience method that combines training and complete dataset storage.
        This provides the same behavior as the old fit_with_validation method.
        
        Returns:
            Dictionary containing all results including complete dataset decomposition
        """
        # Use fit_with_validation with default behavior (store_complete_decomposition=True)
        return self.fit_with_validation(matrices, atom_id, k, train_fraction, 
                                      store_complete_decomposition=True, **fit_kwargs)
    
    def reconstruct_matrix(self, original_matrix: np.ndarray, atom_id: int) -> np.ndarray:
        """
        Reconstruct a matrix by decomposing it with the fitted archetypes and reconstructing.
        
        Args:
            original_matrix: The matrix to reconstruct, shape (n, n)
            atom_id: ID of the atom whose archetypes to use
            
        Returns:
            Reconstructed matrix
        """
        if atom_id not in self.archetypes:
            raise ValueError(f"No archetypes available for atom {atom_id}")
        
        # Decompose the single matrix
        matrix_batch = original_matrix.reshape(1, *original_matrix.shape)
        weights, _ = self.decompose_matrices(matrix_batch, A_archetypes=self.archetypes[atom_id])
        
        # Reconstruct from weights and archetypes
        reconstructed = np.zeros_like(original_matrix)
        for j, w in enumerate(weights[0]):  # weights[0] since we only have one matrix
            reconstructed += w * self.archetypes[atom_id][j]
        
        return reconstructed
    
    def calculate_reconstruction_error(self, original_matrix: np.ndarray, atom_id: int) -> float:
        """
        Calculate reconstruction error for a single matrix.
        
        Args:
            original_matrix: The original matrix, shape (n, n)
            atom_id: ID of the atom whose archetypes to use
            
        Returns:
            Frobenius norm error between original and reconstructed matrix
        """
        reconstructed = self.reconstruct_matrix(original_matrix, atom_id)
        return np.linalg.norm(original_matrix - reconstructed, 'fro')
    
    def get_matrix_from_dataset(self, atom_id: int, matrix_idx: int, dataset: str = 'all') -> np.ndarray:
        """
        Get a specific matrix from the stored datasets.
        
        Args:
            atom_id: ID of the atom
            matrix_idx: Index of the matrix in the dataset
            dataset: 'all', 'train', or 'test'
            
        Returns:
            The matrix at the specified index
        """
        if dataset == 'all':
            if atom_id not in self.matrices_all:
                raise ValueError(f"No matrices stored for atom {atom_id}")
            matrices = self.matrices_all[atom_id]
        elif dataset == 'train':
            if atom_id not in self.matrices_train:
                raise ValueError(f"No training matrices stored for atom {atom_id}")
            matrices = self.matrices_train[atom_id]
        elif dataset == 'test':
            if atom_id not in self.matrices_test:
                raise ValueError(f"No test matrices stored for atom {atom_id}")
            matrices = self.matrices_test[atom_id]
        else:
            raise ValueError("dataset must be 'all', 'train', or 'test'")
            
        if matrix_idx >= len(matrices):
            raise IndexError(f"Matrix index {matrix_idx} out of range for {dataset} dataset")
            
        return matrices[matrix_idx]
    
    def get_weights_from_dataset(self, atom_id: int, matrix_idx: int, dataset: str = 'all') -> np.ndarray:
        """
        Get weights for a specific matrix from the stored datasets.
        
        Args:
            atom_id: ID of the atom
            matrix_idx: Index of the matrix in the dataset
            dataset: 'all', 'train', or 'test'
            
        Returns:
            The weights vector for the specified matrix
        """
        if dataset == 'all':
            if atom_id not in self.weights_all:
                raise ValueError(f"No weights stored for atom {atom_id}")
            weights = self.weights_all[atom_id]
        elif dataset == 'train':
            if atom_id not in self.weights_train:
                raise ValueError(f"No training weights stored for atom {atom_id}")
            weights = self.weights_train[atom_id]
        elif dataset == 'test':
            if atom_id not in self.weights_test:
                raise ValueError(f"No test weights stored for atom {atom_id}")
            weights = self.weights_test[atom_id]
        else:
            raise ValueError("dataset must be 'all', 'train', or 'test'")
            
        if matrix_idx >= len(weights):
            raise IndexError(f"Matrix index {matrix_idx} out of range for {dataset} dataset")
            
        return weights[matrix_idx]
    
    def progressive_multi_atom_training(self, databank: 'DataBank', atom_ids: List[int], 
                                       k_values: Union[int, List[int], Dict[int, int]], 
                                       matrix_type: str = 'combined', 
                                       **kwargs) -> Dict[int, dict]:
        """Progressive training across multiple atoms without side effects."""

        retraining_threshold_factor = self.retraining_threshold_factor
        
        if isinstance(k_values, int):
            k_values = {atom_id: k_values for atom_id in atom_ids}
        elif isinstance(k_values, list):
            k_values = {atom_ids[i]: k_values[i] for i in range(len(atom_ids))}
        
        results = {}
        
        for i, atom_id in enumerate(atom_ids):
            print(f"\n--- Processing Atom {atom_id} (step {i+1}/{len(atom_ids)}) ---")
            
            # Ensure matrices are built
            if atom_id not in databank.matrices:
                databank.build_matrices(atom_id)
            
            current_matrices = databank.get_matrices(atom_id, matrix_type)
            
            if i == 0:
                # First atom: train from scratch
                result = self.fit_with_validation(
                    current_matrices, atom_id, k_values[atom_id], 
                    store_complete_decomposition=True, **kwargs
                )
                result['method'] = 'fresh_training'
                results[atom_id] = result
                
            else:
                # Test with previous archetypes first
                prev_atom_id = atom_ids[i-1]
                prev_archetypes = results[prev_atom_id]['archetypes']
                
                # Test with previous archetypes using decompose_matrices
                W_test, test_error = self.decompose_matrices(
                    current_matrices, A_archetypes=prev_archetypes, mode='non-negative'
                )
                
                avg_error_per_matrix = test_error / len(current_matrices)
                prev_avg_error = results[prev_atom_id]['train_error'] / len(
                    databank.get_matrices(prev_atom_id, matrix_type)
                )
                
                retraining_threshold = retraining_threshold_factor * prev_avg_error
                
                print(f"Testing atom {atom_id} with atom {prev_atom_id} archetypes:")
                print(f"  Average error per matrix: {avg_error_per_matrix:.4f}")
                print(f"  Retraining threshold: {retraining_threshold:.4f}")
                
                if avg_error_per_matrix <= retraining_threshold:
                    # Use previous archetypes - store archetypes and indices for this atom
                    print(f"  -> Using atom {prev_atom_id} archetypes (no retraining needed)")
                    
                    # Store the archetypes for this atom (reused from previous)
                    self.archetypes[atom_id] = prev_archetypes
                    self.k_values[atom_id] = k_values[atom_id]
                    self.n_orbitals[atom_id] = current_matrices.shape[1]
                    
                    # Store the complete decomposition using reused archetypes
                    decomp_result = self.store_complete_decomposition(
                        current_matrices, atom_id, prev_archetypes
                    )
                    
                    results[atom_id] = {
                        'archetypes': prev_archetypes,
                        'weights_test': W_test,
                        'test_error': test_error,
                        'method': 'reused_archetypes',
                        'source_atom': prev_atom_id,
                        'avg_error_per_matrix': avg_error_per_matrix,
                        'retraining_threshold': retraining_threshold,
                        **decomp_result  # Include complete decomposition results
                    }
                else:
                    # Retrain with previous archetypes as initialization
                    print(f"  -> Retraining needed (error too high)")
                    kwargs_with_init = kwargs.copy()
                    kwargs_with_init['starting_guess_archetypes'] = prev_archetypes
                    
                    result = self.fit_with_validation(
                        current_matrices, atom_id, k_values[atom_id], 
                        store_complete_decomposition=True, **kwargs_with_init
                    )
                    
                    result.update({
                        'method': 'retrained',
                        'source_atom': prev_atom_id,
                        'avg_error_per_matrix': avg_error_per_matrix,
                        'retraining_threshold': retraining_threshold
                    })
                    results[atom_id] = result
        
        # Store atom results in the instance
        self.atom_results = results
        
        return results
    
    def save_state(self, filepath: str):
        """
        Save the complete ArchetypeFitter state to JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        def serialize_dict_of_arrays(data_dict):
            """Convert dict of numpy arrays to dict of lists for JSON serialization."""
            return {str(k): v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in data_dict.items()}
        
        state_data = {
            # Core arrays per atom (convert numpy arrays to lists for JSON serialization)
            'archetypes': serialize_dict_of_arrays(self.archetypes),
            'weights_train': serialize_dict_of_arrays(self.weights_train),
            'weights_test': serialize_dict_of_arrays(self.weights_test),
            'weights_all': serialize_dict_of_arrays(self.weights_all),
            'matrices_train': serialize_dict_of_arrays(self.matrices_train),
            'matrices_test': serialize_dict_of_arrays(self.matrices_test),
            'matrices_all': serialize_dict_of_arrays(self.matrices_all),
            
            # Metadata per atom
            'k_values': {str(k): v for k, v in self.k_values.items()},
            'target_trace': {str(k): v for k, v in self.target_trace.items()},
            'n_orbitals': {str(k): v for k, v in self.n_orbitals.items()},
            
            # Training results per atom
            'train_error': {str(k): float(v) if v is not None else None for k, v in self.train_error.items()},
            'test_error': {str(k): float(v) if v is not None else None for k, v in self.test_error.items()},
            'train_indices': serialize_dict_of_arrays(self.train_indices),
            'test_indices': serialize_dict_of_arrays(self.test_indices),
            'convergence_iteration': {str(k): v for k, v in self.convergence_iteration.items()},
            
            # Per-atom results
            'atom_results': self._serialize_atom_results(),
            
            # Training parameters
            'fit_params': self.fit_params
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        print(f"ArchetypeFitter state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """
        Load ArchetypeFitter state from JSON file.
        
        Args:
            filepath: Path to the JSON file
        """
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        def deserialize_dict_of_arrays(data_dict):
            """Convert dict of lists back to dict of numpy arrays."""
            return {int(k): np.array(v) if v is not None else None 
                    for k, v in data_dict.items()}
        
        def deserialize_dict_of_values(data_dict):
            """Convert dict with string keys back to dict with int keys."""
            return {int(k): v for k, v in data_dict.items()}
        
        # Restore core arrays per atom (convert lists back to numpy arrays)
        self.archetypes = deserialize_dict_of_arrays(state_data['archetypes'])
        self.weights_train = deserialize_dict_of_arrays(state_data['weights_train'])
        self.weights_test = deserialize_dict_of_arrays(state_data['weights_test'])
        self.weights_all = deserialize_dict_of_arrays(state_data.get('weights_all', {}))
        self.matrices_train = deserialize_dict_of_arrays(state_data['matrices_train'])
        self.matrices_test = deserialize_dict_of_arrays(state_data['matrices_test'])
        self.matrices_all = deserialize_dict_of_arrays(state_data.get('matrices_all', {}))
        
        # Restore metadata per atom
        self.k_values = deserialize_dict_of_values(state_data['k_values'])
        self.target_trace = deserialize_dict_of_values(state_data['target_trace'])
        self.n_orbitals = deserialize_dict_of_values(state_data['n_orbitals'])
        
        # Restore training results per atom
        self.train_error = deserialize_dict_of_values(state_data['train_error'])
        self.test_error = deserialize_dict_of_values(state_data['test_error'])
        self.train_indices = deserialize_dict_of_arrays(state_data['train_indices'])
        self.test_indices = deserialize_dict_of_arrays(state_data['test_indices'])
        self.convergence_iteration = deserialize_dict_of_values(state_data['convergence_iteration'])
        
        # Restore per-atom results
        self.atom_results = self._deserialize_atom_results(state_data['atom_results'])
        
        # Restore training parameters
        self.fit_params = state_data['fit_params']
        
        print(f"ArchetypeFitter state loaded from {filepath}")
    
    def _serialize_atom_results(self) -> dict:
        """Convert atom_results to JSON-serializable format."""
        serialized = {}
        for atom_id, results in self.atom_results.items():
            atom_data = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    atom_data[key] = value.tolist()
                elif isinstance(value, np.floating):
                    atom_data[key] = float(value)
                elif isinstance(value, np.integer):
                    atom_data[key] = int(value)
                else:
                    atom_data[key] = value
            serialized[str(atom_id)] = atom_data
        return serialized
    
    def _deserialize_atom_results(self, serialized_data: dict) -> dict:
        """Convert JSON data back to atom_results format."""
        deserialized = {}
        for atom_id_str, atom_data in serialized_data.items():
            atom_id = int(atom_id_str)
            result_data = {}
            for key, value in atom_data.items():
                if key in ['archetypes', 'weights_train', 'weights_test', 'train_indices', 'test_indices']:
                    result_data[key] = np.array(value) if value is not None else None
                else:
                    result_data[key] = value
            deserialized[atom_id] = result_data
        return deserialized
    
    def get_summary(self) -> dict:
        """Get a summary of the current ArchetypeFitter state."""
        summary = {
            'fitted': len(self.archetypes) > 0,
            'atoms_fitted': list(self.archetypes.keys()),
            'k_values': self.k_values.copy(),
            'n_orbitals': self.n_orbitals.copy(),
            'target_trace': self.target_trace.copy(),
            'convergence_iteration': self.convergence_iteration.copy(),
            'train_error': self.train_error.copy(),
            'test_error': self.test_error.copy(),
            'n_atoms_processed': len(self.atom_results),
            'fit_params': self.fit_params
        }
        
        # Add per-atom training matrix counts
        summary['n_training_matrices'] = {}
        for atom_id, weights in self.weights_train.items():
            if weights is not None:
                summary['n_training_matrices'][atom_id] = len(weights)
        
        # Add per-atom test matrix counts
        summary['n_test_matrices'] = {}
        for atom_id, weights in self.weights_test.items():
            if weights is not None:
                summary['n_test_matrices'][atom_id] = len(weights)
        
        # Add per-atom total matrix counts (entire dataset)
        summary['n_all_matrices'] = {}
        for atom_id, weights in self.weights_all.items():
            if weights is not None:
                summary['n_all_matrices'][atom_id] = len(weights)
        
        # Add average errors per matrix per atom
        summary['avg_train_error_per_matrix'] = {}
        for atom_id, error in self.train_error.items():
            if error is not None and atom_id in summary['n_training_matrices']:
                summary['avg_train_error_per_matrix'][atom_id] = error / summary['n_training_matrices'][atom_id]
        
        summary['avg_test_error_per_matrix'] = {}
        for atom_id, error in self.test_error.items():
            if error is not None and atom_id in summary['n_test_matrices']:
                summary['avg_test_error_per_matrix'][atom_id] = error / summary['n_test_matrices'][atom_id]
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of the current state."""
        summary = self.get_summary()
        
        print("\n=== ArchetypeFitter Summary ===")
        print(f"Status: {'Fitted' if summary['fitted'] else 'Not fitted'}")
        
        if summary['fitted']:
            print(f"Atoms fitted: {summary['atoms_fitted']}")
            
            for atom_id in summary['atoms_fitted']:
                print(f"\nAtom {atom_id}:")
                print(f"  Number of archetypes (k): {summary['k_values'].get(atom_id, 'N/A')}")
                print(f"  Matrix dimension (n): {summary['n_orbitals'].get(atom_id, 'N/A')}")
                print(f"  Target trace: {summary['target_trace'].get(atom_id, 'N/A')}")
                print(f"  Convergence iteration: {summary['convergence_iteration'].get(atom_id, 'N/A')}")
                
                if atom_id in summary['n_training_matrices']:
                    print(f"  Training matrices: {summary['n_training_matrices'][atom_id]}")
                    if atom_id in summary['train_error']:
                        print(f"  Training error: {summary['train_error'][atom_id]:.4f}")
                    if atom_id in summary['avg_train_error_per_matrix']:
                        print(f"  Avg training error per matrix: {summary['avg_train_error_per_matrix'][atom_id]:.4f}")
                    
                if atom_id in summary['n_test_matrices']:
                    print(f"  Test matrices: {summary['n_test_matrices'][atom_id]}")  
                    if atom_id in summary['test_error']:
                        print(f"  Test error: {summary['test_error'][atom_id]:.4f}")
                    if atom_id in summary['avg_test_error_per_matrix']:
                        print(f"  Avg test error per matrix: {summary['avg_test_error_per_matrix'][atom_id]:.4f}")
                
                if atom_id in summary['n_all_matrices']:
                    print(f"  Total matrices (entire dataset): {summary['n_all_matrices'][atom_id]}")
                
            if summary['n_atoms_processed'] > 0:
                print(f"\nTotal atoms processed: {summary['n_atoms_processed']}")
                
        print("=" * 30)

# %%

class DataBank:
    """
    Handles all data loading, processing, and matrix management operations.
    
    This class is responsible for:
    - Loading JSON/CSV data
    - Processing raw data into structured DataFrames
    - Building matrices from occupation data
    - Matrix format conversions (numpy/pytorch compatibility)
    - Data organization and retrieval
    """
    
    def __init__(self):
        self.raw_data = None
        self.dataframe = None
        self.matrices = {}  # {atom_id: {'up': matrices, 'down': matrices, 'combined': matrices}}
        
        # Flattening/indexing helpers for upper-triangular matrix elements
        self.flat_index_map = None  # dict: (atom_id, spin, i, j) -> flat_index
        self.flat_index_rev = None  # list: [(atom_id, spin, i, j), ...] by flat_index
        self.flat_size = 0
        
    def load_json_data(self, json_file_path: str) -> pd.DataFrame:
        """Load and process JSON data containing calculation results."""
        print(f"Loading data from: {json_file_path}")
        
        try:
            with open(json_file_path, 'r') as f:
                self.raw_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at '{json_file_path}'")
            
        self._process_raw_data()
        print(f"Successfully loaded {len(self.dataframe)} calculations")
        return self.dataframe
        
    def load_csv_data(self, csv_file_path: str) -> pd.DataFrame:
        """Load and process CSV data containing occupation matrices."""
        print(f"Loading CSV data from: {csv_file_path}")
        
        try:
            self.dataframe = pd.read_csv(csv_file_path)
            if 'total_energy_eV' in self.dataframe.columns:
                self.dataframe = self.dataframe.sort_values(by='total_energy_eV')
            print(f"Successfully loaded {len(self.dataframe)} calculations")
            return self.dataframe
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at '{csv_file_path}'")
            
    def _process_raw_data(self):
        """Process raw JSON data into a structured DataFrame."""
        if self.raw_data is None:
            raise ValueError("No raw data loaded")
            
        results = []
        calculations = self.raw_data.get('calculations', {})
        
        for calc_id, calc_data in calculations.items():
            if not isinstance(calc_data, dict):
                continue
                
            properties = self._extract_base_properties(calc_data)
            occupation_data = self._extract_occupation_matrices(calc_data)
            
            result_entry = {
                'calculation_id': calc_id,
                **properties,
                **occupation_data
            }
            results.append(result_entry)
            
        self.dataframe = pd.DataFrame(results)
        if 'total_energy_eV' in self.dataframe.columns:
            self.dataframe = self.dataframe.sort_values(by='total_energy_eV')
            
    def _extract_base_properties(self, calc_data: dict) -> dict:
        """Extract fundamental properties from a single calculation entry."""
        properties = {
            'source': calc_data.get('calculation_source', 'unknown'),
            'total_energy_eV': None,
            'hubbard_energy_eV': None,
        }
        
        output_params = calc_data.get('output_parameters', {})
        if isinstance(output_params, dict):
            properties['total_energy_eV'] = output_params.get('energy')
            properties['hubbard_energy_eV'] = output_params.get('energy_hubbard')
            
        return properties
        
    def _extract_occupation_matrices(self, calc_data: dict) -> dict:
        """Extract the upper triangle of occupation matrices for all atoms."""
        occupation_data = {}
        atomic_occupations = calc_data.get('output_atomic_occupations', {})
        
        if not isinstance(atomic_occupations, dict):
            return occupation_data
            
        for atom_idx, atom_data in enumerate(atomic_occupations.values(), start=1):
            if not (isinstance(atom_data, dict) and 'spin_data' in atom_data):
                continue
                
            spin_data = atom_data['spin_data']
            for spin in ['up', 'down']:
                matrix = spin_data.get(spin, {}).get('occupation_matrix', [])
                if matrix:
                    for i in range(len(matrix)):
                        for j in range(i, len(matrix)):
                            key = f"atom_{atom_idx}_{spin}_occ_{i+1}_{j+1}"
                            occupation_data[key] = matrix[i][j]
                            
        return occupation_data
        
    def get_available_atoms(self) -> List[int]:
        """Get list of available atom IDs in the dataset."""
        if self.dataframe is None:
            return []
            
        atoms = set()
        for col in self.dataframe.columns:
            if col.startswith('atom_') and '_occ_' in col:
                parts = col.split('_')
                if len(parts) >= 2:
                    try:
                        atoms.add(int(parts[1]))
                    except ValueError:
                        continue
        return sorted(list(atoms))
        
    def detect_atom_orbitals(self, atom_id: int) -> int:
        """Auto-detect the number of orbitals for a given atom."""
        if self.dataframe is None:
            raise ValueError("No data loaded")
            
        pattern_cols = [col for col in self.dataframe.columns 
                       if col.startswith(f'atom_{atom_id}_') and '_occ_' in col]
        
        if not pattern_cols:
            return 0
            
        max_orbital = 0
        for col in pattern_cols:
            parts = col.split('_')
            if len(parts) >= 6:
                try:
                    i, j = int(parts[4]), int(parts[5])
                    max_orbital = max(max_orbital, i, j)
                except ValueError:
                    continue
                    
        return max_orbital
        
    def build_matrices(self, atom_id: int, spin: str = 'both') -> Dict[str, np.ndarray]:
        """
        Build symmetric matrices from occupation data for a specific atom.
        
        Args:
            atom_id: The atom ID to process
            spin: 'up', 'down', or 'both' for combined analysis
            
        Returns:
            Dictionary containing matrices for the specified spin(s)
        """
        if self.dataframe is None:
            raise ValueError("No data loaded")
            
        n_orbitals = self.detect_atom_orbitals(atom_id)
        if n_orbitals == 0:
            raise ValueError(f"Could not detect orbitals for atom {atom_id}")
            
        print(f"Building matrices for atom {atom_id} with {n_orbitals} orbitals")
        
        matrices = {}
        spins_to_process = ['up', 'down'] if spin == 'both' else [spin]
        
        for spin_type in spins_to_process:
            spin_matrices = []
            
            for idx, row in self.dataframe.iterrows():
                matrix = self._build_single_matrix(row, atom_id, spin_type, n_orbitals)
                if matrix is not None:
                    spin_matrices.append(matrix)
                    
            if spin_matrices:
                matrices[spin_type] = np.array(spin_matrices)
                print(f"Built {len(spin_matrices)} {spin_type} matrices for atom {atom_id}")
                
        # Create combined matrices (up + down) for joint training
        if 'up' in matrices and 'down' in matrices and len(matrices['up']) == len(matrices['down']):
            combined_matrices = []
            for up_mat, down_mat in zip(matrices['up'], matrices['down']):
                combined_matrices.append(up_mat)
                combined_matrices.append(down_mat)
            matrices['combined'] = np.array(combined_matrices)
            print(f"Built {len(combined_matrices)} combined matrices for atom {atom_id}")
            
        self.matrices[atom_id] = matrices
        return matrices
        
    def _build_single_matrix(self, row, atom_id: int, spin_type: str, n_orbitals: int) -> Optional[np.ndarray]:
        """Build a single symmetric matrix from a data row."""
        matrix = np.zeros((n_orbitals, n_orbitals))
        
        for i in range(n_orbitals):
            for j in range(i, n_orbitals):
                col_name = f'atom_{atom_id}_{spin_type}_occ_{i+1}_{j+1}'
                
                if col_name in row and pd.notna(row[col_name]):
                    value = row[col_name]
                    matrix[i, j] = value
                    if i != j:
                        matrix[j, i] = value
                else:
                    return None
                    
        return matrix
        
    def get_matrices(self, atom_id: int, spin_type: str = None) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Get matrices for a specific atom and spin type.
        
        Args:
            atom_id: The atom ID
            spin_type: 'up', 'down', 'combined', or None for all types
            
        Returns:
            Matrices array or dictionary of matrices
        """
        if atom_id not in self.matrices:
            raise ValueError(f"No matrices found for atom {atom_id}")
            
        if spin_type is None:
            return self.matrices[atom_id]
        elif spin_type in self.matrices[atom_id]:
            return self.matrices[atom_id][spin_type]
        else:
            raise ValueError(f"Spin type '{spin_type}' not available for atom {atom_id}")
            
    def get_matrix(self, atom_id: int, spin_type: str, matrix_idx: int) -> np.ndarray:
        """Get a specific matrix by atom ID, spin type, and matrix index."""
        matrices = self.get_matrices(atom_id, spin_type)
        if matrix_idx >= len(matrices):
            raise IndexError(f"Matrix index {matrix_idx} out of range")
        return matrices[matrix_idx]
        
    # PyTorch compatibility methods
    def matrices_to_vectors(self, matrices: np.ndarray) -> np.ndarray:
        """Convert matrices to flattened vectors for PyTorch/ML compatibility."""
        N, n, _ = matrices.shape
        return matrices.reshape(N, -1)
        
    def vectors_to_matrices(self, vectors: np.ndarray, matrix_shape: Tuple[int, int]) -> np.ndarray:
        """Convert flattened vectors back to matrices."""
        N = vectors.shape[0]
        n, m = matrix_shape
        return vectors.reshape(N, n, m)

    def upper_triangle_indices(self, n_orbitals: int):
        """
        Return a list of (i, j) tuples for the upper-triangle (including diagonal)
        in the requested order: for i in 0..n-1, for j in i..n-1 -> (i, j).
        This ordering is deterministic and reversible.
        """
        idxs = []
        for i in range(n_orbitals):
            for j in range(i, n_orbitals):
                idxs.append((i, j))
        return idxs

    def upper_triangle_length(self, n_orbitals: int) -> int:
        """Return number of elements in the upper-triangle (including diag)."""
        return n_orbitals * (n_orbitals + 1) // 2
        
    def to_pytorch_format(self, atom_id: int, spin_type: str = 'combined') -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert matrices to PyTorch-compatible format.
        
        Returns:
            Tuple of (matrix_vectors, original_shape) for PyTorch processing
        """
        matrices = self.get_matrices(atom_id, spin_type)
        vectors = self.matrices_to_vectors(matrices)
        original_shape = (matrices.shape[1], matrices.shape[2])
        return vectors, original_shape
        
    def from_pytorch_format(self, vectors: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Convert PyTorch vectors back to matrix format."""
        return self.vectors_to_matrices(vectors, original_shape)
        
    def save_data(self, filepath: str):
        """Save the DataBank state to file."""
        save_data = {
            'dataframe': self.dataframe,
            'matrices': self.matrices,
            'raw_data': self.raw_data
        }
        np.savez_compressed(filepath, **save_data)
        print(f"DataBank saved to {filepath}")
        
    def load_data(self, filepath: str):
        """Load previously saved DataBank state."""
        data = np.load(filepath, allow_pickle=True)
        
        if 'dataframe' in data and data['dataframe'].item() is not None:
            self.dataframe = data['dataframe'].item()
        if 'matrices' in data:
            self.matrices = data['matrices'].item()
        if 'raw_data' in data:
            self.raw_data = data['raw_data'].item()
            
        print(f"DataBank loaded from {filepath}")

    # --- Flattening utilities (upper-triangular only) ---
    def build_flat_index_map(self, atom_ids: Optional[List[int]] = None, spins: List[str] = ['up', 'down']):
        """
        Build a one-to-one mapping from (atom_id, spin, i, j) with i<=j to a flat index.

        The ordering follows: for atom in sorted(atom_ids): for spin in spins: iterate
        over upper-triangular elements in row-major order (i from 0..n-1, j from i..n-1).
        
        Args:
            atom_ids: List of atom IDs to include. If None, uses all available atoms.
            spins: List of spin types to include (default: ['up', 'down'])
            
        Returns:
            tuple: (flat_map, flat_rev) where flat_map is the forward mapping and 
                   flat_rev is the reverse mapping list
        """
        if self.dataframe is None:
            raise ValueError("No data loaded")

        if atom_ids is None:
            atom_ids = self.get_available_atoms()

        flat_map = {}
        flat_rev = []
        idx = 0
        
        for atom in sorted(atom_ids):
            n_orb = self.detect_atom_orbitals(atom)
            if n_orb == 0:
                continue
            for spin in spins:
                for i in range(n_orb):
                    for j in range(i, n_orb):  # Upper triangular: i <= j
                        flat_map[(atom, spin, i, j)] = idx
                        flat_rev.append((atom, spin, i, j))
                        idx += 1

        self.flat_index_map = flat_map
        self.flat_index_rev = flat_rev
        self.flat_size = idx
        
        print(f"Built flat index map with {self.flat_size} elements for atoms {sorted(atom_ids)}")
        return flat_map, flat_rev

    def flatten_calculation(self, calc_row_index: int, atom_ids: Optional[List[int]] = None, 
                          spins: List[str] = ['up', 'down']) -> np.ndarray:
        """
        Flatten a single calculation (row index in self.dataframe) into the upper-triangular vector
        following the map built by `build_flat_index_map`.
        
        Args:
            calc_row_index: Row index in the dataframe
            atom_ids: Atom IDs to include (if None, uses all available)
            spins: Spin types to include
            
        Returns:
            1D numpy array of length `flat_size` with flattened matrix elements
        """
        if self.flat_index_map is None:
            self.build_flat_index_map(atom_ids, spins)

        if calc_row_index >= len(self.dataframe):
            raise IndexError(f"calc_row_index {calc_row_index} out of range")

        row = self.dataframe.iloc[calc_row_index]
        vec = np.zeros(self.flat_size, dtype=float)

        # Fill vector using the flat index mapping
        for flat_idx, (atom, spin, i, j) in enumerate(self.flat_index_rev):
            col_name = f'atom_{atom}_{spin}_occ_{i+1}_{j+1}'
            if col_name in row and pd.notna(row[col_name]):
                vec[flat_idx] = row[col_name]
            else:
                # Missing data -> NaN to mark
                vec[flat_idx] = np.nan

        return vec

    def flatten_all_calculations(self, atom_ids: Optional[List[int]] = None, 
                               spins: List[str] = ['up', 'down']) -> np.ndarray:
        """
        Flatten all calculations (rows) into a 2D array (n_calcs, flat_size).
        
        Args:
            atom_ids: Atom IDs to include (if None, uses all available)
            spins: Spin types to include
            
        Returns:
            2D numpy array of shape (n_calculations, flat_size)
        """
        if self.dataframe is None:
            raise ValueError("No data loaded")
            
        self.build_flat_index_map(atom_ids, spins)
        N = len(self.dataframe)
        out = np.zeros((N, self.flat_size), dtype=float)
        
        for r in range(N):
            out[r, :] = self.flatten_calculation(r)
            
        return out

    def unflatten_vector_to_matrices(self, vec: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Given a 1D flattened vector (matching self.flat_index_rev), reconstruct a nested dict
        of matrices: {atom_id: {spin: matrix}}. Missing entries (NaN) are filled with zeros.
        
        Args:
            vec: 1D array of length flat_size with flattened matrix elements
            
        Returns:
            Nested dictionary: {atom_id: {spin: matrix}} with symmetric matrices
        """
        if self.flat_index_rev is None:
            raise ValueError('Flat index map not built. Call build_flat_index_map() first.')
            
        if len(vec) != self.flat_size:
            raise ValueError(f"Vector length {len(vec)} doesn't match flat_size {self.flat_size}")

        # Determine orbitals per atom from the index map
        atom_orbitals = {}
        for atom, spin, i, j in self.flat_index_rev:
            atom_orbitals.setdefault(atom, set()).add(i)
            atom_orbitals[atom].add(j)

        # Initialize zero matrices
        matrices = {}
        for atom, inds in atom_orbitals.items():
            n = max(inds) + 1
            matrices[atom] = {}
            for spin in ['up', 'down']:
                matrices[atom][spin] = np.zeros((n, n), dtype=float)

        # Fill matrices from flattened vector
        for idx, (atom, spin, i, j) in enumerate(self.flat_index_rev):
            val = vec[idx]
            if np.isnan(val):
                val = 0.0
            matrices[atom][spin][i, j] = val
            if i != j:  # Fill symmetric part
                matrices[atom][spin][j, i] = val

        return matrices

    # --- Per-atom flattening helpers (single-atom focus) ---
    def flatten_atom_across_calculations(self, atom_id: int, spins=('up', 'down'), drop_missing=True):
        """
        Flatten only the given atom's spin matrices for every calculation (row) in the
        dataframe. Each returned vector corresponds to a single calculation (calc_index)
        and contains the upper-triangle elements for the requested spins in the order:

            atom_{id}_{spin}_occ_1_1, atom_{id}_{spin}_occ_1_2, ..., atom_{id}_{spin}_occ_2_2, ...

        Args:
            atom_id: integer atom id
            spins: tuple/list indicating the spin ordering
            drop_missing: whether to skip calculations missing any of the required columns

        Returns:
            vectors: np.ndarray shape (N_valid, L)
            calc_indices: list of dataframe row indices corresponding to each vector
        """
        if self.dataframe is None:
            raise ValueError('No data loaded')

        n_orb = self.detect_atom_orbitals(atom_id)
        if n_orb == 0:
            raise ValueError(f'No orbitals detected for atom {atom_id}')

        idxs = self.upper_triangle_indices(n_orb)
        col_names = []
        for spin in spins:
            for (i, j) in idxs:
                col_names.append(f'atom_{atom_id}_{spin}_occ_{i+1}_{j+1}')

        # Verify all columns exist globally (if missing and drop_missing True -> error)
        missing_cols = [c for c in col_names if c not in self.dataframe.columns]
        if missing_cols:
            raise ValueError(f'Missing required columns for atom {atom_id}: {missing_cols}')

        rows = []
        calc_indices = []
        for idx, row in self.dataframe.iterrows():
            vals = [row[c] for c in col_names]
            if any(pd.isna(v) for v in vals):
                if drop_missing:
                    continue
            rows.append(vals)
            calc_indices.append(idx)

        if len(rows) == 0:
            return np.zeros((0, len(col_names))), []

        vectors = np.array(rows, dtype=float)
        return vectors, calc_indices

    def unflatten_atom_vector(self, vector: np.ndarray, atom_id: int, spins=('up', 'down')) -> Dict[str, np.ndarray]:
        """
        Given a single flattened vector produced by flatten_atom_across_calculations,
        reconstruct and return a dict mapping spin -> (n_orb, n_orb) symmetric matrix.
        """
        n_orb = self.detect_atom_orbitals(atom_id)
        if n_orb == 0:
            raise ValueError(f'No orbitals detected for atom {atom_id}')

        idxs = self.upper_triangle_indices(n_orb)
        per_spin = len(idxs)
        expected = per_spin * len(spins)
        if vector.size != expected:
            raise ValueError(f'Vector length {vector.size} does not match expected {expected}')

        out = {}
        offset = 0
        for spin in spins:
            mat = np.zeros((n_orb, n_orb), dtype=float)
            for k, (i, j) in enumerate(idxs):
                val = vector[offset + k]
                mat[i, j] = val
                if i != j:
                    mat[j, i] = val
            out[spin] = mat
            offset += per_spin

        return out

    def atom_flat_index(self, atom_id: int, spin: str, i: int, j: int, spins=('up', 'down')) -> int:
        """
        Compute the index within the per-atom flattened vector for the element (i,j)
        (0-based orbital indices) and the given spin. The ordering matches
        flatten_atom_across_calculations above (spins in the order provided).
        """
        n_orb = self.detect_atom_orbitals(atom_id)
        if n_orb == 0:
            raise ValueError(f'No orbitals detected for atom {atom_id}')

        if i > j:
            i, j = j, i

        idxs = self.upper_triangle_indices(n_orb)
        try:
            intra = idxs.index((i, j))
        except ValueError:
            raise ValueError(f'Invalid orbital pair {(i, j)} for atom {atom_id}')

        try:
            spin_pos = spins.index(spin)
        except ValueError:
            raise ValueError(f"Spin '{spin}' not in provided spins {spins}")

        per_spin = len(idxs)
        return spin_pos * per_spin + intra



def prepare_data_for_fit(json_file):
    """
    Processes the JSON file to create a DataFrame with features for linear regression,
    including a Heisenberg m1*m2 term.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file}'.")
        return pd.DataFrame()

    feature_list = []
    print("Starting feature engineering for model fitting...")

    for calc_id, calc_data in data.get('calculations', {}).items():
        # Ensure the calculation has all the necessary data blocks
        if not (isinstance(calc_data, dict)
                and 'so_n_decomposition' in calc_data
                and 'output_atomic_occupations' in calc_data):
            continue

        total_energy = calc_data.get('output_parameters', {}).get('energy')
        hubbard_energy = calc_data.get('output_parameters', {}).get('energy_hubbard')
        if total_energy is None:
            continue

        # Initialize accumulators for the cell
        hubbard_term_cell = 0
        exchange_term_cell = 0
        internal_repulsion_term_cell = 0
        cf_features = {}
        
        # Dictionary to store local moments for the Heisenberg term
        local_moments = {}

        so_n_data = calc_data['so_n_decomposition']['atom_decompositions']
        occupations_data = calc_data['output_atomic_occupations']

        heisenberg_term_value = 1
        
        for atom_id_str, atom_data in so_n_data.items():
            atom_id = int(atom_id_str)
            up_eigs = atom_data.get('up_spin', {}).get('eigenvalues', [])
            down_eigs = atom_data.get('down_spin', {}).get('eigenvalues', [])
            all_eigs_atom = up_eigs + down_eigs

            # Term 1: Hubbard U term
            hubbard_term_cell += sum(eig * (1 - eig) for eig in all_eigs_atom)

            # Term 2: Exchange J term (M^2)
            local_moment = sum(up_eigs) - sum(down_eigs)
            exchange_term_cell += local_moment**2

            heisenberg_term_value *= local_moment
            
            # Store the local moment for the Heisenberg term calculation
            local_moments[atom_id_str] = local_moment

            # Term 3: Internal U term
            num_electrons_atom = sum(all_eigs_atom)
            internal_repulsion_term_cell += num_electrons_atom * (num_electrons_atom - 1)
            
            # Term 4: Crystal Field terms (full occupation matrix elements)
            atom_occ_data = occupations_data.get(atom_id_str)
            if atom_occ_data:
                n_up = np.array(atom_occ_data.get('spin_data', {}).get('up', {}).get('occupation_matrix', []))
                n_down = np.array(atom_occ_data.get('spin_data', {}).get('down', {}).get('occupation_matrix', []))

                #diagonalize n_up and n_down
                eig_up = np.linalg.eigvalsh(n_up) 
                eig_down = np.linalg.eigvalsh(n_down) 

                if n_up.size > 0 and n_down.size > 0:
                    n_total = n_up + n_down
                    for i in range(n_total.shape[0]):
                        # quadratic_feature_name = f'cf_atom{atom_id}_eig_{i+1}_sq'
                        # cf_features[quadratic_feature_name] = (eig_up[i] + eig_down[i])**2

                        for j in range(i, n_total.shape[1]):
                            feature_name = f'cf_atom{atom_id}_n_{i+1}_{j+1}'
                            cf_features[feature_name] = n_total[i, j]

        # hubbard_term_energy  = 5/2 * hubbard_term_cell  # Assuming U=5 eV
        # Store all features for this calculation
        calc_features = {
            'calculation_id': calc_id,
            # 'total_energy_eV': total_energy- hubbard_term_energy,
            'total_energy_eV': total_energy,
            'hubbard_term': hubbard_term_cell,
            'exchange_term_M2': exchange_term_cell,
            'internal_repulsion_term_N_N_1': internal_repulsion_term_cell,
            # 'heisenberg_term_m1_m2': heisenberg_term_value,
            # **cf_features
        }
        feature_list.append(calc_features)
        
    print(f"Feature engineering complete. Created {len(feature_list)} data points.")
    return pd.DataFrame(feature_list)