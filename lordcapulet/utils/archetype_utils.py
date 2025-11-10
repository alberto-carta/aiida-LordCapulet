import numpy as np
import pandas as pd
import json
from pathlib import Path

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

def fit_psd_factorization(C_data, k, target_trace=1.0, n_iter=1000, tol=1e-5, use_lasso=False, lasso_alpha=0.01, starting_guess_archetypes='random'):
    """
    Performs Non-negative PSD Factorization with trace-normalized archetypes.
    
    Finds W (weights) and A (archetypes) such that C_data[i] approx. sum(W[i,j] * A[j])
    and tr(A[j]) = target_trace.
    
    Args:
        C_data (np.array): Dataset of shape (N, n, n).
        k (int): Number of archetypes.
        target_trace (float): The desired trace for each archetype (e.g., 1.0).
        n_iter (int): Maximum iterations.
        tol (float): Convergence tolerance.
        use_lasso (bool): Whether to use Lasso regularization.
        lasso_alpha (float): Lasso regularization parameter.
        starting_guess_archetypes (str or np.array): 'random' for random initialization,
                                                   or array of shape (k, n, n) for custom initialization.
        
    Returns:
        W (np.array): Non-negative weights, shape (N, k).
        A (np.array): PSD archetypes with fixed trace, shape (k, n, n).
    """
    
    N, n, _ = C_data.shape
    
    # --- 1. Initialization ---
    if isinstance(starting_guess_archetypes, str) and starting_guess_archetypes == 'random':
        # Random initialization from data matrices
        idx = np.random.choice(N, k, replace=False)
        A = C_data[idx].copy()
    else:
        # Use provided archetypes as initialization
        if not isinstance(starting_guess_archetypes, np.ndarray):
            raise ValueError("starting_guess_archetypes must be 'random' or a numpy array of shape (k, n, n)")
        if starting_guess_archetypes.shape != (k, n, n):
            raise ValueError(f"starting_guess_archetypes shape {starting_guess_archetypes.shape} must be (k={k}, n={n}, n={n})")
        A = starting_guess_archetypes.copy()
        print(f"Using provided archetypes as initialization")
    
    # Normalize initial archetypes
    for j in range(k):
        tr = np.trace(A[j])
        if tr > 1e-9:
            A[j] = A[j] * (target_trace / tr)
            
    W = np.zeros((N, k))
    C_vec = C_data.reshape(N, -1)
    
    prev_error = np.inf
    
    print(f"Starting optimization for k={k} with trace={target_trace}...")
    
    for i in range(n_iter):
        # --- 2. Update W (Archetypes A fixed) ---
        # This step is unchanged. 
        # nnls will find the best W for our trace-normalized A.
        A_vec_T = A.reshape(k, -1).T
        
        for j in range(N):
            y = C_vec[j]
            W[j, :], _ = nnls(A_vec_T, y)
            # res = lsq_linear(A_vec_T, y, bounds=(0, 1)) # keep to max 1
            # W[j, :] = res.x
            
        # --- 3. Update A (Weights W fixed) ---
        # Solve the unconstrained problem
        if not use_lasso:
            A_vec_unc, _, _, _ = np.linalg.lstsq(W, C_vec, rcond=None)
        else:
            sparsity_penalty = lasso_alpha
            lasso_model = Lasso(alpha=sparsity_penalty, fit_intercept=False, max_iter=1000)
            lasso_model.fit(W, C_vec)
            A_vec_unc = lasso_model.coef_.T
        
        # Reshape unconstrained solution
        A_unc = A_vec_unc.reshape(k, n, n)
        
        # Project AND normalize each archetype
        for j in range(k):
            # First, project onto the PSD cone
            A_psd = project_psd(A_unc[j, :, :])
            
            # --- START: NEW NORMALIZATION STEP ---
            current_trace = np.trace(A_psd)
            
            # Avoid division by zero if archetype collapses
            if current_trace > 1e-9:
                A[j, :, :] = A_psd * (target_trace / current_trace)
            else:
                A[j, :, :] = A_psd # Keep it as a zero matrix
            # --- END: NEW NORMALIZATION STEP ---
            
        # --- 4. Check Convergence ---
        current_error = np.linalg.norm(C_vec - W @ A.reshape(k, -1), 'fro')
        
        if (prev_error - current_error) < tol:
            print(f"Converged at iteration {i+1} with error {current_error:.4f}")
            break
            
        prev_error = current_error
        
        if (i+1) % 10 == 0:
            current_error_per_matrix = current_error / N
            print(f"Iteration {i+1}/{n_iter}, quadratic error per matrix: {current_error_per_matrix:.4f}")
            
    return W, A, current_error


def decompose_psd_matrices(C_data_new, A_archetypes, mode='non-negative', max_weight=1.0):
    """
    Decomposes a new set of matrices into weights based on fixed archetypes.

    Args:
        C_data_new (np.array): The new (N_test, n, n) matrices to analyze.
        A_archetypes (np.array): The fixed (k, n, n) archetypes from training.
        mode (str): The constraint to use for solving:
            'non-negative': w >= 0 (Matches the 'nnls' in your training function)
            'capped': 0 <= w <= max_weight (Matches your 'lsq_linear' comment)
        max_weight (float): The upper bound to use when mode='capped'.

    Returns:
        np.array: The k-dimensional weight vectors, shape (N_test, k).
    """
    
    N_test, n, _ = C_data_new.shape
    k = A_archetypes.shape[0]

    # 1. Vectorize the problem
    C_vec_new = C_data_new.reshape(N_test, -1)
    A_vec_T = A_archetypes.reshape(k, -1).T # Shape (n*n, k)
    
    # Initialize the new weight matrix
    W_new = np.zeros((N_test, k))

    # 2. Loop over each new matrix and solve for its weights
    for j in range(N_test):
        y = C_vec_new[j] # The (n*n,) vector for this matrix
        
        if mode == 'non-negative':
            # This matches the 'nnls' in your fit_psd_factorization
            w, _ = nnls(A_vec_T, y)
        
        elif mode == 'capped':
            # This matches your 'lsq_linear' commented-out line
            res = lsq_linear(A_vec_T, y, bounds=(0, max_weight))
            w = res.x
        
        else:
            raise ValueError(f"Unknown mode: '{mode}'. Must be 'non-negative' or 'capped'.")
        
        # scale the fit error by number of elements 
        
        W_new[j, :] = w
    
    fit_error = np.linalg.norm(C_vec_new - W_new @ A_archetypes.reshape(k, -1), 'fro')

    return W_new, fit_error

def save_archetypes(filepath, A_archetypes, **other_data):
    """
    Saves the archetypes (and any other data) to a single .npz file.

    Args:
        filepath (str): The path to the file (e..g, "my_model.npz").
        A_archetypes (np.array): The (k, n, n) array of archetypes.
        **other_data: Any other variables you want to save,
                      e.g., k=5, target_trace=1.0
    """
    # Use savez_compressed for efficiency
    # The main archetypes are saved under the key 'A'
    # Any other data is saved with its keyword name
    np.savez_compressed(filepath, A=A_archetypes, **other_data)
    print(f"Archetypes successfully saved to {filepath}")

def load_archetypes(filepath):
    """
    Loads archetypes (and any other data) from an .npz file.

    Args:
        filepath (str): The path to the file (e.g., "my_model.npz").

    Returns:
        np.load: A dictionary-like object with all the saved data.
                 - The archetypes are in data['A']
                 - Other saved data is in data['key_name']
    """
    # np.load returns a "lazy" NpzFile object
    data = np.load(filepath)
    print(f"Archetypes successfully loaded from {filepath}")
    return data


# file I/O
# %%

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

    
class ArchetypeAnalyzer:
    """
    A comprehensive class for archetype analysis of occupation matrices from magnetic calculations.
    
    Handles data loading from JSON, matrix construction, archetype fitting, and progressive 
    multi-atom training with configurable retraining thresholds.
    """
    
    def __init__(self, retraining_threshold_factor=3.0):
        """
        Initialize the ArchetypeAnalyzer.
        
        Args:
            retraining_threshold_factor (float): Factor to multiply average residue 
                                               to determine retraining threshold
        """
        self.raw_data = None  # Raw JSON data
        self.dataframe = None  # Processed pandas DataFrame
        self.matrices = {}  # Store matrices by atom: {atom_id: {'up': matrices, 'down': matrices, 'combined': matrices}}
        self.archetypes = {}  # Store fitted archetypes by atom: {atom_id: A_matrices}
        self.weights = {}  # Store weights by atom: {atom_id: W_matrices}
        self.training_errors = {}  # Store training errors by atom
        self.retraining_threshold_factor = retraining_threshold_factor
        
    def load_csv_data(self, csv_file_path):
        """
        Load and process CSV data containing occupation matrices.
        
        Args:
            csv_file_path (str): Path to the CSV file
        """
        print(f"Loading CSV data from: {csv_file_path}")
        
        try:
            self.dataframe = pd.read_csv(csv_file_path)
            # Sort by energy for consistency
            if 'total_energy_eV' in self.dataframe.columns:
                self.dataframe = self.dataframe.sort_values(by='total_energy_eV')
            print(f"Successfully loaded {len(self.dataframe)} calculations")
            return self.dataframe
        except FileNotFoundError:
            print(f"Error: CSV file not found at '{csv_file_path}'.")
            return None

    def load_json_data(self, json_file_path):
        """
        Load and process JSON data containing calculation results.
        
        Args:
            json_file_path (str): Path to the JSON file
        """
        print(f"Loading data from: {json_file_path}")
        
        try:
            with open(json_file_path, 'r') as f:
                self.raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at '{json_file_path}'.")
            return
            
        # Process the data into a DataFrame
        self._process_raw_data()
        print(f"Successfully loaded {len(self.dataframe)} calculations")
        return self.dataframe
        
    def _process_raw_data(self):
        """Process raw JSON data into a structured DataFrame with occupation matrices."""
        if self.raw_data is None:
            print("No raw data loaded. Please load JSON data first.")
            return
            
        results = []
        calculations = self.raw_data.get('calculations', {})
        
        for calc_id, calc_data in calculations.items():
            if not isinstance(calc_data, dict):
                continue
                
            # Extract base properties
            properties = self._extract_base_properties(calc_data)
            
            # Extract occupation matrices
            occupation_data = self._extract_occupation_matrices(calc_data)
            
            # Combine results
            result_entry = {
                'calculation_id': calc_id,
                **properties,
                **occupation_data
            }
            results.append(result_entry)
            
        self.dataframe = pd.DataFrame(results)
        # Sort by energy for consistency
        if 'total_energy_eV' in self.dataframe.columns:
            self.dataframe = self.dataframe.sort_values(by='total_energy_eV')
            
    def _extract_base_properties(self, calc_data):
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
        
    def _extract_occupation_matrices(self, calc_data):
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
                if matrix:  # Only process if matrix exists
                    for i in range(len(matrix)):
                        for j in range(i, len(matrix)):  # Upper triangle including diagonal
                            key = f"atom_{atom_idx}_{spin}_occ_{i+1}_{j+1}"
                            occupation_data[key] = matrix[i][j]
                            
        return occupation_data
        
    def detect_atom_orbitals(self, atom_id):
        """
        Auto-detect the number of orbitals for a given atom by examining column names.
        
        Args:
            atom_id (int): The atom ID to examine
            
        Returns:
            int: Number of orbitals detected
        """
        if self.dataframe is None:
            print("No data loaded. Please load JSON data first.")
            return 0
            
        # Look for columns matching pattern atom_{atom_id}_{spin}_occ_{i}_{j}
        pattern_cols = [col for col in self.dataframe.columns 
                       if col.startswith(f'atom_{atom_id}_') and '_occ_' in col]
        
        if not pattern_cols:
            print(f"No occupation data found for atom {atom_id}")
            return 0
            
        # Find the maximum orbital index
        max_orbital = 0
        for col in pattern_cols:
            parts = col.split('_')
            if len(parts) >= 6:  # atom_X_spin_occ_i_j
                try:
                    i, j = int(parts[4]), int(parts[5])
                    max_orbital = max(max_orbital, i, j)
                except ValueError:
                    continue
                    
        return max_orbital
        
    def build_matrices_from_occupations(self, atom_id, spin='both'):
        """
        Build symmetric matrices from occupation data for a specific atom.
        
        Args:
            atom_id (int): The atom ID to process
            spin (str): 'up', 'down', or 'both' for combined analysis
            
        Returns:
            dict: Dictionary containing matrices for the specified spin(s)
        """
        if self.dataframe is None:
            print("No data loaded. Please load JSON data first.")
            return {}
            
        n_orbitals = self.detect_atom_orbitals(atom_id)
        if n_orbitals == 0:
            print(f"Could not detect orbitals for atom {atom_id}")
            return {}
            
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
        
    def _build_single_matrix(self, row, atom_id, spin_type, n_orbitals):
        """Build a single symmetric matrix from a data row."""
        matrix = np.zeros((n_orbitals, n_orbitals))
        
        for i in range(n_orbitals):
            for j in range(i, n_orbitals):
                col_name = f'atom_{atom_id}_{spin_type}_occ_{i+1}_{j+1}'
                
                if col_name in row and pd.notna(row[col_name]):
                    value = row[col_name]
                    matrix[i, j] = value
                    if i != j:  # Fill symmetric part
                        matrix[j, i] = value
                else:
                    # If data is missing, return None to skip this matrix
                    return None
                    
        return matrix
        
    def fit_archetypes(self, atom_id, k, matrix_type='combined', target_trace=1.0, 
                      train_test_split=0.6, **kwargs):
        """
        Fit archetypes for a specific atom using the specified matrix type.
        
        Args:
            atom_id (int): The atom ID to fit archetypes for
            k (int): Number of archetypes
            matrix_type (str): 'up', 'down', or 'combined'
            target_trace (float): Target trace for archetypes
            train_test_split (float): Fraction of data to use for training
            **kwargs: Additional arguments for fit_psd_factorization
            
        Returns:
            tuple: (W_weights, A_archetypes, training_error, test_error)
        """
        if atom_id not in self.matrices:
            print(f"No matrices found for atom {atom_id}. Please build matrices first.")
            return None, None, None, None
            
        if matrix_type not in self.matrices[atom_id]:
            print(f"Matrix type '{matrix_type}' not available for atom {atom_id}")
            return None, None, None, None
            
        matrices = self.matrices[atom_id][matrix_type]
        
        # Split into training and testing
        np.random.seed(42)  # For reproducible results
        shuffled_indices = np.random.permutation(len(matrices))
        split_idx = int(len(matrices) * train_test_split)
        
        train_matrices = matrices[shuffled_indices[:split_idx]]
        test_matrices = matrices[shuffled_indices[split_idx:]]
        
        print(f"Training on {len(train_matrices)} matrices, testing on {len(test_matrices)} matrices")
        
        # Fit archetypes on training data
        W_train, A_archetypes, train_error = fit_psd_factorization(
            train_matrices, k=k, target_trace=target_trace, **kwargs
        )
        
        # Test on validation set
        if len(test_matrices) > 0:
            W_test, test_error = decompose_psd_matrices(test_matrices, A_archetypes, mode='non-negative')
        else:
            W_test, test_error = None, None
            
        # Store results
        self.archetypes[atom_id] = A_archetypes
        self.weights[atom_id] = {'train': W_train, 'test': W_test}
        self.training_errors[atom_id] = {'train': train_error, 'test': test_error}
        
        print(f"Atom {atom_id}: Training error = {train_error:.4f}")
        if test_error is not None:
            print(f"Atom {atom_id}: Test error = {test_error:.4f}")
        
        # After training, fit ALL matrices (up and down separately) for easy access
        print(f"Fitting archetypes to all matrices for atom {atom_id}...")
        self._fit_all_matrices_with_archetypes(atom_id)
            
        return W_train, A_archetypes, train_error, test_error
    
    
    def _fit_all_matrices_with_archetypes(self, atom_id):
        """
        Internal method to fit all matrices (up, down, combined) with the learned archetypes.
        This is called automatically after training to provide complete weight matrices.
        """
        if atom_id not in self.archetypes or atom_id not in self.matrices:
            return
            
        A_archetypes = self.archetypes[atom_id]
        matrices = self.matrices[atom_id]
        
        # Store complete weights for each matrix type
        complete_weights = {}
        complete_errors = {}
        
        for matrix_type in ['up', 'down', 'combined']:
            if matrix_type in matrices:
                print(f"  Fitting {matrix_type} matrices...")
                W_complete, error = decompose_psd_matrices(
                    matrices[matrix_type], A_archetypes, mode='non-negative'
                )
                complete_weights[matrix_type] = W_complete
                complete_errors[matrix_type] = error
                avg_error = error / len(matrices[matrix_type]) if len(matrices[matrix_type]) > 0 else 0
                print(f"    {matrix_type}: {len(matrices[matrix_type])} matrices, "
                      f"total error: {error:.4f}, avg error per matrix: {avg_error:.4f}")
        
        # Store the complete results
        if not hasattr(self, 'complete_weights'):
            self.complete_weights = {}
        if not hasattr(self, 'complete_errors'):
            self.complete_errors = {}
            
        self.complete_weights[atom_id] = complete_weights
        self.complete_errors[atom_id] = complete_errors
    
    # Now use the archetypes to fit the entire set of matrices

    def get_archetypes_weights(self, atom_id):
        """
        Retrieve the archetypes and weights for a specific atom for both up and down spins.
        
        Args:
            atom_id (int): The atom ID to retrieve data for
        Returns:
            weights_up (np.array): Weights for up spin matrices
            weights_down (np.array): Weights for down spin matrices
        """
        if atom_id not in self.archetypes:
            print(f"No archetypes found for atom {atom_id}. Please fit archetypes first.")
            return None, None
            
        # Try to use precomputed complete weights first
        if (hasattr(self, 'complete_weights') and 
            atom_id in self.complete_weights and
            'up' in self.complete_weights[atom_id] and 
            'down' in self.complete_weights[atom_id]):
            
            return (self.complete_weights[atom_id]['up'], 
                   self.complete_weights[atom_id]['down'])
        
        # Fallback: compute weights on demand (for backward compatibility)
        print(f"Complete weights not found for atom {atom_id}. Computing on demand...")
        A_archetypes = self.archetypes[atom_id]
        
        if atom_id not in self.matrices:
            print(f"No matrices found for atom {atom_id}. Please build matrices first.")
            return None, None
            
        matrices = self.matrices[atom_id]
        
        if 'up' not in matrices or 'down' not in matrices:
            print(f"Both 'up' and 'down' matrices are required for atom {atom_id}.")
            return None, None
            
        up_matrices = matrices['up']
        down_matrices = matrices['down']
        
        # Decompose up and down matrices separately
        W_up, _ = decompose_psd_matrices(up_matrices, A_archetypes, mode='non-negative')
        W_down, _ = decompose_psd_matrices(down_matrices, A_archetypes, mode='non-negative')
        
        return W_up, W_down
    
    def get_weights_by_spin(self, atom_id, spin_type):
        """
        Get weights for a specific atom and spin type.
        
        Args:
            atom_id (int): The atom ID
            spin_type (str): 'up', 'down', or 'combined'
            
        Returns:
            np.array: Weights matrix of shape (n_matrices, k) or None if not found
        """
        if not hasattr(self, 'complete_weights') or atom_id not in self.complete_weights:
            print(f"Complete weights not found for atom {atom_id}. Please fit archetypes first.")
            return None
            
        if spin_type not in self.complete_weights[atom_id]:
            print(f"Spin type '{spin_type}' not available for atom {atom_id}")
            print(f"Available types: {list(self.complete_weights[atom_id].keys())}")
            return None
            
        return self.complete_weights[atom_id][spin_type]
    
    def get_matrix_weights(self, atom_id, spin_type, matrix_idx):
        """
        Get weights for a specific matrix by atom ID, spin type, and matrix index.
        
        Args:
            atom_id (int): The atom ID
            spin_type (str): 'up', 'down', or 'combined' 
            matrix_idx (int): Index of the matrix
            
        Returns:
            np.array: Weight vector for the specific matrix or None if not found
        """
        weights = self.get_weights_by_spin(atom_id, spin_type)
        if weights is None:
            return None
            
        if matrix_idx >= len(weights):
            print(f"Matrix index {matrix_idx} out of range. Available: 0-{len(weights)-1}")
            return None
            
        return weights[matrix_idx]
    
    def get_archetypes(self, atom_id):
        """
        Get the archetypes for a specific atom.
        
        Args:
            atom_id (int): The atom ID
            
        Returns:
            np.array: Archetype matrices of shape (k, n_orbitals, n_orbitals) or None if not found
        """
        if atom_id not in self.archetypes:
            print(f"No archetypes found for atom {atom_id}. Please fit archetypes first.")
            return None
            
        return self.archetypes[atom_id]
    
    def get_matrix(self, atom_id, spin_type, matrix_idx):
        """
        Get a specific matrix by atom ID, spin type, and matrix index.
        
        Args:
            atom_id (int): The atom ID
            spin_type (str): 'up', 'down', or 'combined'
            matrix_idx (int): Index of the matrix
            
        Returns:
            np.array: The specific matrix or None if not found
        """
        if atom_id not in self.matrices:
            print(f"No matrices found for atom {atom_id}")
            return None
            
        if spin_type not in self.matrices[atom_id]:
            print(f"Spin type '{spin_type}' not available for atom {atom_id}")
            return None
            
        matrices = self.matrices[atom_id][spin_type]
        if matrix_idx >= len(matrices):
            print(f"Matrix index {matrix_idx} out of range. Available: 0-{len(matrices)-1}")
            return None
            
        return matrices[matrix_idx]
        
    def progressive_multi_atom_training(self, atom_ids, k_values, matrix_type='combined', 
                                      target_trace=1.0, **kwargs):
        """
        Train archetypes progressively across multiple atoms with optional retraining.
        
        When retraining is needed, uses the previous atom's archetypes as initialization
        to potentially achieve better convergence and consistency.
        
        Args:
            atom_ids (list): List of atom IDs to process
            k_values (dict, int, or list): Number of archetypes per atom. 
                                         - dict: {atom_id: k_value}
                                         - int: same k for all atoms
                                         - list: k_values[i] for atom_ids[i]
            matrix_type (str): 'up', 'down', or 'combined'
            target_trace (float): Target trace for archetypes
            **kwargs: Additional arguments for fit_psd_factorization
        """
        if isinstance(k_values, int):
            k_values = {atom_id: k_values for atom_id in atom_ids}
        elif isinstance(k_values, list):
            # Convert list to dictionary mapping atom_ids to k_values
            if len(k_values) != len(atom_ids):
                raise ValueError(f"k_values list length ({len(k_values)}) must match atom_ids length ({len(atom_ids)})")
            k_values = {atom_ids[i]: k_values[i] for i in range(len(atom_ids))}
            
        results = {}
        
        for i, atom_id in enumerate(atom_ids):
            print(f"\n--- Processing Atom {atom_id} (step {i+1}/{len(atom_ids)}) ---")
            
            if i == 0:
                # First atom: train from scratch
                W, A, train_err, test_err = self.fit_archetypes(
                    atom_id, k_values[atom_id], matrix_type, target_trace, **kwargs
                )
                results[atom_id] = {
                    'method': 'fresh_training',
                    'train_error': train_err,
                    'test_error': test_err,
                    'retraining_threshold': None
                }
                
            else:
                # Subsequent atoms: test with previous archetypes first
                prev_atom_id = atom_ids[i-1]
                if prev_atom_id not in self.archetypes:
                    print(f"Warning: No archetypes found for previous atom {prev_atom_id}")
                    continue
                    
                # Test current atom with previous archetypes
                current_matrices = self.matrices[atom_id][matrix_type]
                prev_archetypes = self.archetypes[prev_atom_id]
                
                W_test, test_error = decompose_psd_matrices(
                    current_matrices, prev_archetypes, mode='non-negative'
                )
                
                # Calculate average error per matrix
                avg_error_per_matrix = test_error / len(current_matrices)
                
                # Compare with previous atom's average error
                prev_train_error = self.training_errors[prev_atom_id]['train']
                prev_matrices_count = len(self.matrices[prev_atom_id][matrix_type])
                prev_avg_error = prev_train_error / prev_matrices_count
                
                retraining_threshold = self.retraining_threshold_factor * prev_avg_error
                
                print(f"Testing atom {atom_id} with atom {prev_atom_id} archetypes:")
                print(f"  Average error per matrix: {avg_error_per_matrix:.4f}")
                print(f"  Retraining threshold: {retraining_threshold:.4f}")
                
                if avg_error_per_matrix <= retraining_threshold:
                    # Use previous archetypes
                    print(f"  -> Using atom {prev_atom_id} archetypes (no retraining needed)")
                    self.archetypes[atom_id] = prev_archetypes
                    self.weights[atom_id] = {'test': W_test}
                    self.training_errors[atom_id] = {'test': test_error}

                    # make the decomposition for all matrices for the atom 
                    print(f"Fitting archetypes to all matrices for atom {atom_id}...")
                    self._fit_all_matrices_with_archetypes(atom_id)
                    
                    results[atom_id] = {
                        'method': 'reused_archetypes',
                        'source_atom': prev_atom_id,
                        'test_error': test_error,
                        'avg_error_per_matrix': avg_error_per_matrix,
                        'retraining_threshold': retraining_threshold
                    }
                    
                else:
                    # Retrain with previous archetypes as initialization
                    print(f"  -> Retraining needed (error too high)")
                    
                    # Use previous archetypes as starting guess for retraining
                    kwargs_with_init = kwargs.copy()
                    kwargs_with_init['starting_guess_archetypes'] = prev_archetypes
                    
                    W, A, train_err, test_err = self.fit_archetypes(
                        atom_id, k_values[atom_id], matrix_type, target_trace, **kwargs_with_init
                    )
                    
                    results[atom_id] = {
                        'method': 'retrained',
                        'source_atom': prev_atom_id,
                        'train_error': train_err,
                        'test_error': test_err,
                        'avg_error_per_matrix': avg_error_per_matrix,
                        'retraining_threshold': retraining_threshold
                    }
                    
        return results
        
    def get_available_atoms(self):
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
        
    def save_results(self, filepath, include_matrices=False):
        """
        Save analysis results to a file.
        
        Args:
            filepath (str): Path to save the results
            include_matrices (bool): Whether to include the full matrices
        """
        save_data = {
            'archetypes': self.archetypes,
            'training_errors': self.training_errors,
            'retraining_threshold_factor': self.retraining_threshold_factor,
        }
        
        # Always save complete weights and errors if available
        if hasattr(self, 'complete_weights'):
            save_data['complete_weights'] = self.complete_weights
        if hasattr(self, 'complete_errors'):
            save_data['complete_errors'] = self.complete_errors
        
        if include_matrices:
            save_data['matrices'] = self.matrices
            save_data['weights'] = self.weights  # Keep old training/test weights for backward compatibility
            
        np.savez_compressed(filepath, **save_data)
        print(f"Results saved to {filepath}")
        
    def load_results(self, filepath):
        """Load previously saved analysis results."""
        data = np.load(filepath, allow_pickle=True)
        
        # Load with proper conversion from numpy arrays to dicts
        self.archetypes = data['archetypes'].item() if 'archetypes' in data else {}
        self.training_errors = data['training_errors'].item() if 'training_errors' in data else {}
        self.retraining_threshold_factor = float(data['retraining_threshold_factor']) if 'retraining_threshold_factor' in data else 2.0
        
        if 'matrices' in data:
            self.matrices = data['matrices'].item()
        if 'weights' in data:
            self.weights = data['weights'].item()
        if 'complete_weights' in data['complete_weights'].item():
            self.complete_weights = data['complete_weights'].item()
        if 'complete_errors' in data:
            self.complete_errors = data['complete_errors'].item()
            
        print(f"Results loaded from {filepath}")
        
    def visualize_archetype(self, atom_id, archetype_idx, figsize=(6, 5)):
        """
        Visualize a specific archetype as a heatmap.
        
        Args:
            atom_id (int): The atom ID
            archetype_idx (int): Index of the archetype to visualize
            figsize (tuple): Figure size for the plot
        """
        import matplotlib.pyplot as plt
        
        if atom_id not in self.archetypes:
            print(f"No archetypes found for atom {atom_id}")
            return
            
        A = self.archetypes[atom_id]
        if archetype_idx >= len(A):
            print(f"Archetype index {archetype_idx} out of range. Available: 0-{len(A)-1}")
            return
            
        archetype = A[archetype_idx]
        
        plt.figure(figsize=figsize)
        plt.title(f'Atom {atom_id} - Archetype {archetype_idx + 1}')
        im = plt.imshow(abs(archetype), cmap='viridis', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(archetype.shape[0]):
            for j in range(archetype.shape[1]):
                plt.text(j, i, f"{archetype[i,j]:.2f}", ha='center', va='center',
                        color='white' if archetype[i,j] < (archetype.max()/2) else 'black',
                        fontdict={'fontsize':10})
        plt.colorbar(im)
        plt.tight_layout()
        plt.show()
        
    def visualize_all_archetypes(self, atom_id, figsize=(16, 8)):
        """
        Visualize all archetypes for a specific atom.
        
        Args:
            atom_id (int): The atom ID
            figsize (tuple): Figure size for the plot
        """
        import matplotlib.pyplot as plt
        
        if atom_id not in self.archetypes:
            print(f"No archetypes found for atom {atom_id}")
            return
            
        A = self.archetypes[atom_id]
        k = len(A)
        
        # Calculate subplot layout
        ncols = min(4, k)
        nrows = (k + ncols - 1) // ncols
        
        plt.figure(figsize=figsize)
        for i in range(k):
            plt.subplot(nrows, ncols, i + 1)
            plt.title(f'Archetype {i + 1}')
            im = plt.imshow(abs(A[i]), cmap='viridis', vmin=0, vmax=1)
            
            # Add text annotations
            for m in range(A[i].shape[0]):
                for n in range(A[i].shape[1]):
                    plt.text(n, m, f"{A[i][m,n]:.2f}", ha='center', va='center',
                            color='white' if A[i][m,n] < (A[i].max()/2) else 'black',
                            fontdict={'fontsize':8})
            plt.colorbar(im)
            
        plt.tight_layout()
        plt.show()
        
            
    def get_matrices_by_spin(self, atom_id):
        """
        Get matrices organized by spin channel for analysis.
        
        Args:
            atom_id (int): The atom ID
            
        Returns:
            dict: Matrices organized by spin channel
        """
        if atom_id not in self.matrices:
            print(f"No matrices found for atom {atom_id}")
            return {}
            
        return self.matrices[atom_id]
        
    def compare_atoms_errors(self):
        """Compare training/test errors across all atoms."""
        if not self.training_errors:
            print("No training errors available")
            return
            
        print("\n--- Error Comparison Across Atoms ---")
        for atom_id, errors in self.training_errors.items():
            print(f"Atom {atom_id}:")
            for error_type, error_val in errors.items():
                if error_val is not None:
                    n_matrices = len(self.matrices[atom_id].get('combined', []))
                    avg_error = error_val / n_matrices if n_matrices > 0 else error_val
                    print(f"  {error_type}: {error_val:.4f} (avg per matrix: {avg_error:.4f})")
                    
    def reconstruct_and_compare(self, atom_id, matrix_idx=0, matrix_type='combined'):
        """
        Reconstruct a specific matrix and show comparison with original.
        
        Args:
            atom_id (int): The atom ID
            matrix_idx (int): Index of the matrix to reconstruct
            matrix_type (str): Type of matrix ('combined', 'up', 'down')
        """
        import matplotlib.pyplot as plt
        
        if atom_id not in self.matrices or matrix_type not in self.matrices[atom_id]:
            print(f"No {matrix_type} matrices found for atom {atom_id}")
            return
            
        if atom_id not in self.archetypes:
            print(f"No archetypes found for atom {atom_id}")
            return
            
        matrices = self.matrices[atom_id][matrix_type]
        archetypes = self.archetypes[atom_id]
        
        if matrix_idx >= len(matrices):
            print(f"Matrix index {matrix_idx} out of range. Available: 0-{len(matrices)-1}")
            return
            
        # Get weights using the new complete weights system
        weights = self.get_matrix_weights(atom_id, matrix_type, matrix_idx)
        if weights is None:
            print("No weights available for reconstruction")
            return
            
        # Reconstruct
        original = matrices[matrix_idx]
        reconstructed = np.zeros_like(original)
        for j, w in enumerate(weights):
            reconstructed += w * archetypes[j]
            
        error = np.linalg.norm(original - reconstructed, 'fro')

        vmax = max(np.abs(original).max(), np.abs(reconstructed).max())
        vmin = min(np.abs(original).min(), np.abs(reconstructed).min())
        
        # Plot comparison
        plt.figure(figsize=(15, 4))
        
        plt.subplot(1, 4, 1)
        plt.title('Original Matrix')
        plt.imshow(original, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar()
        
        plt.subplot(1, 4, 2)
        plt.title('Reconstructed Matrix')
        plt.imshow(reconstructed, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar()
        
        plt.subplot(1, 4, 3)
        plt.title('Error Matrix')
        plt.imshow(original - reconstructed, cmap='bwr')
        plt.colorbar()
        
        plt.subplot(1, 4, 4)
        plt.title('Weights')
        plt.bar(range(len(weights)), weights)
        plt.xlabel('Archetype')
        plt.ylabel('Weight')

        # add all ticks for last plot
        plt.xticks(range(len(weights)), [f'A{j+1}' for j in range(len(weights))])
        
        plt.suptitle(f'Atom {atom_id} - Matrix {matrix_idx} ({matrix_type}) - Error: {error:.4f}')
        plt.tight_layout()
        plt.show()
        
        return error

    def reconstruct_matrix(self, atom_id, spin_type, matrix_idx):
        """
        Reconstruct a specific matrix using the learned archetypes and weights.
        
        Args:
            atom_id (int): The atom ID
            spin_type (str): 'up', 'down', or 'combined'
            matrix_idx (int): Index of the matrix to reconstruct
            
        Returns:
            tuple: (original_matrix, reconstructed_matrix, reconstruction_error) or (None, None, None)
        """
        # Get original matrix
        original = self.get_matrix(atom_id, spin_type, matrix_idx)
        if original is None:
            return None, None, None
            
        # Get weights
        weights = self.get_matrix_weights(atom_id, spin_type, matrix_idx)
        if weights is None:
            return None, None, None
            
        # Get archetypes
        archetypes = self.get_archetypes(atom_id)
        if archetypes is None:
            return None, None, None
            
        # Reconstruct
        reconstructed = np.zeros_like(original)
        for j, w in enumerate(weights):
            reconstructed += w * archetypes[j]
            
        error = np.linalg.norm(original - reconstructed, 'fro')
        
        return original, reconstructed, error
    
    def get_reconstruction_quality(self, atom_id, spin_type=None):
        """
        Get reconstruction quality statistics for an atom.
        
        Args:
            atom_id (int): The atom ID
            spin_type (str): 'up', 'down', 'combined', or None for all types
            
        Returns:
            dict: Statistics about reconstruction quality
        """
        if not hasattr(self, 'complete_errors') or atom_id not in self.complete_errors:
            print(f"Complete errors not found for atom {atom_id}")
            return {}
            
        results = {}
        
        if spin_type is None:
            # Get stats for all available spin types
            for spin_t in self.complete_errors[atom_id]:
                if spin_t in self.matrices[atom_id]:
                    n_matrices = len(self.matrices[atom_id][spin_t])
                    total_error = self.complete_errors[atom_id][spin_t]
                    avg_error = total_error / n_matrices if n_matrices > 0 else 0
                    
                    results[spin_t] = {
                        'total_error': total_error,
                        'avg_error_per_matrix': avg_error,
                        'n_matrices': n_matrices
                    }
        else:
            # Get stats for specific spin type
            if spin_type in self.complete_errors[atom_id] and spin_type in self.matrices[atom_id]:
                n_matrices = len(self.matrices[atom_id][spin_type])
                total_error = self.complete_errors[atom_id][spin_type]
                avg_error = total_error / n_matrices if n_matrices > 0 else 0
                
                results = {
                    'total_error': total_error,
                    'avg_error_per_matrix': avg_error,
                    'n_matrices': n_matrices
                }
                
        return results
    
    def plot_weight_distributions(self, atom_id, spin_type=None, figsize=(12, 8)):
        """
        Plot weight distributions for different archetypes.
        
        Args:
            atom_id (int): The atom ID
            spin_type (str): 'up', 'down', 'combined', or None for all available types
            figsize (tuple): Figure size
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'complete_weights') or atom_id not in self.complete_weights:
            print(f"Complete weights not found for atom {atom_id}")
            return
            
        # Determine which spin types to plot
        if spin_type is None:
            spin_types = list(self.complete_weights[atom_id].keys())
        else:
            spin_types = [spin_type] if spin_type in self.complete_weights[atom_id] else []
            
        if not spin_types:
            print(f"No valid spin types found for atom {atom_id}")
            return
            
        n_spin_types = len(spin_types)
        fig, axes = plt.subplots(n_spin_types, 1, figsize=figsize)
        if n_spin_types == 1:
            axes = [axes]
            
        for i, spin_t in enumerate(spin_types):
            weights = self.complete_weights[atom_id][spin_t]
            n_archetypes = weights.shape[1]
            
            # Create box plots for each archetype
            weight_data = [weights[:, j] for j in range(n_archetypes)]
            
            axes[i].boxplot(weight_data, labels=[f'A{j+1}' for j in range(n_archetypes)])
            axes[i].set_title(f'Weight Distributions - Atom {atom_id} - {spin_t.upper()} Spin')
            axes[i].set_xlabel('Archetype')
            axes[i].set_ylabel('Weight Value')
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
    
   
    
  

#%%
