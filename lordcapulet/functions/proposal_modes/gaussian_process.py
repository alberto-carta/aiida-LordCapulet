
"""
Gaussian Process mode for generating occupation matrix proposals.

This module implements Bayesian optimization using Gaussian Processes
to propose occupation matrices that minimize DFT+U total energy.
"""


from pyexpat import model
import numpy as np
import torch
import warnings
import time
from typing import List, Dict, Any, Optional
from functools import partial

# lordcapulet imports
from lordcapulet.data_structures import OccupationMatrixData, DataBank
from .shared_functionality import  create_patchwork_guess

# Bayesian components
from .Bayesian.gp_model import create_gp_model, print_kernel_diagnostics, train_gp_model, evaluate_loo_cv
from .Bayesian.mean_functions import VectorizedPhysicsMean
from .Bayesian.kernels import build_kernel
from .Bayesian.gp_model import print_kernel_diagnostics
from .Bayesian.acquisition import AnalyticCustomPreference, compute_total_preference_fast, BatchedAcqFunc

# BoTorch components
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.generation.sampling import BoltzmannSampling
from botorch.exceptions import InputDataWarning
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.constraints import GreaterThan




# Suppress the BoTorch float32 warning
warnings.filterwarnings("ignore", category=InputDataWarning)



def propose_gaussian_process_constraints(
    occ_matr_list: list[OccupationMatrixData],
    energies: list[float],
    natoms: int,
    N: int,
    gp_config: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    reporter=None,
    **kwargs) -> List[OccupationMatrixData]:
    """
    Generate N occupation matrix proposals using Gaussian Process optimization.
    
    Strategy:
    1. Create DataBank from input matrices and energies
    2. Train a Gaussian Process model to predict energy from occupation matrices
    3. Validate using Leave-One-Out Cross-Validation
    4. Generate proposals (placeholder for now - returns best training samples)
    
    :param occ_matr_list: List of OccupationMatrixData objects from previous calculations
    :param natoms: Number of atoms in the system
    :param N: Number of proposals to generate
    :param debug: Whether to reporter debug information
    :param energies: List of total energies corresponding to each occupation matrix (required)
    :param reporter: Optional callable for logging (if None, uses reporter)
    :param kwargs: Additional parameters (device, mean_config, kernel_config, etc.)
    
    :return: List of N OccupationMatrixData objects (proposals)
    
    :raises ValueError: If energies not provided or length mismatch
    """
    
    # Setup reporter
    if reporter is None:
        reporter = print
    
    # ========================================================================
    # STEP 1: Input Validation
    # ========================================================================
    
    # check if gp_config is provided in the kwargs, if not use default
    if gp_config is None:
        reporter(f"No configuration provided, using default settings.")
        gp_config = {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                
                # Physics Mean Function
                "mean": {
                    "type": "VectorizedPhysicsMean",
                    # "type": "HubbardUMean",
                    "J_prior": {"mean": 0.5, "std": 0.1},
                    "U_prior": {"mean": 5.0, "std": 0.2},
                },
                
                # Kernel Configuration
                "kernel": {
                    "local": {
                        "matern": {"enabled": True, "nu": 2.5, "outputscale_prior": {"mean": 0.2, "std": 0.02}},
                    },
                    "nonlocal": {
                        "residual": {"enabled": True, "outputscale_prior": {"mean": 0.05, "std": 0.01}},
                    },
                    "spin_flip_invariant": True,
                },
                
                # Acquisition Function
                "acquisition": {
                    "beta": 0.3,  # Exploration parameter (0=pure exploitation)
                    "use_preference": True,
                    "trace_target": None,  # Target electron count
                    "trace_sigma": None,  # Increased to allow more variation 
                    "use_eigenvalue_preference": False,  # Eigenvalue constraints
                    "eigenvalue_k": 20000.0,
                    "supergaussian_index": 4,  # Super-Gaussian index for trace preference
                },
                
                # Optimization
                "optimization": {
                    "optim_strategy": "Boltzmann",  # "optimize" or "Boltzmann"
                    "init_strategy": "patchwork",  # "best_train", "patchwork", or None
                    "ensamble_size": 100000,  # Number of candidates to generate for acquisition function, 
                    # this should be high for Boltzmann sampling (>10000), but can be 
                    # lower if you want to optimize the acquisition function instead
                    "patchwork_params": {
                        "apply_rotation": True,
                        "rotation_prob": 0.2,
                        "rotation_type": "Mixed",  # "SO(3)", "SO(N)" or "Mixed"
                    },
                    "Boltzmann_config": {
                        "eta": 30,  # Higher eta = more exploitation (lower temperature), eta is in 1/eV
                    },
                }

                }
        
    
    # report configuration
    reporter(f"--- Gaussian Process Configuration Parameters ---")
    for key, value in gp_config.items():
        reporter(f"  {key}: {value}")
    
    if energies is None:
        raise ValueError("Energies must be provided for Gaussian Process proposal mode")
    
    if len(energies) != len(occ_matr_list):
        raise ValueError(
            f"Length mismatch: {len(energies)} energies but {len(occ_matr_list)} matrices"
        )
    
    if len(occ_matr_list) < 3:
        raise ValueError(
            f"Need at least 3 samples for GP training, got {len(occ_matr_list)}"
        )
    
    reporter(f"{'='*60}")
    reporter(f"GAUSSIAN PROCESS PROPOSAL MODE")
    reporter(f"{'='*60}")
    reporter(f"Training samples: {len(occ_matr_list)}")
    reporter(f"Energy range: [{min(energies):.4f}, {max(energies):.4f}] eV")
    reporter(f"Proposals to generate: {N}")
    
    # ========================================================================
    # STEP 2: Data Preparation - Create DataBank
    # ========================================================================
    
    reporter(f"--- Data Preparation ---")
    
    # Extract metadata from first matrix
    first_occ_data = occ_matr_list[0]
    atom_labels = first_occ_data.get_atom_labels()
    atom_species = {label: first_occ_data[label]['specie'] for label in atom_labels}
    atom_shells = {label: first_occ_data[label]['shell'] for label in atom_labels}
    
    # Create DataBank
    databank = DataBank.from_matrices(
        occ_matrices=occ_matr_list,
        energies=energies,
        pks=None,  # No PKs needed for internal use
        converged=[True] * len(occ_matr_list),  # Assume all converged
        energy_uncertainties=None,
        metadata=None,
        include_electron_number=False,
        include_moment=False
    )
    
    reporter(f"Created DataBank with {len(databank)} entries")
    reporter(f"Atom IDs: {databank.atom_ids}")
    
    # ========================================================================
    # STEP 3: Convert to PyTorch Tensors
    # ========================================================================
    
    # Detect device: try GPU first, fall back to CPU
    if 'device' in kwargs:
        device = kwargs['device']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    local_device = torch.device(device)
    
    atoms = databank.atom_ids
    x_data = databank.to_pytorch(
        atom_ids=atoms,
        spins=['up', 'down'],
        include_energies=False,
        device=local_device
    )
    y_data = torch.tensor(energies, dtype=torch.float32, device=local_device).unsqueeze(-1)
    
    reporter(f"Tensor shapes: X={x_data.shape}, Y={y_data.shape}")
    reporter(f"Device: {local_device}")
    
    
    
    # Get target number of electrons for and their standard deviations
    # to compute the preference trace values

    trace_targets = gp_config['acquisition'].get('trace_target', None)
    trace_sigma = gp_config['acquisition'].get('trace_sigma', None)
    
    if trace_targets is None:
        trace_targets = [databank.get_electron_number(atom_id).mean() for atom_id in atoms]
    
    if trace_sigma is None:
        trace_sigma = [databank.get_electron_number(atom_id).std() for atom_id in atoms]
    
    for atom_id, target, sigma in zip(atoms, trace_targets, trace_sigma):
        reporter(f"  Atom {atom_id}: mean n. electrons = {target:.2f}, std_dev = {sigma:.2f}")
    
    # ========================================================================
    # STEP 4: Create and Train GP Model
    # ========================================================================
    
    reporter(f"--- GP Model Training ---")
    
    mean_config = kwargs.get('mean_config', gp_config['mean'])
    kernel_config = kwargs.get('kernel_config', gp_config['kernel'])


    
    # Create GP model
    model = create_gp_model(
        train_X=x_data,
        train_Y=y_data,
        databank=databank,
        atom_ids=atoms,
        mean_config=mean_config,
        kernel_config=kernel_config,
        device=local_device
    )
    
    # Add numerical stability constraint, basically this prevents situations when you have
    # duplicate points with very similar input features from causing singular matrix inversion issues
    noise_lower_bound = torch.tensor(1e-4, device=local_device)
    model.likelihood.noise_covar.register_constraint(
        "raw_noise", GreaterThan(noise_lower_bound)
    )

    
    # Train the model
    model = train_gp_model(
        model=model,
        train_X=x_data,
        train_Y=y_data,
        training_config={"method": "fit_gpytorch_mll"}
    )
    
    reporter(f"Learned parameters:")
    reporter(f"  U = {model.mean_module.U.item():.4f}")
    
    if hasattr(model.mean_module, 'J'):
            reporter(f"  J = {model.mean_module.J.item():.4f}")
            reporter(f"  Constant = {model.mean_module.constant.item():.4f} eV")


    # also reporter diagnostics 
    print_kernel_diagnostics(model, reporter=reporter)

    # ========================================================================
    # STEP 5: Leave-One-Out Cross-Validation
    # ========================================================================
    
    # perform LOO CV evaluation using the function
    loo_results = evaluate_loo_cv(model, x_data, y_data.unsqueeze(-1), tikhonov_reg=1e-3)

    reporter(f"Leave-one-out cross-validation: Q² = {loo_results['q2']:.4f}, the closer to 1 the better")
    reporter(f"Leave-one-out cross-validation: RMSE = {loo_results['rmse']:.4f} eV, the lower the better")


    
    # ========================================================================
    # STEP 6: Generate Proposals 
    # ========================================================================

    # Create acquisition function
    base_acqf = UpperConfidenceBound(
        model=model,
        beta=gp_config["acquisition"]["beta"],
        posterior_transform=ScalarizedPosteriorTransform(weights=torch.tensor([-1.0])),
        maximize=True
    )

    if gp_config["acquisition"]["use_preference"]:
        pref_func = partial(
            compute_total_preference_fast,
            databank=databank,
            atom_ids=atoms,
            trace_target=trace_targets,
            trace_sigma=trace_sigma,
            use_eigenvalue_preference=gp_config["acquisition"].get("use_eigenvalue_preference", False),
            eig_k=gp_config["acquisition"].get("eigenvalue_k", 20000.0),
            supergaussian_index=gp_config["acquisition"]["supergaussian_index"],
        )
        acqf = AnalyticCustomPreference(model=model, base_acqf=base_acqf,
                                        compute_preference_func=pref_func)
    else:
        acqf = base_acqf

    acqf = base_acqf
    batched_acqf = BatchedAcqFunc(acqf, batch_size=100)

    # check optimization strategy
    optim_strategy = gp_config["optimization"].get("optim_strategy", None)
    
    if optim_strategy not in ["Boltzmann"]:
        raise ValueError(f'Optimization strategy "{optim_strategy}" not implemented. Please choose "Boltzmann"')


    if optim_strategy == "Boltzmann":
        # check if the "Boltzmann subdictionary" exists, if not raise error
        if "Boltzmann_config" not in gp_config["optimization"]:
            raise ValueError('Boltzmann_config not found in optimization configuration, please provide "Boltzmann_config" subdictionary inside "optimization"')

        reporter(f"--- Generating Proposals via Boltzmann Sampling ---")
        reporter(f"inverse temperature eta = {gp_config['optimization']['Boltzmann_config']['eta']} 1/eV")
        
         
        
        # get the necessary parameters
        ensamble_size = gp_config["optimization"].get("ensamble_size", 10000)
        eta = gp_config["optimization"]["Boltzmann_config"].get("eta", 30)

        candidate_list = torch.empty((N, x_data.shape[1]), device=local_device)
        ensamble_batch = torch.empty((ensamble_size, x_data.shape[1]), device=local_device)

        
        # get patchwork parameters
        apply_rotation = gp_config["optimization"]["patchwork_params"].get("apply_rotation", True)
        rotation_prob = gp_config["optimization"]["patchwork_params"].get("rotation_prob", 0.2)
        rotation_type = gp_config["optimization"]["patchwork_params"].get("rotation_type", "Mixed")
        
        # time ensamble generation
        start_ensamble = time.time()
        for iguess in range(ensamble_size):
            guess = create_patchwork_guess(
                        databank, x_data, atoms, local_device,
                        apply_rotation=apply_rotation,
                        rotation_prob=rotation_prob,
                        rotation_type=rotation_type,
                        use_torch=True,
                    )
            ensamble_batch[iguess,:] = guess.squeeze(0)
        end_ensamble = time.time()
        reporter(f"Generated ensamble of {ensamble_size} candidates in {end_ensamble - start_ensamble:.2f} seconds")

        sampler = BoltzmannSampling(acq_func=batched_acqf, eta =eta, replacement=False) # high eta, low T
        start_sampling = time.time()

        with torch.no_grad():
            candidates = sampler(ensamble_batch, num_samples = N)

        end_sampling = time.time()

        candidates_list = [candidate.unsqueeze(0) for candidate in candidates]
        acqf_values = [acqf(candidate).item() for candidate in candidates_list]

        reporter(f"Sampled {len(candidates_list)} candidates in {end_sampling - start_sampling:.2f} seconds")


        reporter("="*80)
        reporter("CANDIDATE SUMMARY")
        reporter("="*80)
        reporter(f"{'#':<4} {'Acqf':<12} {'Energy (eV)':<20} {'Uncertainty':<12}")
        reporter("-"*80)

        with torch.no_grad():
            for i, (cand, acqf_val) in enumerate(zip(candidates_list, acqf_values)):
                posterior = model.posterior(cand)
                mean_energy = posterior.mean.item()
                std_energy = posterior.variance.sqrt().item()
                reporter(f"{i:<4} {acqf_val:<12.4f} {mean_energy:<12.4f} ± {std_energy:<12.4f}")

            best_idx = np.argmax(acqf_values)
            reporter(f'Best candidate: #{best_idx} with Acqf = {acqf_values[best_idx]:.4f}')

            # print the up and down occupation matrices of the best candidate
            best_cand = candidates_list[best_idx]


            best_cand = databank.from_pytorch(
            candidates_list[best_idx][0].detach(), 
            atom_ids=atoms, 
            spins=['up', 'down']
            )[0]

            for atom_id in atoms:
                up_mat = np.array(best_cand.get_occupation_matrix(atom_id, 'up'))
                down_mat = np.array(best_cand.get_occupation_matrix(atom_id, 'down'))

                tot_electrons = np.trace(up_mat) + np.trace(down_mat)
                tot_moment = np.trace(up_mat) - np.trace(down_mat)
                reporter(f"Atom {atom_id} - Total electrons: {tot_electrons:.4f}, Total moment: {tot_moment:.4f}")
                reporter(f"Atom {atom_id} - Up matrix")
                # print matrix with 4 decimal points
                torch.set_printoptions(precision=4)
                reporter(str(torch.tensor(up_mat)))
                reporter(f"Atom {atom_id} - Down matrix")
                reporter(str(torch.tensor(down_mat)))


            reporter("="*80)
        
        # convert candidates to OccupationMatrixData
        proposals = databank.from_pytorch( 
            matrices=candidates,
            atom_ids=atoms,
            spins=['up', 'down'])


    return proposals