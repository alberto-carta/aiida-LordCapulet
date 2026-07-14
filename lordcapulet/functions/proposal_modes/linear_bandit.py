"""
Linear bandit proposal mode: Bayesian Ridge + ARD regression.

No PyTorch dependency — uses sklearn exclusively.

Provides:
    propose_linear_bandit_constraints() — generates occupation matrix proposals
    using Bayesian linear regression with LCB acquisition + Boltzmann sampling.

Config structure (passed as bandit_config for API consistency):
    {
        "method": "ridge",          # "ridge" or "ard"
        "model_kwargs": {           # passed to BayesianRidge / ARDRegression
            "n_iter": 300,
            "tol": 1e-6,
            "alpha_1": 1e-6,
            "alpha_2": 1e-6,
            "lambda_1": 1e-6,
            "lambda_2": 1e-6,
            "fit_intercept": True,
        },
        "acquisition": {
            "beta": 0.5,            # exploration parameter
            "eta": 30,              # inverse temperature for Boltzmann
        },
        "optimization": {
            "ensamble_size": 50000,
            "patchwork_params": {
                "apply_rotation": True,
                "rotation_prob": 0.2,
                "rotation_type": "Mixed",  # "SO(3)", "SO(N)", "Mixed"
            },
        },
        "features": {
            # Raw occupation matrix elements (upper triangle, per atom, per spin).
            # These capture the crystal-field / orbital arrangement.
            "include_raw_occ": True,
            # Hubbard term: tr[n(1-n)] = tr(n) - tr(n²) per atom (summed over spins).
            # Models the U correction: E_U = (U/2) * Σ tr[n(1-n)].
            "include_hubbard": True,
            # Hund term: M² per atom where M = tr_up - tr_down.
            # Models the J correction: E_J = -(J/4) * Σ M².
            "include_hund_per_atom": True,
            # Heisenberg term: m_i · m_j for each atom pair (i < j).
            # Captures inter-atom magnetic coupling.
            "include_heisenberg": True,
            # Trace: tr(n) per (atom, spin) — raw electron count per orbital.
            "include_trace_per_spin": False,
            # Magnetic moment: M = tr_up - tr_down per atom.
            "include_moment_per_atom": False,
            # Pair products: n_{ii}^{(a)} · n_{jj}^{(b)} for all diagonal elements
            # across all atom pairs. Mimics the GP's non-local kernel products.
            "include_pair_products": False,
        },
    }
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional

from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error

from lordcapulet.data_structures import OccupationMatrixData, DataBank
from lordcapulet.functions.proposal_modes.shared_functionality import create_patchwork_guess
from lordcapulet.functions.proposal_modes.Bandits_shared import boltzmann_sample, lcb_acquisition


def propose_linear_bandit_constraints(
    occ_matr_list: List[OccupationMatrixData],
    energies: List[float],
    natoms: int,
    N: int,
    bandit_config: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    reporter=None,
    **kwargs,
) -> List[OccupationMatrixData]:
    """
    Generate N occupation matrix proposals using Bayesian linear bandits.

    Strategy:
    1. Create DataBank, featurize (raw occ + physics terms).
    2. Fit BayesianRidge or ARDRegression with built-in uncertainty.
    3. Generate patchwork ensemble (numpy).
    4. Featurize ensemble → predict μ, σ → LCB → Boltzmann sample.
    5. Return top N as OccupationMatrixData.

    Args:
        occ_matr_list: Previous calculation occupation matrices.
        energies: Total energies (eV) for each calculation.
        natoms: Number of atoms (for API consistency, inferred if needed).
        N: Number of proposals to generate.
        bandit_config: Configuration dict (see module docstring).
        debug: If True, print/report extra diagnostics.
        reporter: Callable for logging (defaults to print).

    Returns:
        List of N OccupationMatrixData proposals.
    """
    if reporter is None:
        reporter = print

    # --- Default config -------------------------------------------------------
    if bandit_config is None:
        bandit_config = {}

    method = bandit_config.get("method", "ard")  # ARD with safe pruning is the default
    if method not in ("ridge", "ard"):
        raise ValueError(f"Unknown method '{method}'. Use 'ridge' or 'ard'.")

    model_kwargs = bandit_config.get("model_kwargs", {})
    acq_cfg = bandit_config.get("acquisition", {})
    opt_cfg = bandit_config.get("optimization", {})
    feat_cfg = bandit_config.get("features", {})

    beta = acq_cfg.get("beta", 0.5)
    eta = acq_cfg.get("eta", 30.0)
    ensamble_size = opt_cfg.get("ensamble_size", 50000)
    patchwork_params = opt_cfg.get("patchwork_params", {})

    reporter(f"{'='*60}")
    reporter(f"LINEAR BANDIT PROPOSAL MODE  (method={method})")
    reporter(f"{'='*60}")
    reporter(f"Training samples: {len(occ_matr_list)}")
    reporter(f"Energy range: [{min(energies):.4f}, {max(energies):.4f}] eV")
    reporter(f"Proposals: {N}  |  Ensemble: {ensamble_size}  |  β={beta}, η={eta}")

    # --- Step 1: Build DataBank -----------------------------------------------
    databank = DataBank.from_matrices(
        occ_matrices=occ_matr_list,
        energies=energies,
        converged=[True] * len(occ_matr_list),
    )
    atoms = databank.atom_ids

    # --- Step 2: Featurize ----------------------------------------------------
    X, feature_names = databank.to_feature_matrix(
        atom_ids=atoms,
        **feat_cfg,
    )
    y = databank.energies  # numpy array

    reporter(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    reporter(f"  raw_occ={feat_cfg.get('include_raw_occ', True)}, "
             f"hubbard={feat_cfg.get('include_hubbard', True)}, "
             f"hund={feat_cfg.get('include_hund_per_atom', True)}, "
             f"heisenberg={feat_cfg.get('include_heisenberg', True)}, "
             f"pair_products={feat_cfg.get('include_pair_products', False)}")

    # --- Step 3: Drop near-constant features (std ≈ 0) before scaling ---------
    # StandardScaler divides by std, so near-zero-variance features (common for
    # crystal-field terms on unoccupied atoms) get blown up to astronomical
    # values.  Drop them early so they never pollute predictions.
    # Threshold 1e-12 → scaled values up to ~1e12 (bad).
    # Threshold 1e-8  → scaled values up to ~1e8  (tolerable for linear models).
    # feature_std = np.std(X, axis=0)
    # STD_THRESHOLD = 1e-8
    # keep_mask = feature_std > STD_THRESHOLD
    # n_dropped = int(np.sum(~keep_mask))
    # if n_dropped > 0:
    #     X = X[:, keep_mask]
    #     feature_names = [fn for fn, k in zip(feature_names, keep_mask) if k]
    #     reporter(f"  Dropped {n_dropped} near-constant features (std < {STD_THRESHOLD}) "
    #              f"→ {X.shape[1]} features remain")

    # --- Step 4: Scale & fit model --------------------------------------------
    # Scaling physical informed features works meh, commenting out for now

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    X_scaled = X.copy()


    if method == "ridge":
        model = BayesianRidge(**model_kwargs)
    else:
        # ARDRegression has threshold_lambda for pruning
        model_kwargs.setdefault("threshold_lambda", 100000.0)
        model = ARDRegression(**model_kwargs)

    model.fit(X_scaled, y)

    # --- Step 4: Training metrics ---------------------------------------------
    y_pred, y_std = model.predict(X_scaled, return_std=True)
    r2_train = r2_score(y, y_pred)
    rmse_train = root_mean_squared_error(y, y_pred)

    reporter(f"--- Training Metrics ---")
    reporter(f"  In-sample R² = {r2_train:.4f}")
    reporter(f"  In-sample RMSE = {rmse_train:.4f} eV")

    # alpha_ and lambda_ are scalars for BayesianRidge, arrays for ARD
    alpha_val = model.alpha_
    lambda_val = model.lambda_
    if isinstance(alpha_val, np.ndarray):
        reporter(f"  Noise precision α = {np.mean(alpha_val):.4f}  (array, mean)")
    else:
        reporter(f"  Noise precision α = {alpha_val:.4f}")
    if isinstance(lambda_val, np.ndarray):
        reporter(f"  Prior precision λ = {np.mean(lambda_val):.4f}  (array, mean)")
    else:
        reporter(f"  Prior precision λ = {lambda_val:.4f}")

    if method == "ard":
        n_active = int(np.sum(np.abs(model.coef_) > 1e-6))
        n_total = len(model.coef_)
        reporter(f"  Features retained: {n_active} / {n_total}  "
                 f"({n_total - n_active} pruned)")

    # Show top coefficients by magnitude
    if debug:
        coefs = np.abs(model.coef_)
        top_idx = np.argsort(coefs)[-10:][::-1]
        reporter(f"  Top-10 features (by |coeff|):")
        for idx in top_idx:
            reporter(f"    {feature_names[idx]:50s} {model.coef_[idx]:+.6e}")

    reporter(f"  Intercept: {model.intercept_:.4f} eV")
    # Extract physics parameters from linear coefficients
    _report_physics_coefficients(model, feature_names, method, reporter)

    # --- Step 5: Generate patchwork ensemble ----------------------------------
    reporter(f"--- Generating patchwork ensemble ({ensamble_size} candidates) ---")

    # Get raw occupation matrix data for patchwork generation
    x_raw = databank.to_numpy(atom_ids=atoms, spins=['up', 'down'])
    ensamble_raw = np.empty((ensamble_size, x_raw.shape[1]))

    apply_rotation = patchwork_params.get("apply_rotation", True)
    rotation_prob = patchwork_params.get("rotation_prob", 0.2)
    rotation_type = patchwork_params.get("rotation_type", "Mixed")

    t0 = time.time()
    for i in range(ensamble_size):
        guess = create_patchwork_guess(
            databank, x_raw, atoms,
            apply_rotation=apply_rotation,
            rotation_prob=rotation_prob,
            rotation_type=rotation_type,
        )
        ensamble_raw[i, :] = guess
    t1 = time.time()
    reporter(f"  Generated in {t1 - t0:.2f} s")

    # --- Step 6: Featurize ensemble & predict ---------------------------------
    # To featurize the ensemble, we need OccupationMatrixData objects.
    # We batch-convert the raw ensemble to OccupationMatrixData, then featurize.
    reporter(f"--- Featurizing ensemble ---")
    t0 = time.time()

    # Build a temporary DataBank for the ensemble to reuse to_feature_matrix
    ensamble_occ_list = databank.from_numpy(ensamble_raw, atom_ids=atoms, spins=['up', 'down'])
    ensamble_db = DataBank.from_matrices(
        occ_matrices=ensamble_occ_list,
        energies=[0.0] * ensamble_size,  # placeholder
    )
    X_ensamble, _ = ensamble_db.to_feature_matrix(atom_ids=atoms, **feat_cfg)

    # Apply same feature mask used during training
    # if n_dropped > 0:
    #     X_ensamble = X_ensamble[:, keep_mask]
    # X_ensamble_scaled = scaler.transform(X_ensamble)

    X_ensamble_scaled = X_ensamble.copy()

    mu_ensamble, sigma_ensamble = model.predict(X_ensamble_scaled, return_std=True)
    t1 = time.time()
    reporter(f"  Featurized & predicted in {t1 - t0:.2f} s")

    # --- Step 7: Boltzmann sample ---------------------------------------------
    acq_scores = lcb_acquisition(mu_ensamble, sigma_ensamble, beta=beta)
    chosen_idx = boltzmann_sample(acq_scores, eta=eta, num_samples=N, replacement=False)

    # --- Step 8: Convert back to OccupationMatrixData -------------------------
    proposals = [ensamble_occ_list[i] for i in chosen_idx]

    # --- Step 9: Summary ------------------------------------------------------
    reporter(f"\n{'='*80}")
    reporter(f"CANDIDATE SUMMARY")
    reporter(f"{'='*80}")
    reporter(f"{'#':<4} {'LCB':<12} {'μ (eV)':<14} {'σ (eV)':<14} {'Acq Score':<12}")
    reporter(f"{'-'*80}")
    for rank, idx in enumerate(chosen_idx):
        reporter(f"{rank:<4} {lcb_acquisition(mu_ensamble[idx:idx+1], sigma_ensamble[idx:idx+1], beta=beta)[0]:<12.4f} "
                 f"{mu_ensamble[idx]:<14.4f} {sigma_ensamble[idx]:<14.4f} "
                 f"{acq_scores[idx]:<12.4f}")
    reporter(f"{'='*80}")

    return proposals


def _report_physics_coefficients(model, feature_names, method, reporter):
    """Extract and report physics parameters from linear coefficients."""
    coeff_dict = dict(zip(feature_names, model.coef_))

    # Hubbard U: coefficient of hubbard_* features → U = 2 * coeff (from U/2 * tr[n(1-n)])
    hubbard_coeffs = [v for k, v in coeff_dict.items() if k.startswith("hubbard_")]
    if hubbard_coeffs:
        U_eff = 2.0 * np.mean(hubbard_coeffs)
        reporter(f"  U_eff (from hubbard coeffs) ≈ {U_eff:.4f} eV  "
                 f"(individual: {[f'{c*2:.2f}' for c in hubbard_coeffs]})")

    # Hund J: coefficient of hund_M2_* features → J = -4 * coeff (from -(J/4) * M²)
    hund_coeffs = [v for k, v in coeff_dict.items() if k.startswith("hund_M2_")]
    if hund_coeffs:
        J_eff = -4.0 * np.mean(hund_coeffs)
        reporter(f"  J_eff (from hund coeffs) ≈ {J_eff:.4f} eV  "
                 f"(individual: {[f'{-c*4:.2f}' for c in hund_coeffs]})")
