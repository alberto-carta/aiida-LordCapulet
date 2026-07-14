"""
Forest bandit proposal mode: Random Forest regression.

No PyTorch dependency — uses sklearn exclusively.

Provides:
    propose_forest_bandit_constraints() — generates occupation matrix proposals
    using Random Forest with ensemble uncertainty + LCB + Boltzmann sampling.

Config structure (passed as bandit_config for API consistency):
    {
        "model_kwargs": {
            "n_estimators": 300,
            "max_depth": None,
            "max_leaf_nodes": None,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "oob_score": True,
        },
        "acquisition": {
            "beta": 0.5,
            "eta": 30,
        },
        "optimization": {
            "ensamble_size": 50000,
            "patchwork_params": {
                "apply_rotation": True,
                "rotation_prob": 0.2,
                "rotation_type": "Mixed",
            },
        },
        "features": {
            "include_raw_occ": True,
            "include_hubbard": True,
            "include_hund_per_atom": True,
            "include_heisenberg": True,
            "include_trace_per_spin": False,
            "include_moment_per_atom": False,
            "include_pair_products": False,
        },
    }
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

from lordcapulet.data_structures import OccupationMatrixData, DataBank
from lordcapulet.functions.proposal_modes.shared_functionality import create_patchwork_guess
from lordcapulet.functions.proposal_modes.Bandits_shared import boltzmann_sample, lcb_acquisition


def propose_forest_bandit_constraints(
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
    Generate N occupation matrix proposals using Random Forest bandits.

    Strategy:
    1. Create DataBank, featurize.
    2. Fit RandomForestRegressor (no scaling needed).
    3. Generate patchwork ensemble (numpy).
    4. Featurize ensemble → predict μ from ensemble mean, σ from tree std.
    5. LCB → Boltzmann sample → return proposals.

    Args:
        occ_matr_list: Previous calculation occupation matrices.
        energies: Total energies (eV).
        natoms: Number of atoms.
        N: Number of proposals.
        bandit_config: Configuration dict (see module docstring).
        debug: Extra diagnostics.
        reporter: Logging callable.

    Returns:
        List of N OccupationMatrixData proposals.
    """
    if reporter is None:
        reporter = print

    # --- Default config -------------------------------------------------------
    if bandit_config is None:
        bandit_config = {}

    model_kwargs = bandit_config.get("model_kwargs", {})
    acq_cfg = bandit_config.get("acquisition", {})
    opt_cfg = bandit_config.get("optimization", {})
    feat_cfg = bandit_config.get("features", {})

    beta = acq_cfg.get("beta", 0.5)
    eta = acq_cfg.get("eta", 30.0)
    ensamble_size = opt_cfg.get("ensamble_size", 50000)
    patchwork_params = opt_cfg.get("patchwork_params", {})

    model_kwargs.setdefault("n_estimators", 300)
    model_kwargs.setdefault("n_jobs", -1)
    model_kwargs.setdefault("oob_score", True)
    model_kwargs.setdefault("random_state", 42)

    reporter(f"{'='*60}")
    reporter(f"FOREST BANDIT PROPOSAL MODE")
    reporter(f"{'='*60}")
    reporter(f"Training samples: {len(occ_matr_list)}")
    reporter(f"Energy range: [{min(energies):.4f}, {max(energies):.4f}] eV")
    reporter(f"Proposals: {N}  |  Ensemble: {ensamble_size}  |  β={beta}, η={eta}")
    reporter(f"Trees: {model_kwargs['n_estimators']}  |  "
             f"min_samples_leaf={model_kwargs.get('min_samples_leaf', 1)}")

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
    y = databank.energies

    reporter(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    reporter(f"  raw_occ={feat_cfg.get('include_raw_occ', True)}, "
             f"raw_occ_total={feat_cfg.get('include_raw_occ_total', False)}, "
             f"raw_occ_offdiag={feat_cfg.get('include_raw_occ_offdiag', True)}, "
             f"hubbard={feat_cfg.get('include_hubbard', True)}, "
             f"hubbard_global={feat_cfg.get('include_hubbard_global', False)}, "
             f"hund={feat_cfg.get('include_hund_per_atom', True)}, "
             f"hund_global={feat_cfg.get('include_hund_global', False)}, "
             f"heisenberg={feat_cfg.get('include_heisenberg', True)}, "
             f"pair_products={feat_cfg.get('include_pair_products', False)}")

    # --- Step 3: Fit Random Forest --------------------------------------------
    # RF doesn't need feature scaling
    model = RandomForestRegressor(**model_kwargs)
    model.fit(X, y)

    # --- Step 4: Training metrics ---------------------------------------------
    # OOB score (out-of-bag R²) — a built-in cross-validation metric
    oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else None

    # Also compute in-sample R²/RMSE for comparison
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)

    reporter(f"--- Training Metrics ---")
    if oob_score is not None:
        reporter(f"  OOB R² = {oob_score:.4f}  (out-of-bag, unbiased estimate)")
    reporter(f"  In-sample R² = {r2:.4f}")
    reporter(f"  In-sample RMSE = {rmse:.4f} eV")

    # Feature importances (top-10)
    if debug:
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        reporter(f"  Top-10 feature importances:")
        for idx in top_idx:
            reporter(f"    {feature_names[idx]:50s} {importances[idx]:.4f}")

    # --- Step 5: Generate patchwork ensemble ----------------------------------
    reporter(f"--- Generating patchwork ensemble ({ensamble_size} candidates) ---")

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
    reporter(f"--- Featurizing ensemble ---")
    t0 = time.time()

    ensamble_occ_list = databank.from_numpy(ensamble_raw, atom_ids=atoms, spins=['up', 'down'])
    ensamble_db = DataBank.from_matrices(
        occ_matrices=ensamble_occ_list,
        energies=[0.0] * ensamble_size,
    )
    X_ensamble, _ = ensamble_db.to_feature_matrix(atom_ids=atoms, **feat_cfg)

    # RF uncertainty: std of individual tree predictions
    tree_preds = np.array([
        tree.predict(X_ensamble) for tree in model.estimators_
    ])  # shape: [n_trees, ensamble_size]
    mu_ensamble = tree_preds.mean(axis=0)
    sigma_ensamble = tree_preds.std(axis=0)
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
    reporter(f"{'#':<4} {'μ (eV)':<14} {'σ (eV)':<14} {'Acq Score':<12}")
    reporter(f"{'-'*80}")
    for rank, idx in enumerate(chosen_idx):
        reporter(f"{rank:<4} {mu_ensamble[idx]:<14.4f} {sigma_ensamble[idx]:<14.4f} "
                 f"{acq_scores[idx]:<12.4f}")
    reporter(f"{'='*80}")

    return proposals
