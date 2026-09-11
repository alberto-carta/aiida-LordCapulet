"""
feature_engineering.py — Computes features from parsed CalculationRecord objects.

This module knows NOTHING about JSON. It only works with the stable data objects
defined in data_parser.py. If you change the featurization logic, the parser
stays untouched — and vice versa.

Energy model:
    E_total ≈ (U/2) * Σ Tr[n(1-n)]  -  (J/4) * Σ M²  +  J_H * Σ_{i<j} m_i*m_j  +  Σ CF terms  +  C

Features produced:
  - hubbard_term:       Σ_i Σ_α n_{iα}(1 - n_{iα})    (from SO(N) eigenvalues)
  - hund_term_M2:       Σ_i M_i²                       (M_i = local magnetic moment)
  - heisenberg_{i}_{j}: m_i * m_j for each atom pair i<j (binary coupling term)
  - cf_atom{k}_n_{i}_{i}: diagonal elements of total occupation matrix per atom
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from data_parser import AtomData, CalculationRecord, parse_json_file


def _hubbard_term(atom: AtomData) -> float:
    """Σ n_α(1-n_α) for one atom using SO(N) eigenvalues."""
    return sum(eig * (1.0 - eig) for eig in atom.all_so_n_eigenvalues)


def _hund_term(atom: AtomData) -> float:
    """M² for one atom (Hund's J coupling)."""
    return atom.local_moment ** 2


def _local_moment(atom: AtomData) -> float:
    """M_i = Σ up_eigs - Σ down_eigs for one atom."""
    return atom.local_moment


def _crystal_field_features(atom: AtomData, include_offdiagonal: bool = False) -> Dict[str, float]:
    """
    Elements of the total (up+down) occupation matrix.

    Returns:
      - Diagonal:     f'cf_atom{idx}_n_{i}_{i}' for each diagonal element
      - Off-diagonal: f'cf_atom{idx}_n_{i}_{j}' for upper triangle i < j (if include_offdiagonal=True)
    """
    cf: Dict[str, float] = {}
    mat_up = atom.occ_matrix_up
    mat_down = atom.occ_matrix_down

    if mat_up is None or mat_down is None or mat_up.size == 0:
        return cf

    n_total = mat_up + mat_down
    n = n_total.shape[0]
    for i in range(n):
        key = f"cf_atom{atom.atom_index}_n_{i+1}_{i+1}"
        cf[key] = float(n_total[i, i])

    if include_offdiagonal:
        for i in range(n):
            for j in range(i + 1, n):
                key = f"cf_atom{atom.atom_index}_n_{i+1}_{j+1}"
                cf[key] = float(n_total[i, j])

    return cf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def featurize(record: CalculationRecord, include_cf_offdiagonal: bool = False) -> Dict[str, float]:
    """
    Compute all features for a single CalculationRecord.

    Args:
        record: Parsed calculation data.
        include_cf_offdiagonal: If True, include upper-triangle off-diagonal
            elements of the occupation matrix as features.

    Returns a dict mapping feature_name → value.
    The target 'total_energy_eV' and metadata 'calculation_id' are also included
    for convenience when building a DataFrame.
    """
    features: Dict[str, float] = {
        "calculation_id": record.calc_id,
        "total_energy_eV": record.total_energy if record.total_energy is not None else np.nan,
        "hubbard_term": 0.0,
        "hund_term_M2": 0.0,
    }

    # Per-atom terms
    moments: List[float] = []
    for atom in record.atoms:
        features["hubbard_term"] += _hubbard_term(atom)
        features["hund_term_M2"] += _hund_term(atom)
        moments.append(_local_moment(atom))

    # Binary Heisenberg: one feature per atom pair (i < j)
    for i in range(len(moments)):
        for j in range(i + 1, len(moments)):
            key = f"heisenberg_{i+1}_{j+1}"
            features[key] = moments[i] * moments[j]

    # Crystal-field features
    for atom in record.atoms:
        features.update(_crystal_field_features(atom, include_offdiagonal=include_cf_offdiagonal))

    return features


def featurize_all(records: List[CalculationRecord], include_cf_offdiagonal: bool = False) -> pd.DataFrame:
    """
    Featurize a list of CalculationRecords into a pandas DataFrame.

    Columns include: calculation_id, total_energy_eV, hubbard_term,
    hund_term_M2, heisenberg_i_j, and cf_atom* features.
    """
    rows = [featurize(r, include_cf_offdiagonal=include_cf_offdiagonal) for r in records]
    df = pd.DataFrame(rows)
    df.dropna(inplace=True)
    df.sort_values("total_energy_eV", inplace=True)
    print(f"Featurization complete: {len(df)} data points, {len(df.columns)} columns")
    return df


def featurize_from_json(json_path: str) -> pd.DataFrame:
    """
    Convenience: parse JSON → featurize → DataFrame in one call.
    """
    records = parse_json_file(json_path)
    return featurize_all(records)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feature_engineering.py <json_file>")
        sys.exit(1)

    df = featurize_from_json(sys.argv[1])
    print("\nFeature columns:", list(df.columns))
    print(df.describe())
