"""
data_parser.py — Parses raw JSON calculation data into a stable internal representation.

This is the ONLY module that knows about the JSON schema. If the schema changes,
only this file needs updating — the featurization and analysis code is untouched.

JSON format: output_atomic_occupations + so_n_decomposition
  - output_atomic_occupations: spin_data.up/down with occupation_matrix + eigenvalues
  - so_n_decomposition.atom_decompositions: up_spin/down_spin eigenvalues
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Stable data objects (the contract between parser and featurizer)
# ---------------------------------------------------------------------------

@dataclass
class AtomData:
    """Per-atom data extracted from a calculation, format-agnostic."""
    atom_index: int
    # Occupation-matrix eigenvalues (diagonalized occupation) — from occupations block
    occ_eigenvalues_up: List[float] = field(default_factory=list)
    occ_eigenvalues_down: List[float] = field(default_factory=list)
    # Raw occupation matrices (if available; may be empty)
    occ_matrix_up: Optional[np.ndarray] = None
    occ_matrix_down: Optional[np.ndarray] = None
    # SO(N) decomposition eigenvalues (from so_n_decomposition block)
    so_n_eigenvalues_up: List[float] = field(default_factory=list)
    so_n_eigenvalues_down: List[float] = field(default_factory=list)
    # Magnetic moment as reported by the calculation
    magnetic_moment: Optional[float] = None
    # Total trace / electron count
    total_trace: Optional[float] = None

    @property
    def all_so_n_eigenvalues(self) -> List[float]:
        """Concatenated up + down SO(N) eigenvalues."""
        return self.so_n_eigenvalues_up + self.so_n_eigenvalues_down

    @property
    def local_moment(self) -> float:
        """M = Σ n_up - Σ n_down from SO(N) eigenvalues."""
        return sum(self.so_n_eigenvalues_up) - sum(self.so_n_eigenvalues_down)

    @property
    def num_electrons(self) -> float:
        """Total electron count from SO(N) eigenvalues."""
        return sum(self.all_so_n_eigenvalues)


@dataclass
class CalculationRecord:
    """One calculation, fully parsed and ready for featurization."""
    calc_id: str
    source: str = "unknown"
    total_energy: Optional[float] = None
    hubbard_energy: Optional[float] = None
    atoms: List[AtomData] = field(default_factory=list)

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------

def _parse_occupations(occ_block: dict) -> Dict[int, dict]:
    """
    Parse v1 output_atomic_occupations block.
    Returns {atom_index: {eig_up, eig_down, mat_up, mat_down, mag_moment, trace}}.
    """
    atoms: Dict[int, dict] = {}
    for atom_id_str, atom_data in occ_block.items():
        if not isinstance(atom_data, dict):
            continue
        idx = int(atom_id_str)

        spin_data = atom_data.get("spin_data", {})
        up = spin_data.get("up", {})
        down = spin_data.get("down", {})

        eig_up = up.get("eigenvalues", [])
        eig_down = down.get("eigenvalues", [])
        mat_up = np.array(up.get("occupation_matrix", []))
        mat_down = np.array(down.get("occupation_matrix", []))

        mag_moment = atom_data.get("magnetic_moment")
        trace = atom_data.get("occupations", {}).get("total")

        atoms[idx] = {
            "eig_up": list(eig_up) if eig_up else [],
            "eig_down": list(eig_down) if eig_down else [],
            "mat_up": mat_up if mat_up.size > 0 else None,
            "mat_down": mat_down if mat_down.size > 0 else None,
            "mag_moment": mag_moment,
            "trace": trace,
        }
    return atoms


def _parse_so_n_decomposition(so_n_block: dict) -> Dict[int, dict]:
    """
    Parse so_n_decomposition block (common to all formats).
    Returns {atom_index: {eig_up, eig_down}}.
    """
    atoms: Dict[int, dict] = {}
    atom_decomps = so_n_block.get("atom_decompositions", {})
    if not isinstance(atom_decomps, dict):
        return atoms

    for atom_id_str, atom_data in atom_decomps.items():
        if not isinstance(atom_data, dict):
            continue
        idx = int(atom_id_str)
        up = atom_data.get("up_spin", {})
        down = atom_data.get("down_spin", {})

        atoms[idx] = {
            "eig_up": list(up.get("eigenvalues", [])),
            "eig_down": list(down.get("eigenvalues", [])),
        }
    return atoms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_calculation(calc_data: dict, calc_id: str) -> Optional[CalculationRecord]:
    """
    Parse a single calculation dict into a CalculationRecord.

    Requires 'so_n_decomposition' and 'output_atomic_occupations' blocks.
    Returns None if required data is missing.
    """
    if not isinstance(calc_data, dict):
        return None

    # Must have both data blocks
    if not ("so_n_decomposition" in calc_data and "output_atomic_occupations" in calc_data):
        return None

    # --- Occupations ---
    occ_atoms = _parse_occupations(calc_data.get("output_atomic_occupations", {}))

    # --- SO(N) decomposition ---
    so_n_atoms = _parse_so_n_decomposition(calc_data["so_n_decomposition"])

    # --- Output parameters ---
    out_params = calc_data.get("output_parameters", {})
    total_energy = out_params.get("energy") if isinstance(out_params, dict) else None
    hubbard_energy = out_params.get("energy_hubbard") if isinstance(out_params, dict) else None

    if total_energy is None:
        return None

    # --- Assemble AtomData objects ---
    all_indices = sorted(set(occ_atoms.keys()) | set(so_n_atoms.keys()))
    atoms: List[AtomData] = []
    for idx in all_indices:
        occ = occ_atoms.get(idx, {})
        so_n = so_n_atoms.get(idx, {})

        atoms.append(AtomData(
            atom_index=idx,
            occ_eigenvalues_up=occ.get("eig_up", []),
            occ_eigenvalues_down=occ.get("eig_down", []),
            occ_matrix_up=occ.get("mat_up"),
            occ_matrix_down=occ.get("mat_down"),
            so_n_eigenvalues_up=so_n.get("eig_up", []),
            so_n_eigenvalues_down=so_n.get("eig_down", []),
            magnetic_moment=occ.get("mag_moment"),
            total_trace=occ.get("trace"),
        ))

    source = calc_data.get("calculation_source", "unknown")

    return CalculationRecord(
        calc_id=calc_id,
        source=source,
        total_energy=total_energy,
        hubbard_energy=hubbard_energy,
        atoms=atoms,
    )


def parse_json_file(filepath: Union[str, Path]) -> List[CalculationRecord]:
    """
    Parse an entire JSON file of calculations.

    Args:
        filepath: Path to the JSON file.

    Returns:
        List of CalculationRecord objects (invalid calculations are skipped).
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        data = json.load(f)

    records: List[CalculationRecord] = []
    calculations = data.get("calculations", {})
    if not isinstance(calculations, dict):
        print(f"Warning: 'calculations' key is not a dict in {filepath}")
        return records

    for calc_id, calc_data in calculations.items():
        record = parse_calculation(calc_data, str(calc_id))
        if record is not None and record.total_energy is not None:
            records.append(record)

    print(f"Parsed {len(records)} valid calculations from {filepath.name} "
          f"(skipped {len(calculations) - len(records)} invalid)")
    return records


# ---------------------------------------------------------------------------
# CLI for quick inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_parser.py <json_file>")
        sys.exit(1)

    records = parse_json_file(sys.argv[1])
    if records:
        r = records[0]
        print(f"\nFirst record: {r.calc_id}")
        print(f"  Source: {r.source}")
        print(f"  Total energy: {r.total_energy:.4f} eV")
        print(f"  Hubbard energy: {r.hubbard_energy}")
        print(f"  Number of atoms: {r.num_atoms}")
        for atom in r.atoms:
            print(f"  Atom {atom.atom_index}: "
                  f"moment={atom.local_moment:.4f}, "
                  f"n_electrons={atom.num_electrons:.4f}, "
                  f"SO(N) eigs_up={len(atom.so_n_eigenvalues_up)}, "
                  f"occ_mat shape={atom.occ_matrix_up.shape if atom.occ_matrix_up is not None else 'N/A'}")
