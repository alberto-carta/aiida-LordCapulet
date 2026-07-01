"""Distance-clustered atomic occupation matrix templates.

The template library is intentionally AiiDA-free.  It takes converged
``OccupationMatrixData`` records, clusters atomic occupation matrices within
``(specie, shell)`` categories by Euclidean distance, and keeps compact metadata
for later proposal modes.

Template schema notes:
* ``min_energy`` is the minimum total cell energy among all calculations where
  an atom belonged to the template cluster.  It is a ranking heuristic, not an
  atomic energy decomposition.
* ``energy_rank`` is assigned globally by ascending ``min_energy``.
* ``source_energies``, ``source_pks``, and ``source_records`` preserve the
  primitive-cell provenance for every atom merged into a template.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .databank import DataBank
from .occupation_matrix import OccupationMatrixData, compute_occupation_distance


def normalize_specie(specie: str, strip_numeric_suffix: bool = True) -> str:
    """Return the specie key used for template categories."""
    if not strip_numeric_suffix:
        return specie
    return re.sub(r"\d+$", "", str(specie))


def atomic_template_distance(template_a: Dict[str, Any], template_b: Dict[str, Any]) -> float:
    """Euclidean distance between two atomic occupation templates."""
    occ_a = OccupationMatrixData({"Atom_1": _template_atom_data(template_a)})
    occ_b = OccupationMatrixData({"Atom_1": _template_atom_data(template_b)})
    return compute_occupation_distance(occ_a, occ_b)


def occupation_fingerprint(occ_data: OccupationMatrixData, decimals: int = 8) -> Tuple[Any, ...]:
    """Hashable rounded fingerprint for exact duplicate bookkeeping."""
    pieces: List[Any] = []
    for atom in sorted(occ_data.get_atom_labels(), key=_atom_sort_key):
        atom_data = occ_data[atom]
        pieces.append(atom_data.get("specie"))
        pieces.append(atom_data.get("shell"))
        for spin in ("up", "down"):
            arr = np.asarray(atom_data["occupation_matrix"][spin], dtype=float)
            pieces.append(tuple(np.round(arr, decimals=decimals).ravel()))
    return tuple(pieces)


class AtomicTemplateLibrary:
    """A distance-clustered library of atomic occupation matrix templates.

    The library stores atomic templates, but the energy attached to a template
    is the minimum total energy of the source cells that contained matching
    atomic occupations.
    """

    def __init__(
        self,
        templates: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.templates = templates or []
        self.metadata = metadata or {}

    @classmethod
    def from_json_files(
        cls,
        json_paths: Iterable[Union[str, Path]],
        *,
        distance_threshold: float = 1e-3,
        only_converged: bool = True,
        strip_numeric_suffix: bool = True,
    ) -> "AtomicTemplateLibrary":
        """Build a template library from gathered workchain JSON files."""
        library = cls(
            metadata={
                "distance_threshold": distance_threshold,
                "only_converged": only_converged,
                "strip_numeric_suffix": strip_numeric_suffix,
                "source_files": [str(Path(p)) for p in json_paths],
                "n_source_records": 0,
            }
        )

        for json_path in json_paths:
            db = DataBank.from_json(json_path, only_converged=only_converged)
            library.add_databank(
                db,
                distance_threshold=distance_threshold,
                source=str(json_path),
                strip_numeric_suffix=strip_numeric_suffix,
            )

        library.assign_energy_ranks()
        return library

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicTemplateLibrary":
        """Create a library from JSON-compatible data."""
        return cls(
            templates=deepcopy(data.get("templates", [])),
            metadata=deepcopy(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "AtomicTemplateLibrary":
        """Load a library from disk."""
        with open(path, "r") as handle:
            return cls.from_dict(json.load(handle))

    def as_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible data."""
        return {
            "metadata": deepcopy(self.metadata),
            "templates": deepcopy(self.templates),
        }

    def to_json(self, path: Union[str, Path], *, indent: int = 2) -> None:
        """Write the library to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.as_dict(), handle, indent=indent)
            handle.write("\n")

    def add_databank(
        self,
        databank: DataBank,
        *,
        distance_threshold: float,
        source: Optional[str] = None,
        strip_numeric_suffix: bool = True,
    ) -> None:
        """Add all atomic matrices from a DataBank."""
        for record in databank._records:
            self.add_record(
                record,
                distance_threshold=distance_threshold,
                source=source,
                strip_numeric_suffix=strip_numeric_suffix,
            )
            self.metadata["n_source_records"] = self.metadata.get("n_source_records", 0) + 1

    def add_record(
        self,
        record: Dict[str, Any],
        *,
        distance_threshold: float,
        source: Optional[str] = None,
        strip_numeric_suffix: bool = True,
    ) -> None:
        """Add atomic matrices from one calculation record."""
        occ_data = record["occ_data"]
        for atom_label in occ_data.get_atom_labels():
            candidate = _template_from_atom(
                occ_data,
                atom_label,
                pk=record.get("pk"),
                energy=record.get("energy"),
                source=source,
                record_metadata=record.get("metadata", {}),
                strip_numeric_suffix=strip_numeric_suffix,
            )
            self._add_template(candidate, distance_threshold=distance_threshold)

    def templates_for(
        self,
        specie: str,
        shell: str,
        *,
        strip_numeric_suffix: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Return templates matching a ``(specie, shell)`` category."""
        if strip_numeric_suffix is None:
            strip_numeric_suffix = bool(self.metadata.get("strip_numeric_suffix", True))
        key = normalize_specie(specie, strip_numeric_suffix)
        return [
            template
            for template in self.templates
            if template["category_specie"] == key and template["shell"] == shell
        ]

    def assign_energy_ranks(self) -> None:
        """Assign rank 1 to the template with the lowest minimum cell energy."""
        ranked = sorted(
            enumerate(self.templates),
            key=lambda item: (
                _rank_energy(item[1].get("min_energy")),
                item[1]["category_specie"],
                item[1]["shell"],
                item[1]["template_id"],
            ),
        )
        for rank, (idx, _) in enumerate(ranked, start=1):
            self.templates[idx]["energy_rank"] = rank

    def summary(self) -> str:
        """Human-readable summary for experiment logs."""
        by_category: Dict[Tuple[str, str], int] = {}
        for template in self.templates:
            key = (template["category_specie"], template["shell"])
            by_category[key] = by_category.get(key, 0) + 1

        lines = [
            "Atomic occupation template library",
            f"source records: {self.metadata.get('n_source_records', 0)}",
            f"templates: {len(self.templates)}",
            f"distance threshold: {self.metadata.get('distance_threshold')}",
            "categories:",
        ]
        for (specie, shell), count in sorted(by_category.items()):
            lines.append(f"  {specie} / {shell}: {count}")
        return "\n".join(lines)

    def _add_template(self, candidate: Dict[str, Any], *, distance_threshold: float) -> None:
        matching = [
            template
            for template in self.templates
            if (
                template["category_specie"] == candidate["category_specie"]
                and template["shell"] == candidate["shell"]
            )
        ]
        for template in matching:
            distance = atomic_template_distance(template, candidate)
            if distance <= distance_threshold:
                _merge_template(template, candidate, distance)
                return

        candidate["template_id"] = _make_template_id(candidate, len(self.templates))
        candidate["count"] = 1
        candidate["source_pks"] = [candidate["representative_pk"]]
        candidate["source_energies"] = [candidate["min_energy"]]
        candidate["source_records"] = [candidate.pop("_source_record")]
        candidate["min_distance_merged"] = None
        self.templates.append(candidate)


def _template_from_atom(
    occ_data: OccupationMatrixData,
    atom_label: str,
    *,
    pk: Optional[int],
    energy: Optional[float],
    source: Optional[str],
    record_metadata: Dict[str, Any],
    strip_numeric_suffix: bool,
) -> Dict[str, Any]:
    atom_data = occ_data[atom_label]
    specie = atom_data["specie"]
    shell = atom_data["shell"]
    category_specie = normalize_specie(specie, strip_numeric_suffix)
    matrix = deepcopy(atom_data["occupation_matrix"])
    return {
        "template_id": None,
        "category_specie": category_specie,
        "source_specie": specie,
        "shell": shell,
        "occupation_matrix": matrix,
        "min_energy": energy,
        "representative_pk": pk,
        "representative_atom_label": atom_label,
        "_source_record": {
            "pk": pk,
            "energy": energy,
            "source": source,
            "atom_label": atom_label,
            "source_specie": specie,
            "calculation_source": record_metadata.get("source"),
            "process_type": record_metadata.get("process_type"),
        },
    }


def _merge_template(template: Dict[str, Any], candidate: Dict[str, Any], distance: float) -> None:
    template["count"] += 1
    template["source_pks"].append(candidate["representative_pk"])
    template["source_energies"].append(candidate["min_energy"])
    template["source_records"].append(candidate["_source_record"])

    if template.get("min_distance_merged") is None:
        template["min_distance_merged"] = distance
    else:
        template["min_distance_merged"] = min(template["min_distance_merged"], distance)

    min_energy = _rank_energy(template.get("min_energy"))
    candidate_energy = _rank_energy(candidate.get("min_energy"))
    if candidate_energy < min_energy:
        template["source_specie"] = candidate["source_specie"]
        template["occupation_matrix"] = deepcopy(candidate["occupation_matrix"])
        template["min_energy"] = candidate["min_energy"]
        template["representative_pk"] = candidate["representative_pk"]
        template["representative_atom_label"] = candidate["representative_atom_label"]


def _template_atom_data(template: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "specie": template.get("category_specie", template.get("source_specie")),
        "shell": template["shell"],
        "occupation_matrix": template["occupation_matrix"],
    }


def _make_template_id(template: Dict[str, Any], index: int) -> str:
    specie = re.sub(r"[^A-Za-z0-9]+", "_", template["category_specie"]).strip("_")
    shell = re.sub(r"[^A-Za-z0-9]+", "_", template["shell"]).strip("_")
    return f"{specie}_{shell}_{index:04d}"


def _rank_energy(energy: Optional[float]) -> float:
    return float("inf") if energy is None else float(energy)


def _atom_sort_key(atom_label: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", atom_label)
    if match:
        return int(match.group(1)), atom_label
    return 0, atom_label
