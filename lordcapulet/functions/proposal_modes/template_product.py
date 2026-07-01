"""Template-library proposal modes for supercell occupation matrices."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from math import prod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from lordcapulet.data_structures import (
    AtomicTemplateLibrary,
    OccupationMatrixData,
    clip_occupation_numbers,
    compute_occupation_distance,
    occupation_fingerprint,
)


def propose_template_product_constraints(
    occ_matr_list: List[OccupationMatrixData],
    natoms: int,
    N: int,
    *,
    template_library: Optional[Union[AtomicTemplateLibrary, Dict[str, Any]]] = None,
    template_library_path: Optional[str] = None,
    distance_threshold: Optional[float] = None,
    duplicate_penalty: float = 1000.0,
    max_combinations: int = 10000,
    debug: bool = False,
    reporter=None,
    **kwargs,
) -> List[OccupationMatrixData]:
    """Generate supercell occupations by assigning atomic templates to atoms.

    Candidate score is deliberately minimal:

    ``score = sum(template.energy_rank) + duplicate_penalty``

    The duplicate penalty is applied when the full generated occupation is
    within ``distance_threshold`` of an already seen occupation with matching
    atom labels.
    """
    if reporter is None:
        reporter = print

    if natoms < 1:
        raise ValueError("natoms must be greater than or equal to 1")

    reference = occ_matr_list[0]
    atom_labels = reference.get_atom_labels()
    if len(atom_labels) != natoms:
        raise ValueError(
            f"natoms={natoms} does not match reference occupation with {len(atom_labels)} atoms"
        )

    library = _load_library(template_library, template_library_path)
    if distance_threshold is None:
        distance_threshold = float(library.metadata.get("distance_threshold", 1e-3))

    pools = _template_pools_for_atoms(reference, atom_labels, library)
    total_combinations = prod(len(pool) for pool in pools)
    combinations = _ranked_template_combinations(pools, max_combinations=max_combinations)

    if debug:
        reporter(
            "Template product mode: "
            f"{len(atom_labels)} atoms, {len(library.templates)} templates, "
            f"{total_combinations} possible combinations, scoring {len(combinations)}"
        )

    seen = list(occ_matr_list)
    generated_fingerprints = set()
    scored: List[Tuple[float, float, int, Tuple[str, ...], OccupationMatrixData]] = []

    for rank_sum, templates in combinations:
        proposal = _proposal_from_templates(reference, atom_labels, templates)
        fingerprint = occupation_fingerprint(proposal)
        if fingerprint in generated_fingerprints:
            continue
        generated_fingerprints.add(fingerprint)

        is_duplicate = _is_duplicate(
            proposal,
            seen,
            distance_threshold=distance_threshold,
        )
        penalty = duplicate_penalty if is_duplicate else 0.0
        score = rank_sum + penalty
        ids = tuple(template["template_id"] for template in templates)
        scored.append((score, rank_sum, int(is_duplicate), ids, proposal))

    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    proposals = [item[-1] for item in scored[:N]]

    if len(proposals) < N:
        raise ValueError(
            f"Only generated {len(proposals)} unique template proposals, requested {N}. "
            "Increase template diversity or lower the duplicate threshold."
        )

    if debug:
        reporter(f"Template product mode generated {len(proposals)} proposals")

    return proposals


def _load_library(
    template_library: Optional[Union[AtomicTemplateLibrary, Dict[str, Any]]],
    template_library_path: Optional[str],
) -> AtomicTemplateLibrary:
    if isinstance(template_library, AtomicTemplateLibrary):
        return template_library
    if isinstance(template_library, dict):
        return AtomicTemplateLibrary.from_dict(template_library)
    if template_library_path is not None:
        return AtomicTemplateLibrary.from_json(template_library_path)
    raise ValueError("template_library or template_library_path is required")


def _template_pools_for_atoms(
    reference: OccupationMatrixData,
    atom_labels: Sequence[str],
    library: AtomicTemplateLibrary,
) -> List[List[Dict[str, Any]]]:
    pools: List[List[Dict[str, Any]]] = []
    for atom_label in atom_labels:
        atom_data = reference[atom_label]
        pool = library.templates_for(atom_data["specie"], atom_data["shell"])
        if not pool:
            raise ValueError(
                "No templates found for "
                f"{atom_label} category ({atom_data['specie']}, {atom_data['shell']})"
            )
        pools.append(sorted(pool, key=lambda template: template.get("energy_rank", float("inf"))))
    return pools


def _ranked_template_combinations(
    pools: Sequence[Sequence[Dict[str, Any]]],
    *,
    max_combinations: int,
) -> List[Tuple[float, Tuple[Dict[str, Any], ...]]]:
    """Return low-energy-rank combinations, pruning when the product is large."""
    if max_combinations < 1:
        raise ValueError("max_combinations must be greater than or equal to 1")

    total_combinations = prod(len(pool) for pool in pools)
    if total_combinations <= max_combinations:
        return sorted(
            (
                (_template_rank_sum(combo), tuple(combo))
                for combo in product(*pools)
            ),
            key=lambda item: (item[0], tuple(t["template_id"] for t in item[1])),
        )

    beam: List[Tuple[float, Tuple[Dict[str, Any], ...]]] = [(0.0, tuple())]
    for pool in pools:
        next_beam: List[Tuple[float, Tuple[Dict[str, Any], ...]]] = []
        for rank_sum, partial in beam:
            for template in pool:
                next_beam.append(
                    (rank_sum + _template_rank(template), partial + (template,))
                )
        next_beam.sort(key=lambda item: (item[0], tuple(t["template_id"] for t in item[1])))
        beam = next_beam[:max_combinations]
    return beam


def _proposal_from_templates(
    reference: OccupationMatrixData,
    atom_labels: Sequence[str],
    templates: Sequence[Dict[str, Any]],
) -> OccupationMatrixData:
    data: Dict[str, Any] = {}
    for atom_label, template in zip(atom_labels, templates):
        atom_data = reference[atom_label]
        data[atom_label] = {
            "specie": atom_data["specie"],
            "shell": atom_data["shell"],
            "occupation_matrix": deepcopy(template["occupation_matrix"]),
        }
    return clip_occupation_numbers(OccupationMatrixData(data))


def _is_duplicate(
    proposal: OccupationMatrixData,
    seen: Sequence[OccupationMatrixData],
    *,
    distance_threshold: float,
) -> bool:
    for previous in seen:
        try:
            if compute_occupation_distance(proposal, previous) <= distance_threshold:
                return True
        except ValueError:
            continue
    return False


def _template_rank(template: Dict[str, Any]) -> float:
    rank = template.get("energy_rank")
    return float("inf") if rank is None else float(rank)


def _template_rank_sum(templates: Sequence[Dict[str, Any]]) -> float:
    return sum(_template_rank(template) for template in templates)
