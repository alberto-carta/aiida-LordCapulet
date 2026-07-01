"""Tests for template-product occupation proposal mode."""

import numpy as np

from lordcapulet.data_structures import AtomicTemplateLibrary, DataBank, OccupationMatrixData
from lordcapulet.functions.proposal_modes.template_product import (
    propose_template_product_constraints,
)
from lordcapulet.functions.propose import propose_new_constraints


def _primitive_occ(value, energy_specie="Fe1"):
    matrix = (np.eye(2) * value).tolist()
    return OccupationMatrixData({
        "Atom_1": {
            "specie": energy_specie,
            "shell": "d",
            "occupation_matrix": {
                "up": matrix,
                "down": matrix,
            },
        }
    })


def _supercell_reference(v1=1.0, v2=1.0):
    return OccupationMatrixData({
        "Atom_1": {
            "specie": "Fe3",
            "shell": "d",
            "occupation_matrix": {
                "up": (np.eye(2) * v1).tolist(),
                "down": (np.eye(2) * v1).tolist(),
            },
        },
        "Atom_2": {
            "specie": "Fe4",
            "shell": "d",
            "occupation_matrix": {
                "up": (np.eye(2) * v2).tolist(),
                "down": (np.eye(2) * v2).tolist(),
            },
        },
    })


def _library():
    db = DataBank.from_matrices(
        [_primitive_occ(1.0), _primitive_occ(0.5)],
        energies=[-2.0, -1.0],
        pks=[100, 101],
    )
    library = AtomicTemplateLibrary()
    library.add_databank(db, distance_threshold=1e-6)
    library.assign_energy_ranks()
    return library


def test_template_product_generates_supercell_occupations_from_atomic_templates():
    reference = _supercell_reference(v1=0.0, v2=0.0)

    proposals = propose_template_product_constraints(
        [reference],
        natoms=2,
        N=2,
        template_library=_library(),
        distance_threshold=1e-6,
        debug=False,
    )

    assert len(proposals) == 2
    for proposal in proposals:
        assert isinstance(proposal, OccupationMatrixData)
        assert proposal.get_atom_labels() == ["Atom_1", "Atom_2"]
        assert proposal.data["Atom_1"]["specie"] == "Fe3"
        assert proposal.data["Atom_2"]["specie"] == "Fe4"

    best = proposals[0]
    assert np.allclose(best.get_occupation_matrix_as_numpy("Atom_1", "up"), np.eye(2))
    assert np.allclose(best.get_occupation_matrix_as_numpy("Atom_2", "up"), np.eye(2))


def test_template_product_clips_occupation_numbers_to_valid_range():
    reference = _supercell_reference(v1=0.0, v2=0.0)
    db = DataBank.from_matrices(
        [_primitive_occ(-0.2), _primitive_occ(1.2)],
        energies=[-2.0, -1.0],
        pks=[100, 101],
    )
    library = AtomicTemplateLibrary()
    library.add_databank(db, distance_threshold=1e-6)
    library.assign_energy_ranks()

    proposals = propose_template_product_constraints(
        [reference],
        natoms=2,
        N=2,
        template_library=library,
        distance_threshold=1e-6,
        debug=False,
    )

    for proposal in proposals:
        for atom_label in proposal.get_atom_labels():
            for spin in ("up", "down"):
                matrix = proposal.get_occupation_matrix_as_numpy(atom_label, spin)
                assert np.all(matrix >= -1.0)
                assert np.all(matrix <= 1.0)


def test_template_product_duplicate_penalty_can_push_seen_candidate_down():
    reference = _supercell_reference(v1=1.0, v2=1.0)

    proposals = propose_template_product_constraints(
        [reference],
        natoms=2,
        N=1,
        template_library=_library(),
        distance_threshold=1e-6,
        duplicate_penalty=100.0,
        debug=False,
    )

    proposal = proposals[0]
    values = [
        proposal.get_occupation_matrix_as_numpy("Atom_1", "up")[0, 0],
        proposal.get_occupation_matrix_as_numpy("Atom_2", "up")[0, 0],
    ]
    assert values != [1.0, 1.0]


def test_template_product_dispatch_from_propose_new_constraints():
    reference = _supercell_reference(v1=0.0, v2=0.0)

    proposals = propose_new_constraints(
        [reference],
        N=1,
        mode="template_product",
        template_library=_library(),
        distance_threshold=1e-6,
        debug=False,
    )

    assert len(proposals) == 1
    assert isinstance(proposals[0], OccupationMatrixData)
