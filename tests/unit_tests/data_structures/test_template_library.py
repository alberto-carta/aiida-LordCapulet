"""Tests for distance-clustered atomic occupation template libraries."""

import numpy as np

from lordcapulet.data_structures import AtomicTemplateLibrary, DataBank, OccupationMatrixData


def _single_atom_occ(specie="Fe1", shell="d", value=1.0):
    matrix = (np.eye(2) * value).tolist()
    return OccupationMatrixData({
        "Atom_1": {
            "specie": specie,
            "shell": shell,
            "occupation_matrix": {
                "up": matrix,
                "down": matrix,
            },
        }
    })


def test_atomic_templates_deduplicate_by_distance_within_category():
    occs = [
        _single_atom_occ(value=1.0),
        _single_atom_occ(value=1.0001),
    ]
    db = DataBank.from_matrices(occs, energies=[-1.0, -2.0], pks=[10, 11])

    library = AtomicTemplateLibrary()
    library.add_databank(db, distance_threshold=1e-3)
    library.assign_energy_ranks()

    assert len(library.templates) == 1
    template = library.templates[0]
    assert template["category_specie"] == "Fe"
    assert template["shell"] == "d"
    assert template["count"] == 2
    assert template["source_pks"] == [10, 11]
    assert template["min_energy"] == -2.0
    assert template["representative_pk"] == 11
    assert template["energy_rank"] == 1


def test_atomic_templates_keep_species_shell_categories_separate():
    occs = [
        _single_atom_occ(specie="Fe1", shell="d", value=1.0),
        _single_atom_occ(specie="Ni1", shell="d", value=1.0),
        _single_atom_occ(specie="Fe1", shell="p", value=1.0),
    ]
    db = DataBank.from_matrices(occs, energies=[-3.0, -2.0, -1.0])

    library = AtomicTemplateLibrary()
    library.add_databank(db, distance_threshold=10.0)
    library.assign_energy_ranks()

    categories = {
        (template["category_specie"], template["shell"]) for template in library.templates
    }
    assert categories == {("Fe", "d"), ("Ni", "d"), ("Fe", "p")}
    assert len(library.templates) == 3


def test_template_library_round_trips_json(tmp_path):
    db = DataBank.from_matrices(
        [_single_atom_occ(value=1.0), _single_atom_occ(value=2.0)],
        energies=[-2.0, -1.0],
    )
    library = AtomicTemplateLibrary()
    library.add_databank(db, distance_threshold=1e-6)
    library.assign_energy_ranks()

    path = tmp_path / "templates.json"
    library.to_json(path)
    loaded = AtomicTemplateLibrary.from_json(path)

    assert len(loaded.templates) == 2
    assert loaded.templates[0]["occupation_matrix"] == library.templates[0]["occupation_matrix"]
