"""Tests for ``prepare_hubbard_corr_info`` and ``prepare_hubbard_structure``.

``prepare_hubbard_corr_info`` is pure-Python and does not need an AiiDA profile.
``prepare_hubbard_structure`` creates AiiDA ``StructureData`` and
``HubbardStructureData`` nodes, so those tests request the ``aiida_profile``
fixture to ensure an active profile is present.
"""
import pytest


# =============================================================================
# ASE structure helpers
# =============================================================================

def _ase_feo():
    """FeO rock-salt unit cell (Fe at origin, O at body centre)."""
    from ase import Atoms
    a = 4.334
    return Atoms(
        symbols=['Fe', 'O'],
        positions=[(0, 0, 0), (a / 2, a / 2, a / 2)],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True,
    )


def _ase_ofe():
    """Same cell but O listed first - good for reorder test."""
    from ase import Atoms
    a = 4.334
    return Atoms(
        symbols=['O', 'Fe'],
        positions=[(a / 2, a / 2, a / 2), (0, 0, 0)],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True,
    )


def _ase_fe2o3():
    """Simplified structure with 2 Fe and 3 O atoms."""
    from ase import Atoms
    a = 5.0
    return Atoms(
        symbols=['Fe', 'Fe', 'O', 'O', 'O'],
        positions=[
            (0,       0,       0),
            (a / 2,   a / 2,   0),
            (a / 4,   a / 4,   a / 4),
            (3*a/4,   a / 4,   a / 4),
            (a / 4,   3*a/4,   a / 4),
        ],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True,
    )


# =============================================================================
# prepare_hubbard_corr_info (pure Python - no AiiDA needed)
# =============================================================================

class TestPrepareHubbardCorrInfo:
    """Tests for ``prepare_hubbard_corr_info``: the wrapper that calls the three helpers."""

    def test_returns_three_tuple(self):
        from lordcapulet.utils.preprocessing.submission import prepare_hubbard_corr_info
        atoms = _ase_feo()
        result = prepare_hubbard_corr_info(atoms)
        assert isinstance(result, tuple) and len(result) == 3

    def test_feo_single_tm(self):
        from lordcapulet.utils.preprocessing.submission import prepare_hubbard_corr_info
        atoms = _ase_feo()
        hubbard_corr_atoms, hubbard_corr_manifolds, hubbard_corr_dimensions = prepare_hubbard_corr_info(atoms)
        assert hubbard_corr_atoms == ['Fe1']
        assert hubbard_corr_manifolds == ['3d']
        assert hubbard_corr_dimensions == [50]   # 5*5*2

    def test_tag_is_set_on_atoms(self):
        """prepare_hubbard_corr_info must set atom.tag on the input Atoms in-place."""
        from lordcapulet.utils.preprocessing.submission import prepare_hubbard_corr_info
        atoms = _ase_feo()
        prepare_hubbard_corr_info(atoms)
        fe = next(a for a in atoms if a.symbol == 'Fe')
        assert fe.tag == 1

    def test_two_tm_sequential_tags(self):
        from lordcapulet.utils.preprocessing.submission import prepare_hubbard_corr_info
        atoms = _ase_fe2o3()
        hubbard_corr_atoms, _manifolds, _dims = prepare_hubbard_corr_info(atoms)
        assert hubbard_corr_atoms == ['Fe1', 'Fe2']
        assert _manifolds == ['3d', '3d']
        assert _dims == [50, 50]

    def test_custom_table_overrides_default(self):
        """Treating O as the Hubbard atom via a custom table."""
        from lordcapulet.utils.preprocessing.submission import prepare_hubbard_corr_info
        atoms = _ase_feo()
        hubbard_corr_atoms, hubbard_corr_manifolds, _dims = prepare_hubbard_corr_info(atoms, table={'O'})
        assert hubbard_corr_atoms == ['O1']
        assert hubbard_corr_manifolds == ['2p']

    def test_empty_structure_returns_empty_lists(self):
        from ase import Atoms
        from lordcapulet.utils.preprocessing.submission import prepare_hubbard_corr_info
        atoms = Atoms()  # completely empty
        hubbard_corr_atoms, hubbard_corr_manifolds, hubbard_corr_dimensions = prepare_hubbard_corr_info(atoms)
        assert hubbard_corr_atoms == []
        assert hubbard_corr_manifolds == []
        assert hubbard_corr_dimensions == []

    def test_no_tm_in_structure(self):
        """A structure with no TM atoms returns empty lists."""
        from ase import Atoms
        from lordcapulet.utils.preprocessing.submission import prepare_hubbard_corr_info
        a = 4.0
        atoms = Atoms(
            symbols=['O', 'H'],
            positions=[(0, 0, 0), (1, 0, 0)],
            cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        )
        hubbard_corr_atoms, _m, _d = prepare_hubbard_corr_info(atoms)
        assert hubbard_corr_atoms == []


# =============================================================================
# prepare_hubbard_structure (needs AiiDA profile)
# =============================================================================

def _hs_feo(aiida_profile, **kwargs):
    """Helper: tag FeO atoms, call prepare_hubbard_structure, return result."""
    from lordcapulet.utils.preprocessing.submission import (
        prepare_hubbard_structure,
        prepare_hubbard_corr_info,
    )
    atoms = _ase_feo()
    hubbard_corr_atoms, hubbard_corr_manifolds, _ = prepare_hubbard_corr_info(atoms)
    return prepare_hubbard_structure(atoms, hubbard_corr_atoms, hubbard_corr_manifolds, **kwargs)


def _hs_fe2o3(aiida_profile, **kwargs):
    """Helper: tag Fe2O3-like atoms, call prepare_hubbard_structure."""
    from lordcapulet.utils.preprocessing.submission import (
        prepare_hubbard_structure,
        prepare_hubbard_corr_info,
    )
    atoms = _ase_fe2o3()
    hubbard_corr_atoms, hubbard_corr_manifolds, _ = prepare_hubbard_corr_info(atoms)
    return prepare_hubbard_structure(atoms, hubbard_corr_atoms, hubbard_corr_manifolds, **kwargs)


class TestPrepareHubbardStructureOnsite:
    """Tests for onsite Hubbard U parameters."""

    def test_returns_hubbard_structure_data(self, aiida_profile):
        from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
        hs = _hs_feo(aiida_profile)
        assert isinstance(hs, HubbardStructureData)

    def test_default_u_value_5(self, aiida_profile):
        """Default U=5.0 should appear in the single onsite parameter."""
        hs = _hs_feo(aiida_profile)
        onsite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'Ueff']
        assert len(onsite) == 1
        assert onsite[0].value == pytest.approx(5.0)

    def test_custom_scalar_u(self, aiida_profile):
        """Scalar U override is applied to every TM site."""
        hs = _hs_feo(aiida_profile, U_values=7.5)
        onsite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'Ueff']
        assert len(onsite) == 1
        assert onsite[0].value == pytest.approx(7.5)

    def test_list_u_single_site(self, aiida_profile):
        """List with one value still applied correctly."""
        hs = _hs_feo(aiida_profile, U_values=[3.5])
        onsite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'Ueff']
        assert len(onsite) == 1
        assert onsite[0].value == pytest.approx(3.5)

    def test_list_u_two_sites(self, aiida_profile):
        """Different U per TM atom via list."""
        hs = _hs_fe2o3(aiida_profile, U_values=[3.0, 7.0])
        onsite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'Ueff']
        assert len(onsite) == 2
        values = {round(p.value, 6) for p in onsite}
        assert 3.0 in values
        assert 7.0 in values

    def test_scalar_u_applied_uniformly_two_sites(self, aiida_profile):
        """Single scalar U broadcasts to all TM atoms."""
        hs = _hs_fe2o3(aiida_profile, U_values=6.0)
        onsite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'Ueff']
        assert len(onsite) == 2
        for p in onsite:
            assert p.value == pytest.approx(6.0)

    def test_u_values_length_mismatch_raises(self, aiida_profile):
        from lordcapulet.utils.preprocessing.submission import (
            prepare_hubbard_structure,
            prepare_hubbard_corr_info,
        )
        atoms = _ase_feo()
        hubbard_corr_atoms, hubbard_corr_manifolds, _ = prepare_hubbard_corr_info(atoms)
        with pytest.raises(ValueError, match='lengths must match'):
            prepare_hubbard_structure(atoms, hubbard_corr_atoms, hubbard_corr_manifolds, U_values=[1.0, 2.0])

    def test_onsite_manifold_recorded(self, aiida_profile):
        """The manifold in the stored parameter must match hubbard_corr_manifolds."""
        hs = _hs_feo(aiida_profile)
        onsite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'Ueff']
        assert onsite[0].atom_manifold == '3d'

    def test_no_intersite_by_default(self, aiida_profile):
        """Without neighbors, no intersite V terms should be present."""
        hs = _hs_feo(aiida_profile)
        intersite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'V']
        assert intersite == []


class TestPrepareHubbardStructureOrdering:
    """Tests that Hubbard (TM) atoms precede ligands after reorder_atoms."""

    def test_fe_before_o_in_feo(self, aiida_profile):
        """For FeO with Fe@0 O@1: Fe should still be first after reorder."""
        hs = _hs_feo(aiida_profile)
        symbols = [s.kind_name for s in hs.sites]
        fe_indices = [i for i, n in enumerate(symbols) if n.startswith('Fe')]
        o_indices = [i for i, n in enumerate(symbols) if n.startswith('O')]
        assert fe_indices, "Fe atom missing from structure"
        assert o_indices, "O atom missing from structure"
        assert max(fe_indices) < min(o_indices)

    def test_fe_before_o_when_o_comes_first(self, aiida_profile):
        """Even if O is listed first in ASE atoms, reorder puts Fe first."""
        from lordcapulet.utils.preprocessing.submission import (
            prepare_hubbard_structure,
            prepare_hubbard_corr_info,
        )
        atoms = _ase_ofe()   # O is atom[0], Fe is atom[1]
        hubbard_corr_atoms, hubbard_corr_manifolds, _ = prepare_hubbard_corr_info(atoms)
        hs = prepare_hubbard_structure(atoms, hubbard_corr_atoms, hubbard_corr_manifolds)

        symbols = [s.kind_name for s in hs.sites]
        fe_indices = [i for i, n in enumerate(symbols) if n.startswith('Fe')]
        o_indices = [i for i, n in enumerate(symbols) if n.startswith('O')]
        assert max(fe_indices) < min(o_indices)

    def test_all_two_fe_before_o_in_fe2o3(self, aiida_profile):
        """Both Fe atoms must appear before all O atoms."""
        hs = _hs_fe2o3(aiida_profile)
        symbols = [s.kind_name for s in hs.sites]
        fe_indices = [i for i, n in enumerate(symbols) if n.startswith('Fe')]
        o_indices = [i for i, n in enumerate(symbols) if n.startswith('O')]
        assert len(fe_indices) == 2
        assert len(o_indices) == 3
        assert max(fe_indices) < min(o_indices)

    def test_site_count_preserved(self, aiida_profile):
        """Reordering must not add or remove atoms."""
        hs = _hs_feo(aiida_profile)
        assert len(hs.sites) == 2

    def test_fe2o3_site_count_preserved(self, aiida_profile):
        hs = _hs_fe2o3(aiida_profile)
        assert len(hs.sites) == 5


class TestPrepareHubbardStructureIntersite:
    """Tests for the intersite Hubbard V functionality."""

    def test_tuple_neighbors_adds_v_terms(self, aiida_profile):
        """Providing neighbors as tuples creates intersite V parameters."""
        hs = _hs_feo(aiida_profile, neighbors=[('O1', '2p')], intersite_V_values=1.5)
        intersite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'V']
        assert len(intersite) >= 1
        for p in intersite:
            assert p.value == pytest.approx(1.5)

    def test_dict_neighbors_same_result_as_tuples(self, aiida_profile):
        """Dict format and tuple format should produce the same parameter count."""
        from lordcapulet.utils.preprocessing.submission import (
            prepare_hubbard_structure,
            prepare_hubbard_corr_info,
        )

        atoms_t = _ase_feo()
        hubbard_corr_atoms_t, manifolds_t, _ = prepare_hubbard_corr_info(atoms_t)
        hs_t = prepare_hubbard_structure(
            atoms_t, hubbard_corr_atoms_t, manifolds_t, neighbors=[('O1', '2p')], intersite_V_values=2.0
        )

        atoms_d = _ase_feo()
        hubbard_corr_atoms_d, manifolds_d, _ = prepare_hubbard_corr_info(atoms_d)
        hs_d = prepare_hubbard_structure(
            atoms_d, hubbard_corr_atoms_d, manifolds_d,
            neighbors=[{'name': 'O1', 'manifold': '2p'}],
            intersite_V_values=2.0,
        )

        n_t = sum(1 for p in hs_t.hubbard.parameters if p.hubbard_type == 'V')
        n_d = sum(1 for p in hs_d.hubbard.parameters if p.hubbard_type == 'V')
        assert n_t == n_d

    def test_default_v_is_1_when_none(self, aiida_profile):
        """If intersite_V_values is None, V defaults to 1.0."""
        hs = _hs_feo(aiida_profile, neighbors=[('O1', '2p')])
        intersite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'V']
        assert len(intersite) >= 1
        for p in intersite:
            assert p.value == pytest.approx(1.0)

    def test_scalar_v_broadcast_over_all_neighbors(self, aiida_profile):
        """Float intersite_V_values is broadcast to every neighbor entry."""
        hs = _hs_feo(aiida_profile, neighbors=[('O1', '2p')], intersite_V_values=3.3)
        intersite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'V']
        for p in intersite:
            assert p.value == pytest.approx(3.3)

    def test_v_values_length_mismatch_raises(self, aiida_profile):
        from lordcapulet.utils.preprocessing.submission import (
            prepare_hubbard_structure,
            prepare_hubbard_corr_info,
        )
        atoms = _ase_feo()
        hubbard_corr_atoms, hubbard_corr_manifolds, _ = prepare_hubbard_corr_info(atoms)
        with pytest.raises(ValueError, match='lengths must match'):
            prepare_hubbard_structure(
                atoms, hubbard_corr_atoms, hubbard_corr_manifolds,
                neighbors=[('O1', '2p')],
                intersite_V_values=[1.0, 2.0],  # 2 values for 1 neighbor
            )

    def test_two_tm_both_get_v_terms(self, aiida_profile):
        """Each TM atom gets a V term for each neighbor entry."""
        hs = _hs_fe2o3(aiida_profile, neighbors=[('O1', '2p')], intersite_V_values=0.5)
        intersite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'V']
        # 2 TM atoms x 1 neighbor = at least 2 intersite parameters
        assert len(intersite) >= 2
        for p in intersite:
            assert p.value == pytest.approx(0.5)

    def test_list_v_per_neighbor(self, aiida_profile):
        """List V_values maps one value per neighbor entry."""
        hs = _hs_feo(
            aiida_profile,
            neighbors=[('O1', '2p'), ('O1', '2p')],
            intersite_V_values=[1.0, 2.0],
        )
        intersite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'V']
        assert len(intersite) >= 2
        values_present = {round(p.value, 6) for p in intersite}
        assert 1.0 in values_present
        assert 2.0 in values_present

    def test_onsite_preserved_when_intersite_present(self, aiida_profile):
        """Adding intersite V must not remove or change the onsite U."""
        hs = _hs_feo(aiida_profile, U_values=4.0, neighbors=[('O1', '2p')])
        onsite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'Ueff']
        assert len(onsite) == 1
        assert onsite[0].value == pytest.approx(4.0)

    def test_neighbors_none_produces_no_intersite(self, aiida_profile):
        """Default neighbors=None means zero V parameters."""
        hs = _hs_feo(aiida_profile, neighbors=None)
        intersite = [p for p in hs.hubbard.parameters if p.hubbard_type == 'V']
        assert intersite == []
