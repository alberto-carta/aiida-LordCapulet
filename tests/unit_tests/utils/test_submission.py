"""Tests for preprocessing submission utilities."""

import pytest

from lordcapulet.utils.preprocessing.submission import (
    default_dimensions,
    default_manifold,
    get_default_manifolds,
    get_dimensions,
    tag_and_list_atoms,
)


class _MockAtom:
    """Minimal mock for ASE-like atom objects."""

    def __init__(self, symbol):
        self.symbol = symbol
        self.tag = 0


class TestTagAndListAtoms:
    """Test ``tag_and_list_atoms``."""

    def test_iron_oxide(self):
        """Fe should be tagged as TM, O should not appear in result."""
        atoms = [_MockAtom('Fe'), _MockAtom('O')]
        hubbard_corr_atoms = tag_and_list_atoms(atoms)

        assert hubbard_corr_atoms == ['Fe1']
        assert atoms[0].tag == 1

    def test_multiple_tm(self):
        """Multiple TM atoms should get sequential tags."""
        atoms = [_MockAtom('Fe'), _MockAtom('Ni'), _MockAtom('O'), _MockAtom('Fe')]
        hubbard_corr_atoms = tag_and_list_atoms(atoms)

        assert hubbard_corr_atoms == ['Fe1', 'Ni1', 'Fe2']

    def test_no_tm(self):
        """Structure with no TM atoms should return empty list."""
        atoms = [_MockAtom('O'), _MockAtom('H')]
        hubbard_corr_atoms = tag_and_list_atoms(atoms)

        assert hubbard_corr_atoms == []

    def test_custom_table(self):
        """Custom TM table should override the default."""
        atoms = [_MockAtom('O'), _MockAtom('Fe')]
        # Only O in custom table
        hubbard_corr_atoms = tag_and_list_atoms(atoms, table={'O'})

        assert hubbard_corr_atoms == ['O1']

    def test_all_3d_transition_metals(self):
        """All 3d TM elements should be recognized."""
        elements_3d = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
        atoms = [_MockAtom(el) for el in elements_3d]
        hubbard_corr_atoms = tag_and_list_atoms(atoms)

        assert len(hubbard_corr_atoms) == 10
        for el in elements_3d:
            assert f'{el}1' in hubbard_corr_atoms


class TestGetDefaultManifolds:
    """Test ``get_default_manifolds``."""

    def test_iron(self):
        assert get_default_manifolds(['Fe1']) == ['3d']

    def test_uranium(self):
        assert get_default_manifolds(['U1']) == ['5f']

    def test_nickel(self):
        assert get_default_manifolds(['Ni1']) == ['3d']

    def test_multiple(self):
        result = get_default_manifolds(['Fe1', 'Ni2', 'U1'])
        assert result == ['3d', '3d', '5f']

    def test_lanthanide(self):
        assert get_default_manifolds(['Ce1']) == ['4f']

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match='No default manifold found'):
            get_default_manifolds(['Xx1'])


class TestGetDimensions:
    """Test ``get_dimensions``."""

    def test_d_orbital(self):
        """d-orbital: 5 * 5 * 2 = 50."""
        assert get_dimensions(['3d']) == [50]

    def test_f_orbital(self):
        """f-orbital: 7 * 7 * 2 = 98."""
        assert get_dimensions(['5f']) == [98]

    def test_p_orbital(self):
        """p-orbital: 3 * 3 * 2 = 18."""
        assert get_dimensions(['2p']) == [18]

    def test_s_orbital(self):
        """s-orbital: 1 * 1 * 2 = 2."""
        assert get_dimensions(['1s']) == [2]

    def test_multiple_manifolds(self):
        result = get_dimensions(['3d', '5f', '2p'])
        assert result == [50, 98, 18]

    def test_unknown_manifold_raises(self):
        with pytest.raises(ValueError, match='No default dimension found'):
            get_dimensions(['3x'])


class TestDefaultManifoldCompleteness:
    """Test that the default_manifold dictionary is complete."""

    def test_all_3d_present(self):
        elements_3d = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
        for el in elements_3d:
            assert el in default_manifold
            assert default_manifold[el] == '3d'

    def test_all_4d_present(self):
        elements_4d = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
        for el in elements_4d:
            assert el in default_manifold
            assert default_manifold[el] == '4d'

    def test_actinides_present(self):
        actinides = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am']
        for el in actinides:
            assert el in default_manifold
            assert default_manifold[el] == '5f'

    def test_lanthanides_present(self):
        lanthanides = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        for el in lanthanides:
            assert el in default_manifold
            assert default_manifold[el] == '4f'

    def test_default_dimensions_complete(self):
        """All orbital types should be in default_dimensions."""
        assert 's' in default_dimensions
        assert 'p' in default_dimensions
        assert 'd' in default_dimensions
        assert 'f' in default_dimensions
