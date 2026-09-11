"""Tests for the ``propose_new_constraints`` dispatch function."""

import numpy as np
import pytest

from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData


def _make_sample_occ_list(n_matrices=3, n_atoms=2, dim=5):
    """Helper to create a list of OccupationMatrixData for testing proposals."""
    occ_list = []
    for _ in range(n_matrices):
        data = {}
        for iatom in range(n_atoms):
            label = f'atom{iatom + 1}'
            data[label] = {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': np.diag(np.random.rand(dim)).tolist(),
                    'down': np.diag(np.random.rand(dim)).tolist(),
                },
            }
        occ_list.append(OccupationMatrixData(data))
    return occ_list


class TestProposeNewConstraints:
    """Test the ``propose_new_constraints`` function (AiiDA-free dispatch logic)."""

    def test_random_mode_returns_correct_type(self):
        """Random mode should return a list of OccupationMatrixData."""
        from lordcapulet.functions.propose import propose_new_constraints

        occ_list = _make_sample_occ_list()
        proposals = propose_new_constraints(occ_list, N=3, mode='random', debug=False)

        assert isinstance(proposals, list)
        assert len(proposals) == 3
        for p in proposals:
            assert isinstance(p, OccupationMatrixData)

    def test_random_so_n_mode_returns_correct_type(self):
        """random_so_n mode should return a list of OccupationMatrixData."""
        from lordcapulet.functions.propose import propose_new_constraints

        occ_list = _make_sample_occ_list()
        proposals = propose_new_constraints(occ_list, N=3, mode='random_so_n', debug=False)

        assert isinstance(proposals, list)
        assert len(proposals) == 3
        for p in proposals:
            assert isinstance(p, OccupationMatrixData)

    def test_invalid_mode_raises(self):
        """An invalid mode should raise an error (no match statement case)."""
        from lordcapulet.functions.propose import propose_new_constraints

        occ_list = _make_sample_occ_list()

        # Python match/case does not raise by default for unmatched cases,
        # but propose_new_constraints should return None or raise
        # Based on code: no default case, so proposals will be undefined -> UnboundLocalError
        with pytest.raises(Exception):
            propose_new_constraints(occ_list, N=3, mode='nonexistent', debug=False)

    def test_read_mode_raises_not_implemented(self):
        """The 'read' mode should raise NotImplementedError."""
        from lordcapulet.functions.propose import propose_new_constraints

        occ_list = _make_sample_occ_list()

        with pytest.raises(NotImplementedError):
            propose_new_constraints(occ_list, N=3, mode='read', debug=False)

    def test_returns_correct_count(self):
        """Should return exactly N proposals."""
        from lordcapulet.functions.propose import propose_new_constraints

        occ_list = _make_sample_occ_list()

        for n in [1, 5, 10]:
            proposals = propose_new_constraints(occ_list, N=n, mode='random', debug=False)
            assert len(proposals) == n

    def test_n_less_than_1_raises(self):
        """N < 1 should raise ValueError."""
        from lordcapulet.functions.propose import propose_new_constraints

        occ_list = _make_sample_occ_list()

        with pytest.raises(ValueError, match='N must be greater than or equal to 1'):
            propose_new_constraints(occ_list, N=0, mode='random', debug=False)

    def test_gp_mode_generation_zero_falls_back_to_random(self):
        """GP mode at generation 0 should produce random proposals."""
        from lordcapulet.functions.propose import propose_new_constraints

        occ_list = _make_sample_occ_list()
        energies = [-100.0, -99.5, -101.0]

        proposals = propose_new_constraints(
            occ_list, N=3, mode='gp', debug=False,
            energies=energies, current_generation=0,
        )

        assert isinstance(proposals, list)
        assert len(proposals) == 3

    def test_proposals_preserve_atom_labels(self):
        """Proposals should preserve atom labels from input."""
        from lordcapulet.functions.propose import propose_new_constraints

        occ_list = _make_sample_occ_list(n_atoms=2)
        proposals = propose_new_constraints(occ_list, N=3, mode='random', debug=False)

        input_labels = set(occ_list[0].get_atom_labels())
        for p in proposals:
            assert set(p.get_atom_labels()) == input_labels

    def test_proposals_preserve_species(self):
        """Proposals should preserve atom species from input."""
        from lordcapulet.functions.propose import propose_new_constraints

        occ_list = _make_sample_occ_list()
        proposals = propose_new_constraints(occ_list, N=3, mode='random', debug=False)

        input_species = occ_list[0].get_atom_species()
        for p in proposals:
            assert p.get_atom_species() == input_species

class TestProposalConfigResolution:
    """Unit tests for the proposal-config key normalization."""

    def test_none_when_absent(self):
        """No config key -> None, and kwargs are left untouched."""
        from lordcapulet.functions.propose import _pop_proposal_config

        kwargs = {'energies': [1.0, 2.0]}
        assert _pop_proposal_config(kwargs) is None
        assert kwargs == {'energies': [1.0, 2.0]}

    @pytest.mark.parametrize('key', [
        'proposal_config',
        'gp_config',
        'bandit_config',
        'linear_bandit_config',
        'rf_config',
        'forest_bandit_config',
    ])
    def test_each_accepted_key_is_popped(self, key):
        """Every accepted key is removed from kwargs and returned."""
        from lordcapulet.functions.propose import _pop_proposal_config

        config = {'beta': 0.5}
        kwargs = {key: config, 'energies': [1.0, 2.0]}

        assert _pop_proposal_config(kwargs) == config
        assert kwargs == {'energies': [1.0, 2.0]}

    def test_multiple_keys_raise(self):
        """Supplying two config keys is ambiguous and should raise."""
        from lordcapulet.functions.propose import _pop_proposal_config

        with pytest.raises(ValueError, match='Multiple proposal configurations'):
            _pop_proposal_config({'proposal_config': {}, 'bandit_config': {}})


class TestBanditConfigPlumbing:
    """The proposal config must actually reach the bandit proposers.

    Regression test: the dispatcher used to pop ``linear_bandit_config`` /
    ``rf_config`` and forward them as ``gp_config``, so the bandit config was
    silently dropped and the bandit fell back to random proposals.
    """

    @staticmethod
    def _small_config():
        """Minimal bandit config (no 'features' section, small ensemble)."""
        return {
            'model_kwargs': {'tol': 1e-6},
            'acquisition': {'beta': 0.5, 'eta': 10},
            'optimization': {
                'ensamble_size': 200,
                'patchwork_params': {'apply_rotation': False},
            },
        }

    @pytest.mark.parametrize('key', ['proposal_config', 'bandit_config', 'linear_bandit_config'])
    def test_linear_bandit_config_reaches_proposer(self, key):
        """Each accepted key should run the bandit (not silently fall back)."""
        pytest.importorskip('sklearn')

        from lordcapulet.functions.propose import propose_new_constraints

        occ_list = _make_sample_occ_list(n_matrices=8)
        energies = [-10.0 + 0.1 * i for i in range(len(occ_list))]
        messages = []

        proposals = propose_new_constraints(
            occ_list,
            N=2,
            mode='linear_bandit',
            debug=False,
            reporter=messages.append,
            energies=energies,
            current_generation=1,
            **{key: self._small_config()},
        )

        assert len(proposals) == 2
        assert not any('Error in linear bandit' in msg for msg in messages), (
            f"config passed as '{key}' never reached the bandit: {messages}"
        )

    def test_legacy_n_iter_kwarg_is_translated(self):
        """A stale 'n_iter' model kwarg is translated to 'max_iter' (sklearn >= 1.4)."""
        pytest.importorskip('sklearn')

        from lordcapulet.functions.proposal_modes.linear_bandit import (
            propose_linear_bandit_constraints,
        )

        occ_list = _make_sample_occ_list(n_matrices=8)
        energies = [-10.0 + 0.1 * i for i in range(len(occ_list))]

        proposals = propose_linear_bandit_constraints(
            occ_list,
            energies,
            natoms=2,
            N=2,
            bandit_config={
                'model_kwargs': {'n_iter': 300, 'tol': 1e-6},
                'acquisition': {'beta': 0.5, 'eta': 10},
                'optimization': {'ensamble_size': 200,
                                 'patchwork_params': {'apply_rotation': False}},
            },
            reporter=lambda msg: None,
        )

        assert len(proposals) == 2


class TestBanditDefaultFeatures:
    """Bandits must work when the config omits the 'features' section."""

    def test_linear_bandit_runs_without_features_section(self):
        pytest.importorskip('sklearn')

        from lordcapulet.functions.proposal_modes.linear_bandit import (
            propose_linear_bandit_constraints,
        )

        occ_list = _make_sample_occ_list(n_matrices=8)
        energies = [-10.0 + 0.1 * i for i in range(len(occ_list))]

        proposals = propose_linear_bandit_constraints(
            occ_list,
            energies,
            natoms=2,
            N=2,
            bandit_config={
                'acquisition': {'beta': 0.5, 'eta': 10},
                'optimization': {'ensamble_size': 200,
                                 'patchwork_params': {'apply_rotation': False}},
            },
            reporter=lambda msg: None,
        )

        assert len(proposals) == 2
        assert proposals[0].get_atom_labels() == occ_list[0].get_atom_labels()