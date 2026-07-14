"""Pure-Python tests for the convergence-rate helpers in
``lordcapulet.utils.postprocessing.gather_workchain_data``.

These tests exercise the labeling logic without an AiiDA profile: they
build synthetic ``calc_data_list`` dicts in the shape produced by
``_extract_calc_data`` and feed them to the helper directly.
"""
import pytest

from lordcapulet.utils.postprocessing.gather_workchain_data import (
    WorkchainDataExtractor,
    _constrained_label,
    _proposal_mode_to_kind,
)


def _calc(source: str, generation_number, converged: bool, proposal_source=None) -> dict:
    return {
        'calculation_source': source,
        'generation_number': generation_number,
        'converged': converged,
        'proposal_source': proposal_source,
    }


class TestProposalModeToKind:
    @pytest.mark.parametrize('raw,kind', [
        ('gp', 'gp'),
        ('gaussian_process', 'gp'),
        ('random', 'random'),
        ('random_so_n', 'random'),
        ('read', 'plain'),
        (None, 'plain'),
        ('bogus', 'plain'),
    ])
    def test_normalization(self, raw, kind):
        assert _proposal_mode_to_kind(raw) == kind


class TestConstrainedLabel:
    def test_gp_gen_zero_is_gp(self):
        assert _constrained_label('gp', 0) == 'GP proposal'

    def test_gp_gen_one_is_random(self):
        assert _constrained_label('gp', 1) == 'random proposal'

    def test_gp_gen_two_is_gp(self):
        assert _constrained_label('gp', 2) == 'GP proposal'

    def test_random_always_random(self):
        assert _constrained_label('random', 0) == 'random proposal'
        assert _constrained_label('random', 3) == 'random proposal'

    def test_plain_is_constrained_scan(self):
        assert _constrained_label('plain', 0) == 'constrained scan'
        assert _constrained_label('plain', None) == 'constrained scan'


class TestConvergenceRatesBySource:
    def _extractor(self):
        # WorkchainDataExtractor __init__ does not touch AiiDA.
        return WorkchainDataExtractor(include_non_converged=True)

    def test_empty_list(self):
        ex = self._extractor()
        assert ex._convergence_rates_by_source([], 'gp') == {}

    def test_afm_only(self):
        ex = self._extractor()
        calcs = [
            _calc('afm_workchain', None, True),
            _calc('afm_workchain', None, False),
            _calc('afm_workchain', None, True),
        ]
        stats = ex._convergence_rates_by_source(calcs, 'gp')
        assert set(stats) == {'standard magnetic scan'}
        bucket = stats['standard magnetic scan']
        assert bucket == {'converged': 2, 'total': 3, 'rate': 2 / 3}

    def test_gp_splits_by_generation(self):
        ex = self._extractor()
        calcs = [
            _calc('constrained_scan', 1, True),
            _calc('constrained_scan', 1, False),
            _calc('constrained_scan', 1, True),
            _calc('constrained_scan', 2, False),
        ]
        stats = ex._convergence_rates_by_source(calcs, 'gaussian_process')
        assert stats['random proposal'] == {'converged': 2, 'total': 3, 'rate': 2 / 3}
        assert stats['GP proposal'] == {'converged': 0, 'total': 1, 'rate': 0.0}

    def test_explicit_proposal_source_overrides_generation_inference(self):
        ex = self._extractor()
        calcs = [
            _calc('constrained_scan', 2, True, proposal_source='random_fallback'),
            _calc('constrained_scan', 1, True, proposal_source='gp'),
        ]
        stats = ex._convergence_rates_by_source(calcs, 'gaussian_process')
        assert stats['random fallback'] == {'converged': 1, 'total': 1, 'rate': 1.0}
        assert stats['GP proposal'] == {'converged': 1, 'total': 1, 'rate': 1.0}

    def test_random_mode_collapses_all_gens(self):
        ex = self._extractor()
        calcs = [
            _calc('constrained_scan', 0, True),
            _calc('constrained_scan', 1, False),
            _calc('constrained_scan', 2, True),
        ]
        stats = ex._convergence_rates_by_source(calcs, 'random_so_n')
        assert set(stats) == {'random proposal'}
        assert stats['random proposal'] == {'converged': 2, 'total': 3, 'rate': 2 / 3}

    def test_plain_mode_constrained_scan_label(self):
        ex = self._extractor()
        calcs = [
            _calc('constrained_scan', 0, True),
            _calc('constrained_scan', 0, False),
        ]
        stats = ex._convergence_rates_by_source(calcs, None)
        assert set(stats) == {'constrained scan'}
        assert stats['constrained scan'] == {'converged': 1, 'total': 2, 'rate': 0.5}

    def test_mixed_afm_and_gp(self):
        ex = self._extractor()
        calcs = [
            _calc('afm_workchain', None, True),
            _calc('afm_workchain', None, False),
            _calc('constrained_scan', 1, True),
            _calc('constrained_scan', 1, False),
            _calc('constrained_scan', 2, True),
        ]
        stats = ex._convergence_rates_by_source(calcs, 'gp')
        assert stats['standard magnetic scan']['rate'] == 0.5
        assert stats['random proposal'] == {'converged': 1, 'total': 2, 'rate': 0.5}
        assert stats['GP proposal'] == {'converged': 1, 'total': 1, 'rate': 1.0}

    def test_all_converged_rate_is_one(self):
        ex = self._extractor()
        calcs = [_calc('constrained_scan', 2, True) for _ in range(4)]
        stats = ex._convergence_rates_by_source(calcs, 'gp')
        assert stats['GP proposal']['rate'] == 1.0

    def test_all_failed_rate_is_zero(self):
        ex = self._extractor()
        calcs = [_calc('constrained_scan', 2, False) for _ in range(4)]
        stats = ex._convergence_rates_by_source(calcs, 'gp')
        assert stats['GP proposal']['rate'] == 0.0


class TestConstructorValidation:
    def test_valid_modes_accepted(self):
        for mode in ('random', 'random_so_n', 'gaussian_process', 'gp', 'read', None):
            WorkchainDataExtractor(proposal_mode_override=mode)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match='proposal_mode_override'):
            WorkchainDataExtractor(proposal_mode_override='bogus')


class TestOutputMetadata:
    def test_active_profile_is_recorded(self, monkeypatch):
        class Profile:
            name = 'presto-pg'

        monkeypatch.setattr(
            'lordcapulet.utils.postprocessing.gather_workchain_data.aiida.get_profile',
            lambda: Profile(),
        )

        ex = WorkchainDataExtractor()
        data = ex._build_output_data(
            calc_data_list=[],
            workchain_info={'pk': 123, 'process_type': 'test', 'node_type': 'WorkChainNode'},
        )

        assert data['metadata']['profile'] == 'presto-pg'
