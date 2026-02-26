"""Tests for the ``GlobalConstrainedSearchWorkChain`` class."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from aiida.orm import Dict, Float, Int, Bool, List, Str

from lordcapulet.workflows.global_constrained_search import GlobalConstrainedSearchWorkChain


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _global_inputs(structure, kpoints, pw_code, constrained_code,
                   Nmax=10, N=3):
    """Minimal valid GlobalConstrainedSearchWorkChain inputs."""
    return {
        'afm': {
            'structure': structure,
            'parameters': Dict({'CONTROL': {'calculation': 'scf'},
                                'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
            'kpoints': kpoints,
            'code': pw_code,
            'tm_atoms': List(list=['Fe']),
        },
        'constrained': {
            'structure': structure,
            'parameters': Dict({'CONTROL': {'calculation': 'scf'},
                                'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
            'kpoints': kpoints,
            'code': constrained_code,
            'tm_atoms': List(list=['Fe']),
            'oscdft_card': Dict({'nconstr': 1}),
        },
        'Nmax': Int(Nmax),
        'N': Int(N),
    }


def _stored_list(values):
    """Create, store and return an AiiDA ``List`` node.

    ``process_afm_results`` checks ``len(afm_matrices.get_list())`` and
    ``gather_final_results`` calls ``.get_list()`` on the context lists.
    The nodes must be *stored* so they have a PK and can be returned as
    real workflow outputs.
    """
    node = List(list=values)
    node.store()
    return node


class _FakeOutputs:
    """Lightweight output-namespace stub.

    ``GlobalConstrainedSearchWorkChain.process_afm_results`` does::

        if 'converged_matrix_pks' not in self.ctx.afm_wc.outputs:
            return self.exit_codes.ERROR_AFM_SEARCH_FAILED

    A plain ``MagicMock`` object returns ``True`` for *any* ``__contains__``
    check because ``MagicMock.__contains__`` is itself a mock.  That would
    silently skip the early-return branch, making the test always pass even
    when the guard is broken.

    ``_FakeOutputs`` implements ``__contains__`` via ``hasattr``, so only
    attributes that were explicitly set are considered present – matching
    real AiiDA output-namespace behaviour.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)


def _mock_afm_wc(matrix_pks, calc_pks, all_calc_pks=None):
    """Build a fake finished ``AFMScanWorkChain`` process node.

    ``process_afm_results`` reads three outputs off ``ctx.afm_wc``:
    ``converged_matrix_pks``, ``converged_calculation_pks``, and
    ``all_calculation_pks``.  We bind stored ``List`` nodes to these
    attributes through ``_FakeOutputs`` so attribute access and
    ``'key' in outputs`` checks both work correctly.
    """
    mock = MagicMock()
    mock.is_finished_ok = True
    mock.exit_status = 0
    mock.outputs = _FakeOutputs(
        converged_matrix_pks=_stored_list(matrix_pks),
        converged_calculation_pks=_stored_list(calc_pks),
        all_calculation_pks=_stored_list(
            all_calc_pks if all_calc_pks is not None else calc_pks
        ),
    )
    return mock


def _mock_constrained_wc(matrix_pks, calc_pks, converged_calc_pks=None):
    """Build a fake finished ``ConstrainedScanWorkChain`` process node.

    ``process_constrained_results`` reads three outputs off
    ``ctx.constrained_wc``: ``converged_matrix_pks``, ``all_calculation_pks``,
    and ``converged_calculation_pks``.
    """
    mock = MagicMock()
    mock.is_finished_ok = True
    mock.exit_status = 0
    mock.outputs = _FakeOutputs(
        converged_matrix_pks=_stored_list(matrix_pks),
        all_calculation_pks=_stored_list(calc_pks),
        converged_calculation_pks=_stored_list(
            converged_calc_pks if converged_calc_pks is not None else calc_pks
        ),
    )
    return mock


def _capture_out_factory():
    """Return a (store_dict, side_effect_fn) pair to intercept ``self.out`` calls.

    ``gather_final_results`` calls ``self.out('key', stored_node)`` for each
    output.  Patching ``out`` with this side-effect populates ``store`` so
    tests can assert on the emitted nodes without the workchain being
    connected to a real process graph.
    """
    store = {}

    def _capture(key, value):
        store[key] = value

    return store, _capture


class TestGlobalSearchDefine:
    """Test the process spec definition."""

    def test_inputs_defined(self):
        """Verify all expected global-search-specific inputs are in the spec."""
        from lordcapulet.workflows.global_constrained_search import GlobalConstrainedSearchWorkChain

        spec = GlobalConstrainedSearchWorkChain.spec()

        # Global search specific inputs
        assert 'Nmax' in spec.inputs
        assert 'N' in spec.inputs
        assert 'proposal_mode' in spec.inputs
        assert 'proposal_debug' in spec.inputs
        assert 'proposal_holistic' in spec.inputs
        assert 'proposal_kwargs' in spec.inputs

        # Optional walltime overrides
        assert 'afm_walltime_hours' in spec.inputs
        assert 'constrained_walltime_hours' in spec.inputs

    def test_outputs_defined(self):
        """Verify all expected outputs are in the spec."""
        from lordcapulet.workflows.global_constrained_search import GlobalConstrainedSearchWorkChain

        spec = GlobalConstrainedSearchWorkChain.spec()

        assert 'converged_matrix_pks' in spec.outputs
        assert 'converged_calculation_pks' in spec.outputs
        assert 'all_calculation_pks' in spec.outputs
        assert 'generation_summary' in spec.outputs

    def test_exit_codes_defined(self):
        """Verify all expected exit codes are defined."""
        from lordcapulet.workflows.global_constrained_search import GlobalConstrainedSearchWorkChain

        exit_codes = GlobalConstrainedSearchWorkChain.exit_codes

        assert hasattr(exit_codes, 'ERROR_AFM_SEARCH_FAILED')
        assert exit_codes.ERROR_AFM_SEARCH_FAILED.status == 400

        assert hasattr(exit_codes, 'ERROR_CONSTRAINED_SCAN_FAILED')
        assert exit_codes.ERROR_CONSTRAINED_SCAN_FAILED.status == 401

        assert hasattr(exit_codes, 'ERROR_PROPOSAL_FAILED')
        assert exit_codes.ERROR_PROPOSAL_FAILED.status == 402

    def test_exposed_afm_namespace(self):
        """Verify the AFM namespace is exposed."""
        from lordcapulet.workflows.global_constrained_search import GlobalConstrainedSearchWorkChain

        spec = GlobalConstrainedSearchWorkChain.spec()
        assert 'afm' in spec.inputs

    def test_exposed_constrained_namespace(self):
        """Verify the constrained namespace is exposed."""
        from lordcapulet.workflows.global_constrained_search import GlobalConstrainedSearchWorkChain

        spec = GlobalConstrainedSearchWorkChain.spec()
        assert 'constrained' in spec.inputs


class TestShouldContinueSearch:
    """Test the ``should_continue_search`` step."""

    def test_continues_under_limit(
        self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family,
    ):
        """should_continue_search returns True when N_cumulative < Nmax."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        pw_code = fixture_code('quantumespresso.pw')
        constrained_code = fixture_code('lordcapulet.constrained_pw')

        inputs = {
            'afm': {
                'structure': structure,
                'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
                'kpoints': kpoints,
                'code': pw_code,
                'tm_atoms': List(list=['Fe']),
            },
            'constrained': {
                'structure': structure,
                'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
                'kpoints': kpoints,
                'code': constrained_code,
                'tm_atoms': List(list=['Fe']),
                'oscdft_card': Dict({'nconstr': 1}),
            },
            'Nmax': Int(100),
            'N': Int(10),
        }

        process = generate_workchain(GlobalConstrainedSearchWorkChain, inputs)

        # Manually set cumulative below limit
        process.ctx.N_cumulative = 50

        assert process.should_continue_search() is True

    def test_stops_at_limit(
        self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family,
    ):
        """should_continue_search returns False when N_cumulative >= Nmax."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        pw_code = fixture_code('quantumespresso.pw')
        constrained_code = fixture_code('lordcapulet.constrained_pw')

        inputs = {
            'afm': {
                'structure': structure,
                'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
                'kpoints': kpoints,
                'code': pw_code,
                'tm_atoms': List(list=['Fe']),
            },
            'constrained': {
                'structure': structure,
                'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
                'kpoints': kpoints,
                'code': constrained_code,
                'tm_atoms': List(list=['Fe']),
                'oscdft_card': Dict({'nconstr': 1}),
            },
            'Nmax': Int(100),
            'N': Int(10),
        }

        process = generate_workchain(GlobalConstrainedSearchWorkChain, inputs)

        # Set cumulative at limit
        process.ctx.N_cumulative = 100

        assert process.should_continue_search() is False

    def test_stops_above_limit(
        self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family,
    ):
        """should_continue_search returns False when N_cumulative > Nmax."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        pw_code = fixture_code('quantumespresso.pw')
        constrained_code = fixture_code('lordcapulet.constrained_pw')

        inputs = {
            'afm': {
                'structure': structure,
                'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
                'kpoints': kpoints,
                'code': pw_code,
                'tm_atoms': List(list=['Fe']),
            },
            'constrained': {
                'structure': structure,
                'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
                'kpoints': kpoints,
                'code': constrained_code,
                'tm_atoms': List(list=['Fe']),
                'oscdft_card': Dict({'nconstr': 1}),
            },
            'Nmax': Int(100),
            'N': Int(10),
        }

        process = generate_workchain(GlobalConstrainedSearchWorkChain, inputs)

        process.ctx.N_cumulative = 150

        assert process.should_continue_search() is False


class TestUpdateCounters:
    """Test the ``update_counters`` step."""

    def test_increments_cumulative(
        self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family,
    ):
        """update_counters should add the generation's n_calculations to N_cumulative."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        pw_code = fixture_code('quantumespresso.pw')
        constrained_code = fixture_code('lordcapulet.constrained_pw')

        inputs = {
            'afm': {
                'structure': structure,
                'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
                'kpoints': kpoints,
                'code': pw_code,
                'tm_atoms': List(list=['Fe']),
            },
            'constrained': {
                'structure': structure,
                'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
                'kpoints': kpoints,
                'code': constrained_code,
                'tm_atoms': List(list=['Fe']),
                'oscdft_card': Dict({'nconstr': 1}),
            },
            'Nmax': Int(100),
            'N': Int(10),
        }

        process = generate_workchain(GlobalConstrainedSearchWorkChain, inputs)

        # Inject context state
        process.ctx.N_cumulative = 20
        process.ctx.generation = 2
        process.ctx.generation_results = {
            2: {
                'type': 'constrained',
                'n_calculations': 10,
                'n_successful': 8,
                'n_failed': 2,
            }
        }

        process.update_counters()

        assert process.ctx.N_cumulative == 30  # 20 + 10


# ---------------------------------------------------------------------------
# New comprehensive data-flow tests
# ---------------------------------------------------------------------------

class TestProcessAfmResults:
    """Test the ``process_afm_results`` step.

    Strategy
    --------
    ``process_afm_results`` reads outputs from ``ctx.afm_wc`` and then calls
    ``aiida_propose_occ_matrices_from_results`` to get the first batch of
    proposals.  Both are replaced in tests:

    * ``ctx.afm_wc`` is set to a ``_mock_afm_wc(...)`` stub that carries
      pre-built ``List`` nodes as outputs – no real AFM calculation needed.

    * ``aiida_propose_occ_matrices_from_results`` is patched at the
      *import site* in the workflow module so that it returns a stored ``List``
      of dummy PKs instead of running any actual proposal logic.

    After calling the step we inspect ``ctx`` directly to verify that all
    counters, lists, and generation-0 metadata were initialised correctly.
    """

    def _make_process(self, generate_workchain, generate_structure,
                      generate_kpoints_mesh, fixture_code, pseudo_family,
                      Nmax=10, N=3):
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        pw_code = fixture_code('quantumespresso.pw')
        constrained_code = fixture_code('lordcapulet.constrained_pw')
        return generate_workchain(
            GlobalConstrainedSearchWorkChain,
            _global_inputs(structure, kpoints, pw_code, constrained_code, Nmax=Nmax, N=N),
        )

    def test_initialises_n_cumulative_to_zero(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """process_afm_results must set ctx.N_cumulative = 0."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        process.ctx.afm_wc = _mock_afm_wc([1, 2], [10, 11])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([101, 102, 103])), \
             patch.object(process, 'report'):
            process.process_afm_results()

        assert process.ctx.N_cumulative == 0

    def test_initialises_generation_to_zero(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """process_afm_results must set ctx.generation = 0."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        process.ctx.afm_wc = _mock_afm_wc([1, 2], [10, 11])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([101, 102, 103])), \
             patch.object(process, 'report'):
            process.process_afm_results()

        assert process.ctx.generation == 0

    def test_stores_generation_0_results(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """process_afm_results must populate ctx.generation_results[0] with AFM data."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        process.ctx.afm_wc = _mock_afm_wc([1, 2], [10, 11])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([101, 102, 103])), \
             patch.object(process, 'report'):
            process.process_afm_results()

        assert 0 in process.ctx.generation_results
        gen0 = process.ctx.generation_results[0]
        assert gen0['type'] == 'afm'
        assert gen0['n_calculations'] == 2
        assert gen0['converged_matrix_pks'] == [1, 2]
        assert gen0['converged_calculation_pks'] == [10, 11]

    def test_sets_ctx_converged_matrix_pks_from_afm(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """ctx.converged_matrix_pks should be seeded with the AFM matrix PKs."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        process.ctx.afm_wc = _mock_afm_wc([1, 2], [10, 11])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([101, 102, 103])), \
             patch.object(process, 'report'):
            process.process_afm_results()

        assert process.ctx.converged_matrix_pks == [1, 2]

    def test_sets_ctx_all_calculation_pks_from_afm(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """ctx.all_calculation_pks should be seeded with the AFM calculation PKs."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        process.ctx.afm_wc = _mock_afm_wc([1, 2], [10, 11], all_calc_pks=[10, 11])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([101, 102, 103])), \
             patch.object(process, 'report'):
            process.process_afm_results()

        assert process.ctx.all_calculation_pks == [10, 11]

    def test_calls_proposal_function(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """process_afm_results must invoke aiida_propose_occ_matrices_from_results exactly once."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        process.ctx.afm_wc = _mock_afm_wc([1, 2], [10, 11])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([101, 102, 103])) as mock_propose, \
             patch.object(process, 'report'):
            process.process_afm_results()

        assert mock_propose.called
        assert mock_propose.call_count == 1

    def test_sets_current_proposals_from_proposal_function(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """ctx.current_proposals must equal the list returned by the proposal function."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        process.ctx.afm_wc = _mock_afm_wc([1, 2], [10, 11])
        proposals = [301, 302, 303]

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list(proposals)), \
             patch.object(process, 'report'):
            process.process_afm_results()

        assert process.ctx.current_proposals == proposals

    def test_returns_error_if_afm_failed(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """process_afm_results must return ERROR_AFM_SEARCH_FAILED when afm_wc fails."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        failed_wc = MagicMock()
        failed_wc.is_finished_ok = False
        failed_wc.exit_status = 1
        process.ctx.afm_wc = failed_wc

        with patch.object(process, 'report'):
            result = process.process_afm_results()

        assert result == process.exit_codes.ERROR_AFM_SEARCH_FAILED


class TestRunConstrainedBatch:
    """Test the ``run_constrained_batch`` step.

    Strategy
    --------
    ``run_constrained_batch`` increments the generation counter, slices
    ``ctx.current_proposals`` down to the remaining budget, builds a
    ``ConstrainedScanWorkChain`` builder, and calls ``self.submit``.

    We inject ``ctx`` state directly (generation, N_cumulative, proposals)
    and patch ``submit`` to capture the builder without launching anything.
    The key assertions are:

    * Generation counter incremented.
    * ``occupation_matrices_list`` = first N (or remaining budget) proposals.
    * ``submit`` called exactly once per batch.
    """

    def _make_process(self, generate_workchain, generate_structure,
                      generate_kpoints_mesh, fixture_code, pseudo_family,
                      Nmax=10, N=3):
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        pw_code = fixture_code('quantumespresso.pw')
        constrained_code = fixture_code('lordcapulet.constrained_pw')
        return generate_workchain(
            GlobalConstrainedSearchWorkChain,
            _global_inputs(structure, kpoints, pw_code, constrained_code, Nmax=Nmax, N=N),
        )

    def _init_ctx(self, process, generation=0, n_cumulative=0, proposals=None):
        """Inject the minimal ctx state that run_constrained_batch reads.

        ``generation``   -- the step starts by incrementing this.
        ``n_cumulative`` -- controls the remaining budget = Nmax - n_cumulative,
                            which determines how many proposals are actually used.
        ``current_proposals`` -- list of PK integers that the step slices to
                                  feed the ConstrainedScanWorkChain.
        """
        process.ctx.generation = generation
        process.ctx.N_cumulative = n_cumulative
        process.ctx.current_proposals = proposals or list(range(101, 110))  # 9 fake PKs

    def test_increments_generation_counter(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """run_constrained_batch must increment ctx.generation by 1."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process, generation=0)

        with patch.object(process, 'submit', return_value=MagicMock()), \
             patch.object(process, 'report'):
            process.run_constrained_batch()

        assert process.ctx.generation == 1

    def test_increments_from_nonzero_generation(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """run_constrained_batch increments generation regardless of starting value."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process, generation=2)

        with patch.object(process, 'submit', return_value=MagicMock()), \
             patch.object(process, 'report'):
            process.run_constrained_batch()

        assert process.ctx.generation == 3

    def test_submit_called_exactly_once(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """run_constrained_batch must call submit exactly once (one ConstrainedScan per batch)."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process)

        with patch.object(process, 'submit', return_value=MagicMock()) as mock_sub, \
             patch.object(process, 'report'):
            process.run_constrained_batch()

        mock_sub.assert_called_once()

    def test_proposals_sliced_to_n(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """occupation_matrices_list passed to ConstrainedScan must be capped at N proposals."""
        # N=3, Nmax=10, N_cumulative=0 → n_proposals = min(3, 10-0) = 3
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family,
                                     Nmax=10, N=3)
        proposals = list(range(201, 210))  # 9 proposals, only 3 should be used
        self._init_ctx(process, n_cumulative=0, proposals=proposals)

        captured = {}

        def _capture(builder):
            captured['occ_list'] = builder.occupation_matrices_list.get_list()
            return MagicMock()

        with patch.object(process, 'submit', side_effect=_capture), \
             patch.object(process, 'report'):
            process.run_constrained_batch()

        assert len(captured['occ_list']) == 3
        assert captured['occ_list'] == proposals[:3]

    def test_proposals_capped_when_near_nmax(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """When remaining budget < N, proposals must be capped at remaining budget."""
        # N=3, Nmax=10, N_cumulative=8 → n_proposals = min(3, 10-8) = 2
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family,
                                     Nmax=10, N=3)
        proposals = list(range(201, 210))
        self._init_ctx(process, n_cumulative=8, proposals=proposals)

        captured = {}

        def _capture(builder):
            captured['occ_list'] = builder.occupation_matrices_list.get_list()
            return MagicMock()

        with patch.object(process, 'submit', side_effect=_capture), \
             patch.object(process, 'report'):
            process.run_constrained_batch()

        assert len(captured['occ_list']) == 2
        assert captured['occ_list'] == proposals[:2]


class TestProcessConstrainedResults:
    """Test the ``process_constrained_results`` step.

    Strategy
    --------
    ``process_constrained_results`` reads outputs from
    ``ctx.constrained_wc``, updates several ``ctx`` lists, and—if the
    cumulative count is below ``Nmax``—calls
    ``aiida_propose_occ_matrices_from_results`` to generate the next batch.

    We set ``ctx.constrained_wc`` to a ``_mock_constrained_wc`` stub and
    patch the proposal function.  The tests then check:

    * ``ctx.generation_results`` is populated with the correct metadata.
    * ``ctx.converged_matrix_pks`` and ``ctx.all_calculation_pks`` are
      extended (not replaced) with the new PKs.
    * The proposal function is called IFF ``N_cumulative + n_total < Nmax``.
    * ``ctx.current_proposals`` is updated from the proposal-function return.
    """

    def _make_process(self, generate_workchain, generate_structure,
                      generate_kpoints_mesh, fixture_code, pseudo_family,
                      Nmax=10, N=3):
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        pw_code = fixture_code('quantumespresso.pw')
        constrained_code = fixture_code('lordcapulet.constrained_pw')
        return generate_workchain(
            GlobalConstrainedSearchWorkChain,
            _global_inputs(structure, kpoints, pw_code, constrained_code, Nmax=Nmax, N=N),
        )

    def _init_ctx(self, process, *, generation=1, n_cumulative=0,
                  converged_matrix_pks=None, converged_calc_pks=None,
                  all_calc_pks=None):
        """Inject the ctx state that process_constrained_results reads and extends.

        We seed each cumulative list with at least one pre-existing entry (from
        the fake generation 0 AFM step) so that tests verifying *extension*
        (not replacement) of the lists can check the original entry is still
        present alongside the new ones added by the constrained step.
        """
        process.ctx.generation = generation
        # N_cumulative is the number of calculations run *before* this step;
        # the proposal-function call guard is: N_cumulative + n_new < Nmax
        process.ctx.N_cumulative = n_cumulative
        # Cumulative lists from all previous generations (AFM + constrained)
        process.ctx.converged_matrix_pks = list(converged_matrix_pks or [1])
        process.ctx.converged_calculation_pks = list(converged_calc_pks or [10])
        process.ctx.all_calculation_pks = list(all_calc_pks or [10])
        # generation_results[0] represents the AFM generation (type='afm')
        process.ctx.generation_results = {
            0: {'type': 'afm', 'n_calculations': 1,
                'converged_matrix_pks': [1], 'converged_calculation_pks': [10]}
        }

    def test_stores_generation_results(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """process_constrained_results must store results under ctx.generation_results[generation]."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process, generation=1, n_cumulative=0)
        process.ctx.constrained_wc = _mock_constrained_wc([201, 202], [301, 302])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([401, 402, 403])), \
             patch.object(process, 'report'):
            process.process_constrained_results()

        assert 1 in process.ctx.generation_results
        gen1 = process.ctx.generation_results[1]
        assert gen1['type'] == 'constrained'
        assert gen1['n_calculations'] == 2
        assert gen1['matrix_pks'] == [201, 202]

    def test_extends_converged_matrix_pks(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """ctx.converged_matrix_pks must grow by the new converged matrices."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process, generation=1, n_cumulative=0, converged_matrix_pks=[1])
        process.ctx.constrained_wc = _mock_constrained_wc([201, 202], [301, 302])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([401, 402, 403])), \
             patch.object(process, 'report'):
            process.process_constrained_results()

        assert 201 in process.ctx.converged_matrix_pks
        assert 202 in process.ctx.converged_matrix_pks
        assert 1 in process.ctx.converged_matrix_pks  # original AFM entry retained

    def test_extends_all_calculation_pks(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """ctx.all_calculation_pks must include the new calculation PKs."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process, generation=1, n_cumulative=0,
                       all_calc_pks=[10])
        process.ctx.constrained_wc = _mock_constrained_wc([201, 202], [301, 302])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([401, 402, 403])), \
             patch.object(process, 'report'):
            process.process_constrained_results()

        assert 301 in process.ctx.all_calculation_pks
        assert 302 in process.ctx.all_calculation_pks
        assert 10 in process.ctx.all_calculation_pks  # original retained

    def test_proposal_function_called_when_below_nmax(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """Proposal function must be called when N_cumulative + n_calc_total < Nmax."""
        # N_cumulative=0, n_total=2, Nmax=10 → 0+2=2 < 10 → proposal called
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family,
                                     Nmax=10)
        self._init_ctx(process, generation=1, n_cumulative=0)
        process.ctx.constrained_wc = _mock_constrained_wc([201, 202], [301, 302])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([401, 402, 403])) as mock_propose, \
             patch.object(process, 'report'):
            process.process_constrained_results()

        assert mock_propose.called

    def test_proposal_function_not_called_when_at_nmax(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """Proposal function must NOT be called when N_cumulative + n_calc_total >= Nmax."""
        # N_cumulative=8, n_total=2, Nmax=10 → 8+2=10 NOT < 10 → proposal skipped
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family,
                                     Nmax=10)
        self._init_ctx(process, generation=1, n_cumulative=8)
        process.ctx.constrained_wc = _mock_constrained_wc([201, 202], [301, 302])

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list([401, 402, 403])) as mock_propose, \
             patch.object(process, 'report'):
            process.process_constrained_results()

        assert not mock_propose.called

    def test_updates_current_proposals_from_proposal_function(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """ctx.current_proposals must be updated from the proposal function return value."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process, generation=1, n_cumulative=0)
        process.ctx.constrained_wc = _mock_constrained_wc([201, 202], [301, 302])
        new_proposals = [401, 402, 403]

        with patch('lordcapulet.workflows.global_constrained_search.aiida_propose_occ_matrices_from_results',
                   return_value=_stored_list(new_proposals)), \
             patch.object(process, 'report'):
            process.process_constrained_results()

        assert process.ctx.current_proposals == new_proposals


class TestGatherFinalResults:
    """Test the ``gather_final_results`` step.

    Strategy
    --------
    ``gather_final_results`` does not interact with the scheduler or any
    sub-workchains.  It only reads from ``ctx`` and creates / stores new
    ``List`` and ``Dict`` nodes, then calls ``self.out`` to register them.

    We initialise ``ctx`` with pre-built lists of PKs (using plain integers
    – no real nodes needed, the step just stores them in new List nodes) and
    patch ``self.out`` with ``_capture_out_factory`` to intercept the four
    emitted outputs.
    """

    def _make_process(self, generate_workchain, generate_structure,
                      generate_kpoints_mesh, fixture_code, pseudo_family):
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        pw_code = fixture_code('quantumespresso.pw')
        constrained_code = fixture_code('lordcapulet.constrained_pw')
        return generate_workchain(
            GlobalConstrainedSearchWorkChain,
            _global_inputs(structure, kpoints, pw_code, constrained_code),
        )

    def _init_ctx(self, process):
        process.ctx.N_cumulative = 5
        process.ctx.converged_matrix_pks = [1, 2, 3]
        process.ctx.converged_calculation_pks = [10, 11, 12]
        process.ctx.all_calculation_pks = [10, 11, 12]
        process.ctx.generation_results = {
            0: {'type': 'afm', 'n_calculations': 2,
                'converged_matrix_pks': [1, 2], 'converged_calculation_pks': [10, 11]},
            1: {'type': 'constrained', 'n_calculations': 3, 'n_successful': 1,
                'n_failed': 2, 'matrix_pks': [3], 'calculation_pks': [10, 11, 12],
                'converged_calculation_pks': [12]},
        }

    def test_all_four_output_keys_emitted(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """gather_final_results must emit all four outputs: converged_matrix_pks,
        converged_calculation_pks, all_calculation_pks, generation_summary."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process)

        outputs, capture = _capture_out_factory()
        with patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_final_results()

        assert 'converged_matrix_pks' in outputs
        assert 'converged_calculation_pks' in outputs
        assert 'all_calculation_pks' in outputs
        assert 'generation_summary' in outputs

    def test_converged_matrix_pks_correct_values(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """converged_matrix_pks output must contain the PKs accumulated in ctx."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process)

        outputs, capture = _capture_out_factory()
        with patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_final_results()

        assert outputs['converged_matrix_pks'].get_list() == [1, 2, 3]

    def test_converged_calculation_pks_correct_values(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """converged_calculation_pks output must contain the accumulated converged PKs."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process)

        outputs, capture = _capture_out_factory()
        with patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_final_results()

        assert outputs['converged_calculation_pks'].get_list() == [10, 11, 12]

    def test_all_calculation_pks_correct_values(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """all_calculation_pks output must contain the full calculation history."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process)

        outputs, capture = _capture_out_factory()
        with patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_final_results()

        assert outputs['all_calculation_pks'].get_list() == [10, 11, 12]

    def test_generation_summary_contains_all_generations(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """generation_summary must include an entry for every generation (0 and 1)."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process)

        outputs, capture = _capture_out_factory()
        with patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_final_results()

        summary = outputs['generation_summary'].get_dict()
        assert 'Generation 0' in summary
        assert 'Generation 1' in summary

    def test_all_output_nodes_are_stored(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """Every output node emitted by gather_final_results must be stored (have a pk)."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        self._init_ctx(process)

        outputs, capture = _capture_out_factory()
        with patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_final_results()

        for key, node in outputs.items():
            assert node.pk is not None, f"Output '{key}' node was not stored"
