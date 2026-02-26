"""Tests for the ``GlobalConstrainedSearchWorkChain`` class."""

import pytest
from aiida.orm import Dict, Float, Int, Bool, List, Str

from lordcapulet.workflows.global_constrained_search import GlobalConstrainedSearchWorkChain


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
