"""Tests for the ``StandardMagneticScanWorkChain`` class."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from aiida.orm import Dict, Float, List, Code, KpointsData, Int


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _afm_inputs(structure, kpoints, code, hubbard_corr_atoms=None):
    """Minimal valid StandardMagneticScanWorkChain inputs dict."""
    return {
        'structure': structure,
        'parameters': Dict({'CONTROL': {'calculation': 'scf'},
                            'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
        'kpoints': kpoints,
        'code': code,
        'hubbard_corr_atoms': List(list=hubbard_corr_atoms or ['Fe']),
    }


def _mock_occ():
    """Return a minimal OccupationMatrixData for patching extract_occupations_from_calc."""
    from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
    return OccupationMatrixData({
        'atom1': {
            'specie': 'Fe', 'shell': '3d',
            'occupation_matrix': {
                'up': np.eye(5).tolist(),
                'down': np.eye(5).tolist(),
            },
        }
    })


def _capture_out_factory():
    """Return a (dict, side_effect) pair to intercept ``self.out`` calls."""
    store = {}

    def _capture(key, value):
        store[key] = value

    return store, _capture


class TestAFMScanDefine:
    """Test the process spec definition."""

    def test_inputs_exist(self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family):
        """Verify all expected inputs are defined in the spec."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')

        inputs = {
            'structure': structure,
            'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
            'kpoints': kpoints,
            'code': code,
            'hubbard_corr_atoms': List(list=['Fe']),
            'magnitude': Float(0.5),
            'walltime_hours': Float(1.0),
        }

        process = generate_workchain('lordcapulet.standard_magnetic_scan', inputs)

        # Check that the process was created
        assert process is not None

    def test_outputs_defined(self):
        """Verify all expected outputs are defined in the spec."""
        from lordcapulet.workflows.standard_magnetic_scan import StandardMagneticScanWorkChain

        spec = StandardMagneticScanWorkChain.spec()

        assert 'converged_matrix_pks' in spec.outputs
        assert 'converged_calculation_pks' in spec.outputs
        assert 'all_calculation_pks' in spec.outputs


class TestAFMScanPrepareConfigs:
    """Test the ``prepare_configs`` step."""

    def test_correct_count_single_atom(self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family):
        """With 1 TM atom, prepare_configs should generate 2^1 = 2 configurations."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')

        inputs = {
            'structure': structure,
            'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
            'kpoints': kpoints,
            'code': code,
            'hubbard_corr_atoms': List(list=['Fe']),
        }

        process = generate_workchain('lordcapulet.standard_magnetic_scan', inputs)
        process.prepare_configs()

        assert len(process.ctx.magnetic_configs) == 2

    def test_correct_count_two_atoms(self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family):
        """With 2 TM atoms, prepare_configs should generate 2^2 = 4 configurations."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')

        inputs = {
            'structure': structure,
            'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
            'kpoints': kpoints,
            'code': code,
            'hubbard_corr_atoms': List(list=['Fe1', 'Fe2']),
        }

        process = generate_workchain('lordcapulet.standard_magnetic_scan', inputs)
        process.prepare_configs()

        assert len(process.ctx.magnetic_configs) == 4

    def test_correct_count_three_atoms(self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family):
        """With 3 TM atoms, prepare_configs should generate 2^3 = 8 configurations."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')

        inputs = {
            'structure': structure,
            'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
            'kpoints': kpoints,
            'code': code,
            'hubbard_corr_atoms': List(list=['Fe1', 'Fe2', 'Ni1']),
        }

        process = generate_workchain('lordcapulet.standard_magnetic_scan', inputs)
        process.prepare_configs()

        assert len(process.ctx.magnetic_configs) == 8

    def test_max_configurations_caps_exhaustive_scan(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """max_configurations should cap the otherwise exhaustive 2^N scan."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')

        inputs = {
            'structure': structure,
            'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
            'kpoints': kpoints,
            'code': code,
            'hubbard_corr_atoms': List(list=['Fe1', 'Fe2', 'Fe3', 'Fe4']),
            'max_configurations': Int(3),
        }

        process = generate_workchain('lordcapulet.standard_magnetic_scan', inputs)
        with patch.object(process, 'report'):
            process.prepare_configs()

        assert len(process.ctx.magnetic_configs) == 3

    def test_sign_patterns(self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family):
        """Each configuration should have a unique sign pattern."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')

        inputs = {
            'structure': structure,
            'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
            'kpoints': kpoints,
            'code': code,
            'hubbard_corr_atoms': List(list=['Fe1', 'Fe2']),
            'magnitude': Float(0.5),
        }

        process = generate_workchain('lordcapulet.standard_magnetic_scan', inputs)
        process.prepare_configs()

        # Extract sign patterns (positive or negative for each atom)
        sign_patterns = set()
        for config in process.ctx.magnetic_configs:
            signs = tuple(1 if v > 0 else -1 for v in config.values())
            sign_patterns.add(signs)

        # All 4 patterns should be unique
        assert len(sign_patterns) == 4

    def test_magnitude_values(self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family):
        """All magnetization values should have the specified magnitude."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')
        magnitude = 0.7

        inputs = {
            'structure': structure,
            'parameters': Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
            'kpoints': kpoints,
            'code': code,
            'hubbard_corr_atoms': List(list=['Fe1', 'Fe2']),
            'magnitude': Float(magnitude),
        }

        process = generate_workchain('lordcapulet.standard_magnetic_scan', inputs)
        process.prepare_configs()

        for config in process.ctx.magnetic_configs:
            for value in config.values():
                assert abs(abs(value) - magnitude) < 1e-10


class TestAFMScanRunAll:
    """Test the ``run_all`` step - submit count and builder contents.

    Strategy
    --------
    ``run_all`` iterates over ``ctx.magnetic_configs`` and for each config
    calls::

        self.to_context(calcs=append_(self.submit(builder)))

    We cannot let ``self.submit`` actually run because that would try to talk
    to the AiiDA daemon and launch real DFT jobs.  Instead we replace it with
    a ``MagicMock`` and inspect what it was called with.

    Two extra patches are needed alongside ``submit``:

    * ``append_`` - AiiDA’s ``append_`` helper validates that its argument is
      an *awaitable* (a real process future).  If ``submit`` returns a
      ``MagicMock``, ``append_`` raises ``ValueError`` before
      ``to_context`` is reached.  Patching it out lets the call pass through.

    * ``to_context`` - the step returns ``ToContext(...)``.  When called
      directly (not via the engine) this method tries to interact with the
      running event loop, which does not exist in tests.  Patching it
      prevents that.

    * ``report`` - calls ``self.report(...)`` which logs to the AiiDA node;
      the node is not stored in tests, so we silence it.
    """

    def test_submit_called_once_per_config_single_atom(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """With 1 TM atom (2 configs) run_all should call submit exactly 2 times."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')

        # Instantiate the workchain in-process (no engine, no daemon).
        # ``prepare_configs`` populates ``ctx.magnetic_configs`` with 2^1 = 2
        # entries, which run_all must iterate over exactly once each.
        process = generate_workchain(
            'lordcapulet.standard_magnetic_scan',
            _afm_inputs(structure, kpoints, code, hubbard_corr_atoms=['Fe']),
        )
        process.prepare_configs()  # builds ctx.magnetic_configs (2 configs)

        fake_future = MagicMock()
        with patch.object(process, 'submit', return_value=fake_future) as mock_submit, \
             patch('lordcapulet.workflows.standard_magnetic_scan.append_'), \
             patch.object(process, 'to_context'), \
             patch.object(process, 'report'):  # see class docstring for why each patch is needed
            process.run_all()

        assert mock_submit.call_count == 2

    def test_submit_called_once_per_config_two_atoms(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """With 2 TM atoms (4 configs) run_all should call submit exactly 4 times."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')

        # 2 TM atoms gives 2^2 = 4 magnetic configurations, so run_all must
        # call submit 4 times (once per configuration).
        process = generate_workchain(
            'lordcapulet.standard_magnetic_scan',
            _afm_inputs(structure, kpoints, code, hubbard_corr_atoms=['Fe1', 'Fe2']),
        )
        process.prepare_configs()

        fake_future = MagicMock()
        with patch.object(process, 'submit', return_value=fake_future) as mock_submit, \
             patch('lordcapulet.workflows.standard_magnetic_scan.append_'), \
             patch.object(process, 'to_context'), \
             patch.object(process, 'report'):
            process.run_all()

        assert mock_submit.call_count == 4

    def test_builders_have_distinct_starting_magnetizations(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """Each builder submitted to run_all should carry a unique starting_magnetization."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')

        process = generate_workchain(
            'lordcapulet.standard_magnetic_scan',
            _afm_inputs(structure, kpoints, code, hubbard_corr_atoms=['Fe1', 'Fe2']),
        )
        process.prepare_configs()

        captured = []

        def _capture(builder):
            # Intercept each builder before it would be submitted.
            # Extract the starting_magnetization dict (keyed by atom label)
            # and normalise values to plain Python floats because they may be
            # stored as AiiDA Float nodes inside the Dict, which are
            # unhashable and cannot be placed in a set directly.
            raw = builder.parameters.get_dict()['SYSTEM']['starting_magnetization']
            mag = {k: float(v) for k, v in raw.items()}
            captured.append(tuple(sorted(mag.items())))
            return MagicMock()

        with patch.object(process, 'submit', side_effect=_capture), \
             patch('lordcapulet.workflows.standard_magnetic_scan.append_'), \
             patch.object(process, 'to_context'), \
             patch.object(process, 'report'):
            process.run_all()

        # All 4 magnetization patterns must be unique (no duplicates)
        assert len(set(captured)) == 4


class TestAFMScanGatherResults:
    """Test the ``gather_results`` step - output lists, extras, and failure handling.

    Strategy
    --------
    ``gather_results`` iterates over ``ctx.calcs``, calls
    ``load_node(calc.pk)`` on each one, checks the node’s ``is_finished``
    and ``exit_status``, then calls ``extract_occupations_from_calc`` to parse
    occupation matrices from QE output files.

    Two fixtures / patches make this testable without any real calculations:

    * ``make_finished_calc_node`` - creates a ``CalcJobNode`` that is stored
      in the ephemeral AiiDA DB (so ``load_node`` works) and has
      ``process_state='finished'`` / ``exit_status`` attributes set directly,
      so ``node.is_finished`` and ``node.exit_status`` return the expected
      values without the plumpy engine having run.

    * ``patch('lordcapulet.workflows.standard_magnetic_scan.extract_occupations_from_calc')``
      - replaces the parser that normally reads binary/text QE output files
      with a function returning a ready-made ``OccupationMatrixData`` object.
      This avoids the need for real QE output files in the test suite.

    * ``patch.object(process, 'out', side_effect=capture)`` - intercepts
      ``self.out('key', node)`` calls so we can inspect what was emitted
      without the workchain being registered in the graph.
    """

    def _make_process(self, generate_workchain, generate_structure,
                      generate_kpoints_mesh, fixture_code, pseudo_family):
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('quantumespresso.pw')
        return generate_workchain(
            'lordcapulet.standard_magnetic_scan',
            _afm_inputs(structure, kpoints, code),
        )

    def test_all_three_output_keys_emitted(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """gather_results must emit converged_matrix_pks, converged_calculation_pks, all_calculation_pks."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc = make_finished_calc_node(exit_status=0)
        process.ctx.calcs = [calc]

        outputs, capture = _capture_out_factory()
        with patch('lordcapulet.workflows.standard_magnetic_scan.extract_occupations_from_calc',
                   return_value=_mock_occ()), \
             patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_results()

        assert 'all_calculation_pks' in outputs
        assert 'converged_calculation_pks' in outputs
        assert 'converged_matrix_pks' in outputs

    def test_successful_calc_in_all_and_converged_lists(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """Calculations with exit_status=0 must appear in both all_ and converged_ lists."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc1 = make_finished_calc_node(exit_status=0)
        calc2 = make_finished_calc_node(exit_status=0)
        process.ctx.calcs = [calc1, calc2]

        outputs, capture = _capture_out_factory()
        with patch('lordcapulet.workflows.standard_magnetic_scan.extract_occupations_from_calc',
                   return_value=_mock_occ()), \
             patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_results()

        all_pks = outputs['all_calculation_pks'].get_list()
        conv_pks = outputs['converged_calculation_pks'].get_list()

        assert len(all_pks) == 2
        assert len(conv_pks) == 2
        assert calc1.pk in all_pks
        assert calc2.pk in all_pks
        assert calc1.pk in conv_pks
        assert calc2.pk in conv_pks

    def test_failed_calc_excluded_from_converged_lists(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """Calculations with non-zero exit_status must not appear in converged_ lists."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc_ok = make_finished_calc_node(exit_status=0)
        calc_fail = make_finished_calc_node(exit_status=2)
        process.ctx.calcs = [calc_ok, calc_fail]

        outputs, capture = _capture_out_factory()
        with patch('lordcapulet.workflows.standard_magnetic_scan.extract_occupations_from_calc',
                   return_value=_mock_occ()), \
             patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_results()

        all_pks = outputs['all_calculation_pks'].get_list()
        conv_pks = outputs['converged_calculation_pks'].get_list()
        matrix_pks = outputs['converged_matrix_pks'].get_list()

        assert len(all_pks) == 2
        assert len(conv_pks) == 1
        assert calc_ok.pk in conv_pks
        assert calc_fail.pk not in conv_pks
        # Only one matrix stored (for the successful calc)
        assert len(matrix_pks) == 1

    def test_occupation_matrix_pk_extra_set_on_successful_calc(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """After gather_results, the successful calc node must carry 'occupation_matrix_pk' extra."""
        from aiida.orm import load_node

        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc = make_finished_calc_node(exit_status=0)
        process.ctx.calcs = [calc]

        with patch('lordcapulet.workflows.standard_magnetic_scan.extract_occupations_from_calc',
                   return_value=_mock_occ()), \
             patch.object(process, 'out'), \
             patch.object(process, 'report'):
            process.gather_results()

        # Reload the node and verify the extra was persisted
        fresh = load_node(calc.pk)
        assert 'occupation_matrix_pk' in fresh.base.extras.all
        occ_pk = fresh.base.extras.get('occupation_matrix_pk')
        occ_node = load_node(occ_pk)
        assert occ_node is not None

    def test_occupation_matrix_pk_extra_not_set_on_failed_calc(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """Failed calculations must NOT have 'occupation_matrix_pk' set as an extra."""
        from aiida.orm import load_node

        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc_fail = make_finished_calc_node(exit_status=1)
        process.ctx.calcs = [calc_fail]

        with patch('lordcapulet.workflows.standard_magnetic_scan.extract_occupations_from_calc',
                   return_value=_mock_occ()), \
             patch.object(process, 'out'), \
             patch.object(process, 'report'):
            process.gather_results()

        fresh = load_node(calc_fail.pk)
        assert 'occupation_matrix_pk' not in fresh.base.extras.all

    def test_converged_matrix_pks_contains_stored_nodes(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """Each PK in converged_matrix_pks must refer to a real persisted AiiDA node."""
        from aiida.orm import load_node

        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc = make_finished_calc_node(exit_status=0)
        process.ctx.calcs = [calc]

        outputs, capture = _capture_out_factory()
        with patch('lordcapulet.workflows.standard_magnetic_scan.extract_occupations_from_calc',
                   return_value=_mock_occ()), \
             patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_results()

        for pk in outputs['converged_matrix_pks'].get_list():
            node = load_node(pk)
            assert node is not None
