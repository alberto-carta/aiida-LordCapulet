"""Tests for the ``ConstrainedScanWorkChain`` class."""

import pytest
from unittest.mock import patch, MagicMock
import copy
import numpy as np
from aiida.orm import Dict, Float, List, Code, KpointsData, JsonableData


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _occ_node():
    """Create, store and return a JsonableData occupation matrix node."""
    from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
    occ_data = {
        'atom1': {
            'specie': 'Fe', 'shell': '3d',
            'occupation_matrix': {
                'up': np.eye(5).tolist(),
                'down': np.eye(5).tolist(),
            },
        }
    }
    node = JsonableData(OccupationMatrixData(copy.deepcopy(occ_data)))
    del node._obj  # force from_dict reconstruction on next access
    node.store()
    return node


def _constrained_inputs(structure, kpoints, code, pks):
    """Minimal valid ConstrainedScanWorkChain inputs dict."""
    return {
        'structure': structure,
        'parameters': Dict({'CONTROL': {'calculation': 'scf'},
                            'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2}}),
        'kpoints': kpoints,
        'code': code,
        'tm_atoms': List(list=['Fe']),
        'oscdft_card': Dict({'nconstr': 1}),
        'occupation_matrices_list': List(list=pks),
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
    store = {}

    def _capture(key, value):
        store[key] = value

    return store, _capture


class TestConstrainedScanDefine:
    """Test the process spec definition."""

    def test_inputs_exist(self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family):
        """Verify all expected inputs are defined, including oscdft_card and occupation_matrices_list."""
        from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData

        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('lordcapulet.constrained_pw')

        # Create a dummy occupation matrix and store as JsonableData
        occ_data = {
            'atom1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': np.diag([1.0, 0.8, 0.5, 0.2, 0.0]).tolist(),
                    'down': np.diag([0.9, 0.7, 0.4, 0.1, 0.0]).tolist(),
                },
            }
        }
        occ_matrix_data = OccupationMatrixData(occ_data)
        occ_node = JsonableData(occ_matrix_data)
        occ_node.store()

        inputs = {
            'structure': structure,
            'parameters': Dict({
                'CONTROL': {'calculation': 'scf'},
                'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2},
            }),
            'kpoints': kpoints,
            'code': code,
            'tm_atoms': List(list=['Fe']),
            'oscdft_card': Dict({'nconstr': 1}),
            'occupation_matrices_list': List(list=[occ_node.pk]),
        }

        process = generate_workchain('lordcapulet.constrained_scan', inputs)
        assert process is not None

    def test_outputs_defined(self):
        """Verify all expected outputs are defined in the spec."""
        from lordcapulet.workflows.constrained_scan import ConstrainedScanWorkChain

        spec = ConstrainedScanWorkChain.spec()

        assert 'converged_matrix_pks' in spec.outputs
        assert 'converged_calculation_pks' in spec.outputs
        assert 'all_calculation_pks' in spec.outputs

    def test_spec_has_oscdft_card(self):
        """Verify oscdft_card input is in the spec."""
        from lordcapulet.workflows.constrained_scan import ConstrainedScanWorkChain

        spec = ConstrainedScanWorkChain.spec()
        assert 'oscdft_card' in spec.inputs

    def test_spec_has_occupation_matrices_list(self):
        """Verify occupation_matrices_list input is in the spec."""
        from lordcapulet.workflows.constrained_scan import ConstrainedScanWorkChain

        spec = ConstrainedScanWorkChain.spec()
        assert 'occupation_matrices_list' in spec.inputs


class TestConstrainedScanPrepareCalculations:
    """Test the ``prepare_calculations`` step."""

    def test_sets_correct_count(self, generate_workchain, generate_structure, generate_kpoints_mesh, fixture_code, pseudo_family):
        """prepare_calculations should set n_calculations equal to the number of target matrices."""
        from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData

        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('lordcapulet.constrained_pw')

        # Create 3 dummy occupation matrix nodes
        pks = []
        for _ in range(3):
            occ_data = {
                'atom1': {
                    'specie': 'Fe',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.diag([1.0, 0.8, 0.5, 0.2, 0.0]).tolist(),
                        'down': np.diag([0.9, 0.7, 0.4, 0.1, 0.0]).tolist(),
                    },
                }
            }
            occ_node = JsonableData(OccupationMatrixData(occ_data))
            occ_node.store()
            pks.append(occ_node.pk)

        inputs = {
            'structure': structure,
            'parameters': Dict({
                'CONTROL': {'calculation': 'scf'},
                'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240, 'nspin': 2},
            }),
            'kpoints': kpoints,
            'code': code,
            'tm_atoms': List(list=['Fe']),
            'oscdft_card': Dict({'nconstr': 1}),
            'occupation_matrices_list': List(list=pks),
        }

        process = generate_workchain('lordcapulet.constrained_scan', inputs)
        process.prepare_calculations()

        assert process.ctx.n_calculations == 3
        assert len(process.ctx.target_matrices) == 3


class TestConstrainedScanRunAll:
    """Test the ``run_all`` step – submit count and builder configuration.

    Strategy
    --------
    ``run_all`` iterates over ``ctx.target_matrices`` (a list of PKs), loads
    each occupation-matrix node with ``load_node(pk)``, attaches it as
    ``builder.target_matrix``, and submits via::

        self.to_context(calcs=append_(self.submit(builder)))

    The same three patches as in ``TestAFMScanRunAll`` are required:

    * ``submit``     — replaced with a ``MagicMock`` / side-effect function
                       to capture builders without launching a real job.
    * ``append_``    — patched out because it rejects non-awaitable arguments
                       (i.e. ``MagicMock``) before ``to_context`` is called.
    * ``to_context`` — patched out to avoid event-loop interactions.
    * ``report``     — silenced to avoid node-logging errors.
    """

    def _make_process(self, generate_workchain, generate_structure,
                      generate_kpoints_mesh, fixture_code, pseudo_family, n=3):
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('lordcapulet.constrained_pw')
        pks = [_occ_node().pk for _ in range(n)]
        process = generate_workchain(
            'lordcapulet.constrained_scan',
            _constrained_inputs(structure, kpoints, code, pks),
        )
        process.prepare_calculations()
        return process

    def test_submit_called_once_per_target_matrix(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """run_all must call submit exactly n_calculations times (once per target matrix)."""
        process = self._make_process(
            generate_workchain, generate_structure, generate_kpoints_mesh,
            fixture_code, pseudo_family, n=3,
        )

        # submit returns a fake future; we only care about how many times it
        # was called, not what it returns.
        fake_future = MagicMock()
        with patch.object(process, 'submit', return_value=fake_future) as mock_submit, \
             patch('lordcapulet.workflows.constrained_scan.append_'), \
             patch.object(process, 'to_context'), \
             patch.object(process, 'report'):
            process.run_all()

        assert mock_submit.call_count == 3

    def test_each_builder_receives_correct_target_matrix(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family,
    ):
        """Each builder passed to submit must carry the correct target_matrix node."""
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('lordcapulet.constrained_pw')

        # Two occupation-matrix nodes stored in the ephemeral DB.  Their PKs
        # form the ``occupation_matrices_list`` input.  run_all loads each by
        # PK and attaches it to the builder.
        nodes = [_occ_node() for _ in range(2)]
        pks = [n.pk for n in nodes]

        process = generate_workchain(
            'lordcapulet.constrained_scan',
            _constrained_inputs(structure, kpoints, code, pks),
        )
        process.prepare_calculations()

        captured_pks = []

        def _capture(builder):
            # Record the PK of the target_matrix node attached to each builder.
            captured_pks.append(builder.target_matrix.pk)
            return MagicMock()

        with patch.object(process, 'submit', side_effect=_capture), \
             patch('lordcapulet.workflows.constrained_scan.append_'), \
             patch.object(process, 'to_context'), \
             patch.object(process, 'report'):
            process.run_all()

        assert len(captured_pks) == 2
        assert set(captured_pks) == set(pks)


class TestConstrainedScanGatherResults:
    """Test the ``gather_results`` step – output lists, node extras, and failure handling.

    Strategy
    --------
    Identical approach to ``TestAFMScanGatherResults``:

    * ``make_finished_calc_node`` creates fake but persisted ``CalcJobNode``\s
      with the desired ``exit_status`` so that ``load_node(calc.pk)`` and
      ``fresh_calc.is_finished`` / ``fresh_calc.exit_status`` work as expected.

    * ``patch('lordcapulet.workflows.constrained_scan.extract_occupations_from_calc')``
      replaces the QE output parser with a stub returning a minimal
      ``OccupationMatrixData`` – no real output files needed.

    * ``patch.object(process, 'out', side_effect=capture)`` captures every
      ``self.out('key', node)`` call into a plain dict so we can assert on
      the emitted output nodes.
    """

    def _make_process(self, generate_workchain, generate_structure,
                      generate_kpoints_mesh, fixture_code, pseudo_family):
        structure = generate_structure('feo')
        kpoints = generate_kpoints_mesh(4)
        code = fixture_code('lordcapulet.constrained_pw')
        pks = [_occ_node().pk for _ in range(2)]
        return generate_workchain(
            'lordcapulet.constrained_scan',
            _constrained_inputs(structure, kpoints, code, pks),
        )

    def test_all_three_output_keys_emitted(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """gather_results must emit all three output list keys."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc = make_finished_calc_node(exit_status=0)
        process.ctx.calcs = [calc]

        outputs, capture = _capture_out_factory()
        with patch('lordcapulet.workflows.constrained_scan.extract_occupations_from_calc',
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
        """Successful calculations appear in both all_ and converged_ output lists."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc1 = make_finished_calc_node(exit_status=0)
        calc2 = make_finished_calc_node(exit_status=0)
        process.ctx.calcs = [calc1, calc2]

        outputs, capture = _capture_out_factory()
        with patch('lordcapulet.workflows.constrained_scan.extract_occupations_from_calc',
                   return_value=_mock_occ()), \
             patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_results()

        all_pks = outputs['all_calculation_pks'].get_list()
        conv_pks = outputs['converged_calculation_pks'].get_list()
        matrix_pks = outputs['converged_matrix_pks'].get_list()

        assert len(all_pks) == 2
        assert len(conv_pks) == 2
        assert len(matrix_pks) == 2
        assert calc1.pk in conv_pks
        assert calc2.pk in conv_pks

    def test_failed_calc_excluded_from_converged_lists(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """Calculations with non-zero exit_status must not appear in converged lists."""
        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc_ok = make_finished_calc_node(exit_status=0)
        calc_fail = make_finished_calc_node(exit_status=3)
        process.ctx.calcs = [calc_ok, calc_fail]

        outputs, capture = _capture_out_factory()
        with patch('lordcapulet.workflows.constrained_scan.extract_occupations_from_calc',
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
        assert len(matrix_pks) == 1

    def test_occupation_matrix_pk_extra_set_on_successful_calc(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """Successful calc node should have 'occupation_matrix_pk' extra after gather_results."""
        from aiida.orm import load_node

        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc = make_finished_calc_node(exit_status=0)
        process.ctx.calcs = [calc]

        with patch('lordcapulet.workflows.constrained_scan.extract_occupations_from_calc',
                   return_value=_mock_occ()), \
             patch.object(process, 'out'), \
             patch.object(process, 'report'):
            process.gather_results()

        fresh = load_node(calc.pk)
        assert 'occupation_matrix_pk' in fresh.base.extras.all
        occ_pk = fresh.base.extras.get('occupation_matrix_pk')
        assert load_node(occ_pk) is not None

    def test_occupation_matrix_pk_extra_not_set_on_failed_calc(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """Failed calc nodes must NOT have 'occupation_matrix_pk' set as an extra."""
        from aiida.orm import load_node

        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc_fail = make_finished_calc_node(exit_status=2)
        process.ctx.calcs = [calc_fail]

        with patch('lordcapulet.workflows.constrained_scan.extract_occupations_from_calc',
                   return_value=_mock_occ()), \
             patch.object(process, 'out'), \
             patch.object(process, 'report'):
            process.gather_results()

        fresh = load_node(calc_fail.pk)
        assert 'occupation_matrix_pk' not in fresh.base.extras.all

    def test_stored_matrix_nodes_loadable(
        self, generate_workchain, generate_structure, generate_kpoints_mesh,
        fixture_code, pseudo_family, make_finished_calc_node,
    ):
        """Every PK in converged_matrix_pks must correspond to a loadable AiiDA node."""
        from aiida.orm import load_node

        process = self._make_process(generate_workchain, generate_structure,
                                     generate_kpoints_mesh, fixture_code, pseudo_family)
        calc = make_finished_calc_node(exit_status=0)
        process.ctx.calcs = [calc]

        outputs, capture = _capture_out_factory()
        with patch('lordcapulet.workflows.constrained_scan.extract_occupations_from_calc',
                   return_value=_mock_occ()), \
             patch.object(process, 'out', side_effect=capture), \
             patch.object(process, 'report'):
            process.gather_results()

        for pk in outputs['converged_matrix_pks'].get_list():
            assert load_node(pk) is not None
