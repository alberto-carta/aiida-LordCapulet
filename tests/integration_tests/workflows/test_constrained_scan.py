"""Tests for the ``ConstrainedScanWorkChain`` class."""

import pytest
from aiida.orm import Dict, Float, List, Code, KpointsData, JsonableData

import numpy as np


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
