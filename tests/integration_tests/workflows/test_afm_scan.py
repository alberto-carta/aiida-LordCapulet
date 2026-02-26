"""Tests for the ``AFMScanWorkChain`` class."""

import pytest
from aiida.orm import Dict, Float, List, Code, KpointsData


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
            'tm_atoms': List(list=['Fe']),
            'magnitude': Float(0.5),
            'walltime_hours': Float(1.0),
        }

        process = generate_workchain('lordcapulet.afm_scan', inputs)

        # Check that the process was created
        assert process is not None

    def test_outputs_defined(self):
        """Verify all expected outputs are defined in the spec."""
        from lordcapulet.workflows.afm_scan import AFMScanWorkChain

        spec = AFMScanWorkChain.spec()

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
            'tm_atoms': List(list=['Fe']),
        }

        process = generate_workchain('lordcapulet.afm_scan', inputs)
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
            'tm_atoms': List(list=['Fe1', 'Fe2']),
        }

        process = generate_workchain('lordcapulet.afm_scan', inputs)
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
            'tm_atoms': List(list=['Fe1', 'Fe2', 'Ni1']),
        }

        process = generate_workchain('lordcapulet.afm_scan', inputs)
        process.prepare_configs()

        assert len(process.ctx.magnetic_configs) == 8

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
            'tm_atoms': List(list=['Fe1', 'Fe2']),
            'magnitude': Float(0.5),
        }

        process = generate_workchain('lordcapulet.afm_scan', inputs)
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
            'tm_atoms': List(list=['Fe1', 'Fe2']),
            'magnitude': Float(magnitude),
        }

        process = generate_workchain('lordcapulet.afm_scan', inputs)
        process.prepare_configs()

        for config in process.ctx.magnetic_configs:
            for value in config.values():
                assert abs(abs(value) - magnitude) < 1e-10
