"""Tests for shared proposal mode functionality."""

import numpy as np
import pytest

from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData


def _make_occ_data(dim=5, n_atoms=2, seed=None):
    """Helper to create a single OccupationMatrixData with random diagonal matrices."""
    if seed is not None:
        np.random.seed(seed)
    data = {}
    for i in range(n_atoms):
        label = f'atom{i + 1}'
        data[label] = {
            'specie': 'Fe',
            'shell': '3d',
            'occupation_matrix': {
                'up': np.diag(np.random.rand(dim)).tolist(),
                'down': np.diag(np.random.rand(dim)).tolist(),
            },
        }
    return OccupationMatrixData(data)


class TestCalculateAverageTraces:
    """Test ``calculate_average_traces``."""

    def test_single_matrix(self):
        """With a single matrix, average trace equals its own trace."""
        from lordcapulet.functions.proposal_modes.shared_functionality import calculate_average_traces

        occ = _make_occ_data(dim=5, n_atoms=2, seed=42)
        avg = calculate_average_traces([occ], natoms=2)

        # Verify each atom's average trace
        for i, label in enumerate(occ.get_atom_labels()):
            up = np.array(occ.get_occupation_matrix(label, 'up'))
            down = np.array(occ.get_occupation_matrix(label, 'down'))
            expected = np.trace(up) + np.trace(down)
            assert abs(avg[i] - expected) < 1e-10

    def test_multiple_matrices(self):
        """Average trace should be the mean across all matrices."""
        from lordcapulet.functions.proposal_modes.shared_functionality import calculate_average_traces

        occ1 = _make_occ_data(dim=3, n_atoms=1, seed=10)
        occ2 = _make_occ_data(dim=3, n_atoms=1, seed=20)
        occ3 = _make_occ_data(dim=3, n_atoms=1, seed=30)

        avg = calculate_average_traces([occ1, occ2, occ3], natoms=1)

        # Manual calculation
        total = 0
        for occ in [occ1, occ2, occ3]:
            label = occ.get_atom_labels()[0]
            up = np.array(occ.get_occupation_matrix(label, 'up'))
            down = np.array(occ.get_occupation_matrix(label, 'down'))
            total += np.trace(up) + np.trace(down)

        assert abs(avg[0] - total / 3) < 1e-10


class TestCreateRandomDiagonalMatrices:
    """Test ``create_random_diagonal_matrices``."""

    def test_correct_shape(self):
        """Output should have shape (2, dim, dim)."""
        from lordcapulet.functions.proposal_modes.shared_functionality import create_random_diagonal_matrices

        result = create_random_diagonal_matrices(dim=5, target_electrons=6)
        assert result.shape == (2, 5, 5)

    def test_diagonal_structure(self):
        """Off-diagonal elements should be zero."""
        from lordcapulet.functions.proposal_modes.shared_functionality import create_random_diagonal_matrices

        result = create_random_diagonal_matrices(dim=5, target_electrons=6)

        for spin in range(2):
            for i in range(5):
                for j in range(5):
                    if i != j:
                        assert result[spin, i, j] == 0

    def test_electron_count(self):
        """Total number of electrons (sum of all diagonals) should match target."""
        from lordcapulet.functions.proposal_modes.shared_functionality import create_random_diagonal_matrices

        for target in [1, 3, 5, 7, 10]:
            result = create_random_diagonal_matrices(dim=5, target_electrons=target)
            total = np.trace(result[0]).real + np.trace(result[1]).real
            assert int(round(total)) == target

    def test_values_binary(self):
        """Diagonal elements should be 0 or 1."""
        from lordcapulet.functions.proposal_modes.shared_functionality import create_random_diagonal_matrices

        result = create_random_diagonal_matrices(dim=5, target_electrons=6)

        for spin in range(2):
            diag = np.diag(result[spin]).real
            for val in diag:
                assert val in [0, 1]

    def test_max_electrons_capped(self):
        """Should not exceed 2*dim electrons."""
        from lordcapulet.functions.proposal_modes.shared_functionality import create_random_diagonal_matrices

        result = create_random_diagonal_matrices(dim=5, target_electrons=15)
        total = np.trace(result[0]).real + np.trace(result[1]).real
        assert int(round(total)) == 10  # capped at 2*5


class TestApplyRandomRotation:
    """Test ``apply_random_rotation``."""

    def test_output_shape_preserved(self):
        """Output should have the same shape as input."""
        from lordcapulet.functions.proposal_modes.shared_functionality import apply_random_rotation

        matrices = np.zeros((2, 5, 5), dtype=complex)
        matrices[0] = np.diag([1.0, 0.8, 0.5, 0.2, 0.0])
        matrices[1] = np.diag([0.9, 0.7, 0.4, 0.1, 0.0])

        result = apply_random_rotation(matrices)
        assert result.shape == matrices.shape

    def test_trace_preserved(self):
        """Rotation should preserve the trace of each spin channel."""
        from lordcapulet.functions.proposal_modes.shared_functionality import apply_random_rotation

        matrices = np.zeros((2, 5, 5), dtype=complex)
        matrices[0] = np.diag([1.0, 0.8, 0.5, 0.2, 0.0])
        matrices[1] = np.diag([0.9, 0.7, 0.4, 0.1, 0.0])

        result = apply_random_rotation(matrices)

        assert abs(np.trace(result[0]).real - np.trace(matrices[0]).real) < 1e-8
        assert abs(np.trace(result[1]).real - np.trace(matrices[1]).real) < 1e-8

    def test_hermiticity_preserved(self):
        """Rotation should preserve Hermiticity."""
        from lordcapulet.functions.proposal_modes.shared_functionality import apply_random_rotation

        matrices = np.zeros((2, 5, 5), dtype=complex)
        matrices[0] = np.diag([1.0, 0.8, 0.5, 0.2, 0.0])
        matrices[1] = np.diag([0.9, 0.7, 0.4, 0.1, 0.0])

        result = apply_random_rotation(matrices)

        for spin in range(2):
            diff = result[spin] - result[spin].conj().T
            assert np.allclose(diff, 0, atol=1e-10)
