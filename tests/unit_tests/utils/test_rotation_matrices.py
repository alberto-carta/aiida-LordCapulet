"""
Tests for rotation matrix utilities.

This module tests the spherical to cubic harmonic rotation matrices
used for converting occupation matrices between different basis sets.
"""

import numpy as np
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a dummy pytest for standalone execution
    class pytest:
        @staticmethod
        def raises(exception_type, match=None):
            class ContextManager:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {exception_type} to be raised")
                    return isinstance(exc_val, exception_type)
            return ContextManager()

from lordcapulet.utils.rotation_matrices import spherical_to_cubic_rotation, get_angular_momentum_operators


class TestSphericalToCubicRotation:
    """Test suite for spherical to cubic rotation matrices."""

    def test_function_exists(self):
        """Test that the function can be imported and called."""
        rot_matrix = spherical_to_cubic_rotation(dim=5, convention='qe')
        assert rot_matrix is not None
        assert rot_matrix.shape == (5, 5)

    def test_invalid_convention(self):
        """Test that invalid conventions raise ValueError."""
        with pytest.raises(ValueError, match="Only 'qe' convention is supported for now."):
            spherical_to_cubic_rotation(dim=5, convention='invalid')

    def test_invalid_dimension(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Only dimension 5 or 7 is supported for Hubbard orbitals."):
            spherical_to_cubic_rotation(dim=3, convention='qe')

    def test_matrix_properties(self):
        """Test that the rotation matrix has correct mathematical properties."""
        rot_matrix = spherical_to_cubic_rotation(dim=5, convention='qe')
        
        # Check matrix is unitary (T @ T† = I)
        identity = rot_matrix @ rot_matrix.T.conj()
        np.testing.assert_allclose(identity, np.eye(5), atol=1e-10)
        
        # Check matrix elements are as expected for QE convention
        sqrt2_inv = 1 / np.sqrt(2)
        expected_matrix = np.array([
                [0, 0, 1, 0, 0],                                     # r^2-3z^2 ~ Y_2^0
                [0, sqrt2_inv, 0, -sqrt2_inv, 0],                    # xz ~ 1/sqrt(2) * (Y_2^-1 - Y_2^1)
                [0, 1j * sqrt2_inv, 0, 1j * sqrt2_inv, 0],           # yz ~ i/sqrt(2) * (Y_2^-1 + Y_2^1)
                [1j * sqrt2_inv, 0, 0, 0, -1j * sqrt2_inv],          # xy ~ i/sqrt(2) * (Y_2^-2 - Y_2^2)
                [sqrt2_inv, 0, 0, 0, sqrt2_inv]                      # x^2-y^2 ~ 1/sqrt(2) * (Y_2^-2 + Y_2^2)
            ])

        np.testing.assert_allclose(rot_matrix, expected_matrix, atol=1e-10)

    def test_simple_density_matrix_rotation(self):
        """Test rotation of a simple density matrix with weight on z^2 orbital."""
        rot_matrix = spherical_to_cubic_rotation(dim=5, convention='qe')
        
        # Density matrix with weight only on Y_2^0 (z^2 in spherical basis)
        rho_spherical = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],  # Y_2^0 component
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # Rotate to cubic basis
        rho_cubic = rot_matrix @ rho_spherical @ rot_matrix.T.conj()
        
        # In cubic basis, this should give weight only on the first orbital (r^2-3z^2)
        expected_cubic = np.array([
            [1, 0, 0, 0, 0],  # r^2-3z^2 component
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        np.testing.assert_allclose(rho_cubic, expected_cubic, atol=1e-10)

    def test_complex_density_matrix_rotation(self):
        """Test rotation of a more complex density matrix."""
        rot_matrix = spherical_to_cubic_rotation(dim=5, convention='qe')
        
        # More complex diagonal density matrix in spherical basis
        rho_spherical = np.array([
            [1, 0, 0, 0, 0],    # Y_2^-2
            [0, 0.5, 0, 0, 0],  # Y_2^-1
            [0, 0, 1, 0, 0],    # Y_2^0
            [0, 0, 0, 0.5, 0],  # Y_2^1
            [0, 0, 0, 0, 1]     # Y_2^2
        ])
        
        # Rotate to cubic basis
        rho_cubic = rot_matrix @ rho_spherical @ rot_matrix.T.conj()
        
        # Check that the trace is preserved
        np.testing.assert_allclose(np.trace(rho_cubic), np.trace(rho_spherical), atol=1e-10)
        
        # Check that the matrix is still Hermitian
        np.testing.assert_allclose(rho_cubic, rho_cubic.T.conj(), atol=1e-10)

    def test_matrix_hermiticity_preservation(self):
        """Test that Hermitian matrices remain Hermitian after rotation."""
        rot_matrix = spherical_to_cubic_rotation(dim=5, convention='qe')
        
        # Create a random Hermitian matrix
        np.random.seed(42)  # For reproducible tests
        A = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
        rho_spherical = A + A.T.conj()  # Make it Hermitian
        
        # Rotate to cubic basis
        rho_cubic = rot_matrix @ rho_spherical @ rot_matrix.T.conj()
        
        # Check that it's still Hermitian
        np.testing.assert_allclose(rho_cubic, rho_cubic.T.conj(), atol=1e-10)


class TestAngularMomentumOperators:
    """Test suite for angular momentum operators."""

    def test_function_exists(self):
        """Test that the angular momentum function can be imported and called."""
        Lx, Ly, Lz, Lp, Lm = get_angular_momentum_operators(1)
        assert Lx is not None
        assert Ly is not None
        assert Lz is not None
        assert Lp is not None
        assert Lm is not None
        
        # Check dimensions for l=1 (3x3 matrices)
        assert Lx.shape == (3, 3)
        assert Ly.shape == (3, 3)
        assert Lz.shape == (3, 3)
        assert Lp.shape == (3, 3)
        assert Lm.shape == (3, 3)

    def test_invalid_l_values(self):
        """Test that invalid l values raise ValueError."""
        with pytest.raises(ValueError, match="l must be a non-negative integer"):
            get_angular_momentum_operators(-1)
        
        with pytest.raises(ValueError, match="l must be a non-negative integer"):
            get_angular_momentum_operators(0.3)  # Not integer or half-integer

    def commutator(self, A, B):
        """Helper function to compute commutator [A, B] = AB - BA."""
        return A @ B - B @ A

    def test_angular_momentum_l_half(self):
        """Test angular momentum operators for l=1/2."""
        Lx, Ly, Lz, Lp, Lm = get_angular_momentum_operators(0.5)
        
        # Check dimensions
        assert Lx.shape == (2, 2)
        assert Ly.shape == (2, 2)
        assert Lz.shape == (2, 2)
        
        # Check commutation relation [Lx, Ly] = i*Lz
        np.testing.assert_allclose(
            self.commutator(Lx, Ly), 1j * Lz, atol=1e-10
        )

    def test_angular_momentum_l_one(self):
        """Test angular momentum operators for l=1."""
        Lx, Ly, Lz, Lp, Lm = get_angular_momentum_operators(1)
        
        # Check dimensions
        assert Lx.shape == (3, 3)
        assert Ly.shape == (3, 3)
        assert Lz.shape == (3, 3)
        
        # Check Lz is diagonal with eigenvalues [-1, 0, 1]
        expected_Lz = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(Lz, expected_Lz, atol=1e-10)
        
        # Check commutation relation [Lx, Ly] = i*Lz
        np.testing.assert_allclose(
            self.commutator(Lx, Ly), 1j * Lz, atol=1e-10
        )

    def test_angular_momentum_l_two(self):
        """Test angular momentum operators for l=2."""
        Lx, Ly, Lz, Lp, Lm = get_angular_momentum_operators(2)
        
        # Check dimensions
        assert Lx.shape == (5, 5)
        assert Ly.shape == (5, 5)
        assert Lz.shape == (5, 5)
        
        # Check Lz is diagonal with eigenvalues [-2, -1, 0, 1, 2]
        expected_Lz = np.array([
            [-2, 0, 0, 0, 0],
            [0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 2]
        ])
        np.testing.assert_allclose(Lz, expected_Lz, atol=1e-10)
        
        # Check commutation relation [Lx, Ly] = i*Lz
        np.testing.assert_allclose(
            self.commutator(Lx, Ly), 1j * Lz, atol=1e-10
        )


if __name__ == "__main__":
    # Run basic tests when executed directly
    test_suite = TestSphericalToCubicRotation()
    angular_test_suite = TestAngularMomentumOperators()
    
    print("Running rotation matrix tests...")
    
    print("✓ Testing function existence...")
    test_suite.test_function_exists()
    
    print("✓ Testing invalid inputs...")
    test_suite.test_invalid_convention()
    test_suite.test_invalid_dimension()
    
    print("✓ Testing matrix properties...")
    test_suite.test_matrix_properties()
    
    print("✓ Testing simple density matrix rotation...")
    test_suite.test_simple_density_matrix_rotation()
    
    print("✓ Testing complex density matrix rotation...")
    test_suite.test_complex_density_matrix_rotation()
    
    print("✓ Testing Hermiticity preservation...")
    test_suite.test_matrix_hermiticity_preservation()
    
    print("\nRunning angular momentum operator tests...")
    
    print("✓ Testing angular momentum function existence...")
    angular_test_suite.test_function_exists()
    
    print("✓ Testing invalid l values...")
    angular_test_suite.test_invalid_l_values()
    
    print("✓ Testing l=1/2 operators...")
    angular_test_suite.test_angular_momentum_l_half()
    
    print("✓ Testing l=1 operators...")
    angular_test_suite.test_angular_momentum_l_one()
    
    print("✓ Testing l=2 operators...")
    angular_test_suite.test_angular_momentum_l_two()
    
    print("\nAll tests passed! ✅")
