"""Tests for SO(N) and O(N) matrix decomposition utilities."""

import numpy as np
import pytest
from lordcapulet.utils.so_n_decomposition import (
    get_so_n_lie_basis,
    euler_angles_to_rotation,
    rotation_to_euler_angles,
    canonicalize_angles
)


class TestSONDecomposition:
    """Test suite for SO(N) decomposition functions."""
    
    def test_lie_basis_dimension(self):
        """Test that the Lie basis has the correct dimension."""
        for norb in [3, 4, 5, 6]:
            generators = get_so_n_lie_basis(norb)
            expected_dim = norb * (norb - 1) // 2
            assert len(generators) == expected_dim
            
            # Check that all generators are antisymmetric
            for gen in generators:
                assert np.allclose(gen, -gen.T)
                assert gen.shape == (norb, norb)
    
    def test_so_n_round_trip(self):
        """Test round-trip: angles -> matrix -> angles for SO(N)."""
        np.random.seed(42)
        
        for norb in [3, 5, 7]:  # Test odd dimensions
            generators = get_so_n_lie_basis(norb)
            original_angles = np.random.uniform(-np.pi, np.pi, len(generators))
            
            # Forward: angles -> matrix
            R = euler_angles_to_rotation(original_angles, generators)
            
            # Check properties of SO(N) matrix
            assert np.allclose(R @ R.T, np.eye(norb))  # Orthogonal
            assert np.allclose(np.linalg.det(R), 1.0)  # det = 1
            
            # Backward: matrix -> angles
            extracted_angles, has_reflection = rotation_to_euler_angles(R, generators)
            
            assert not has_reflection  # Should be SO(N), not O(N)
            
            # Verify reconstruction (angles might differ but matrix should be the same)
            R_reconstructed = euler_angles_to_rotation(extracted_angles, generators)
            assert np.allclose(R, R_reconstructed, atol=1e-12)
            
            # Optional: check if angles are close (they might not be due to multiple representations)
            if not np.allclose(original_angles, extracted_angles, atol=1e-10):
                print(f"Note: Different but equivalent angle representation found for norb={norb}")
                print(f"Original: {original_angles[:3]}")
                print(f"Extracted: {extracted_angles[:3]}")
    
    def test_o_n_reflection_case(self):
        """Test O(N) matrices with det = -1 (reflection case)."""
        np.random.seed(42)
        
        for norb in [3, 5, 7]:  # Test odd dimensions only
            generators = get_so_n_lie_basis(norb)
            angles = np.random.uniform(-np.pi, np.pi, len(generators))
            
            # Create O(N) matrix with reflection
            R_reflection = euler_angles_to_rotation(angles, generators, reflection=True)
            
            # Check properties of O(N) matrix
            assert np.allclose(R_reflection @ R_reflection.T, np.eye(norb))  # Orthogonal
            assert np.allclose(np.linalg.det(R_reflection), -1.0)  # det = -1
            
            # Note: The new implementation raises ValueError for det=-1 matrices
            # This is intentional as decomposition of reflection matrices requires special handling
            with pytest.raises(ValueError, match="Matrix has det = -1"):
                rotation_to_euler_angles(R_reflection, generators)
            
            print(f"Reflection case correctly raises ValueError for norb={norb}")
    
    def test_even_dimension_reflection_error(self):
        """Test that reflection with even N raises appropriate error."""
        norb = 4  # Even dimension
        generators = get_so_n_lie_basis(norb)
        angles = np.random.uniform(-np.pi, np.pi, len(generators))
        
        with pytest.raises(ValueError, match="Reflection using -I only valid for odd N"):
            euler_angles_to_rotation(angles, generators, reflection=True)
    
    def test_invalid_matrix_errors(self):
        """Test error handling for invalid input matrices."""
        generators = get_so_n_lie_basis(3)
        
        # Non-orthogonal matrix
        bad_matrix = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="Matrix is not orthogonal"):
            rotation_to_euler_angles(bad_matrix, generators)
        
        # Matrix with invalid determinant
        bad_det_matrix = 2 * np.eye(3)  # det = 8
        with pytest.raises(ValueError, match="expected ±1 for orthogonal matrix"):
            rotation_to_euler_angles(bad_det_matrix, generators, check_orthogonal=False)
    
    def test_angle_generator_mismatch(self):
        """Test error when number of angles doesn't match generators."""
        generators = get_so_n_lie_basis(3)
        wrong_angles = np.array([1.0, 2.0])  # Should be 3 angles for SO(3)
        
        with pytest.raises(ValueError, match="Number of Euler angles .* must match"):
            euler_angles_to_rotation(wrong_angles, generators)
    
    def test_quantum_espresso_example(self):
        """Test with realistic quantum espresso density matrix."""
        # Example d-orbital density matrix
        spin_up_density_matrix = np.array([
            [ 0.575,  0.054,  0.054, -0.0,   0.108],
            [ 0.054,  0.962,  0.013,  0.094, -0.013],
            [ 0.054,  0.013,  0.962, -0.094, -0.013],
            [-0.0,    0.094, -0.094,  0.575, -0.0],
            [ 0.108, -0.013, -0.013, -0.0,   0.962]
        ])
        
        # Diagonalize to get eigenvectors (orthogonal matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(spin_up_density_matrix)
        
        # Check determinant to see if we need special handling
        det_eigenvectors = np.linalg.det(eigenvectors)
        print(f"Eigenvector determinant: {det_eigenvectors:.6f}")
        
        generators = get_so_n_lie_basis(5)
        
        if det_eigenvectors < 0:
            # If det = -1, the function should raise ValueError
            with pytest.raises(ValueError, match="Matrix has det = -1"):
                rotation_to_euler_angles(eigenvectors, generators)
            print("QE example correctly raises ValueError for det=-1 case")
        else:
            # If det = +1, decomposition should work
            angles, need_regularization = rotation_to_euler_angles(eigenvectors, generators)
            
            # Verify reconstruction
            eigenvectors_reconstructed = euler_angles_to_rotation(angles, generators)
            assert np.allclose(eigenvectors, eigenvectors_reconstructed, atol=1e-12)
            
            # Verify full density matrix reconstruction
            density_reconstructed = (
                eigenvectors_reconstructed @ np.diag(eigenvalues) @ eigenvectors_reconstructed.T
            )
            assert np.allclose(spin_up_density_matrix, density_reconstructed, atol=1e-12)
    
    def test_basis_orthogonality(self):
        """Test that basis matrices have expected properties."""
        generators = get_so_n_lie_basis(4)
        
        for i, gen in enumerate(generators):
            # Each generator should be antisymmetric
            assert np.allclose(gen, -gen.T)
            
            # Each generator should have norm^2 = 2 (Frobenius norm)
            norm_squared = np.sum(gen * gen)
            assert np.allclose(norm_squared, 2.0)
            
            # Generators should be linearly independent (checked via rank)
        
        # Stack all generators and check rank
        gen_matrix = np.array([gen.flatten() for gen in generators])
        assert np.linalg.matrix_rank(gen_matrix) == len(generators)
    
    def test_canonicalize_angles(self):
        """Test angle canonicalization functionality."""
        np.random.seed(42)
        
        for norb in [3, 4, 5]:
            generators = get_so_n_lie_basis(norb)
            
            # Test with random angles
            original_angles = np.random.uniform(-np.pi, np.pi, len(generators))
            
            # Canonicalize the angles
            canonical_angles = canonicalize_angles(original_angles, generators)
            
            # Both should produce the same rotation matrix
            R_original = euler_angles_to_rotation(original_angles, generators)
            R_canonical = euler_angles_to_rotation(canonical_angles, generators)
            assert np.allclose(R_original, R_canonical, atol=1e-12)
            
            # Canonical angles should match what rotation_to_euler_angles returns
            extracted_angles, _ = rotation_to_euler_angles(R_original, generators)
            assert np.allclose(canonical_angles, extracted_angles, atol=1e-12)
    
    def test_canonicalize_with_long_path(self):
        """Test canonicalization with angles that represent a 'long path' rotation."""
        generators = get_so_n_lie_basis(3)  # SO(3) for simplicity
        
        # Create a case where we know the original and canonical will differ
        # Using the seed from the example that produces different representations
        np.random.seed(42)
        long_path_angles = np.random.uniform(-np.pi, np.pi, len(generators))
        
        # Get canonical representation
        canonical_angles = canonicalize_angles(long_path_angles, generators)
        
        # They should produce identical matrices
        R_long = euler_angles_to_rotation(long_path_angles, generators)
        R_canonical = euler_angles_to_rotation(canonical_angles, generators)
        assert np.allclose(R_long, R_canonical, atol=1e-12)
        
        # The canonical angles should be what we get from matrix decomposition
        extracted_angles, _ = rotation_to_euler_angles(R_long, generators)
        assert np.allclose(canonical_angles, extracted_angles, atol=1e-12)
        
        # In this specific case, we expect the angles to be different
        # (this validates our test case actually demonstrates the issue)
        angle_differences = np.abs(long_path_angles - canonical_angles)
        assert np.any(angle_differences > 1e-10), "Test case should show different angle representations"
    
    def test_canonicalize_idempotent(self):
        """Test that canonicalizing already canonical angles is idempotent."""
        np.random.seed(123)
        
        for norb in [3, 4, 5]:
            generators = get_so_n_lie_basis(norb)
            
            # Start with some random angles and canonicalize them
            angles = np.random.uniform(-np.pi, np.pi, len(generators))
            canonical_once = canonicalize_angles(angles, generators)
            
            # Canonicalizing again should give the same result
            canonical_twice = canonicalize_angles(canonical_once, generators)
            assert np.allclose(canonical_once, canonical_twice, atol=1e-12)
            
            # Also test with extracted angles (which are already canonical)
            R = euler_angles_to_rotation(angles, generators) 
            extracted_angles, _ = rotation_to_euler_angles(R, generators)
            canonical_extracted = canonicalize_angles(extracted_angles, generators)
            assert np.allclose(extracted_angles, canonical_extracted, atol=1e-12)


if __name__ == "__main__":
    # Run basic tests if executed directly
    test_suite = TestSONDecomposition()
    
    print("Running SO(N) decomposition tests...")
    
    test_suite.test_lie_basis_dimension()
    print("Lie basis dimension test passed (^_^)")
    
    test_suite.test_so_n_round_trip()
    print("SO(N) round-trip test passed (^_^)")
    
    test_suite.test_o_n_reflection_case()
    print("O(N) reflection test passed (^_^)")
    
    test_suite.test_quantum_espresso_example()
    print("Quantum Espresso example test passed (^_^)")
    
    test_suite.test_canonicalize_angles()
    print("Angle canonicalization test passed (^_^)")
    
    test_suite.test_canonicalize_with_long_path()
    print("Long path canonicalization test passed (^_^)")
    
    test_suite.test_canonicalize_idempotent()
    print("Canonicalization idempotent test passed (^_^)")
    
    print("All tests passed!")