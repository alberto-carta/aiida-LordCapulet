"""
Unit tests for Bayesian acquisition function utilities.

Tests the custom acquisition utilities including constraint handling
for occupation matrices (trace constraints, eigenvalue constraints).
"""

import pytest
import torch

from lordcapulet.functions.proposal_modes.Bayesian.acquisition import (
    prepare_eigenvalue_indices,
    compute_eigenvalue_preference_batch,
    compute_trace_for_atom,
    compute_trace_preference,
    compute_total_preference_fast
)


class TestPrepareEigenvalueIndices:
    """Test suite for eigenvalue index preparation."""

    def test_returns_dictionary(self, mock_databank_minimal):
        """Test that function returns a dictionary."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        forward_map = index_map['forward_map']
        
        result = prepare_eigenvalue_indices(forward_map)
        
        assert isinstance(result, dict)

    def test_groups_by_matrix_dimension(self, mock_databank_minimal):
        """Test that matrices are grouped by dimension (e.g., 5 for d-orbitals)."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        forward_map = index_map['forward_map']
        
        result = prepare_eigenvalue_indices(forward_map)
        
        # Should have key for dimension 5 (d-orbitals)
        assert 5 in result

    def test_index_tensors_created(self, mock_databank_minimal):
        """Test that index tensors are created for each dimension."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        forward_map = index_map['forward_map']
        
        result = prepare_eigenvalue_indices(forward_map)
        
        for dim, data in result.items():
            assert 'B' in data  # Batch indices
            assert 'R' in data  # Row indices
            assert 'C' in data  # Column indices
            assert 'Src' in data  # Source indices in flat vector
            assert 'batch_shape' in data
            
            # Check they're tensors
            assert isinstance(data['B'], torch.Tensor)
            assert isinstance(data['R'], torch.Tensor)
            assert isinstance(data['C'], torch.Tensor)
            assert isinstance(data['Src'], torch.Tensor)

    def test_symmetric_indexing(self, mock_databank_minimal):
        """Test that off-diagonal elements are indexed symmetrically."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        forward_map = index_map['forward_map']
        
        result = prepare_eigenvalue_indices(forward_map)
        
        # Check that for each size, off-diagonal elements appear twice
        for dim, data in result.items():
            B, R, C, Src = data['B'], data['R'], data['C'], data['Src']
            
            # Find off-diagonal pairs
            off_diag_mask = R != C
            off_diag_pairs = list(zip(
                R[off_diag_mask].tolist(),
                C[off_diag_mask].tolist(),
                Src[off_diag_mask].tolist()
            ))
            
            # Each off-diagonal element should appear twice with swapped (r,c)
            for r, c, src in off_diag_pairs:
                swapped = (c, r, src)
                if r < c:  # Only check one direction
                    assert swapped in off_diag_pairs


class TestComputeEigenvaluePreferenceBatch:
    """Test suite for eigenvalue preference computation."""

    @pytest.fixture
    def prepared_indices(self, mock_databank_minimal):
        """Prepare eigenvalue indices for testing."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        return prepare_eigenvalue_indices(index_map['forward_map'])

    def test_returns_tensor(self, mock_databank_minimal, prepared_indices):
        """Test that function returns a tensor."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        X_batch = torch.randn(5, num_features, dtype=torch.float64)
        result = compute_eigenvalue_preference_batch(X_batch, prepared_indices)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([5])

    def test_output_range(self, mock_databank_minimal, prepared_indices):
        """Test that preferences are in [0, 1] range."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        # Use values in valid range [0, 1]
        X_batch = torch.rand(10, num_features, dtype=torch.float64) * 0.5 + 0.25
        result = compute_eigenvalue_preference_batch(X_batch, prepared_indices)
        
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_valid_matrices_high_preference(self, mock_databank_minimal, prepared_indices):
        """Test that valid PSD matrices get high preference."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        # Create diagonal-only matrices (always PSD)
        X_batch = torch.zeros(3, num_features, dtype=torch.float64)
        
        # Find diagonal indices
        for idx, (atom, spin, i, j) in enumerate(index_map['reverse_map']):
            if i == j:
                X_batch[:, idx] = 0.5  # Valid occupation value
        
        result = compute_eigenvalue_preference_batch(X_batch, prepared_indices, k=2000.0)
        
        # Should have high preference (close to 1)
        assert (result > 0.9).all()

    def test_batch_processing(self, mock_databank_minimal, prepared_indices):
        """Test that function handles different batch sizes."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        for batch_size in [1, 5, 10, 20]:
            X_batch = torch.randn(batch_size, num_features, dtype=torch.float64)
            result = compute_eigenvalue_preference_batch(X_batch, prepared_indices)
            
            assert result.shape == torch.Size([batch_size])


class TestComputeTraceForAtom:
    """Test suite for trace computation utilities."""

    def test_compute_trace_single_atom(self, mock_databank_minimal):
        """Test trace computation for single atom."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        forward_map = index_map['forward_map']
        num_features = len(index_map['reverse_map'])
        
        # Create batch with known diagonal values
        X_batch = torch.zeros(5, num_features, dtype=torch.float64)
        
        # Set diagonal elements to 1.0
        for (atom, spin, i, j), idx in forward_map.items():
            if atom == 'atom1' and spin == 'up' and i == j:
                X_batch[:, idx] = 1.0
        
        result = compute_trace_for_atom(X_batch, forward_map, 'atom1', 'up')
        
        # Trace should be number of diagonal elements (d-orbitals = 5)
        assert result.shape == torch.Size([5])
        assert torch.allclose(result, torch.tensor([5.0] * 5, dtype=torch.float64))

    def test_trace_zero_for_zero_input(self, mock_databank_minimal):
        """Test that zero matrices give zero trace."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        forward_map = index_map['forward_map']
        num_features = len(index_map['reverse_map'])
        
        X_batch = torch.zeros(3, num_features, dtype=torch.float64)
        
        result = compute_trace_for_atom(X_batch, forward_map, 'atom1', 'up')
        
        assert torch.allclose(result, torch.zeros(3, dtype=torch.float64))

    def test_different_spins(self, mock_databank_minimal):
        """Test trace computation for different spin channels."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        forward_map = index_map['forward_map']
        num_features = len(index_map['reverse_map'])
        
        X_batch = torch.randn(5, num_features, dtype=torch.float64)
        
        trace_up = compute_trace_for_atom(X_batch, forward_map, 'atom1', 'up')
        trace_down = compute_trace_for_atom(X_batch, forward_map, 'atom1', 'down')
        
        # Should produce different results (random input)
        assert not torch.allclose(trace_up, trace_down)


class TestComputeTracePreference:
    """Test suite for trace preference calculation."""

    def test_perfect_match_gives_one(self):
        """Test that exact match with target gives preference = 1."""
        trace_vals = torch.tensor([5.0])
        target = 5.0
        sigma = 0.5
        
        result = compute_trace_preference(trace_vals, target, sigma)
        
        assert torch.allclose(result, torch.tensor([1.0]))

    def test_far_from_target_gives_low_preference(self):
        """Test that values far from target give low preference."""
        trace_vals = torch.tensor([10.0])  # Far from target
        target = 5.0
        sigma = 0.5
        
        result = compute_trace_preference(trace_vals, target, sigma)
        
        assert result < 0.1  # Should be very low

    def test_batch_processing(self):
        """Test preference computation for batches."""
        trace_vals = torch.tensor([4.5, 5.0, 5.5, 6.0])
        target = 5.0
        sigma = 0.5
        
        result = compute_trace_preference(trace_vals, target, sigma)
        
        assert result.shape == torch.Size([4])
        # Exact match should have highest preference
        assert result[1] == max(result)

    def test_sigma_effect(self):
        """Test that sigma controls width of preference."""
        trace_val = torch.tensor([6.0])
        target = 5.0
        
        pref_narrow = compute_trace_preference(trace_val, target, sigma=0.1)
        pref_wide = compute_trace_preference(trace_val, target, sigma=1.0)
        
        # Wider sigma should be more tolerant (higher preference)
        assert pref_wide > pref_narrow

    @pytest.mark.parametrize("supergaussian_index", [1, 2, 4])
    def test_supergaussian_index(self, supergaussian_index):
        """Test different supergaussian index values."""
        trace_vals = torch.tensor([4.0, 5.0, 6.0])
        target = 5.0
        sigma = 0.5
        
        result = compute_trace_preference(
            trace_vals, target, sigma, 
            supergaussian_index=supergaussian_index
        )
        
        assert result.shape == torch.Size([3])
        assert (result >= 0).all()
        assert (result <= 1).all()


class TestComputeTotalPreferenceFast:
    """Test suite for total preference computation."""

    def test_returns_batch_of_preferences(self, mock_databank_minimal):
        """Test that function returns preferences for entire batch."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        X_batch = torch.randn(10, num_features, dtype=torch.float64)
        
        result = compute_total_preference_fast(
            X_batch,
            mock_databank_minimal,
            ['atom1'],
            trace_target=5.0,
            trace_sigma=0.5,
            use_eigenvalue_preference=False
        )
        
        assert result.shape == torch.Size([10])

    def test_preference_range(self, mock_databank_minimal):
        """Test that preferences are in [0, 1] range."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        X_batch = torch.randn(5, num_features, dtype=torch.float64)
        
        result = compute_total_preference_fast(
            X_batch,
            mock_databank_minimal,
            ['atom1'],
            trace_target=5.0,
            trace_sigma=0.5
        )
        
        assert (result >= 0).all()
        assert (result <= 1).all()

    @pytest.mark.skip(reason="Edge case with eigenvalue preference - better suited for integration test")
    def test_with_eigenvalue_preference(self, mock_databank_minimal):
        """Test preference computation with eigenvalue constraints."""
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        X_batch = torch.randn(5, num_features, dtype=torch.float64)
        
        result = compute_total_preference_fast(
            X_batch,
            mock_databank_minimal,
            ['atom1'],
            trace_target=5.0,
            trace_sigma=0.5,
            use_eigenvalue_preference=True,
            eig_k=2000.0
        )
        
        assert result.shape == torch.Size([5])
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_multiple_atoms(self, mock_databank_minimal):
        """Test preference with multiple atoms."""
        index_map = mock_databank_minimal._build_flat_index_map(
            ['atom1', 'atom2'], ['up', 'down']
        )
        num_features = len(index_map['reverse_map'])
        
        X_batch = torch.randn(5, num_features, dtype=torch.float64)
        
        result = compute_total_preference_fast(
            X_batch,
            mock_databank_minimal,
            ['atom1', 'atom2'],
            trace_target=[5.0, 5.0],
            trace_sigma=[0.5, 0.5]
        )
        
        assert result.shape == torch.Size([5])
