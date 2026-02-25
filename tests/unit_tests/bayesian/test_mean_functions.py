"""
Unit tests for Bayesian mean functions.

Tests the custom physics-based mean functions that encode
domain knowledge about magnetic systems.
"""

import pytest
import torch

from lordcapulet.functions.proposal_modes.Bayesian.mean_functions import (
    VectorizedPhysicsMean
)


class TestVectorizedPhysicsMean:
    """Test suite for VectorizedPhysicsMean class."""

    def test_initialization(self, mock_databank_minimal):
        """Test that mean function initializes correctly."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1'],
            J_prior_mean=0.5,
            J_prior_std=0.2,
            U_prior_mean=4.5,
            U_prior_std=1.0
        )
        
        assert mean_fn is not None
        assert mean_fn.atom_ids == ['atom1']
        
        # Check parameters are initialized
        assert hasattr(mean_fn, 'J')
        assert hasattr(mean_fn, 'U')
        assert hasattr(mean_fn, 'constant')

    def test_parameters_initialized_at_prior_means(self, mock_databank_minimal):
        """Test that parameters are initialized at prior mean values."""
        J_mean = 0.8
        U_mean = 5.0
        const_mean = -100.0
        
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1'],
            J_prior_mean=J_mean,
            U_prior_mean=U_mean,
            constant_mean=const_mean
        )
        
        assert abs(mean_fn.J.item() - J_mean) < 1e-6
        assert abs(mean_fn.U.item() - U_mean) < 1e-6
        assert abs(mean_fn.constant.item() - const_mean) < 1e-6

    def test_priors_registered(self, mock_databank_minimal):
        """Test that priors are properly registered."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1']
        )
        
        # Check that priors exist
        assert hasattr(mean_fn, 'J_prior')
        assert hasattr(mean_fn, 'U_prior')

    def test_constraints_registered(self, mock_databank_minimal):
        """Test that parameter constraints are enforced."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1']
        )
        
        # Check constraints exist
        assert hasattr(mean_fn, 'constraint_for_parameter_name')
        constraints = mean_fn.constraint_for_parameter_name('J')
        assert constraints is not None
        
        constraints_U = mean_fn.constraint_for_parameter_name('U')
        assert constraints_U is not None

    def test_index_buffers_created(self, mock_databank_minimal):
        """Test that index buffers are created for each atom and spin."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1', 'atom2']
        )
        
        # Check buffers exist for each atom and spin
        for atom in ['atom1', 'atom2']:
            for spin in ['up', 'down']:
                assert hasattr(mean_fn, f'diag_idx_{atom}_{spin}')
                assert hasattr(mean_fn, f'off_diag_idx_{atom}_{spin}')
                
                diag_buf = getattr(mean_fn, f'diag_idx_{atom}_{spin}')
                off_diag_buf = getattr(mean_fn, f'off_diag_idx_{atom}_{spin}')
                
                assert isinstance(diag_buf, torch.Tensor)
                assert isinstance(off_diag_buf, torch.Tensor)

    def test_forward_pass_shape(self, mock_databank_minimal):
        """Test that forward pass returns correct shape."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1']
        )
        
        # Get number of features
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        # Create batch of inputs
        X = torch.randn(10, num_features, dtype=torch.float64)
        
        # Forward pass
        result = mean_fn(X)
        
        # Check output shape
        assert result.shape == torch.Size([10])

    def test_forward_pass_returns_finite_values(self, mock_databank_minimal):
        """Test that forward pass returns finite values."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1']
        )
        
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        X = torch.randn(5, num_features, dtype=torch.float64)
        result = mean_fn(X)
        
        # Check all values are finite
        assert torch.isfinite(result).all()

    def test_multiple_atoms(self, mock_databank_minimal):
        """Test mean function with multiple atoms."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1', 'atom2']
        )
        
        index_map = mock_databank_minimal._build_flat_index_map(
            ['atom1', 'atom2'], ['up', 'down']
        )
        num_features = len(index_map['reverse_map'])
        
        X = torch.randn(5, num_features, dtype=torch.float64)
        result = mean_fn(X)
        
        assert result.shape == torch.Size([5])
        assert torch.isfinite(result).all()

    def test_batch_dimensions(self, mock_databank_minimal):
        """Test that mean function handles different batch dimensions."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1']
        )
        
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        # Test different batch shapes
        test_shapes = [
            (5, num_features),           # 2D
            (3, 4, num_features),        # 3D
            (2, 3, 4, num_features),     # 4D
        ]
        
        for shape in test_shapes:
            X = torch.randn(*shape, dtype=torch.float64)
            result = mean_fn(X)
            
            # Output should have same batch dims, without feature dim
            expected_shape = shape[:-1]
            assert result.shape == expected_shape

    def test_zero_input(self, mock_databank_minimal):
        """Test mean function behavior with zero occupation matrices."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1']
        )
        
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        # Zero input
        X = torch.zeros(5, num_features, dtype=torch.float64)
        result = mean_fn(X)
        
        # Should return constant term only (no magnetization or Hubbard contrib)
        expected = mean_fn.constant.item()
        assert torch.allclose(result, torch.full_like(result, expected))

    def test_parameters_are_learnable(self, mock_databank_minimal):
        """Test that parameters have requires_grad=True."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1']
        )
        
        assert mean_fn.J.requires_grad
        assert mean_fn.U.requires_grad
        assert mean_fn.constant.requires_grad

    def test_gradient_flow(self, mock_databank_minimal):
        """Test that gradients can flow through the mean function."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1']
        )
        
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        X = torch.randn(5, num_features, dtype=torch.float64, requires_grad=True)
        result = mean_fn(X)
        
        # Compute loss and backprop
        loss = result.sum()
        loss.backward()
        
        # Check gradients exist
        assert X.grad is not None
        assert mean_fn.J.grad is not None
        assert mean_fn.U.grad is not None


class TestMeanFunctionPhysics:
    """Test suite for physics correctness of mean function."""

    def test_magnetization_contribution_sign(self, mock_databank_minimal):
        """Test that magnetization lowers energy (negative J contribution)."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1'],
            J_prior_mean=1.0,
            U_prior_mean=0.0,  # Turn off U contribution
            constant_mean=0.0
        )
        
        # Force J to positive value
        mean_fn.J.data = torch.tensor(1.0)
        mean_fn.U.data = torch.tensor(0.0)
        
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        # Create occupation with magnetization (more up than down)
        X = torch.zeros(1, num_features, dtype=torch.float64)
        
        # Set diagonal elements to create magnetization
        diag_up = getattr(mean_fn, 'diag_idx_atom1_up')
        diag_down = getattr(mean_fn, 'diag_idx_atom1_down')
        
        if len(diag_up) > 0:
            X[0, diag_up[0]] = 1.0  # One up electron
        if len(diag_down) > 0:
            X[0, diag_down[0]] = 0.0  # No down electrons
        
        result = mean_fn(X)
        
        # Magnetization contribution should be negative (J > 0, M^2 > 0)
        # -(J/4) * M^2 < 0
        assert result.item() < mean_fn.constant.item()

    def test_constant_parameter_effect(self, mock_databank_minimal):
        """Test that constant parameter shifts energy."""
        constant_value = -100.0
        
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1'],
            J_prior_mean=0.0,  # No J contribution
            U_prior_mean=0.0,  # No U contribution
            constant_mean=constant_value
        )
        
        # Force parameters to zero
        mean_fn.J.data = torch.tensor(0.0)
        mean_fn.U.data = torch.tensor(0.0)
        
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        X = torch.randn(3, num_features, dtype=torch.float64)
        result = mean_fn(X)
        
        # With J=0 and U=0, should return constant only
        assert torch.allclose(result, torch.full_like(result, constant_value), atol=1e-5)

    @pytest.mark.parametrize("J_val,U_val", [
        (0.5, 4.5),
        (1.0, 5.0),
        (0.0, 0.0),
    ])
    def test_different_parameter_values(self, mock_databank_minimal, J_val, U_val):
        """Test mean function with different parameter values."""
        mean_fn = VectorizedPhysicsMean(
            databank=mock_databank_minimal,
            atom_ids=['atom1'],
            J_prior_mean=J_val,
            U_prior_mean=U_val
        )
        
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        X = torch.randn(5, num_features, dtype=torch.float64)
        result = mean_fn(X)
        
        # Should produce finite results for any reasonable parameters
        assert torch.isfinite(result).all()
