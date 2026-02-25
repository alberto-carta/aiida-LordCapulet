"""
Unit tests for Bayesian kernel functions.

Tests the custom kernels and kernel building utilities,
using mock databank fixtures to avoid dependency on real data.
"""

import pytest
import torch
from gpytorch.kernels import MaternKernel, LinearKernel

from lordcapulet.functions.proposal_modes.Bayesian.kernels import (
    get_spin_indices,
    get_all_indices_for_atom,
    SpinFlipInvariantKernel,
    build_kernel
)


class TestSpinIndicesExtraction:
    """Test suite for extracting spin indices from databank."""

    def test_get_spin_indices(self, mock_databank_minimal):
        """Test extraction of up and down spin indices for an atom."""
        up_indices, down_indices = get_spin_indices(mock_databank_minimal, 'atom1')
        
        # Check they're tensors
        assert isinstance(up_indices, torch.Tensor)
        assert isinstance(down_indices, torch.Tensor)
        
        # Check they have the same length (one spin up per spin down)
        assert len(up_indices) == len(down_indices)
        
        # Check all are unique
        assert len(torch.unique(up_indices)) == len(up_indices)
        assert len(torch.unique(down_indices)) == len(down_indices)

    def test_get_all_indices_for_atom(self, mock_databank_minimal):
        """Test extraction of all indices for an atom."""
        all_indices = get_all_indices_for_atom(mock_databank_minimal, 'atom1')
        
        # Check it's a tensor
        assert isinstance(all_indices, torch.Tensor)
        
        # Should have both spins (up and down)
        up_indices, down_indices = get_spin_indices(mock_databank_minimal, 'atom1')
        expected_length = len(up_indices) + len(down_indices)
        assert len(all_indices) == expected_length

    def test_multiple_atoms(self, mock_databank_minimal):
        """Test indices for multiple atoms don't overlap."""
        atom1_indices = get_all_indices_for_atom(mock_databank_minimal, 'atom1')
        atom2_indices = get_all_indices_for_atom(mock_databank_minimal, 'atom2')
        
        # Indices should be disjoint (no overlap)
        all_indices = torch.cat([atom1_indices, atom2_indices])
        assert len(torch.unique(all_indices)) == len(all_indices)


class TestSpinFlipInvariantKernel:
    """Test suite for spin-flip invariant kernel wrapper."""

    @pytest.fixture
    def base_kernel(self):
        """Create a simple base kernel for testing."""
        return MaternKernel(nu=2.5, ard_num_dims=4)

    @pytest.fixture
    def spin_indices(self):
        """Create sample spin indices (2 features per spin)."""
        up_indices = torch.tensor([0, 1], dtype=torch.long)
        down_indices = torch.tensor([2, 3], dtype=torch.long)
        return up_indices, down_indices

    def test_initialization(self, base_kernel, spin_indices):
        """Test that kernel initializes correctly."""
        up_idx, down_idx = spin_indices
        kernel = SpinFlipInvariantKernel(base_kernel, up_idx, down_idx)
        
        assert kernel.base_kernel is base_kernel
        assert torch.equal(kernel.up_indices, up_idx)
        assert torch.equal(kernel.down_indices, down_idx)

    def test_mismatched_spin_indices_raises_error(self, base_kernel):
        """Test that mismatched spin indices raise ValueError."""
        up_indices = torch.tensor([0, 1], dtype=torch.long)
        down_indices = torch.tensor([2, 3, 4], dtype=torch.long)  # Different length
        
        with pytest.raises(ValueError, match="up_indices and down_indices must have the same length"):
            SpinFlipInvariantKernel(base_kernel, up_indices, down_indices)

    def test_spin_swap(self, base_kernel, spin_indices):
        """Test that spin swapping works correctly."""
        up_idx, down_idx = spin_indices
        kernel = SpinFlipInvariantKernel(base_kernel, up_idx, down_idx)
        
        # Create sample input
        X = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        
        # Swap spins
        X_flipped = kernel._swap_spins(X)
        
        # Check that up and down are swapped
        assert torch.equal(X_flipped[..., up_idx], X[..., down_idx])
        assert torch.equal(X_flipped[..., down_idx], X[..., up_idx])

    def test_symmetry_property(self, base_kernel, spin_indices):
        """Test that kernel is symmetric under spin flip."""
        up_idx, down_idx = spin_indices
        kernel = SpinFlipInvariantKernel(base_kernel, up_idx, down_idx)
        
        # Create sample inputs
        X1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        X2 = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
        
        # Compute kernel values and evaluate to dense
        k_12 = kernel(X1, X2).to_dense()
        
        # Flip spins and compute again
        X1_flipped = kernel._swap_spins(X1)
        X2_flipped = kernel._swap_spins(X2)
        k_flipped = kernel(X1_flipped, X2_flipped).to_dense()
        
        # Should be equal (invariant under simultaneous spin flip)
        assert torch.allclose(k_12, k_flipped, rtol=1e-5)

    def test_forward_returns_kernel_tensor(self, base_kernel, spin_indices):
        """Test that forward pass returns a kernel tensor."""
        up_idx, down_idx = spin_indices
        kernel = SpinFlipInvariantKernel(base_kernel, up_idx, down_idx)
        
        X1 = torch.randn(5, 4)
        X2 = torch.randn(3, 4)
        
        result = kernel(X1, X2)
        
        # GPyTorch kernels return LazyTensor objects, convert to check shape
        result_dense = result.to_dense()
        assert result_dense.shape == torch.Size([5, 3])


class TestBuildKernel:
    """Test suite for flexible kernel construction."""

    @pytest.fixture
    def simple_kernel_config(self):
        """Simple kernel configuration for testing."""
        return {
            "local": {
                "matern": {
                    "enabled": True,
                    "nu": 2.5,
                    "outputscale_prior": {"mean": 1.0, "std": 0.2}
                },
                "linear": {
                    "enabled": False
                },
                "polynomial": {
                    "enabled": False
                }
            },
            "nonlocal": {
                "residual": {
                    "enabled": False
                }
            },
            "spin_flip_invariant": False
        }

    def test_build_simple_kernel(self, mock_databank_minimal, simple_kernel_config):
        """Test building a simple Matern kernel."""
        kernel = build_kernel(
            mock_databank_minimal,
            ['atom1'],
            simple_kernel_config
        )
        
        assert kernel is not None

    def test_build_kernel_multiple_atoms(self, mock_databank_minimal, simple_kernel_config):
        """Test building kernel for multiple atoms."""
        kernel = build_kernel(
            mock_databank_minimal,
            ['atom1', 'atom2'],
            simple_kernel_config
        )
        
        assert kernel is not None

    def test_build_kernel_with_spin_flip(self, mock_databank_minimal, simple_kernel_config):
        """Test building kernel with spin-flip invariance."""
        simple_kernel_config["spin_flip_invariant"] = True
        
        kernel = build_kernel(
            mock_databank_minimal,
            ['atom1'],
            simple_kernel_config
        )
        
        assert kernel is not None

    def test_build_kernel_multiple_components(self, mock_databank_minimal):
        """Test building kernel with multiple components."""
        config = {
            "local": {
                "matern": {
                    "enabled": True,
                    "nu": 2.5,
                    "outputscale_prior": {"mean": 1.0, "std": 0.2}
                },
                "linear": {
                    "enabled": True,
                    "outputscale_prior": {"mean": 0.3, "std": 0.1}
                }
            },
            "nonlocal": {},
            "spin_flip_invariant": False
        }
        
        kernel = build_kernel(
            mock_databank_minimal,
            ['atom1'],
            config
        )
        
        assert kernel is not None

    def test_kernel_can_evaluate(self, mock_databank_minimal, simple_kernel_config):
        """Test that built kernel can evaluate on sample data."""
        # Force CPU for this test to avoid device mismatch issues
        simple_kernel_config_cpu = simple_kernel_config.copy()
        
        kernel = build_kernel(
            mock_databank_minimal,
            ['atom1'],
            simple_kernel_config_cpu
        )
        
        # Get number of features
        index_map = mock_databank_minimal._build_flat_index_map(['atom1'], ['up', 'down'])
        num_features = len(index_map['reverse_map'])
        
        # Create sample data on CPU
        X = torch.randn(5, num_features, dtype=torch.float64)
        
        # Evaluate kernel
        result = kernel(X, X).to_dense()
        
        # Check shape
        assert result.shape == torch.Size([5, 5])
        
        # Check it's positive semi-definite (diagonal elements are positive)
        diag = torch.diag(result)
        assert (diag > 0).all()
