"""
Unit tests for Bayesian prior distributions.

Tests the helper functions for creating GP hyperparameter priors,
ensuring correct distributions and constraints are generated.
"""

import pytest
import torch
from math import log, sqrt

from lordcapulet.functions.proposal_modes.Bayesian.priors import (
    get_botorch_lengthscale_prior,
    get_outputscale_prior
)


class TestLengthscalePrior:
    """Test suite for lengthscale prior creation."""

    def test_creates_prior_and_constraint(self):
        """Test that the function returns both prior and constraint."""
        prior, constraint = get_botorch_lengthscale_prior(ard_num_dims=5)
        
        assert prior is not None
        assert constraint is not None

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="ard_num_dims must be > 0"):
            get_botorch_lengthscale_prior(ard_num_dims=0)
        
        with pytest.raises(ValueError, match="ard_num_dims must be > 0"):
            get_botorch_lengthscale_prior(ard_num_dims=-1)

    def test_prior_parameters(self):
        """Test that prior has correct parameters."""
        ard_num_dims = 5
        prior, _ = get_botorch_lengthscale_prior(ard_num_dims)
        
        SQRT2 = sqrt(2.0)
        SQRT3 = sqrt(3.0)
        expected_loc = SQRT2 + log(ard_num_dims) * 0.5
        expected_scale = SQRT3
        
        assert abs(prior.loc - expected_loc) < 1e-6
        assert abs(prior.scale - expected_scale) < 1e-6

    def test_constraint_properties(self):
        """Test that constraint has correct properties."""
        _, constraint = get_botorch_lengthscale_prior(ard_num_dims=5)
        
        # Check lower bound
        assert hasattr(constraint, 'lower_bound')
        assert abs(constraint.lower_bound - 2.5e-2) < 1e-8

    def test_prior_sampling(self):
        """Test that we can sample from the prior."""
        prior, _ = get_botorch_lengthscale_prior(ard_num_dims=5)
        
        # Sample from prior
        samples = prior.sample(sample_shape=torch.Size([100]))
        
        # Check shape
        assert samples.shape == torch.Size([100])
        
        # Check all samples are positive (lognormal property)
        assert (samples > 0).all()

    @pytest.mark.parametrize("dim", [1, 3, 5, 10, 20])
    def test_different_dimensions(self, dim):
        """Test prior creation with different ARD dimensions."""
        prior, constraint = get_botorch_lengthscale_prior(ard_num_dims=dim)
        
        # Prior parameters should scale with dimension
        expected_loc = sqrt(2.0) + log(dim) * 0.5
        assert abs(prior.loc - expected_loc) < 1e-6
        
        # Constraint should be same regardless of dimension
        assert abs(constraint.lower_bound - 2.5e-2) < 1e-8


class TestOutputscalePrior:
    """Test suite for outputscale prior creation."""

    def test_creates_lognormal_prior(self):
        """Test that the function returns a LogNormal prior."""
        prior = get_outputscale_prior(mean=1.0, std=0.5)
        
        assert prior is not None
        # Check it's a LogNormal distribution
        assert hasattr(prior, 'loc')
        assert hasattr(prior, 'scale')

    def test_prior_parameters_from_mean_std(self):
        """Test that prior parameters are correctly computed from mean and std."""
        mean = 2.0
        std = 0.5
        prior = get_outputscale_prior(mean=mean, std=std)
        
        # For LogNormal: loc and scale computed from mean and std
        variance = std ** 2
        var_normalized = variance / (mean ** 2)
        expected_scale_squared = log(1 + var_normalized)
        expected_scale = sqrt(expected_scale_squared)
        expected_loc = log(mean) - 0.5 * expected_scale_squared
        
        assert abs(prior.loc - expected_loc) < 1e-6
        assert abs(prior.scale - expected_scale) < 1e-6

    def test_prior_sampling(self):
        """Test that we can sample from the outputscale prior."""
        prior = get_outputscale_prior(mean=1.0, std=0.5)
        
        samples = prior.sample(sample_shape=torch.Size([1000]))
        
        # Check shape
        assert samples.shape == torch.Size([1000])
        
        # Check all samples are positive (lognormal property)
        assert (samples > 0).all()
        
        # Check empirical mean is close to specified mean (large sample size)
        empirical_mean = samples.mean().item()
        assert abs(empirical_mean - 1.0) < 0.1  # Allow 10% tolerance

    @pytest.mark.parametrize("mean,std", [
        (0.5, 0.1),
        (1.0, 0.5),
        (2.0, 1.0),
        (5.0, 2.0),
    ])
    def test_different_parameters(self, mean, std):
        """Test prior creation with different mean/std combinations."""
        prior = get_outputscale_prior(mean=mean, std=std)
        
        # Check parameters are correctly computed
        variance = std ** 2
        var_normalized = variance / (mean ** 2)
        expected_scale_squared = log(1 + var_normalized)
        expected_scale = sqrt(expected_scale_squared)
        expected_loc = log(mean) - 0.5 * expected_scale_squared
        
        assert abs(prior.loc - expected_loc) < 1e-6
        assert abs(prior.scale - expected_scale) < 1e-6
