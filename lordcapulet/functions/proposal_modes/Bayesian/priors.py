"""
Prior distributions for GP hyperparameters.

This module provides helper functions to create common priors
used in the Bayesian optimization framework, particularly those
recommended by Hvarfner et al. 2024 (vanilla BoTorch).
"""

import torch
from gpytorch.priors.torch_priors import LogNormalPrior, GammaPrior
from gpytorch.constraints.constraints import GreaterThan
from math import log, sqrt

SQRT2 = sqrt(2.0)
SQRT3 = sqrt(3.0)


def get_botorch_lengthscale_prior(ard_num_dims: int):
    """
    Creates the default BoTorch lengthscale prior and constraint
    from Hvarfner et al. 2024 (vanilla BoTorch).
    
    Args:
        ard_num_dims: Number of dimensions for ARD lengthscales
        
    Returns:
        tuple: (prior, constraint)
    """
    if ard_num_dims <= 0:
        raise ValueError("ard_num_dims must be > 0")
        
    # Create the prior
    prior = LogNormalPrior(
        loc=SQRT2 + log(ard_num_dims) * 0.5, 
        scale=SQRT3
    )
    
    # Create the constraint
    constraint = GreaterThan(
        2.5e-2, 
        # 2.5e-5, 
        transform=None, 
        initial_value=prior.mode
    )
    
    return prior, constraint


def get_outputscale_prior(mean: float, std: float):
    """
    Returns a LogNormalPrior for kernel outputscale based on the 
    desired physical mean and standard deviation.
    
    This is useful for setting priors on kernel variances when you
    have physical intuition about the expected variance in eV^2.
    
    Args:
        mean: Desired mean of the outputscale (e.g., 2.0 for 2 eV^2)
        std: Desired standard deviation (e.g., 0.1 for 0.1 eV^2)
        
    Returns:
        LogNormalPrior configured to match the desired statistics
        
    Example:
        >>> # I expect local kernel variance ~2 eV^2, spread by 0.1 eV^2
        >>> prior = get_outputscale_prior(mean=2.0, std=0.1)
    """
    if mean <= 0:
        raise ValueError("Mean must be positive for a LogNormal prior")
    
    # Calculate sigma^2 (variance of the underlying normal)
    # s^2 = ln(1 + Var/Mean^2)
    variance = std ** 2
    var_normalized = variance / (mean ** 2)
    scale_squared = log(1 + var_normalized)
    
    # Calculate scale (sigma)
    scale = sqrt(scale_squared)
    
    # Calculate loc (mu)
    # m = ln(Mean) - s^2 / 2
    loc = log(mean) - 0.5 * scale_squared
    
    return LogNormalPrior(loc=loc, scale=scale)
