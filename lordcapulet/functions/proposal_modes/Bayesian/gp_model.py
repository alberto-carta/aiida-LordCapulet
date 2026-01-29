"""
Gaussian Process model creation and training utilities.

This module provides functions to:
- Create GP models with custom mean and covariance functions
- Train GP models using various strategies
- Evaluate model performance
"""

from pyexpat import model
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
import torch.optim as optim
from sklearn.metrics import r2_score, root_mean_squared_error

from .mean_functions import VectorizedPhysicsMean
from .kernels import build_kernel


def create_gp_model(train_X, train_Y, databank, atom_ids, mean_config, kernel_config, device):
    """
    Create a GP model with custom mean and kernel functions.
    
    Args:
        train_X: Training inputs [n_train, n_features]
        train_Y: Training outputs [n_train, 1]
        databank: DataBank instance
        atom_ids: List of atom IDs to include
        mean_config: Configuration for mean function
        kernel_config: Configuration for kernel
        device: Device to use (CPU or CUDA)
        
    Returns:
        SingleTaskGP model instance
    """
    # Create mean function
    mean_module = None
    if mean_config["type"] == "VectorizedPhysicsMean":
        # When using Standardize transform, the mean function works in standardized space
        # Initialize constant to 0 (will learn the offset in standardized space)
        # constant_mean = 0.0

        #initialize constant mean to average of trainY
        # constant_mean = torch.mean(train_Y).item()
        constant_mean = torch.min(train_Y).item()
        
        mean_module = VectorizedPhysicsMean(
            databank=databank,
            atom_ids=atom_ids,
            J_prior_mean=mean_config.get("J_prior_mean", 0.5),
            J_prior_std=mean_config.get("J_prior_std", 0.2),
            # J_lin_prior_mean=mean_config.get("J_lin_prior_mean", 0.1),
            # J_lin_prior_std=mean_config.get("J_lin_prior_std", 0.05),
            U_prior_mean=mean_config.get("U_prior_mean", 4.5),
            U_prior_std=mean_config.get("U_prior_std", 1.0),
            constant_mean=constant_mean
        )
    
    # Create kernel
    covar_module = build_kernel(databank, atom_ids, kernel_config)
    
    # Use Standardize transform to automatically normalize outputs
    # This prevents issues with large absolute energy values overwhelming the physics-based mean
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        mean_module=mean_module,
        covar_module=covar_module,
        # outcome_transform=Standardize(m=1),  # Standardize outputs to mean=0, std=1
        outcome_transform= None
    )
    
    return model


def train_gp_model(model, train_X, train_Y, training_config):
    """
    Train a GP model using the specified training strategy.
    
    Args:
        model: The GP model to train
        train_X: Training inputs
        train_Y: Training outputs
        training_config: Dictionary with training parameters
        
    Returns:
        Trained model
    """
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll = mll.to(train_X)
    
    method = training_config.get("method", "fit_gpytorch_mll")
    
    if method == "fit_gpytorch_mll":
        # Default: Use BoTorch's fit_gpytorch_mll (L-BFGS-B)
        fit_gpytorch_mll(mll)
        
    elif method == "sgd":
        # Custom SGD training loop (following BoTorch tutorial)
        # _train_with_sgd(model, mll, train_X, train_Y, training_config)
        raise NotImplementedError("SGD training method is currently disabled due to issues with Standardize transform")
        
    elif method == "two_stage":
        # Two-stage training: first mean only, then kernel only
        # _train_two_stage(model, mll, train_X, train_Y, training_config)
        raise NotImplementedError("Two-stage training method is currently disabled due to issues with Standardize transform")
        
    else:
        # Fallback to default
        fit_gpytorch_mll(mll)

    # print value of the optimized loss
    # 2. Compute the final loss manually
    model.train()
    mll.train()

    with torch.no_grad():
        # Pass the training data back through the model
        # Note: model.train_inputs[0] and model.train_targets are stored by the GP
        output = model(model.train_inputs[0])
        loss = -mll(output, model.train_targets)

    print(f"Final Total Loss (NMLL): {loss.item():.4f}") 
    
    return model


def evaluate_model(model, test_X, test_Y):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained GP model
        test_X: Test inputs
        test_Y: Test outputs
        
    Returns:

        Dictionary with evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(test_X)
        y_pred_mean = posterior.mean
        y_pred_variance = posterior.variance

    # Move to numpy for scikit-learn metrics
    y_true_np = test_Y.cpu().numpy().ravel()
    y_pred_np = y_pred_mean.cpu().numpy().ravel()
    y_std_np = torch.sqrt(y_pred_variance).cpu().numpy().ravel()

    # Calculate metrics
    r2 = r2_score(y_true_np, y_pred_np)
    rmse = root_mean_squared_error(y_true_np, y_pred_np)

    return {
        "r2": r2,
        "rmse": rmse,
        "y_true": y_true_np,
        "y_pred": y_pred_np,
        "y_std": y_std_np,
    }

def evaluate_loo_cv(model, train_X, train_Y, tikhonov_reg=1e-5, debug=False):
    """
    Evaluate model performance using Leave-One-Out Cross-Validation (LOO-CV).
    
    Args:
        model: Trained GP model
        train_X: Training inputs
        train_Y: Training outputs
    """


    model.eval() # Ensure we are in eval mode to access fixed parameters
    
    with torch.no_grad():
        # 1. Get the full covariance matrix (Kernel + Noise) at training points and the mean function
        model_output = model(train_X)
        likelihood_output = model.likelihood(model_output)
        
        K_total = likelihood_output.covariance_matrix + tikhonov_reg
        mean_func = model.mean_module(train_X).squeeze()

        # 2. Invert the matrix   
        K_inv = torch.linalg.inv(K_total)
        # K_inv = torch.linalg.pinv(K_total, atol=1e-3)  # Use pseudo-inverse for numerical stability

        
        # 3. Compute Alpha (Weight vector) = K_inv * y
        y_flat = train_Y.squeeze() - mean_func  # Center y by subtracting mean function
        alpha = K_inv @ y_flat
        
        # 4. Extract the diagonal of the inverse 
        diag_inv = torch.diagonal(K_inv)

        loo_error = y_flat - (alpha / diag_inv)
    
        #    Variance = 1 / Diagonal
        loo_vars = 1.0 / diag_inv
        
        # 6. Calculate Metrics (RMSE)
        residuals = y_flat - loo_error
        mse = torch.mean(residuals.pow(2))
        rmse = torch.sqrt(mse)
        
        # Compute PRESS (Predicted Residual Sum of Squares) and TSS (Total Sum of Squares)
        # $$\text{PRESS} = \sum_{i=1}^N (y_i - \mu_{-i})^2 = \sum_{i=1}^N \left( \frac{\alpha_i}{[K^{-1}]_{ii}} \right)^2$$
        # $$\text{TSS} = \sum_{i=1}^N (y_i - \bar{y})^2$$
        press = torch.sum((alpha / diag_inv).pow(2))
        # tss = torch.sum((y_flat - torch.mean(y_flat)).pow(2))
        tss = torch.sum((train_Y - torch.mean(train_Y)).pow(2))
        q2 = 1 - (press / tss)


        loo_predictions = loo_error + mean_func


        if debug:
            print(f"LOO-CV RMSE: {rmse.item():.4f}")
            print(f"LOO-CV Q^2 metric: {q2.item():.4f}")    

        return {
            "rmse": rmse.item(),
            "q2": q2.item(),
            "predictions": loo_predictions.cpu(),
            "vars": loo_vars.cpu(),
        }
    
        


def print_kernel_diagnostics(model, reporter=print):
    """
    Print diagnostic information about kernel components and their learned variances.
    
    Args:
        model: Trained GP model
    """
    import gpytorch
    
    reporter("="*60)
    reporter(f"{'KERNEL TYPE':<30} | {'VARIANCE (Outputscale)':<25}")
    reporter("="*60)
    
    # Get the list of additive components
    if hasattr(model.covar_module, 'kernels'):
        sub_kernels = model.covar_module.kernels
    else:
        sub_kernels = [model.covar_module]

    for i, sub_kernel in enumerate(sub_kernels):
        name = f"Term {i}"
        variance = "N/A"
        
        # Unwrap ScaleKernel
        if isinstance(sub_kernel, gpytorch.kernels.ScaleKernel):
            variance = f"{sub_kernel.outputscale.item():.5f}"
            base = sub_kernel.base_kernel
        else:
            base = sub_kernel
            
        # Unwrap SpinFlipInvariantKernel
        if hasattr(base, 'base_kernel'):
            if "SpinFlip" in base.__class__.__name__:
                base = base.base_kernel

        # Identify the physics type
        if isinstance(base, gpytorch.kernels.ProductKernel):
            k1 = base.kernels[0]
            k2 = base.kernels[1]
            
            if isinstance(k1, gpytorch.kernels.LinearKernel):
                name = "NON-LOCAL: Heisenberg (Lin x Lin)"
            elif isinstance(k1, gpytorch.kernels.PolynomialKernel):
                name = "NON-LOCAL: Kugel-Khomskii (Poly x Poly)"
            elif isinstance(k1, gpytorch.kernels.MaternKernel):
                name = "NON-LOCAL: Residuals (Mat x Mat)"
            else:
                name = "NON-LOCAL: Mixed Product"

        elif isinstance(base, gpytorch.kernels.LinearKernel):
            name = "LOCAL: Linear"
        elif isinstance(base, gpytorch.kernels.PolynomialKernel):
            name = "LOCAL: Poly"
        elif isinstance(base, gpytorch.kernels.MaternKernel):
            name = "LOCAL: Texture (Matern)"
            
        reporter(f"{name:<30} | {variance:<25}")
    
    reporter("="*60)
