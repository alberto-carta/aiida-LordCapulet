"""
Integration tests for GP model training.

Tests the complete GP model creation and training pipeline with real data.
"""

import pytest
import torch
import numpy as np


# =============================================================================
# Configuration Fixtures (matching notebook setup)
# =============================================================================

@pytest.fixture
def standard_mean_config():
    """Standard mean function configuration from notebook."""
    return {
        "type": "VectorizedPhysicsMean",
        "J_prior_mean": 0.5,
        "J_prior_std": 0.1,
        "U_prior_mean": 5.0,
        "U_prior_std": 0.2,
    }


@pytest.fixture
def standard_kernel_config():
    """Standard kernel configuration from notebook: local Matern + nonlocal residual."""
    return {
        "local": {
            "matern": {
                "enabled": True,
                "nu": 2.5,
                "outputscale_prior": {"mean": 0.2, "std": 0.02}
            },
        },
        "nonlocal": {
            "residual": {
                "enabled": True,
                "outputscale_prior": {"mean": 0.05, "std": 0.01}
            },
        },
        "spin_flip_invariant": True
    }


@pytest.fixture
def minimal_mean_config():
    """Minimal mean configuration for fast tests."""
    return {"type": "VectorizedPhysicsMean"}


# =============================================================================
# Test Classes
# =============================================================================

class TestGPModelCreation:
    """Test GP model creation with real DataBank."""
    
    def test_create_gp_model_basic(self, gp_databank_small, standard_mean_config, standard_kernel_config):
        """Test basic GP model creation with real data."""
        from lordcapulet.functions.proposal_modes.Bayesian.gp_model import (
            create_gp_model, print_kernel_diagnostics
        )
        
        print("\n" + "="*60)
        print("TEST: GP Model Creation (Basic)")
        print("="*60)
        
        # Prepare data using databank's methods
        device = torch.device('cpu')
        atoms = gp_databank_small.atom_ids
        
        print(f"\nDataBank Info:")
        print(f"  Total samples: {len(gp_databank_small)}")
        print(f"  Atoms: {atoms}")
        print(f"  Energy range: [{min(gp_databank_small.energies):.4f}, {max(gp_databank_small.energies):.4f}] eV")
        
        x_data = gp_databank_small.to_pytorch(
            atom_ids=atoms,
            spins=['up', 'down'],
            include_energies=False,
            device=device
        )
        
        energies = gp_databank_small.energies
        y_data = torch.tensor(energies, dtype=torch.float32, device=device).unsqueeze(-1)
        
        print(f"\nTensor shapes:")
        print(f"  X: {x_data.shape}")
        print(f"  Y: {y_data.shape}")
        
        # Create model using fixtures
        print("\nCreating GP model...")
        model = create_gp_model(
            train_X=x_data,
            train_Y=y_data,
            databank=gp_databank_small,
            atom_ids=atoms,
            mean_config=standard_mean_config,
            kernel_config=standard_kernel_config,
            device=device
        )
        
        print("\nModel created successfully!")
        print(f"  Mean module type: {type(model.mean_module).__name__}")
        print(f"  Covariance module type: {type(model.covar_module).__name__}")
        
        # Print kernel diagnostics
        print("\nKernel Configuration:")
        print_kernel_diagnostics(model)
        
        # Assertions
        assert model is not None
        assert hasattr(model, 'mean_module')
        assert hasattr(model, 'covar_module')
        assert hasattr(model, 'likelihood')
        assert model.train_inputs[0].shape == x_data.shape
        assert model.train_targets.shape == y_data.squeeze().shape
    
    def test_create_gp_model_minimal(self, gp_databank_small, minimal_mean_config, standard_kernel_config):
        """Test GP model creation with minimal configuration."""
        from lordcapulet.functions.proposal_modes.Bayesian.gp_model import create_gp_model
        
        device = torch.device('cpu')
        atoms = gp_databank_small.atom_ids
        
        x_data = gp_databank_small.to_pytorch(
            atom_ids=atoms,
            spins=['up', 'down'],
            include_energies=False,
            device=device
        )
        
        energies = gp_databank_small.energies
        y_data = torch.tensor(energies, dtype=torch.float32, device=device).unsqueeze(-1)
        
        # Should work even with minimal config
        model = create_gp_model(
            train_X=x_data,
            train_Y=y_data,
            databank=gp_databank_small,
            atom_ids=atoms,
            mean_config=minimal_mean_config,
            kernel_config=standard_kernel_config,
            device=device
        )
        
        assert model is not None
    
    def test_mean_module_initialization(self, gp_databank_small, standard_kernel_config):
        """Test that mean module is properly initialized."""
        from lordcapulet.functions.proposal_modes.Bayesian.gp_model import create_gp_model
        
        device = torch.device('cpu')
        atoms = gp_databank_small.atom_ids
        
        x_data = gp_databank_small.to_pytorch(
            atom_ids=atoms,
            spins=['up', 'down'],
            include_energies=False,
            device=device
        )
        
        energies = gp_databank_small.energies
        y_data = torch.tensor(energies, dtype=torch.float32, device=device).unsqueeze(-1)
        
        mean_config = {
            "type": "VectorizedPhysicsMean",
            "J_prior_mean": 0.5,
            "J_prior_std": 0.1,
        }
        
        model = create_gp_model(
            train_X=x_data,
            train_Y=y_data,
            databank=gp_databank_small,
            atom_ids=atoms,
            mean_config=mean_config,
            kernel_config=standard_kernel_config,
            device=device
        )
        
        # Check mean module has expected attributes
        assert hasattr(model.mean_module, 'J')
        assert hasattr(model.mean_module, 'U')
        assert hasattr(model.mean_module, 'constant')
        
        # Mean module should produce reasonable values
        with torch.no_grad():
            mean_values = model.mean_module(x_data)
        
        # Shape should be (n_samples,) as it returns unqueezed output
        assert mean_values.shape[0] == len(x_data)
        assert torch.all(torch.isfinite(mean_values))


class TestGPModelTraining:
    """Test GP model training."""
    
    def test_train_gp_model_basic(self, gp_databank_small, minimal_mean_config, standard_kernel_config):
        """Test basic GP model training."""
        from lordcapulet.functions.proposal_modes.Bayesian.gp_model import (
            create_gp_model, train_gp_model, evaluate_loo_cv, print_kernel_diagnostics
        )
        
        print("\n" + "="*60)
        print("TEST: GP Model Training (Basic)")
        print("="*60)
        
        device = torch.device('cpu')
        atoms = gp_databank_small.atom_ids
        
        x_data = gp_databank_small.to_pytorch(
            atom_ids=atoms,
            spins=['up', 'down'],
            include_energies=False,
            device=device
        )
        
        energies = gp_databank_small.energies
        y_data = torch.tensor(energies, dtype=torch.float32, device=device).unsqueeze(-1)
        
        model = create_gp_model(
            train_X=x_data,
            train_Y=y_data,
            databank=gp_databank_small,
            atom_ids=atoms,
            mean_config=minimal_mean_config,
            kernel_config=standard_kernel_config,
            device=device
        )
        
        # Train model
        print("\nTraining GP model...")
        training_config = {"method": "fit_gpytorch_mll"}
        trained_model = train_gp_model(
            model=model,
            train_X=x_data,
            train_Y=y_data,
            training_config=training_config
        )
        
        # Model should be trained
        assert trained_model is not None
        
        print("\nLearned Hyperparameters:")
        print_kernel_diagnostics(trained_model)
        
        # Check that parameters have been updated
        # Likelihood noise should be positive
        noise = trained_model.likelihood.noise.item()
        print(f"\nLikelihood noise: {noise:.6f}")
        assert noise > 0
        assert np.isfinite(noise)
        
        # Evaluate LOO cross-validation
        print("\nPerforming Leave-One-Out Cross-Validation...")
        loo_results = evaluate_loo_cv(
            trained_model, x_data, y_data, tikhonov_reg=1e-5, debug=False
        )
        
        print(f"\nLOO-CV Results:")
        print(f"  RMSE: {loo_results['rmse']:.4f} eV")
        print(f"  Q²:   {loo_results['q2']:.4f}")
        
        assert loo_results['rmse'] > 0
        assert -1 <= loo_results['q2'] <= 1  # Q² should be reasonable
    
    def test_trained_model_predictions(self, gp_databank_small, minimal_mean_config, standard_kernel_config):
        """Test that trained model can make predictions."""
        from lordcapulet.functions.proposal_modes.Bayesian.gp_model import (
            create_gp_model, train_gp_model, evaluate_loo_cv
        )
        
        print("\n" + "="*60)
        print("TEST: GP Model Predictions")
        print("="*60)
        
        device = torch.device('cpu')
        atoms = gp_databank_small.atom_ids
        
        x_data = gp_databank_small.to_pytorch(
            atom_ids=atoms,
            spins=['up', 'down'],
            include_energies=False,
            device=device
        )
        
        energies = gp_databank_small.energies
        y_data = torch.tensor(energies, dtype=torch.float32, device=device).unsqueeze(-1)
        
        model = create_gp_model(
            train_X=x_data,
            train_Y=y_data,
            databank=gp_databank_small,
            atom_ids=atoms,
            mean_config=minimal_mean_config,
            kernel_config=standard_kernel_config,
            device=device
        )
        
        training_config = {"method": "fit_gpytorch_mll"}
        trained_model = train_gp_model(
            model=model,
            train_X=x_data,
            train_Y=y_data,
            training_config=training_config
        )
        
        # Make predictions on training data
        print("\nMaking predictions on training data...")
        trained_model.eval()
        with torch.no_grad():
            posterior = trained_model.posterior(x_data)
            predictions = posterior.mean
            uncertainties = posterior.variance
        
        # Compute prediction statistics
        pred_np = predictions.cpu().numpy().flatten()
        true_np = y_data.cpu().numpy().flatten()
        std_np = torch.sqrt(uncertainties).cpu().numpy().flatten()
        
        residuals = pred_np - true_np
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        
        print(f"\nPrediction Statistics:")
        print(f"  MAE:  {mae:.4f} eV")
        print(f"  RMSE: {rmse:.4f} eV")
        print(f"  Mean uncertainty: {std_np.mean():.4f} eV")
        print(f"  Max uncertainty:  {std_np.max():.4f} eV")
        
        # Evaluate LOO-CV for more robust validation
        print("\nLOO-CV Validation:")
        loo_results = evaluate_loo_cv(
            trained_model, x_data, y_data, tikhonov_reg=1e-5, debug=False
        )
        print(f"  LOO RMSE: {loo_results['rmse']:.4f} eV")
        print(f"  LOO Q²:   {loo_results['q2']:.4f}")
        
        # Check predictions
        assert predictions.shape == (len(x_data), 1)
        assert uncertainties.shape == (len(x_data), 1)
        assert torch.all(torch.isfinite(predictions))
        assert torch.all(torch.isfinite(uncertainties))
        assert torch.all(uncertainties > 0)  # Variance should be positive
    
    def test_training_convergence(self, gp_databank_small, minimal_mean_config, standard_kernel_config, capsys):
        """Test that training prints convergence information."""
        from lordcapulet.functions.proposal_modes.Bayesian.gp_model import (
            create_gp_model, train_gp_model
        )
        
        device = torch.device('cpu')
        atoms = gp_databank_small.atom_ids
        
        x_data = gp_databank_small.to_pytorch(
            atom_ids=atoms,
            spins=['up', 'down'],
            include_energies=False,
            device=device
        )
        
        energies = gp_databank_small.energies
        y_data = torch.tensor(energies, dtype=torch.float32, device=device).unsqueeze(-1)
        
        model = create_gp_model(
            train_X=x_data,
            train_Y=y_data,
            databank=gp_databank_small,
            atom_ids=atoms,
            mean_config=minimal_mean_config,
            kernel_config=standard_kernel_config,
            device=device
        )
        
        training_config = {"method": "fit_gpytorch_mll"}
        trained_model = train_gp_model(
            model=model,
            train_X=x_data,
            train_Y=y_data,
            training_config=training_config
        )
        
        # Check that loss was printed
        captured = capsys.readouterr()
        assert "Final Total Loss (NMLL)" in captured.out


class TestGPModelWithFullDataset:
    """Test GP model with full FeO dataset."""
    
    @pytest.mark.slow
    def test_full_dataset_training(self, gp_databank, standard_mean_config, standard_kernel_config):
        """Test training on full FeO dataset (slower)."""
        from lordcapulet.functions.proposal_modes.Bayesian.gp_model import (
            create_gp_model, train_gp_model, evaluate_loo_cv, print_kernel_diagnostics
        )
        
        print("\n" + "="*60)
        print("TEST: Full Dataset Training (FeO)")
        print("="*60)
        
        device = torch.device('cpu')
        atoms = gp_databank.atom_ids
        
        # Use full dataset
        x_data = gp_databank.to_pytorch(
            atom_ids=atoms,
            spins=['up', 'down'],
            include_energies=False,
            device=device
        )
        
        energies = gp_databank.energies
        y_data = torch.tensor(energies, dtype=torch.float32, device=device).unsqueeze(-1)
        
        print(f"\nDataset Information:")
        print(f"  Total samples: {len(x_data)}")
        print(f"  Feature dimension: {x_data.shape[1]}")
        print(f"  Energy range: [{min(energies):.4f}, {max(energies):.4f}] eV")
        print(f"  Energy std: {np.std(energies):.4f} eV")
        
        model = create_gp_model(
            train_X=x_data,
            train_Y=y_data,
            databank=gp_databank,
            atom_ids=atoms,
            mean_config=standard_mean_config,
            kernel_config=standard_kernel_config,
            device=device
        )
        
        print("\nTraining GP model on full dataset...")
        training_config = {"method": "fit_gpytorch_mll"}
        trained_model = train_gp_model(
            model=model,
            train_X=x_data,
            train_Y=y_data,
            training_config=training_config
        )
        
        # Model should be trained
        assert trained_model is not None
        
        print("\nLearned Kernel Components:")
        print_kernel_diagnostics(trained_model)
        
        print(f"\nLikelihood noise: {trained_model.likelihood.noise.item():.6f}")
        
        # Evaluate LOO cross-validation
        print("\nPerforming Leave-One-Out Cross-Validation on full dataset...")
        print("(This may take a while with 419 samples)")
        loo_results = evaluate_loo_cv(
            trained_model, x_data, y_data, tikhonov_reg=1e-5, debug=False
        )
        
        print(f"\nFull Dataset LOO-CV Results:")
        print(f"  RMSE: {loo_results['rmse']:.4f} eV")
        print(f"  Q²:   {loo_results['q2']:.4f}")
        
        # Make predictions on subset
        print("\nTesting predictions on first 10 samples...")
        trained_model.eval()
        with torch.no_grad():
            posterior = trained_model.posterior(x_data[:10])
            predictions = posterior.mean
            uncertainties = torch.sqrt(posterior.variance)
        
        pred_np = predictions.cpu().numpy().flatten()
        true_np = y_data[:10].cpu().numpy().flatten()
        std_np = uncertainties.cpu().numpy().flatten()
        
        print("\nFirst 10 predictions:")
        for i in range(10):
            print(f"  Sample {i}: True={true_np[i]:.4f}, Pred={pred_np[i]:.4f} ± {std_np[i]:.4f} eV")
        
        assert torch.all(torch.isfinite(predictions))
        assert loo_results['q2'] > 0.8, f"Q² = {loo_results['q2']:.4f} is too low for good model"
