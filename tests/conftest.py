"""
Pytest configuration for LordCapulet tests.

Provides shared fixtures for unit and integration tests.
"""

import pytest
import sys
import os
import numpy as np

# Add the parent directory to the path so we can import lordcapulet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Math/Geometry Fixtures
# ============================================================================

@pytest.fixture
def sample_rotation_matrix():
    """Fixture providing a sample rotation matrix for tests."""
    return np.array([
        [0, 0, 1, 0, 0],
        [0, -1j/np.sqrt(2), 0, 1j/np.sqrt(2), 0],
        [0, 1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
        [-1j/np.sqrt(2), 0, 0, 0, 1j/np.sqrt(2)],
        [1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2)]
    ])


# ============================================================================
# DataBank Fixtures (for all tests)
# ============================================================================

@pytest.fixture
def mock_occupation_matrix_data():
    """
    Create simple occupation matrices for testing.
    
    Structure: 2 atoms with 5x5 diagonal matrices (d-orbitals).
    """
    from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
    
    data = {
        'atom1': {
            'specie': 'Fe1',
            'shell': '3d',
            'occupation_matrix': {
                'up': np.diag([1.0, 0.8, 0.5, 0.2, 0.0]),
                'down': np.diag([0.9, 0.7, 0.4, 0.1, 0.0])
            }
        },
        'atom2': {
            'specie': 'Fe2',
            'shell': '3d',
            'occupation_matrix': {
                'up': np.diag([0.8, 0.6, 0.4, 0.2, 0.1]),
                'down': np.diag([0.7, 0.5, 0.3, 0.1, 0.0])
            }
        }
    }
    return OccupationMatrixData(data)


@pytest.fixture
def simple_databank(mock_occupation_matrix_data):
    """
    Real DataBank with simple test data.
    
    Use this for ALL unit tests. It's a real DataBank object,
    just initialized with simple diagonal matrices instead of
    loading from JSON.
    
    Contains:
    - 2 atoms ('atom1', 'atom2')  
    - 2 spins (up, down)
    - 5x5 diagonal matrices (d-orbitals)
    - 2 calculation records
    - All DataBank methods work normally
    """
    from lordcapulet.data_structures.databank import DataBank
    
    records = [
        {
            'pk': 1,
            'energy': -100.5,
            'energy_uncertainty': 0.0,
            'converged': True,
            'occ_data': mock_occupation_matrix_data,
            'metadata': {}
        },
        {
            'pk': 2,
            'energy': -99.8,
            'energy_uncertainty': 0.0,
            'converged': True,
            'occ_data': mock_occupation_matrix_data,
            'metadata': {}
        }
    ]
    
    return DataBank(records=records)


# Backward compatibility aliases (tests still using old names)
mock_databank_minimal = simple_databank
mock_databank_with_data = simple_databank


# ============================================================================
# PyTorch Fixtures
# ============================================================================

@pytest.fixture
def torch_device():
    """Get available torch device (CPU or CUDA)."""
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_train_data(torch_device):
    """Create sample training data for GP models."""
    import torch
    
    # Simple 2D input, 1D output
    train_X = torch.tensor([
        [0.2, 0.3],
        [0.4, 0.5],
        [0.6, 0.7],
        [0.8, 0.9],
    ], dtype=torch.float64, device=torch_device)
    
    train_Y = torch.tensor([
        [-1.5],
        [-1.2],
        [-0.9],
        [-0.6],
    ], dtype=torch.float64, device=torch_device)
    
    return train_X, train_Y


# ============================================================================
# Real Data Fixtures (for integration tests)
# ============================================================================

@pytest.fixture
def real_databank_path():
    """Path to real test databank JSON file."""
    # Use one of the example files
    import os
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, 'examples', 'FeO_scan_data_extractor.json')


@pytest.fixture
def real_databank(real_databank_path):
    """Load a real DataBank from JSON (for integration tests)."""
    from lordcapulet.data_structures.databank import DataBank
    import os
    
    # Only load if file exists (skip if not available)
    if not os.path.exists(real_databank_path):
        pytest.skip(f"Real data file not found: {real_databank_path}")
    
    return DataBank.from_json(real_databank_path, only_converged=True)
