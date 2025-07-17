"""
Pytest configuration for LordCapulet tests.
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import lordcapulet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_rotation_matrix():
    """Fixture providing a sample rotation matrix for tests."""
    import numpy as np
    return np.array([
        [0, 0, 1, 0, 0],
        [0, -1j/np.sqrt(2), 0, 1j/np.sqrt(2), 0],
        [0, 1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
        [-1j/np.sqrt(2), 0, 0, 0, 1j/np.sqrt(2)],
        [1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2)]
    ])
