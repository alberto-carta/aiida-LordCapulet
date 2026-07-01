"""Tests for OccupationMatrixData - essential functionality only."""

import numpy as np
import pytest

from lordcapulet.data_structures.occupation_matrix import (
    OccupationMatrixData,
    compute_occupation_distance
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_2x2_identity():
    """Create a simple 2x2 identity occupation matrix for a single atom."""
    return {
        'Atom_1': {
            'specie': 'Fe',
            'shell': '3d',
            'occupation_matrix': {
                'up': [[1.0, 0.0], [0.0, 1.0]],
                'down': [[1.0, 0.0], [0.0, 1.0]]
            }
        }
    }


@pytest.fixture
def simple_2x2_modified():
    """Create a 2x2 occupation matrix with one element changed."""
    return {
        'Atom_1': {
            'specie': 'Fe',
            'shell': '3d',
            'occupation_matrix': {
                'up': [[2.0, 0.0], [0.0, 1.0]],  # Changed [0,0] from 1.0 to 2.0
                'down': [[1.0, 0.0], [0.0, 1.0]]
            }
        }
    }


@pytest.fixture
def multi_atom_data():
    """Create occupation data with two atoms."""
    return {
        'Atom_1': {
            'specie': 'Fe',
            'shell': '3d',
            'occupation_matrix': {
                'up': [[1.0, 0.0], [0.0, 1.0]],
                'down': [[1.0, 0.0], [0.0, 1.0]]
            }
        },
        'Atom_2': {
            'specie': 'Ni',
            'shell': '3d',
            'occupation_matrix': {
                'up': [[1.0, 0.0], [0.0, 1.0]],
                'down': [[1.0, 0.0], [0.0, 1.0]]
            }
        }
    }


@pytest.fixture
def realistic_fe_3d():
    """Create a realistic 5x5 Fe d-orbital occupation matrix."""
    d_orbital_up = [
        [0.575, 0.054, 0.054, 0.0, 0.108],
        [0.054, 0.962, 0.013, 0.094, -0.013],
        [0.054, 0.013, 0.962, -0.094, -0.013],
        [0.0, 0.094, -0.094, 0.575, 0.0],
        [0.108, -0.013, -0.013, 0.0, 0.962]
    ]
    return {
        'Atom_1': {
            'specie': 'Fe1',
            'shell': '3d',
            'occupation_matrix': {
                'up': d_orbital_up,
                'down': d_orbital_up
            }
        }
    }


# ============================================================================
# Test Classes
# ============================================================================


class TestOccupationMatrixData:
    """Test suite for OccupationMatrixData class - core features only."""
    
    def test_initialization_empty(self):
        """Test creating empty OccupationMatrixData."""
        occ_data = OccupationMatrixData()
        assert occ_data.data == {}
        assert occ_data.as_dict() == {}
    
    def test_initialization_with_data(self, simple_2x2_identity):
        """Test creating OccupationMatrixData with initial data."""
        occ_data = OccupationMatrixData(simple_2x2_identity)
        assert occ_data.data == simple_2x2_identity
        assert occ_data.as_dict() == simple_2x2_identity
    
    def test_from_dict(self, simple_2x2_identity):
        """Test creating from dictionary."""
        occ_data = OccupationMatrixData.from_dict(simple_2x2_identity)
        assert occ_data.data == simple_2x2_identity
    
    def test_get_trace(self):
        """Test getting trace of occupation matrices."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.1], [0.1, 2.0]],
                    'down': [[0.5, 0.0], [0.0, 0.5]]
                }
            }
        }
        occ_data = OccupationMatrixData(test_data)
        
        # Test spin-up
        trace_up = occ_data.get_trace('Atom_1', 'up')
        assert trace_up == pytest.approx(3.0)  # 1.0 + 2.0
        
        # Test spin-down
        trace_down = occ_data.get_trace('Atom_1', 'down')
        assert trace_down == pytest.approx(1.0)  # 0.5 + 0.5
    
    def test_get_electron_number(self):
        """Test getting total electron number (trace_up + trace_down)."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.1], [0.1, 2.0]],
                    'down': [[0.5, 0.0], [0.0, 0.5]]
                }
            }
        }
        occ_data = OccupationMatrixData(test_data)
        electron_num = occ_data.get_electron_number('Atom_1')
        assert electron_num == pytest.approx(4.0)  # 3.0 + 1.0
    
    def test_get_magnetic_moment(self):
        """Test getting magnetic moment (trace_up - trace_down)."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.1], [0.1, 2.0]],
                    'down': [[0.5, 0.0], [0.0, 0.5]]
                }
            }
        }
        occ_data = OccupationMatrixData(test_data)
        moment = occ_data.get_magnetic_moment('Atom_1')
        assert moment == pytest.approx(2.0)  # 3.0 - 1.0
    
    def test_from_aiida_qe_occupations_simple(self):
        """Test conversion from AiiDA-QE get_occupations() format.
        
        atom_index comes from QE as a plain integer; from_aiida_qe_occupations
        must normalise it to the canonical 'Atom_N' string key.
        """
        aiida_format = [
            {
                'atom_index': 1,          # QE returns an integer
                'kind_name': 'Fe1',
                'manifold': '3d',
                'occupations': {
                    'up': np.array([[1.0, 0.1], [0.1, 0.5]]),
                    'down': np.array([[0.5, 0.05], [0.05, 1.0]])
                }
            }
        ]
        
        occ_data = OccupationMatrixData.from_aiida_qe_occupations(aiida_format)
        
        assert 'Atom_1' in occ_data.data
        assert occ_data.data['Atom_1']['specie'] == 'Fe1'
        assert occ_data.data['Atom_1']['shell'] == '3d'
        
        # Check matrices are converted to lists
        assert isinstance(occ_data.data['Atom_1']['occupation_matrix']['up'], list)
        np.testing.assert_allclose(
            occ_data.data['Atom_1']['occupation_matrix']['up'],
            [[1.0, 0.1], [0.1, 0.5]]
        )
    
    def test_from_aiida_qe_occupations_multiple_atoms(self):
        """Test conversion with multiple atoms; both use integer atom_index as QE returns."""
        aiida_format = [
            {
                'atom_index': 1,
                'kind_name': 'Fe1',
                'manifold': '3d',
                'occupations': {
                    'up': np.eye(5),
                    'down': np.eye(5) * 0.5
                }
            },
            {
                'atom_index': 2,
                'kind_name': 'Fe2',
                'manifold': '3d',
                'occupations': {
                    'up': np.eye(5) * 0.8,
                    'down': np.eye(5) * 0.2
                }
            }
        ]
        
        occ_data = OccupationMatrixData.from_aiida_qe_occupations(aiida_format)
        
        assert len(occ_data.data) == 2
        assert 'Atom_1' in occ_data.data
        assert 'Atom_2' in occ_data.data
    
    def test_get_occupation_matrix(self, simple_2x2_identity):
        """Test getting specific occupation matrix."""
        occ_data = OccupationMatrixData(simple_2x2_identity)
        up_matrix = occ_data.get_occupation_matrix('Atom_1', 'up')
        
        assert up_matrix == [[1.0, 0.0], [0.0, 1.0]]
    
    def test_get_occupation_matrix_as_numpy(self, simple_2x2_identity):
        """Test getting occupation matrix as numpy array."""
        occ_data = OccupationMatrixData(simple_2x2_identity)
        up_matrix = occ_data.get_occupation_matrix_as_numpy('Atom_1', 'up')
        
        assert isinstance(up_matrix, np.ndarray)
        np.testing.assert_allclose(up_matrix, [[1.0, 0.0], [0.0, 1.0]])
    
    def test_set_occupation_matrix(self):
        """Test setting occupation matrix."""
        occ_data = OccupationMatrixData()
        
        matrix = [[1.0, 0.0], [0.0, 0.8]]
        occ_data.set_occupation_matrix('Atom_1', 'up', matrix)
        
        assert 'Atom_1' in occ_data.data
        assert occ_data.data['Atom_1']['occupation_matrix']['up'] == matrix
    
    def test_json_serializability(self, simple_2x2_identity):
        """Test that data can be JSON serialized."""
        import json
        
        occ_data = OccupationMatrixData(simple_2x2_identity)
        json_str = json.dumps(occ_data.as_dict())
        loaded_data = json.loads(json_str)
        
        assert loaded_data == simple_2x2_identity
    
    def test_realistic_5x5_d_orbital(self, realistic_fe_3d):
        """Test with realistic 5x5 d-orbital occupation matrix."""
        occ_data = OccupationMatrixData(realistic_fe_3d)
        up_matrix = occ_data.get_occupation_matrix_as_numpy('Atom_1', 'up')
        
        # Verify trace
        trace_up = np.trace(up_matrix)
        d_orbital_up = realistic_fe_3d['Atom_1']['occupation_matrix']['up']
        expected_trace = sum(d_orbital_up[i][i] for i in range(5))
        np.testing.assert_allclose(trace_up, expected_trace, atol=1e-10)


class TestOccupationDistance:
    """Test suite for compute_occupation_distance function."""
    
    def test_identical_matrices_zero_distance_all_atoms(self, multi_atom_data):
        """Test that two identical occupation matrices have zero distance."""
        occ1 = OccupationMatrixData(multi_atom_data)
        occ2 = OccupationMatrixData(multi_atom_data)
        
        distance = compute_occupation_distance(occ1, occ2)
        
        assert distance == 0.0, f"Expected distance 0.0, got {distance}"
    
    def test_identical_matrices_zero_distance_single_atom(self, multi_atom_data):
        """Test that two identical occupation matrices have zero distance for a single atom."""
        occ1 = OccupationMatrixData(multi_atom_data)
        occ2 = OccupationMatrixData(multi_atom_data)
        
        distance = compute_occupation_distance(occ1, occ2, atom_label='Atom_1')
        
        assert distance == 0.0, f"Expected distance 0.0, got {distance}"
    
    def test_simple_difference_known_distance(self, simple_2x2_identity, simple_2x2_modified):
        """Test distance calculation with matrices that differ by a known amount."""
        occ1 = OccupationMatrixData(simple_2x2_identity)
        occ2 = OccupationMatrixData(simple_2x2_modified)
        
        # Expected distance: sqrt((2.0 - 1.0)^2) = 1.0
        distance = compute_occupation_distance(occ1, occ2)
        expected = 1.0
        
        assert np.isclose(distance, expected), f"Expected distance {expected}, got {distance}"
    
    def test_multiple_atoms_distance(self):
        """Test distance calculation with multiple atoms."""
        data1 = {
            'Atom_1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 1.0]],
                    'down': [[1.0, 0.0], [0.0, 1.0]]
                }
            },
            'Atom_2': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 1.0]],
                    'down': [[1.0, 0.0], [0.0, 1.0]]
                }
            }
        }
        
        data2 = {
            'Atom_1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[2.0, 0.0], [0.0, 1.0]],  # Changed [0,0] from 1.0 to 2.0
                    'down': [[1.0, 0.0], [0.0, 1.0]]
                }
            },
            'Atom_2': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 1.0]],
                    'down': [[1.0, 0.0], [0.0, 2.0]]  # Changed [1,1] from 1.0 to 2.0
                }
            }
        }
        
        occ1 = OccupationMatrixData(data1)
        occ2 = OccupationMatrixData(data2)
        
        # Expected total distance: sqrt((2-1)^2 + (2-1)^2) = sqrt(2)
        distance = compute_occupation_distance(occ1, occ2)
        expected = np.sqrt(2.0)
        
        assert np.isclose(distance, expected), f"Expected distance {expected}, got {distance}"
    
    def test_single_atom_distance_in_multi_atom_data(self):
        """Test that single atom distance works correctly when data contains multiple atoms."""
        data1 = {
            'Atom_1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 1.0]],
                    'down': [[1.0, 0.0], [0.0, 1.0]]
                }
            },
            'Atom_2': {
                'specie': 'Ni',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[5.0, 0.0], [0.0, 5.0]],
                    'down': [[5.0, 0.0], [0.0, 5.0]]
                }
            }
        }
        
        data2 = {
            'Atom_1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[2.0, 0.0], [0.0, 1.0]],
                    'down': [[1.0, 0.0], [0.0, 1.0]]
                }
            },
            'Atom_2': {
                'specie': 'Ni',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[5.0, 0.0], [0.0, 5.0]],
                    'down': [[5.0, 0.0], [0.0, 5.0]]
                }
            }
        }
        
        occ1 = OccupationMatrixData(data1)
        occ2 = OccupationMatrixData(data2)
        
        # Distance for Atom_1 only should be 1.0
        distance_atom1 = compute_occupation_distance(occ1, occ2, atom_label='Atom_1')
        assert np.isclose(distance_atom1, 1.0)
        
        # Distance for Atom_2 should be 0.0
        distance_atom2 = compute_occupation_distance(occ1, occ2, atom_label='Atom_2')
        assert distance_atom2 == 0.0
    
    def test_spin_channel_selection(self, simple_2x2_identity):
        """Test distance calculation with specific spin channels."""
        data2 = {
            'Atom_1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[2.0, 0.0], [0.0, 1.0]],  # Different
                    'down': [[1.0, 0.0], [0.0, 1.0]]  # Same
                }
            }
        }
        
        occ1 = OccupationMatrixData(simple_2x2_identity)
        occ2 = OccupationMatrixData(data2)
        
        # Distance for up-spin only should be 1.0
        distance_up = compute_occupation_distance(occ1, occ2, spins=['up'])
        assert np.isclose(distance_up, 1.0)
        
        # Distance for down-spin only should be 0.0
        distance_down = compute_occupation_distance(occ1, occ2, spins=['down'])
        assert distance_down == 0.0
        
        # Distance for both spins should be 1.0
        distance_both = compute_occupation_distance(occ1, occ2, spins=['up', 'down'])
        assert np.isclose(distance_both, 1.0)
    
    def test_error_on_mismatched_atoms(self, simple_2x2_identity):
        """Test that an error is raised when atom sets don't match."""
        data_different = {
            'Atom_1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 1.0]],
                    'down': [[1.0, 0.0], [0.0, 1.0]]
                }
            },
            'Atom_3': {
                'specie': 'Ni',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 1.0]],
                    'down': [[1.0, 0.0], [0.0, 1.0]]
                }
            }
        }
        
        occ1 = OccupationMatrixData(simple_2x2_identity)
        occ2 = OccupationMatrixData(data_different)
        
        with pytest.raises(ValueError, match="Atom labels don't match"):
            compute_occupation_distance(occ1, occ2)
    
    def test_error_on_mismatched_dimensions(self, simple_2x2_identity):
        """Test that an error is raised when matrix dimensions don't match."""
        data_3x3 = {
            'Atom_1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    'down': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                }
            }
        }
        
        occ1 = OccupationMatrixData(simple_2x2_identity)
        occ2 = OccupationMatrixData(data_3x3)
        
        with pytest.raises(ValueError, match="Matrix dimensions don't match"):
            compute_occupation_distance(occ1, occ2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
