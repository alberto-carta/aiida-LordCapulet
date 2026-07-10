"""Tests for proposal mode functions (random and random_so_n)."""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
from lordcapulet.functions.proposal_modes.random_mode import propose_random_constraints
from lordcapulet.functions.proposal_modes.random_so_n_mode import propose_random_so_n_constraints


class TestProposalModesMetadataPreservation:
    """Test that proposal modes preserve specie and shell metadata."""
    
    def setup_method(self):
        """Set up test data for each test."""
        # Create sample occupation matrices with metadata
        self.sample_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': (np.eye(5) * 0.5).tolist()
                    }
                },
                'Atom_2': {
                    'specie': 'Fe2',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': (np.eye(5) * 0.8).tolist(),
                        'down': (np.eye(5) * 0.7).tolist()
                    }
                }
            }),
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': (np.eye(5) * 0.9).tolist(),
                        'down': (np.eye(5) * 0.6).tolist()
                    }
                },
                'Atom_2': {
                    'specie': 'Fe2',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': (np.eye(5) * 0.7).tolist(),
                        'down': (np.eye(5) * 0.8).tolist()
                    }
                }
            })
        ]
    
    def test_random_mode_preserves_metadata(self):
        """Test that random mode preserves specie and shell metadata."""
        proposals = propose_random_constraints(
            self.sample_matrices,
            natoms=2,
            N=3,
            debug=False
        )
        
        assert len(proposals) == 3
        
        for proposal in proposals:
            assert isinstance(proposal, OccupationMatrixData)
            
            # Check that metadata is preserved
            assert proposal.data['Atom_1']['specie'] == 'Fe1'
            assert proposal.data['Atom_1']['shell'] == '3d'
            assert proposal.data['Atom_2']['specie'] == 'Fe2'
            assert proposal.data['Atom_2']['shell'] == '3d'
    
    def test_random_so_n_mode_preserves_metadata(self):
        """Test that random_so_n mode preserves specie and shell metadata."""
        proposals = propose_random_so_n_constraints(
            self.sample_matrices,
            natoms=2,
            N=3,
            debug=False
        )
        
        assert len(proposals) == 3
        
        for proposal in proposals:
            assert isinstance(proposal, OccupationMatrixData)
            
            # Check that metadata is preserved
            assert proposal.data['Atom_1']['specie'] == 'Fe1'
            assert proposal.data['Atom_1']['shell'] == '3d'
            assert proposal.data['Atom_2']['specie'] == 'Fe2'
            assert proposal.data['Atom_2']['shell'] == '3d'


class TestRandomModeProposals:
    """Test suite for random mode proposal generation."""
    
    def test_basic_functionality(self):
        """Test basic proposal generation."""
        # Simple input with one atom
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': (np.eye(5) * 0.5).tolist()
                    }
                }
            })
        ]
        
        proposals = propose_random_constraints(occ_matrices, natoms=1, N=5)
        
        assert len(proposals) == 5
        for proposal in proposals:
            assert isinstance(proposal, OccupationMatrixData)
            assert 'Atom_1' in proposal.data
    
    def test_proposal_matrix_properties(self):
        """Test that proposed matrices have correct properties."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Ni1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': np.eye(5).tolist()
                    }
                }
            })
        ]
        
        np.random.seed(42)  # For reproducibility
        proposals = propose_random_constraints(occ_matrices, natoms=1, N=3)
        
        for proposal in proposals:
            up_matrix = np.array(proposal.data['Atom_1']['occupation_matrix']['up'])
            down_matrix = np.array(proposal.data['Atom_1']['occupation_matrix']['down'])
            
            # Check matrix dimensions
            assert up_matrix.shape == (5, 5)
            assert down_matrix.shape == (5, 5)
            
            # Check matrices are real (for collinear calculations)
            assert np.all(np.isreal(up_matrix))
            assert np.all(np.isreal(down_matrix))
            
            # Check Hermiticity (occupation matrices should be Hermitian)
            assert np.allclose(up_matrix, up_matrix.T, atol=1e-10)
            assert np.allclose(down_matrix, down_matrix.T, atol=1e-10)
    
    def test_target_traces_parameter(self):
        """Test using explicit target_traces parameter."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': np.eye(5).tolist()
                    }
                }
            })
        ]
        
        target_traces = [8.0]  # Specify target electron count
        proposals = propose_random_constraints(
            occ_matrices,
            natoms=1,
            N=2,
            target_traces=target_traces,
            randomize_oxidation=False  # Disable randomization for deterministic test
        )
        
        for proposal in proposals:
            up_matrix = np.array(proposal.data['Atom_1']['occupation_matrix']['up'])
            down_matrix = np.array(proposal.data['Atom_1']['occupation_matrix']['down'])
            
            # Check that trace is close to target
            total_trace = np.trace(up_matrix) + np.trace(down_matrix)
            # With randomize_oxidation=False, should be close to 8.0
            assert abs(total_trace - 8.0) < 0.5  # Small tolerance for rounding
    
    def test_multiple_atoms(self):
        """Test proposal generation with multiple atoms."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': (np.eye(5) * 0.8).tolist()
                    }
                },
                'Atom_2': {
                    'specie': 'Ni1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': (np.eye(5) * 0.9).tolist(),
                        'down': (np.eye(5) * 0.9).tolist()
                    }
                }
            })
        ]
        
        proposals = propose_random_constraints(occ_matrices, natoms=2, N=3)
        
        assert len(proposals) == 3
        for proposal in proposals:
            assert 'Atom_1' in proposal.data
            assert 'Atom_2' in proposal.data
            assert proposal.data['Atom_1']['specie'] == 'Fe1'
            assert proposal.data['Atom_2']['specie'] == 'Ni1'
    
    def test_randomness(self):
        """Test that proposals are actually random (not identical)."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': np.eye(5).tolist()
                    }
                }
            })
        ]
        
        np.random.seed(42)
        proposals = propose_random_constraints(occ_matrices, natoms=1, N=5)
        
        # Check that not all proposals are identical
        matrices = [np.array(p.data['Atom_1']['occupation_matrix']['up']) for p in proposals]
        
        all_identical = True
        for i in range(1, len(matrices)):
            if not np.allclose(matrices[0], matrices[i]):
                all_identical = False
                break
        
        assert not all_identical, "All proposals should not be identical"


class TestRandomSONModeProposals:
    """Test suite for random_so_n mode proposal generation."""
    
    def test_basic_functionality(self):
        """Test basic SO(N) proposal generation."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': (np.eye(5) * 0.5).tolist()
                    }
                }
            })
        ]
        
        proposals = propose_random_so_n_constraints(occ_matrices, natoms=1, N=5)
        
        assert len(proposals) == 5
        for proposal in proposals:
            assert isinstance(proposal, OccupationMatrixData)
            assert 'Atom_1' in proposal.data
    
    def test_so_n_matrix_properties(self):
        """Test that SO(N) proposals have correct mathematical properties."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Ni1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': np.eye(5).tolist()
                    }
                }
            })
        ]
        
        np.random.seed(42)
        proposals = propose_random_so_n_constraints(occ_matrices, natoms=1, N=3)
        
        for proposal in proposals:
            up_matrix = np.array(proposal.data['Atom_1']['occupation_matrix']['up'])
            down_matrix = np.array(proposal.data['Atom_1']['occupation_matrix']['down'])
            
            # Check matrix dimensions
            assert up_matrix.shape == (5, 5)
            assert down_matrix.shape == (5, 5)
            
            # Check matrices are real
            assert np.all(np.isreal(up_matrix))
            assert np.all(np.isreal(down_matrix))
            
            # Check Hermiticity
            assert np.allclose(up_matrix, up_matrix.T, atol=1e-10)
            assert np.allclose(down_matrix, down_matrix.T, atol=1e-10)
    
    def test_target_traces_parameter(self):
        """Test using explicit target_traces parameter with SO(N)."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': np.eye(5).tolist()
                    }
                }
            })
        ]
        
        target_traces = [8.0]
        proposals = propose_random_so_n_constraints(
            occ_matrices,
            natoms=1,
            N=2,
            target_traces=target_traces,
            randomize_oxidation=False
        )
        
        for proposal in proposals:
            up_matrix = np.array(proposal.data['Atom_1']['occupation_matrix']['up'])
            down_matrix = np.array(proposal.data['Atom_1']['occupation_matrix']['down'])
            
            # Check that trace is close to target
            total_trace = np.trace(up_matrix) + np.trace(down_matrix)
            assert abs(total_trace - 8.0) < 0.5
    
    def test_multiple_atoms_so_n(self):
        """Test SO(N) proposal generation with multiple atoms."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': (np.eye(5) * 0.8).tolist()
                    }
                },
                'Atom_2': {
                    'specie': 'Ni1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': (np.eye(5) * 0.9).tolist(),
                        'down': (np.eye(5) * 0.9).tolist()
                    }
                }
            })
        ]
        
        proposals = propose_random_so_n_constraints(occ_matrices, natoms=2, N=3)
        
        assert len(proposals) == 3
        for proposal in proposals:
            assert 'Atom_1' in proposal.data
            assert 'Atom_2' in proposal.data
            assert proposal.data['Atom_1']['specie'] == 'Fe1'
            assert proposal.data['Atom_2']['specie'] == 'Ni1'
    
    def test_so_n_randomness(self):
        """Test that SO(N) proposals are random."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': np.eye(5).tolist()
                    }
                }
            })
        ]
        
        # Generate multiple sets of proposals to ensure randomness
        # With 10 proposals generated twice, it's extremely unlikely they'll all be identical
        np.random.seed(42)
        proposals1 = propose_random_so_n_constraints(occ_matrices, natoms=1, N=10)
        
        np.random.seed(43)
        proposals2 = propose_random_so_n_constraints(occ_matrices, natoms=1, N=10)
        
        # Check that at least one proposal differs between the two sets
        matrices1 = [np.array(p.data['Atom_1']['occupation_matrix']['up']) for p in proposals1]
        matrices2 = [np.array(p.data['Atom_1']['occupation_matrix']['up']) for p in proposals2]
        
        # At least one matrix should be different
        some_different = False
        for m1, m2 in zip(matrices1, matrices2):
            if not np.allclose(m1, m2, atol=1e-6):
                some_different = True
                break
        
        assert some_different, "SO(N) proposals should produce different results with different random seeds"


class TestProposalModesConsistency:
    """Test consistency between random and random_so_n modes."""
    
    def test_same_output_structure(self):
        """Test that both modes produce the same output structure."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {
                    'specie': 'Fe1',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.eye(5).tolist(),
                        'down': (np.eye(5) * 0.8).tolist()
                    }
                }
            })
        ]
        
        random_proposals = propose_random_constraints(occ_matrices, natoms=1, N=2)
        so_n_proposals = propose_random_so_n_constraints(occ_matrices, natoms=1, N=2)
        
        # Both should return same structure
        assert len(random_proposals) == len(so_n_proposals) == 2
        
        for rp, sp in zip(random_proposals, so_n_proposals):
            # Same keys
            assert rp.data.keys() == sp.data.keys()
            
            # Same metadata structure
            assert rp.data['Atom_1']['specie'] == sp.data['Atom_1']['specie']
            assert rp.data['Atom_1']['shell'] == sp.data['Atom_1']['shell']
            
            # Same matrix structure (not necessarily same values)
            assert 'up' in rp.data['Atom_1']['occupation_matrix']
            assert 'down' in rp.data['Atom_1']['occupation_matrix']
            assert 'up' in sp.data['Atom_1']['occupation_matrix']
            assert 'down' in sp.data['Atom_1']['occupation_matrix']
    
    def test_both_preserve_atom_count(self):
        """Test that both modes preserve the number of atoms."""
        occ_matrices = [
            OccupationMatrixData({
                'Atom_1': {'specie': 'Fe1', 'shell': '3d', 'occupation_matrix': {'up': np.eye(5).tolist(), 'down': np.eye(5).tolist()}},
                'Atom_2': {'specie': 'Fe2', 'shell': '3d', 'occupation_matrix': {'up': np.eye(5).tolist(), 'down': np.eye(5).tolist()}},
                'Atom_3': {'specie': 'O1', 'shell': '2p', 'occupation_matrix': {'up': np.eye(3).tolist(), 'down': np.eye(3).tolist()}}
            })
        ]
        
        # Note: Different atoms can have different orbital dimensions
        # For now, both modes require all atoms to have same dimension
        # So we'll test with uniform dimensions
        occ_matrices_uniform = [
            OccupationMatrixData({
                'Atom_1': {'specie': 'Fe1', 'shell': '3d', 'occupation_matrix': {'up': np.eye(5).tolist(), 'down': np.eye(5).tolist()}},
                'Atom_2': {'specie': 'Fe2', 'shell': '3d', 'occupation_matrix': {'up': np.eye(5).tolist(), 'down': np.eye(5).tolist()}},
                'Atom_3': {'specie': 'Fe3', 'shell': '3d', 'occupation_matrix': {'up': np.eye(5).tolist(), 'down': np.eye(5).tolist()}}
            })
        ]
        
        random_proposals = propose_random_constraints(occ_matrices_uniform, natoms=3, N=2)
        so_n_proposals = propose_random_so_n_constraints(occ_matrices_uniform, natoms=3, N=2)
        
        for rp in random_proposals:
            assert len(rp.data) == 3
            assert 'Atom_1' in rp.data and 'Atom_2' in rp.data and 'Atom_3' in rp.data
        
        for sp in so_n_proposals:
            assert len(sp.data) == 3
            assert 'Atom_1' in sp.data and 'Atom_2' in sp.data and 'Atom_3' in sp.data


if __name__ == "__main__":
    # Run tests if executed directly
    print("Running proposal mode tests...\n")
    
    # Metadata preservation tests
    print("=== Testing Metadata Preservation ===")
    test_metadata = TestProposalModesMetadataPreservation()
    test_metadata.setup_method()
    
    test_metadata.test_random_mode_preserves_metadata()
    print("Random mode metadata preservation test passed (^_^)")
    
    test_metadata.test_random_so_n_mode_preserves_metadata()
    print("Random SO(N) mode metadata preservation test passed (^_^)")
    
    # Random mode tests
    print("\n=== Testing Random Mode ===")
    test_random = TestRandomModeProposals()
    
    test_random.test_basic_functionality()
    print("Random mode basic functionality test passed (^_^)")
    
    test_random.test_proposal_matrix_properties()
    print("Random mode matrix properties test passed (^_^)")
    
    test_random.test_target_traces_parameter()
    print("Random mode target traces test passed (^_^)")
    
    test_random.test_multiple_atoms()
    print("Random mode multiple atoms test passed (^_^)")
    
    test_random.test_randomness()
    print("Random mode randomness test passed (^_^)")
    
    # SO(N) mode tests
    print("\n=== Testing Random SO(N) Mode ===")
    test_so_n = TestRandomSONModeProposals()
    
    test_so_n.test_basic_functionality()
    print("SO(N) mode basic functionality test passed (^_^)")
    
    test_so_n.test_so_n_matrix_properties()
    print("SO(N) mode matrix properties test passed (^_^)")
    
    test_so_n.test_target_traces_parameter()
    print("SO(N) mode target traces test passed (^_^)")
    
    test_so_n.test_multiple_atoms_so_n()
    print("SO(N) mode multiple atoms test passed (^_^)")
    
    test_so_n.test_so_n_randomness()
    print("SO(N) mode randomness test passed (^_^)")
    
    # Consistency tests
    print("\n=== Testing Consistency Between Modes ===")
    test_consistency = TestProposalModesConsistency()
    
    test_consistency.test_same_output_structure()
    print("Output structure consistency test passed (^_^)")
    
    test_consistency.test_both_preserve_atom_count()
    print("Atom count preservation test passed (^_^)")
    
    print("\nAll proposal mode tests passed! (^_^)")
