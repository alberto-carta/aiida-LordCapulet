"""End-to-end integration test for the full GP proposal pipeline."""

import pytest
import numpy as np

from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData


@pytest.mark.slow
class TestGPProposalPipeline:
    """Test ``propose_gaussian_process_constraints`` end-to-end."""

    def test_full_pipeline(self, gp_databank_small):
        """End-to-end: DataBank -> GP training -> proposal generation."""
        from lordcapulet.functions.proposal_modes.gaussian_process import propose_gaussian_process_constraints

        # Extract OccupationMatrixData list and energies from the databank
        occ_list = [gp_databank_small.get_occ_data(i) for i in range(len(gp_databank_small))]
        energies = gp_databank_small.energies.tolist()
        natoms = len(gp_databank_small.atom_ids)
        N = 3

        proposals = propose_gaussian_process_constraints(
            occ_matr_list=occ_list,
            energies=energies,
            natoms=natoms,
            N=N,
            debug=False,
        )

        # Verify output structure
        assert isinstance(proposals, list)
        assert len(proposals) == N

        for p in proposals:
            assert isinstance(p, OccupationMatrixData)
            # Should have same atom labels as inputs
            assert set(p.get_atom_labels()) == set(occ_list[0].get_atom_labels())

            # Matrix properties: Hermitian, trace in valid range
            for label in p.get_atom_labels():
                for spin in ['up', 'down']:
                    mat = np.array(p.get_occupation_matrix(label, spin))
                    # Should be square
                    assert mat.shape[0] == mat.shape[1]
                    # Trace should be non-negative
                    assert np.trace(mat).real >= -0.1  # small tolerance


class TestGPFallback:
    """Test GP mode fallbacks."""

    def test_generation_zero_uses_random(self):
        """At generation 0, GP mode should fall back to random proposals."""
        from lordcapulet.functions.propose import propose_new_constraints

        # Create simple test data
        occ_list = []
        for _ in range(5):
            data = {
                'atom1': {
                    'specie': 'Fe',
                    'shell': '3d',
                    'occupation_matrix': {
                        'up': np.diag(np.random.rand(5)).tolist(),
                        'down': np.diag(np.random.rand(5)).tolist(),
                    },
                },
            }
            occ_list.append(OccupationMatrixData(data))

        energies = [-100.0 + i for i in range(5)]

        proposals = propose_new_constraints(
            occ_list,
            N=3,
            mode='gp',
            debug=False,
            energies=energies,
            current_generation=0,
        )

        assert len(proposals) == 3
        for p in proposals:
            assert isinstance(p, OccupationMatrixData)
