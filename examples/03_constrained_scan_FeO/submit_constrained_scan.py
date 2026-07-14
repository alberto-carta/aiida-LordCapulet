#%%
"""
Constrained scan for FeO: multiple occupation-matrix targets via random rotations.

Starting from one reference occupation matrix (read from a QE output), we
generate a set of diverse initial guesses by applying independent random
SO(3) rotations to each Fe site. Each rotated matrix is stored as an
OccupationMatrixAiidaData node and its PK is added to the list passed to
ConstrainedScanWorkChain.

Caching: result JSON ({material}_constrained_scan.json) acts as the cache
marker. On first submit a stub JSON with just the workchain PK is written; the
extractor cell below overwrites it with full results. Re-running reloads the PK
from the JSON and skips submission (and skips regenerating the rotated target
matrices). Delete the JSON to force a fresh submission.
"""

import json
import os
import numpy as np
import aiida
from aiida.orm import JsonableData, load_node
from aiida.engine import submit
from ase.io import read

from lordcapulet.workflows import ConstrainedScanWorkChain
from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
from lordcapulet.functions.proposal_modes.shared_functionality import apply_random_rotation
from lordcapulet.utils import prepare_hubbard_corr_info, prepare_hubbard_structure

# ── Load AiiDA profile ──────────────────────────────────────────────────────
aiida.load_profile()

material_name = 'FeO'
JSON_FILE = f"{material_name}_constrained_scan.json"
N_rotations = 8


def _read_pk_from_json(path):
    with open(path) as f:
        return json.load(f)['metadata']['pk']


#%%
if os.path.exists(JSON_FILE):
    workchain_pk = _read_pk_from_json(JSON_FILE)
    workchain = load_node(workchain_pk)
    print(f"Found existing {JSON_FILE}, skipping submit. PK={workchain_pk}")
else:
    # ── Build the structure ─────────────────────────────────────────────────
    atoms = read('../FeO.scf.in', format='espresso-in')

    hubbard_corr_atoms, hubbard_corr_manifolds, hubbard_corr_dimensions = prepare_hubbard_corr_info(atoms, table={'Fe'})

    print("Tagged transition atoms:", hubbard_corr_atoms)
    print("Corresponding manifolds:", hubbard_corr_manifolds)
    print("Orbital dimensions:", hubbard_corr_dimensions)
    print("Total OSCDFT dimensions:", sum(hubbard_corr_dimensions))

    hubbard_structure = prepare_hubbard_structure(
        atoms, hubbard_corr_atoms, hubbard_corr_manifolds, U_values=5.0
    )

    # ── Reference occupation matrices ───────────────────────────────────────
    # These come from the same QE output used in the single-calculation example.
    # We treat them as the "seed" and generate diverse proposals by rotating them.
    up_matrix = np.array([
        [ 0.216, -0.129, -0.129,  0.000, -0.258],
        [-0.129,  0.550, -0.260, -0.223,  0.260],
        [-0.129, -0.260,  0.550,  0.223,  0.260],
        [ 0.000, -0.223,  0.223,  0.216, -0.000],
        [-0.258,  0.260,  0.260, -0.000,  0.550],
    ])
    down_matrix = np.array([
        [ 0.755,  0.037,  0.037, -0.000,  0.075],
        [ 0.037,  0.968,  0.013,  0.065, -0.013],
        [ 0.037,  0.013,  0.968, -0.065, -0.013],
        [-0.000,  0.065, -0.065,  0.755, -0.000],
        [ 0.075, -0.013, -0.013, -0.000,  0.968],
    ])

    # apply_random_rotation expects shape (2, dim, dim): [spin, orb, orb]
    base_matrices = np.stack([up_matrix, down_matrix], axis=0)

    # ── Generate rotated proposals and store them ───────────────────────────
    target_matrix_pks = []
    for i in range(N_rotations):
        rot_fe1 = apply_random_rotation(base_matrices.copy())
        rot_fe2 = apply_random_rotation(base_matrices.copy())

        occ_data = OccupationMatrixData({
            'Atom_1': {
                'specie': 'Fe1', 'shell': '3d',
                'occupation_matrix': {
                    'up':   rot_fe1[0].real.tolist(),
                    'down': rot_fe1[1].real.tolist(),
                },
            },
            'Atom_2': {
                'specie': 'Fe2', 'shell': '3d',
                'occupation_matrix': {
                    'up':   rot_fe2[0].real.tolist(),
                    'down': rot_fe2[1].real.tolist(),
                },
            },
        })
        node = JsonableData(occ_data)
        node.store()
        target_matrix_pks.append(node.pk)
        print(f"  Proposal {i+1}/{N_rotations}: stored OccupationMatrixAiidaData PK={node.pk}")

    print(f"\nStored {len(target_matrix_pks)} target matrices: {target_matrix_pks}")

    code = aiida.orm.load_code('pw-7.5-fix@prn')  # Adjust to your code

    # Protocol-based builder. Overrides (non-exhaustive):
    #   'kpoints_distance'              - spacing in 1/Å (default 0.4)
    #   'kpoints_mesh'                  - explicit [nx,ny,nz]
    #   'walltime_hours'                - hours per calculation (default 2.0)
    #   'pseudo_family'                 - aiida-pseudo group label
    #   'parameters'                    - nested QE namelist overrides
    #   'oscdft_card'                   - override individual OSCDFT parameters
    builder = ConstrainedScanWorkChain.get_builder_from_protocol(
        code=code,
        structure=hubbard_structure,
        hubbard_corr_atoms=hubbard_corr_atoms,
        occupation_matrices_list=target_matrix_pks,
        overrides={
            'kpoints_mesh': [3, 3, 3],
            'walltime_hours': 2.0,
        },
    )

    workchain = submit(builder)
    workchain_pk = workchain.pk

    print(f"\nSubmitted ConstrainedScanWorkChain with PK: {workchain_pk}")
    print(f"Monitor with: verdi process status {workchain_pk}")

    # Stub JSON so reruns find the PK before extraction completes.
    with open(JSON_FILE, 'w') as f:
        json.dump(
            {'metadata': {
                'pk': workchain_pk,
                'material': material_name,
                'n_rotations': N_rotations,
                'target_matrix_pks': target_matrix_pks,
                'status': 'submitted',
            }},
            f, indent=2,
        )


# %%
if workchain.process_state.value == 'finished' and workchain.exit_status == 0:
    print("Workchain finished successfully!")

    converged_pks = workchain.outputs.converged_calculation_pks
    converged_matrix_pks = workchain.outputs.converged_matrix_pks

    first_matrix_node = load_node(converged_matrix_pks[0])
    first_calc_node = load_node(converged_pks[0])
    print(f"First converged matrix node PK: {first_matrix_node.pk} from calculation PK: {first_calc_node.pk}")

    matrix_data = first_matrix_node.obj.data
    for i in range(2):
        atom_label = matrix_data[f'Atom_{i+1}']['specie']
        up_matrix = np.array(matrix_data[f'Atom_{i+1}']['occupation_matrix']['up']).reshape(5, 5)
        down_matrix = np.array(matrix_data[f'Atom_{i+1}']['occupation_matrix']['down']).reshape(5, 5)
        print(f"\nConverged occupation matrix for {atom_label}:")
        with np.printoptions(precision=3, suppress=True):
            print("Up spin:")
            print(up_matrix)
            print("Down spin:")
            print(down_matrix)
else:
    print("Workchain did not finish successfully.")
    print(f"Process state: {workchain.process_state.value}")
    print(f"Exit status: {workchain.exit_status}")
    if workchain.exit_message:
        print(f"Exit message: {workchain.exit_message}")


# %%
from lordcapulet.utils.postprocessing.gather_workchain_data import WorkchainDataExtractor
aiida.load_profile()

extractor = WorkchainDataExtractor(perform_so_n=True,
                            sanity_check_reconstruct=True,
                            debug=True)

data = extractor.extract_from_workchain(workchain_pk)
extractor.save_to_json(data, JSON_FILE)
