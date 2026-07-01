#%%
"""
Single constrained DFT+U calculation for FeO.

We hard-code one occupation matrix (same for both Fe1 and Fe2) that was
read from a Quantum ESPRESSO output file, store it as an
OccupationMatrixAiidaData node in the AiiDA database, and submit a single
ConstrainedPWCalculation that drives the electronic density towards that target.

Caching: a minimal {material}_single.json (just the calc PK + target matrix
PK) is written on first submit and used as the cache marker. Re-running reloads
the existing calc from the DB instead of submitting a new one. Delete the JSON
to force a fresh submission.
"""

import json
import os
import numpy as np
import aiida
from aiida.orm import (
    Dict, KpointsData, load_group, JsonableData, load_node
)
from aiida.engine import submit
from ase.io import read

from lordcapulet.calculations.constrained_pw import ConstrainedPWCalculation
from lordcapulet.data_structures.occupation_matrix import (
    OccupationMatrixData, extract_occupations_from_calc,
)
from lordcapulet.utils import prepare_tm_info, prepare_hubbard_structure

# ── Load AiiDA profile ──────────────────────────────────────────────────────
aiida.load_profile()

material_name = 'FeO'
JSON_FILE = f"{material_name}_single.json"


def _read_pk_from_json(path):
    with open(path) as f:
        return json.load(f)['metadata']['pk']


#%%
if os.path.exists(JSON_FILE):
    calc_pk = _read_pk_from_json(JSON_FILE)
    calc = load_node(calc_pk)
    print(f"Found existing {JSON_FILE}, skipping submit. PK={calc_pk}")
else:
    # ── Build the structure ─────────────────────────────────────────────────
    atoms = read('../FeO.scf.in', format='espresso-in')

    hubbard_corr_atoms, hubbard_corr_manifolds, hubbard_corr_dimensions = prepare_tm_info(atoms, table={'Fe'})
    total_dimensions = sum(hubbard_corr_dimensions)

    print("Tagged transition atoms:", hubbard_corr_atoms)
    print("Corresponding manifolds:", hubbard_corr_manifolds)
    print("Orbital dimensions:", hubbard_corr_dimensions)
    print("Total OSCDFT dimensions:", total_dimensions)

    hubbard_structure = prepare_hubbard_structure(
        atoms, hubbard_corr_atoms, hubbard_corr_manifolds, U_values=5.0
    )

    # ── Target occupation matrices (same for Fe1 and Fe2) ───────────────────
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

    occ_data = OccupationMatrixData({
        'Atom_1': {
            'specie': 'Fe1', 'shell': '3d',
            'occupation_matrix': {'up': up_matrix.tolist(), 'down': down_matrix.tolist()},
        },
        'Atom_2': {
            'specie': 'Fe2', 'shell': '3d',
            'occupation_matrix': {'up': up_matrix.tolist(), 'down': down_matrix.tolist()},
        },
    })

    target_matrix_node = JsonableData(occ_data)
    target_matrix_node.store()
    print(f"Stored target OccupationMatrixAiidaData with PK: {target_matrix_node.pk}")

    # ── Code and k-points ───────────────────────────────────────────────────
    code = aiida.orm.load_code('pw-7.5-fix@prn')  # adjust to your installation
    # code = aiida.orm.load_code('pw-occ-fix@eiger-uenv-gnu-25.6')  # adjust to your installation
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([3, 3, 3])

    parameters = Dict(dict={
        'CONTROL': {
            'calculation': 'scf', 'restart_mode': 'from_scratch', 'verbosity': 'high',
        },
        'SYSTEM': {
            'ecutwfc': 60.0, 'ecutrho': 480.0,
            'occupations': 'smearing', 'smearing': 'cold', 'degauss': 0.01,
            'nspin': 2,
        },
        'ELECTRONS': {
            'conv_thr': 1.0e-8, 'mixing_beta': 0.1,
            'electron_maxstep': 500, 'mixing_mode': 'local-TF',
        },
    })

    oscdft_card = Dict(dict={
        'oscdft_type': 2,
        'n_oscdft': total_dimensions,
        'constraint_strength': 1.0,
        'constraint_conv_thr': 0.005,
        'constraint_maxstep': 200,
        'constraint_mixing_beta': 0.4,
    })

    builder = ConstrainedPWCalculation.get_builder()
    builder.code = code
    builder.structure = hubbard_structure
    builder.parameters = parameters
    builder.kpoints = kpoints

    pseudo_family = load_group('SSSP/1.3/PBEsol/efficiency')
    builder.pseudos = pseudo_family.get_pseudos(structure=hubbard_structure)

    magnetization_config = {tm: 1e-9 for tm in hubbard_corr_atoms}
    builder.parameters['SYSTEM']['starting_magnetization'] = magnetization_config

    builder.oscdft_card = oscdft_card
    builder.target_matrix = target_matrix_node

    builder.metadata = {
        'options': {
            'resources': {'num_machines': 1},
            'withmpi': True,
            'max_wallclock_seconds': int(2.0 * 3600),
        }
    }
    builder.settings = Dict(dict={'parser_options': {'parse_atomic_occupations': True}})

    calc = submit(builder)
    calc_pk = calc.pk
    print(f"Submitted ConstrainedPWCalculation with PK: {calc_pk}")
    print(f"Monitor with: verdi process status {calc_pk}")

    with open(JSON_FILE, 'w') as f:
        json.dump(
            {'metadata': {
                'pk': calc_pk,
                'material': material_name,
                'target_matrix_pk': target_matrix_node.pk,
                'status': 'submitted',
            }},
            f, indent=2,
        )

# %%
if calc.process_state.value == 'finished' and calc.exit_status == 0:
    print("Calculation finished successfully!")

    # extract_occupations_from_calc handles AiiDA-QE API variants and falls
    # back to parsing the HUBBARD OCCUPATIONS block from QE stdout.
    final_occupations = extract_occupations_from_calc(calc).data
    target_occupations = calc.inputs.target_matrix.obj.data

    with np.printoptions(precision=3, suppress=True):
        for atom_label, atom_data in final_occupations.items():
            specie = atom_data['specie']
            m_final = atom_data['occupation_matrix']
            m_target = target_occupations[atom_label]['occupation_matrix']
            print(f"\n=== {atom_label} ({specie}) ===")
            for spin in ('up', 'down'):
                target = np.array(m_target[spin])
                final = np.array(m_final[spin])
                print(f"  target {spin}:")
                print(target)
                print(f"  final {spin}:")
                print(final)
                print(f"  |final - target| (Frobenius): "
                      f"{np.linalg.norm(final - target):.4f}")
else:
    print("Calculation did not finish successfully.")
    print(f"Process state: {calc.process_state.value}")
    print(f"Exit status: {calc.exit_status}")
    if calc.exit_message:
        print(f"Exit message: {calc.exit_message}")
