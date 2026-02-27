#%%
"""
Constrained scan for FeO: multiple occupation-matrix targets via random rotations.

Starting from one reference occupation matrix (read from a QE output), we
generate a set of diverse initial guesses by applying independent random
SO(3) rotations to each Fe site.  Each rotated matrix is stored as an
OccupationMatrixAiidaData node and its PK is added to the list passed to
ConstrainedScanWorkChain, which then runs a separate ConstrainedPWCalculation
per target and gathers the converged results.
"""

import numpy as np
import aiida
from aiida.orm import (
    Dict, KpointsData, List, Float, load_group, JsonableData, load_node
)
from aiida.engine import submit
from ase.io import read

from aiida_quantumespresso.utils.hubbard import HubbardUtils
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida.orm import StructureData

from lordcapulet.workflows import ConstrainedScanWorkChain
from lordcapulet.data_structures.occupation_matrix import (
    OccupationMatrixData,
    OccupationMatrixAiidaData,
)
from lordcapulet.functions.proposal_modes.shared_functionality import apply_random_rotation
from lordcapulet.utils.preprocessing.submission import (
    tag_and_list_atoms,
    get_default_manifolds,
    get_dimensions,
)

#%%
# ── Load AiiDA profile ──────────────────────────────────────────────────────
aiida.load_profile()

# ── Build the structure ─────────────────────────────────────────────────────
atoms = read('../FeO.scf.in', format='espresso-in')

tm_atoms = tag_and_list_atoms(atoms, table={'Fe'})   # ['Fe1', 'Fe2']
tm_manifolds = get_default_manifolds(tm_atoms)        # ['3d', '3d']
tm_dimensions = get_dimensions(tm_manifolds)          # [5, 5]
total_dimensions = sum(tm_dimensions)                 # 10

print("Tagged transition atoms:", tm_atoms)
print("Corresponding manifolds:", tm_manifolds)
print("Orbital dimensions:", tm_dimensions)
print("Total OSCDFT dimensions:", total_dimensions)

structure = StructureData(ase=atoms)

Uval = 5.0
hubbard_structure = HubbardStructureData.from_structure(structure)
for itm, tm_atom in enumerate(tm_atoms):
    hubbard_structure.initialize_onsites_hubbard(
        atom_name=tm_atom,
        atom_manifold=tm_manifolds[itm],
        value=Uval,
    )
hutils = HubbardUtils(hubbard_structure)
hutils.reorder_atoms()
hubbard_structure = hutils._hubbard_structure

# ── Reference occupation matrices ───────────────────────────────────────────
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
base_matrices = np.stack([up_matrix, down_matrix], axis=0)  # (2, 5, 5)

# ── Generate rotated proposals and store them ────────────────────────────────
# For each proposal we rotate Fe1 and Fe2 *independently* so consecutive
# proposals explore different AF-like configurations.

N_rotations = 8   # total number of constrained calculations to submit; adjust freely

target_matrix_pks = []

for i in range(N_rotations):
    # Independent random rotations for each iron site
    rot_fe1 = apply_random_rotation(base_matrices.copy())   # (2, 5, 5) for Fe1
    rot_fe2 = apply_random_rotation(base_matrices.copy())   # (2, 5, 5) for Fe2

    occ_data = OccupationMatrixData({
        'Atom_1': {
            'specie': 'Fe1',
            'shell': '3d',
            'occupation_matrix': {
                'up':   rot_fe1[0].real.tolist(),   # discard any tiny imaginary part
                'down': rot_fe1[1].real.tolist(),
            },
        },
        'Atom_2': {
            'specie': 'Fe2',
            'shell': '3d',
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

# ── Code and k-points ───────────────────────────────────────────────────────
code = aiida.orm.load_code('pwx_const@daint-general')  # adjust to your installation

kpoints = KpointsData()
kpoints.set_kpoints_mesh([4, 4, 4])

# ── DFT parameters ──────────────────────────────────────────────────────────
parameters = Dict(dict={
    'CONTROL': {
        'calculation': 'scf',
        'restart_mode': 'from_scratch',
        'verbosity': 'high',
    },
    'SYSTEM': {
        'ecutwfc': 60.0,
        'ecutrho': 480.0,
        'occupations': 'smearing',
        'smearing': 'cold',
        'degauss': 0.01,
        'nspin': 2,
        # ConstrainedScanWorkChain sets starting_magnetization per-atom internally
    },
    'ELECTRONS': {
        'conv_thr': 1.0e-8,
        'mixing_beta': 0.1,
        'electron_maxstep': 500,
        'mixing_mode': 'local-TF',
    },
})

# ── OSCDFT card ─────────────────────────────────────────────────────────────
oscdft_card = Dict(dict={
    'oscdft_type': 2,
    'n_oscdft': total_dimensions,       # 10 for 2 × 3d Fe sites
    'constraint_strength': 1.0,
    'constraint_conv_thr': 0.005,
    'constraint_maxstep': 200,
    'constraint_mixing_beta': 0.4,
})

# ── Build and submit ─────────────────────────────────────────────────────────
inputs = {
    'structure': hubbard_structure,
    'parameters': parameters,
    'kpoints': kpoints,
    'code': code,
    'tm_atoms': List(list=tm_atoms),
    'oscdft_card': oscdft_card,
    # Pass the PKs of the stored target matrices as a plain integer List;
    # ConstrainedScanWorkChain.run_all() calls load_node() on each PK.
    'occupation_matrices_list': List(list=target_matrix_pks),
    'walltime_hours': Float(2.0),
}

workchain = submit(ConstrainedScanWorkChain, **inputs)

print(f"\nSubmitted ConstrainedScanWorkChain with PK: {workchain.pk}")
print(f"Monitor with: verdi process status {workchain.pk}")

# Save submission info for post-processing
with open('feo_constrained_scan_info.txt', 'w') as f:
    f.write("=" * 40 + "\n")
    f.write(f"Workchain PK:  {workchain.pk}\n")
    f.write(f"Material:      FeO\n")
    f.write(f"N rotations:   {N_rotations}\n")
    f.write(f"Target PKs:    {target_matrix_pks}\n")
    f.write("=" * 40 + "\n")

# %%

if workchain.process_state.value == 'finished' and workchain.exit_status == 0:
    print("Workchain finished successfully!")

    # getting outputs lists

    converged_pks = workchain.outputs.converged_calculation_pks
    converged_matrix_pks = workchain.outputs.converged_matrix_pks

    # load the first converged occupation matrix and display it

    first_matrix_node = load_node(converged_matrix_pks[0])
    first_calc_node = load_node(converged_pks[0])
    print(f"First converged matrix node PK: {first_matrix_node.pk} from calculation PK: {first_calc_node.pk}")

    matrix_data = first_matrix_node.obj.data
    # here the atom label is a string?

    for i in range(2):
        atom_label = matrix_data[f'{i+1}']['specie']
        up_matrix = np.array(matrix_data[f'{i+1}']['occupation_matrix']['up']).reshape(5,5)
        down_matrix = np.array(matrix_data[f'{i+1}']['occupation_matrix']['down']).reshape(5,5)
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


    







    # get the converged occupation matrix
    # up_matrix = workchain.outputs.converged_occupation_matrix.get_up_matrix()
    # down_matrix = workchain.outputs.converged_occupation_matrix.get_down_matrix()

    # print("Converged occupation matrices:")
    # print("Up spin:")