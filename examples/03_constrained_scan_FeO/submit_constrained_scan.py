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
from aiida.orm import Dict, List, Float, JsonableData, load_node
from aiida.engine import submit
from ase.io import read


from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
from lordcapulet.functions.proposal_modes.shared_functionality import apply_random_rotation
from lordcapulet.utils import prepare_tm_info, prepare_hubbard_structure

# ── Load AiiDA profile ──────────────────────────────────────────────────────
aiida.load_profile()

#%%
# ── Build the structure ─────────────────────────────────────────────────────
atoms = read('../FeO.scf.in', format='espresso-in')

tm_atoms, tm_manifolds, tm_dimensions = prepare_tm_info(atoms, table={'Fe'})

print("Tagged transition atoms:", tm_atoms)
print("Corresponding manifolds:", tm_manifolds)
print("Orbital dimensions:", tm_dimensions)
print("Total OSCDFT dimensions:", sum(tm_dimensions))

# u_values can be a single float (same U for all TM sites) or a per-atom
# list with one entry per site, e.g. u_values=[5.0, 4.0]
hubbard_structure = prepare_hubbard_structure(
    atoms, tm_atoms, tm_manifolds, U_values=5.0
)

# ── Reference occupation matrices ───────────────────────────────────────────
# This part here until XXXXX should not be automated and must be kept in the example
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
# XXXXX
code = aiida.orm.load_code('pwx_const@daint-general')  # adjust to your installation

#%%
# ── Option A: fully manual builder (no protocols) ────────────────────────────
# Use this if you want explicit control over every single input without relying
# on the YAML defaults.  n_oscdft must equal the total number of orbital
# channels across all TM sites (sum of tm_dimensions, which is 10 for 2x3d).
# Uncomment the block below and comment out Option B to use it.
#
# from aiida.orm import Dict, KpointsData, List, Float, Str
#
# kpoints = KpointsData()
# kpoints.set_kpoints_mesh([4, 4, 4])
#
# parameters = Dict(dict={
#     'CONTROL': {
#         'calculation': 'scf',
#         'restart_mode': 'from_scratch',
#         'verbosity': 'high',
#     },
#     'SYSTEM': {
#         'ecutwfc': 60.0,
#         'ecutrho': 480.0,
#         'occupations': 'smearing',
#         'smearing': 'cold',
#         'degauss': 0.01,
#         'nspin': 2,
#     },
#     'ELECTRONS': {
#         'conv_thr': 1.0e-8,
#         'mixing_beta': 0.1,
#         'electron_maxstep': 500,
#         'mixing_mode': 'local-TF',
#     },
# })
#
# oscdft_card = Dict(dict={
#     'oscdft_type': 2,
#     'n_oscdft': sum(tm_dimensions),   # 10 for 2 Fe 3d sites
#     'constraint_strength': 1.0,
#     'constraint_conv_thr': 0.005,
#     'constraint_maxstep': 200,
#     'constraint_mixing_beta': 0.4,
# })
#
# builder = ConstrainedScanWorkChain.get_builder()
# builder.code = code
# builder.structure = hubbard_structure
# builder.kpoints = kpoints
# builder.parameters = parameters
# builder.tm_atoms = List(list=tm_atoms)
# builder.oscdft_card = oscdft_card
# builder.occupation_matrices_list = List(list=target_matrix_pks)
# builder.walltime_hours = Float(2.0)
# builder.pseudo_family_string = Str('SSSP/1.3/PBEsol/efficiency')

# ── Option B: protocol-based builder (recommended) ───────────────────────────
# All DFT parameters, k-points, pseudo family, and OSCDFT defaults are loaded
# from the protocol YAMLs.  n_oscdft is computed automatically from tm_atoms.
# Pass `overrides` as a nested dict to change only what you need.
#
# Available overrides (non-exhaustive):
#   'kpoints_distance'              - spacing in 1/Å (default 0.4)
#   'kpoints_mesh'                  - explicit [nx,ny,nz] (bypasses density logic)
#   'walltime_hours'                - hours per calculation (default 2.0)
#   'pseudo_family'                 - aiida-pseudo group label
#   'parameters'                    - nested QE namelist overrides
#   'oscdft_card'                   - override individual OSCDFT parameters
#                                     (n_oscdft is always set automatically)
builder = ConstrainedScanWorkChain.get_builder_from_protocol(
    code=code,
    structure=hubbard_structure,
    tm_atoms=tm_atoms,
    occupation_matrices_list=target_matrix_pks,
    overrides={
        'kpoints_mesh': [3, 3, 3],
        'walltime_hours': 2.0,
        # Uncomment to change OSCDFT settings:
        # 'oscdft_card': {'constraint_strength': 2.0},
    },
)

workchain = submit(builder)

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

# workchain = load_node(190615)  # replace with your actual PK

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
        atom_label = matrix_data[f'Atom_{i+1}']['specie']
        up_matrix = np.array(matrix_data[f'Atom_{i+1}']['occupation_matrix']['up']).reshape(5,5)
        down_matrix = np.array(matrix_data[f'Atom_{i+1}']['occupation_matrix']['down']).reshape(5,5)
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