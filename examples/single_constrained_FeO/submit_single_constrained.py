#%%
"""
Single constrained DFT+U calculation for FeO.

We hard-code one occupation matrix (same for both Fe1 and Fe2) that was
read from a Quantum ESPRESSO output file, store it as an
OccupationMatrixAiidaData node in the AiiDA database, and submit a single
ConstrainedPWCalculation that drives the electronic density towards that target.
"""

import numpy as np
import aiida
from aiida.orm import (
    Dict, KpointsData, List, Float, load_group, JsonableData
)
from aiida.engine import submit
from ase.io import read

from aiida_quantumespresso.utils.hubbard import HubbardUtils
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida.orm import StructureData

from lordcapulet.calculations.constrained_pw import ConstrainedPWCalculation
from lordcapulet.data_structures.occupation_matrix import (
    OccupationMatrixData,
    OccupationMatrixAiidaData,
)
from lordcapulet.utils.preprocessing.submission import (
    tag_and_list_atoms,
    get_default_manifolds,
    get_dimensions,
)

# ── Load AiiDA profile ──────────────────────────────────────────────────────
aiida.load_profile()

# ── Build the structure ─────────────────────────────────────────────────────
# FeO.scf.in contains two non-equivalent iron sites (Fe1, Fe2) and oxygen
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

# ── Target occupation matrices ──────────────────────────────────────────────
# Both Fe1 and Fe2 are initialised with the same matrix

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

# ── Wrap the matrices in OccupationMatrixData ───────────────────────────────
# The internal format expects keys 'Atom_1', 'Atom_2', ... matching the sort
# order used by to_constrained_matrix_format() (i.e. by atom index).
# Atom_1 → Fe1, Atom_2 → Fe2 (order follows the reordered Hubbard structure).

occ_data = OccupationMatrixData({
    'Atom_1': {
        'specie': 'Fe1',
        'shell': '3d',
        'occupation_matrix': {
            'up':   up_matrix.tolist(),
            'down': down_matrix.tolist(),
        },
    },
    'Atom_2': {
        'specie': 'Fe2',
        'shell': '3d',
        'occupation_matrix': {
            'up':   up_matrix.tolist(),
            'down': down_matrix.tolist(),
        },
    },
})

# Store the node in the AiiDA database so ConstrainedPWCalculation can load it
target_matrix_node = JsonableData(occ_data)
target_matrix_node.store()
print(f"Stored target OccupationMatrixAiidaData with PK: {target_matrix_node.pk}")

# ── Code and k-points ───────────────────────────────────────────────────────
code = aiida.orm.load_code('pwx_const@daint-general')  # adjust to your installation

kpoints = KpointsData()
kpoints.set_kpoints_mesh([3, 3, 3])

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
        # starting_magnetization is set below on the builder copy
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
    'n_oscdft': total_dimensions,       # total number of constrained orbitals (10 for 2×3d)
    'constraint_strength': 1.0,
    'constraint_conv_thr': 0.005,
    'constraint_maxstep': 200,
    'constraint_mixing_beta': 0.4,
})

# ── Build and submit ─────────────────────────────────────────────────────────
builder = ConstrainedPWCalculation.get_builder()
builder.code = code
builder.structure = hubbard_structure
builder.parameters = parameters
builder.kpoints = kpoints

pseudo_family = load_group('SSSP/1.3/PBEsol/efficiency')
builder.pseudos = pseudo_family.get_pseudos(structure=hubbard_structure)

# Set a near-zero starting magnetization so the constrained field drives
# the occupations rather than the initial spin guess
magnetization_config = {tm: 1e-9 for tm in tm_atoms}
builder.parameters['SYSTEM']['starting_magnetization'] = magnetization_config

builder.oscdft_card = oscdft_card
builder.target_matrix = target_matrix_node   # OccupationMatrixAiidaData

builder.metadata = {
    'options': {
        'resources': {'num_machines': 1},
        'withmpi': True,
        'max_wallclock_seconds': int(2.0 * 3600),  # 2 hours
    }
}

builder.settings = Dict(dict={
    'parser_options': {'parse_atomic_occupations': True},
})
#%%
calc = submit(builder)
print(f"Submitted ConstrainedPWCalculation with PK: {calc.pk}")
print(f"Monitor with: verdi process status {calc.pk}")

# %%

if calc.process_state.value == 'finished' and calc.exit_status == 0:
    print("Calculation finished successfully!")

    # get the converged occupation matrix and 

    final_occupations = OccupationMatrixData.from_aiida_qe_occupations(calc.tools.get_occupations()).data
    # here the atom label is an integer?



    # move back to array and print the final occupation matrix for Fe1 and Fe2
    # reshape it to (5, 5) for better readability, since we know it's a 3d shell
    with np.printoptions(precision=3, suppress=True):
        for i in range(2):  # Loop over Fe1 and Fe2
            final_matrix = final_occupations[f"{i+1}"]['occupations']
            atom_label = final_occupations[f"{i+1}"]['specie']
            print(f"\nFinal occupation matrix for {atom_label}:")
            up_matrix = np.array(final_matrix['up']).reshape((5, 5))
            down_matrix = np.array(final_matrix['down']).reshape((5, 5))
            print("Up spin:")
            print(up_matrix)
            print("Down spin:")
            print(down_matrix)
else:
    print("Calculation did not finish successfully.")

    # print error message if available
    print(f"Process state: {calc.process_state.value}")
    print(f"Exit status: {calc.exit_status}")
    if calc.exit_message:
        print(f"Exit message: {calc.exit_message}")
