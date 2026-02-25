#%%
# import all aiida boilerplate
from aiida import load_profile
load_profile()
from aiida.orm import Dict, Code, KpointsData, load_node, Dict, List, Int, Float, Str
from aiida.engine import WorkChain, run
import json 
import numpy as np



# from lordcapulet_functions.propose import aiida_propose_occ_matrices_from_results, propose_new_constraints
from lordcapulet.functions.proposal_modes import propose_random_constraints
from lordcapulet.functions.propose import aiida_propose_occ_matrices_from_results

# load a ConstrainedScanWorkChain
# nodes are hardocded for testing purposes
# workchain_node =  load_node(116434) # NiO 40 points

workchain_node = load_node(117081) # CoO second run, 500 points with so(n) enabled


all_calc_pks = workchain_node.outputs.all_calculation_pks
print(f"Length of all calculation pks: {len(all_calc_pks.get_list())}")

conv_calc_pks = workchain_node.outputs.converged_calculation_pks
print(f"Length of converged calculation pks: {len(conv_calc_pks.get_list())}")

conv_matrices_pks = workchain_node.outputs.converged_matrix_pks
print(f"Length of converged matrices pks: {len(conv_matrices_pks.get_list())}")


# look in the extras of the calculation node 

print("Sanity check: verifying that converged calculation PKs have matching occupation matrix PKs in extras.")

for ipk, pk in enumerate(conv_calc_pks.get_list()):
    calc_node = load_node(pk)
    matrix_in_extra = calc_node.base.extras.get('occupation_matrix_pk', None)

    assert matrix_in_extra == conv_matrices_pks.get_list()[ipk], f"Mismatch in matrix PK for calculation PK {pk}"

print("All converged calculation PKs have matching occupation matrix PKs in extras.")


# calculation pk
single_calc = load_node(116757)
single_calc_energy = single_calc.outputs.output_parameters.get_dict()['energy']

print(f"Single calculation energy: {single_calc_energy} eV")

#%% random proposal test

proposals = aiida_propose_occ_matrices_from_results(
    occ_matr_pks=conv_matrices_pks,
    calc_pks=conv_calc_pks,
    N=5,
    debug=False,
    mode='random_so_n'
)

#%% random so_n proposal test
from importlib import reload
import lordcapulet.functions.propose as propose_module

reload(propose_module)

aiida_propose_occ_matrices_from_results = propose_module.aiida_propose_occ_matrices_from_results



# test creating databank from occ matrices and energies
from lordcapulet.data_structures import DataBank

# Load occupation matrices from PKs
occ_matrices = []
for pk in conv_matrices_pks.get_list():
    node = load_node(pk)
    occ_matrices.append(node.obj)  # JsonableData node containing OccupationMatrixData

# Extract energies from calculation PKs
energies = []
for pk in conv_calc_pks.get_list():
    calc = load_node(pk)
    energy = calc.outputs.output_parameters.get_dict()['energy']
    energies.append(energy)

# Create DataBank from matrices and energies
databank = DataBank.from_matrices(occ_matrices, energies, pks=conv_calc_pks.get_list())

print(f"Created DataBank with {len(databank)} calculations")
print(databank.summary())
print(f"Energy range: {databank.energies.min():.4f} to {databank.energies.max():.4f} eV")

#%%
kwargs_internal = {}

occ_matrices = []
for pk in conv_matrices_pks.get_list():
    node = load_node(pk)
    
    # Handle JsonableData nodes containing OccupationMatrixData (preferred)
    if hasattr(node, 'obj') and hasattr(node.obj, 'as_dict'):
        # This is a JsonableData node containing our OccupationMatrixData
        occupation_matrix_data = node.obj
        occ_matrices.append(occupation_matrix_data)

energies = [ load_node(pk).outputs.output_parameters.get_dict().get('energy') for pk in conv_calc_pks.get_list() ]

kwargs_internal['energies'] = energies 
#%%
import lordcapulet.functions.proposal_modes.gaussian_process as gp_proposal

from importlib import reload

reload(gp_proposal)

propose = gp_proposal.propose_gaussian_process_constraints


proposals_gp = propose(
    occ_matr_list=occ_matrices[:],
    N=50,
    natoms=2,
    debug=True,
    energies=energies[:],
    reporter=print,
    device='cuda'
)




#%%
import lordcapulet.functions.propose as propose_module
from importlib import reload
reload(propose_module)

# add an aiida dictionary with a random number
useless_kwargs = Dict(dict={'random': np.random.randint(1, 1_000_000)})


aiida_propose_occ_matrices_from_results = propose_module.aiida_propose_occ_matrices_from_results

proposals = aiida_propose_occ_matrices_from_results(
    occ_matr_pks=conv_matrices_pks,
    calc_pks=conv_calc_pks,
    N=50,
    debug=False,
    mode='gp',
    useless_kwargs=useless_kwargs
)



#%% get the list of matrices from the proposals pks

import matplotlib.pyplot as plt

matrices = [ load_node(pk).obj for pk in proposals ]

# calculate cell mmagnetization for all matrices

cell_magnetizations = [matrix.get_magnetic_moment('1')+matrix.get_magnetic_moment('2') for matrix in matrices]

matrix = matrices[12]

for atom_id in ['1', '2']:
    up_mat = np.array(matrix.get_occupation_matrix(atom_id, 'up'))
    down_mat = np.array(matrix.get_occupation_matrix(atom_id, 'down'))

    print(f"\n{atom_id}:")
    print(f"  Trace: {np.trace(up_mat) + np.trace(down_mat):.3f}")
    print(f"  Eigenvalues (up): {np.round(np.linalg.eigvalsh(up_mat), 3)}")
    print(f"  Eigenvalues (down): {np.round(np.linalg.eigvalsh(down_mat), 3)}")
    print(f" Moment: {np.trace(up_mat) - np.trace(down_mat):.3f}")    


    # Visualize
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    for idx, (mat, title) in enumerate([(up_mat, 'Up'), (down_mat, 'Down')]):
        im = axs[idx].imshow(np.abs(mat), cmap='viridis', vmin=0, vmax=1)
        axs[idx].set_title(f'{atom_id} {title}')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                axs[idx].text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', color='w')
        fig.colorbar(im, ax=axs[idx])
    
    plt.tight_layout()
    plt.show()

# print cell magnetizations in rows of 5
cell_magnetizations = np.array(cell_magnetizations)
with np.printoptions(precision=3, suppress=True):
    print("Cell Magnetizations for Proposed Matrices:")
    for i in range(0, len(cell_magnetizations), 5):
        print(cell_magnetizations[i:i+5])

# %% use the proposals to submit a new ConstrainedScanWorkChain
import aiida
from aiida.orm import Code, Dict, StructureData, KpointsData, List, Int, Str, Bool, Float, load_node
from aiida.engine import submit
from lordcapulet.workflows import ConstrainedScanWorkChain
from lordcapulet.utils.preprocessing.submission import tag_and_list_atoms, get_default_manifolds, get_dimensions
# import HubbardUtils to rearrange atoms
from aiida_quantumespresso.utils.hubbard import HubbardUtils
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from ase.io import read


# Load AiiDA profile
aiida.load_profile()

# Load structure (adapt this to your system)
atoms = read('CoO.scf.in', format='espresso-in')  # Adjust path as needed


tm_atoms = tag_and_list_atoms(atoms, table={'Co'})
tm_manifolds = get_default_manifolds(tm_atoms)
tm_dimensions = get_dimensions(tm_manifolds) 


structure = StructureData(ase=atoms)
Uval = 5.0  # Example value for Hubbard U
hubbard_structure = HubbardStructureData.from_structure(structure)
for itm, tm_atom in enumerate(tm_atoms):
    hubbard_structure.initialize_onsites_hubbard(
        atom_name=tm_atom,
        # atom_manifold="3d",
        atom_manifold="3d",
        value=Uval  )  # Example: incrementing

# Convert to HubbardStructureData if needed

# Load computational resources
code = aiida.orm.load_code('pwx_const_debug@daint-debug')  # Adjust to your code

# Set up k-points
kpoints = KpointsData()
kpoints.set_kpoints_mesh([6, 6, 6])  # Adjust as needed



total_dimensions = sum(tm_dimensions)

# Define DFT parameters
parameters = Dict(dict={
    'CONTROL': {
        'calculation': 'scf',
        'restart_mode': 'from_scratch',
        'verbosity': 'high',
    },
    'SYSTEM': {
        'ecutwfc': 80.0,    # Adjust as needed
        'ecutrho': 640.0,   # Adjust as needed
        'occupations': 'smearing',
        'smearing': 'gaussian',
        'degauss': 0.01,
        'nspin': 2,
        # Add other system parameters as needed
    },
    'ELECTRONS': {
        'conv_thr': 1.0e-8,
        'mixing_beta': 0.3,
        'electron_maxstep': 500,
    },
})



# OSCDFT parameters
oscdft_card = Dict(dict={
    'oscdft_type': 2,
    'n_oscdft': total_dimensions,
    'constraint_strength': 1.0,
    'constraint_conv_thr': 0.005,
    'constraint_maxstep': 200,
    'constraint_mixing_beta': 0.4,
})


inputs_constrained = {
    'structure': hubbard_structure,
    'parameters': parameters,
    'kpoints': kpoints,
    'code': code,
    'tm_atoms': List(list=tm_atoms),
    'oscdft_card': oscdft_card,
    'occupation_matrices_list': proposals,
}

#%%
# Submit the ConstrainedScanWorkChain
workchain = submit(ConstrainedScanWorkChain, **inputs_constrained)
print(f"Submitted ConstrainedScanWorkChain with PK: {workchain.pk}")
# %%
