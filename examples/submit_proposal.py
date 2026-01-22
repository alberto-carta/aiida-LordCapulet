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
workchain_node =  load_node(116434) # NiO 40 points

# workchain_node = load_node(117081) # CoO second run, 500 points with so(n) enabled


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

aiida_propose_occ_matrices_from_results = propose_module.aiida_propose_occ_matrices_from_results

proposals = aiida_propose_occ_matrices_from_results(
    occ_matr_pks=conv_matrices_pks,
    calc_pks=conv_calc_pks,
    N=59,
    debug=False,
    mode='gp'
)


# %%
