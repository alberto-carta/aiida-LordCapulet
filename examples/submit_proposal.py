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
workchain_node =  load_node(116434)

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

#%% random proposal test

proposals = aiida_propose_occ_matrices_from_results(
    occ_matr_pks=conv_matrices_pks,
    calc_pks=conv_calc_pks,
    N=5,
    debug=False,
    mode='random'
)

#%% random so_n proposal test
from importlib import reload
import lordcapulet.functions.propose as propose_module

reload(propose_module)

aiida_propose_occ_matrices_from_results = propose_module.aiida_propose_occ_matrices_from_results


proposals = aiida_propose_occ_matrices_from_results(
    occ_matr_pks=conv_matrices_pks,
    calc_pks=conv_calc_pks,
    N=5,
    debug=False,
    mode='gp'
)

