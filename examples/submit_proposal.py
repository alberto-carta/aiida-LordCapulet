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

# load a AFMworkchain
# nodes are hardocded for testing purposes
workchain_output =  load_node(1198)

pk_list = workchain_output.get_list()

# for each pk load the node and get the dictionary out and save it to a list

resulting_occ_matrices = []
for pk in pk_list:
    node = load_node(pk)
    if node.__class__.__name__ == "Dict":
        occupation_matrix = node.get_dict()
        resulting_occ_matrices.append(occupation_matrix)

resulting_occ_matrices[2]


        
    
#%%
#load the json manually
with open('NiO_mixing_lTF_beta0.3_oscdft_data.json', 'r') as f:
    loaded_data = json.load(f)

# gett occupation matrices from the first dictionary

occ1 = loaded_data[0]


# prop_list =  propose_new_constraints(resulting_occ_matrices, 2, debug=True,mode='read' ,readfile='NiO_mixing_lTF_beta0.3_oscdft_data.json')
prop_list =  propose_random_constraints(resulting_occ_matrices, mode='random', natoms=2, N=10, debug=True)

#%%
from aiida.engine import run_get_node
# Example usage
out_PK_list = 6113 # output of the AFMScanWorkChain or ConstrainedScanWorkChain
workchain_output =  load_node(out_PK_list)
pk_list = workchain_output.get_list()
# pk_list
aiida_list_proposals = run(
    aiida_propose_occ_matrices_from_results,
    pk_list = workchain_output,  
    N=8,
    debug=True,
    mode='random'
    # mode='read',
    # readfile=Str('NiO_mixing_lTF_beta0.3_oscdft_data.json')
)

# Print the list of Pks of the created Dict nodes

print(f"Created Dict nodes with PKs: " + str(aiida_list_proposals.get_list()))


#%%
# check by loading the first node
first_node = load_node(aiida_list_proposals[0])

with np.printoptions(precision=3, suppress=True):
    print("Loaded occupation numbers from first proposal:")
    print(np.array(occ1['occupation_numbers'])[1,1])

    print("Stored occupation matrix in AiiDA Dict node:")
    print(np.array(first_node.get_dict()['matrix'])[1,1])

# %%
from aiida.engine import calcfunction
from aiida.orm import Int
@calcfunction
def test_report(x, self=None):
    
    print(self)
    if self is not None:
        self.report(f"Hello! Value is {x.value}")
    return x + 1

from aiida.engine import run_get_node
result, node = run_get_node(test_report, x=Int(5))
print("PK:", node.pk)

# %%
