#%%

import json

import aiida
from lordcapulet.utils.postprocessing.gather_workchain_data import WorkchainDataExtractor
from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData 
from lordcapulet.data_structures.databank import DataBank
# aiida profile load
aiida.load_profile()

# %%
material_name = "FeO"
json_filename = f"{material_name}_scan_data_extractor_redone.json"


databank = DataBank.from_json(json_filename)
# %%
databank.to_dataframe()


# %%
occ_matrix = databank.get_occ_data(0)

occ_matrix.get_occupation_matrix('1', 'up')


occ_matrix.get_trace('1', 'up')

#%%
# from already loaded databank, get the total magnetic moment for each entry
databank.get_electron_number()


# now reload the databank adding electron number and magnetic moment to each entry
databank = DataBank.from_json(json_filename, include_electron_number=True, include_moment=True)
# %%
import torch
from lordcapulet.functions.proposal_modes.Bayesian.acquisition import prepare_eigenvalue_indices, compute_eigenvalue_preference
torch_matrix = databank.to_pytorch()[0]
# add violation to test
torch_matrix[1] = 0.16
atom_ids = databank.atom_ids
spins = ['up', 'down']

fw_map = databank.get_forward_index_map()
batch_indices = prepare_eigenvalue_indices(fw_map)

preference_scores = compute_eigenvalue_preference(torch_matrix, batch_indices, k=2000)

preference_scores

#%%
import numpy as np
from lordcapulet.data_structures.occupation_matrix import compute_occupation_distance
# get 2 different occupation matrices and compare their distance
occ_data1 = databank.get_occ_data(0)
occ_data2 = databank.get_occ_data(1)

zerodist = compute_occupation_distance(occ_data1, occ_data1)
dist = compute_occupation_distance(occ_data1, occ_data2)

with np.printoptions(precision=4, suppress=True):
    print(f"Distance between identical matrices: {zerodist:.4f}")
    print(f"Distance between different matrices: {dist:.4f}")


# %%
sem