#%%
import json
import aiida
from lordcapulet.utils.postprocessing.gather_workchain_data import WorkchainDataExtractor
# aiida profile load
aiida.load_profile()



#%%
material_name = "UO2"
# workchain_pk = 74786 # UO2 second run with so(n) enabled
workchain_pk = 96376 # FeO second run with so(n) enabled
workchain_pk = 101440 # NiO second run with so(n) enabled
workchain_pk = 117081 # CoO second run with so(n) enabled
workchain_pk = 125002 # CoO 2 generation of 50 points of gp proposals
# Create extractor with SO(N) decomposition enabled


workchain_pk = 125470 # UO2
workchain_pk = 127812 # UO2 1st generation GP proposals
workchain_pk = 128328 # UO2 2nd generation GP proposals

extractor = WorkchainDataExtractor(perform_so_n=True,
                            sanity_check_reconstruct=True,
                            debug=True)

# Extract data from workchain
data = extractor.extract_from_workchain(workchain_pk)

# Save to JSON
extractor.save_to_json(data, f"{material_name}_scan_gp_gen2.json")
# %%
