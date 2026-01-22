#%%
import json
import aiida
from lordcapulet.utils.postprocessing.gather_workchain_data import WorkchainDataExtractor
# aiida profile load
aiida.load_profile()



#%%
material_name = "CoO"
# workchain_pk = 74786 # UO2 second run with so(n) enabled
workchain_pk = 96376 # FeO second run with so(n) enabled
workchain_pk = 101440 # NiO second run with so(n) enabled
workchain_pk = 117081 # CoO second run with so(n) enabled
# Create extractor with SO(N) decomposition enabled

extractor = WorkchainDataExtractor(perform_so_n=True,
                            sanity_check_reconstruct=True,
                            debug=True)

# Extract data from workchain
data = extractor.extract_from_workchain(workchain_pk)

# Save to JSON
extractor.save_to_json(data, f"{material_name}_scan_data_extractor_redone.json")
# %%
