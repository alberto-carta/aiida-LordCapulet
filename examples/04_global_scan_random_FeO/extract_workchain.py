#%%
import json
import aiida
from lordcapulet.utils.postprocessing.gather_workchain_data import WorkchainDataExtractor
# aiida profile load
aiida.load_profile()


#%%
material_name = "scob"
# workchain_pk = 134153 # Global workchain
# workchain_pk = 134868 # Bigger global workchain
# workchain_pk = 141137 # Constrained GP workchain
workchain_pk = 149298 # Constrained GP workchain gaussian process generation 1

# Create extractor with SO(N) decomposition enabled
extractor = WorkchainDataExtractor(perform_so_n=True,
                            sanity_check_reconstruct=True,
                            debug=True)

# Extract data from workchain
data = extractor.extract_from_workchain(workchain_pk)

# Save to JSON
# extractor.save_to_json(data, f"{material_name}_scan_data.json")
extractor.save_to_json(data, f"{material_name}_scan_random+gp.json")



#
# %%
