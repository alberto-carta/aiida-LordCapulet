#%%
import aiida
from aiida.orm import Code, Dict, StructureData, KpointsData, List, Int, Str, Bool, Float, load_node
from aiida.engine import submit
from lordcapulet.workflows import GlobalConstrainedSearchWorkChain
from lordcapulet.utils.preprocessing.submission import tag_and_list_atoms, get_default_manifolds, get_dimensions
# import HubbardUtils to rearrange atoms
from aiida_quantumespresso.utils.hubbard import HubbardUtils
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from ase.io import read

# Load AiiDA profile
aiida.load_profile()

# load scf file
atoms = read('../FeO.scf.in', format='espresso-in')  # Adjust path as needed

#%%
# tag transition metal atoms and get their manifolds and dimensions
tm_atoms = tag_and_list_atoms(atoms, table={'Fe'})
tm_manifolds = get_default_manifolds(tm_atoms)
tm_dimensions = get_dimensions(tm_manifolds) 

total_dimensions = sum(tm_dimensions)

# print tags
print("Tagged transition atoms:", tm_atoms)
print("Corresponding manifolds:", tm_manifolds)
print("Corresponding dimensions:", tm_dimensions)
print("Total dimensions:", total_dimensions)

structure = StructureData(ase=atoms)

Uval = 5 # Hubbard U
hubbard_structure = HubbardStructureData.from_structure(structure)


for itm, tm_atom in enumerate(tm_atoms):
    hubbard_structure.initialize_onsites_hubbard(
        atom_name=tm_atom,
        atom_manifold=tm_manifolds[itm],
        value=Uval  )  # 

# make sure that the Hubbard atoms are always before the rest of the atoms in the structure
hutils = HubbardUtils(hubbard_structure)
hutils.reorder_atoms()
hubbard_structure = hutils._hubbard_structure

code = aiida.orm.load_code('pwx_const@daint-general')  # Adjust to your code

# Set up k-pointsSearchWorkChain.
kpoints = KpointsData()
kpoints.set_kpoints_mesh([3, 3, 3])  # Adjust as needed

# Define DFT parameters
parameters = Dict(dict={
    'CONTROL': {
        'calculation': 'scf',
        'restart_mode': 'from_scratch',
        'verbosity': 'high',
    },
    'SYSTEM': {
        'ecutwfc': 40.0,    # Adjust as needed
        'ecutrho': 480.0,   # Adjust as needed
        'occupations': 'smearing',
        'smearing': 'cold',
        'degauss': 0.05,
        'nspin': 2,
        # Add other system parameters as needed
    },
    'ELECTRONS': {
        'conv_thr': 1.0e-3,
        'mixing_beta': 0.3,
        'electron_maxstep': 500,
        # 'mixing_mode': 'local-TF',
    },
})




#%%

oscdft_card = Dict(dict={
    'oscdft_type': 2,
    'n_oscdft': total_dimensions,
    'constraint_strength': 1.0,
    'constraint_conv_thr': 0.005,
    'constraint_maxstep': 200,
    'constraint_mixing_beta': 0.4,
})

# Global search parameters
Nmax = 20   # Total number of constrained calculations to perform
N = 4      # Number of proposals per generation

inputs = {
    # AFM search inputs
    'afm': {
        'structure': hubbard_structure,  # or hubbard_structure
        'parameters': parameters,
        'kpoints': kpoints,
        'code': code,
        'tm_atoms': List(list=tm_atoms),
        'magnitude': Float(0.5),  # Magnetization magnitude for AFM
        'walltime_hours': Float(1.0),  # AFM calculations walltime (1 hour)
    },
    
    # Constrained scan inputs
    'constrained': {
        'structure': hubbard_structure,  # or hubbard_structure  
        'parameters': parameters,
        'kpoints': kpoints,
        'code': code,
        'tm_atoms': List(list=tm_atoms),
        'oscdft_card': oscdft_card,
        'walltime_hours': Float(1.0),  # Constrained calculations walltime (1 hour)
    },
    
    # Global search parameters
    'Nmax': Int(Nmax),
    'N': Int(N),
    
    # Proposal function parameters
    'proposal_mode': Str('random_so_n'),  # Use read mode to load from JSON file
    'proposal_debug': Bool(True),
    'proposal_holistic': Bool(False),  # Use Markovian approach by default
    
    # Provide the JSON file for read mode
    'proposal_kwargs': Dict(dict={ 'randomize_oxidation': False,
    }),
}

# Submit the workchain
workchain = submit(GlobalConstrainedSearchWorkChain, **inputs)

print(f"Submitted GlobalConstrainedSearchWorkChain with PK: {workchain.pk}")
print(f"This will perform up to {Nmax} constrained calculations in batches of {N}")
print(f"Monitor progress with: verdi process status {workchain.pk}")

# create a file and save information about the workchain for postprocessing
# append to the file if it already exists, otherwise create a new one
with open('feo_global_scan_info.txt', 'w') as f:
    f.write("="*40 + "\n")
    f.write(f"Workchain PK: {workchain.pk}\n")
    f.write(f"Material: FeO\n")
    f.write(f"Total constrained calculations (Nmax): {Nmax}\n")
    f.write(f"Proposals per generation (N): {N}\n")
    f.write(f"Proposal mode: random_so_n\n")
    f.write(f"Randomize oxidation states in proposals: {inputs['proposal_kwargs'].get('randomize_oxidation', False)}\n")
    f.write("="*40 + "\n")

#%%

GlobalConstrainedSearchWorkChain.get_builder()


# %%
from lordcapulet.utils.postprocessing.gather_workchain_data import WorkchainDataExtractor
# aiida profile load
aiida.load_profile()


material_name = "FeO"
workchain_pk = workchain.pk  
# workchain_pk = 190189

# Create extractor with SO(N) decomposition enabled
extractor = WorkchainDataExtractor(perform_so_n=True,
                            sanity_check_reconstruct=True,
                            debug=True)

# Extract data from workchain
data = extractor.extract_from_workchain(workchain_pk)

# Save to JSON
# extractor.save_to_json(data, f"{material_name}_scan_data.json")
extractor.save_to_json(data, f"{material_name}_scan_random.json")
