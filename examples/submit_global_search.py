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

# Load structure (adapt this to your system)
atoms = read('NiO.scf.in', format='espresso-in')  # Adjust path as needed
# atoms = read('FeO.scf.in', format='espresso-in')  # Adjust path as needed


tm_atoms = tag_and_list_atoms(atoms, table={'Ni'})
tm_manifolds = get_default_manifolds(tm_atoms)
tm_dimensions = get_dimensions(tm_manifolds) 

total_dimensions = sum(tm_dimensions)

# print tags
print("Tagged transition atoms:", tm_atoms)
print("Corresponding manifolds:", tm_manifolds)
print("Corresponding dimensions:", tm_dimensions)
print("Total dimensions:", total_dimensions)

structure = StructureData(ase=atoms)
Uval = 5.0  # Example value for Hubbard U
hubbard_structure = HubbardStructureData.from_structure(structure)
for itm, tm_atom in enumerate(tm_atoms):
    hubbard_structure.initialize_onsites_hubbard(
        atom_name=tm_atom,
        # atom_manifold="3d",
        atom_manifold="3d",
        value=Uval  )  # Example: incrementing

# make sure that the Hubbard atoms are always before the rest of the atoms in the structure
hutils = HubbardUtils(hubbard_structure)
hutils.reorder_atoms()
hubbard_structure = hutils._hubbard_structure

# Load computational resources
code = aiida.orm.load_code('pwx_const_debug@daint-debug')  # Adjust to your code
# code = aiida.orm.load_code('pwx_dev_debug@daint-debug_lowtime')  # Adjust to your code

# Set up k-points
kpoints = KpointsData()
kpoints.set_kpoints_mesh([4, 3, 4])  # Adjust as needed

# Define DFT parameters
parameters = Dict(dict={
    'CONTROL': {
        'calculation': 'scf',
        'restart_mode': 'from_scratch',
        'verbosity': 'high',
    },
    'SYSTEM': {
        'ecutwfc': 80.1,    # Adjust as needed
        'ecutrho': 640.0,   # Adjust as needed
        'occupations': 'smearing',
        'smearing': 'gaussian',
        'degauss': 0.01,
        'nspin': 2,
        # Add other system parameters as needed
    },
    'ELECTRONS': {
        'conv_thr': 1.0e-8,
        'mixing_beta': 0.30,
        'electron_maxstep': 500,
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
Nmax = 40   # Total number of constrained calculations to perform
N = 20      # Number of proposals per generation

json_readfile = '/home/carta_a/Documents/Local_calculations/aiida-LordCapulet/examples/NiO_mixing_lTF_beta0.3_oscdft_data.json'
# Set up the inputs dictionary
inputs = {
    # AFM search inputs
    'afm': {
        'structure': hubbard_structure,  # or hubbard_structure
        'parameters': parameters,
        'kpoints': kpoints,
        'code': code,
        'tm_atoms': List(list=tm_atoms),
        'magnitude': Float(0.5),  # Magnetization magnitude for AFM
        'walltime_hours': Float(0.5),  # time for one AFM calculation
    },
    
    # Constrained scan inputs
    'constrained': {
        'structure': hubbard_structure,  # or hubbard_structure  
        'parameters': parameters,
        'kpoints': kpoints,
        'code': code,
        'tm_atoms': List(list=tm_atoms),
        'oscdft_card': oscdft_card,
        'walltime_hours': Float(0.5),  # time for one constrained calculation
    },
    
    # Global search parameters
    'Nmax': Int(Nmax),
    'N': Int(N),
    
    # Proposal function parameters
    'proposal_mode': Str('random_so_n'), 
    'proposal_debug': Bool(True),
    'proposal_holistic': Bool(False),  # Use Markovian approach by default
    
    # Provide the JSON file for read mode
    'proposal_kwargs': Dict(dict={
        # 'readfile': json_readfile
        # 'readfile': './NiO_mixing_lTF_beta0.3_oscdft_data.json'
    })
}

# Submit the workchain
workchain = submit(GlobalConstrainedSearchWorkChain, **inputs)

print(f"Submitted GlobalConstrainedSearchWorkChain with PK: {workchain.pk}")
print(f"This will perform up to {Nmax} constrained calculations in batches of {N}")
# print(f"Using 'read' mode with JSON file: {json_readfile}")
print(f"Monitor progress with: verdi process status {workchain.pk}")
#%% to run after finishing the workchain
print(f"To analyze results after completion, run:")
print(f"python -c \"")
print(f"import aiida; aiida.load_profile()")
print(f"from aiida.orm import load_node")
print(f"wc = load_node({workchain.pk})")
print(f"print('AFM matrices (OccupationMatrixAiidaData PKs):', len(wc.outputs.all_afm_matrices.get_list()))")
print(f"print('Constrained matrices (OccupationMatrixAiidaData PKs):', len(wc.outputs.all_constrained_matrices.get_list()))")
print(f"print('Total calculations:', len(wc.outputs.all_calculation_pks.get_list()))")
print(f"summary = wc.outputs.generation_summary.get_dict()")
print(f"for gen_id, gen_data in summary.items():")
print(f"    if gen_data['type'] == 'afm':")
print(f"        print(f'Generation {{gen_id}} (AFM): {{gen_data[\\\"n_calculations\\\"]}} calculations')")
print(f"    else:")
print(f"        print(f'Generation {{gen_id}} (Constrained): {{gen_data[\\\"n_successful\\\"]}}/{{gen_data[\\\"n_calculations\\\"]}} successful')")
print(f"\"")

# %%
