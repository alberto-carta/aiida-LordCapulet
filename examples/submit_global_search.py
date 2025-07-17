#%%
import aiida
from aiida.orm import Code, Dict, StructureData, KpointsData, List, Int, Str, Bool, Float, load_node
from aiida.engine import submit
from lordcapulet.workflows import GlobalConstrainedSearchWorkChain
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from ase.io import read

# Load AiiDA profile
aiida.load_profile()

# Load structure (adapt this to your system)
atoms = read('NiO.scf.in', format='espresso-in')  # Adjust path as needed

def tag_and_list_atoms(atoms):
    """
    Tags atoms based on whether they are transition metals or other elements.
    Transition metals get a unique tag (e.g., Ni1, Mn2).
    Other elements get a tag based on their element symbol (e.g., O1, S1).
    These tags are stored in atom.info['custom_tag'].

    Args:
        atoms (list): A list of atom objects, assumed to be ASE Atom objects
                      or similar with 'symbol' and an 'info' dictionary attribute.
    """
    transition_metals = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'
    }

    tm_counts = {}
    other_counts = {}
    tm_atoms = []

    for atom in atoms:

        if atom.symbol in transition_metals:
            if atom.symbol not in tm_counts:
                tm_counts[atom.symbol] = 0
            
            tm_counts[atom.symbol] += 1
            # Store the custom string tag in atom.info
            atom.tag = tm_counts[atom.symbol]
            tm_atoms.append(f"{atom.symbol}{tm_counts[atom.symbol]}")
        else:
            if atom.symbol not in other_counts:
                other_counts[atom.symbol] = 1
            
            # Store the custom string tag in atom.info
            atom.tag = other_counts[atom.symbol]
    
    return tm_atoms


tm_atoms = tag_and_list_atoms(atoms)

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
# hubbard_structure = HubbardStructureData.from_structure(structure)
# Add Hubbard parameters here if needed

# Load computational resources
code = aiida.orm.load_code('pwx_const_debug@daint-debug')  # Adjust to your code
# code = aiida.orm.load_code('pwx_dev_debug@daint-debug_lowtime')  # Adjust to your code

# Set up k-points
kpoints = KpointsData()
kpoints.set_kpoints_mesh([8, 8, 8])  # Adjust as needed

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
        'electron_maxstep': 200,
    },
})
#%%

oscdft_card = Dict(dict={
    'oscdft_type': 2,
    'n_oscdft': 100,
    'constraint_strength': 1.0,
    'constraint_conv_thr': 0.005,
    'constraint_maxstep': 200,
    'constraint_mixing_beta': 0.4,
})

# Global search parameters
Nmax = 61   # Total number of constrained calculations to perform (6 trials)
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
    },
    
    # Constrained scan inputs
    'constrained': {
        'structure': hubbard_structure,  # or hubbard_structure  
        'parameters': parameters,
        'kpoints': kpoints,
        'code': code,
        'tm_atoms': List(list=tm_atoms),
        'oscdft_card': oscdft_card,
    },
    
    # Global search parameters
    'Nmax': Int(Nmax),
    'N': Int(N),
    
    # Proposal function parameters
    'proposal_mode': Str('read'),  # Use read mode to load from JSON file
    'proposal_debug': Bool(True),
    'proposal_holistic': Bool(False),  # Use Markovian approach by default
    
    # Provide the JSON file for read mode
    'proposal_kwargs': Dict(dict={
        'readfile': json_readfile
        # 'readfile': './NiO_mixing_lTF_beta0.3_oscdft_data.json'
    })
}

# Submit the workchain
workchain = submit(GlobalConstrainedSearchWorkChain, **inputs)

print(f"Submitted GlobalConstrainedSearchWorkChain with PK: {workchain.pk}")
print(f"This will perform up to {Nmax} constrained calculations in batches of {N}")
print(f"Using 'read' mode with JSON file: {json_readfile}")
print(f"Monitor progress with: verdi process status {workchain.pk}")
#%% to run after finishing the workchain
print(f"\nTo analyze results after completion, run:")
print(f"python -c \"")
print(f"import aiida; aiida.load_profile()")
print(f"from aiida.orm import load_node")
print(f"wc = load_node({workchain.pk})")
print(f"print('AFM matrices:', len(wc.outputs.all_afm_matrices.get_list()))")
print(f"print('Total calculations:', len(wc.outputs.all_calculation_pks.get_list()))")
print(f"summary = wc.outputs.generation_summary.get_dict()")
print(f"for gen_id, gen_data in summary.items():")
print(f"    if gen_data['type'] == 'afm':")
print(f"        print(f'Generation {{gen_id}} (AFM): {{gen_data[\\\"n_calculations\\\"]}} calculations')")
print(f"    else:")
print(f"        print(f'Generation {{gen_id}} (Constrained): {{gen_data[\\\"n_successful\\\"]}}/{{gen_data[\\\"n_calculations\\\"]}} successful')")
print(f"\"")

# %%
