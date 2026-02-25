#%%
from lordcapulet.calculations import ConstrainedPWCalculation
from aiida.orm import Code, Dict, StructureData, KpointsData, load_group
from aiida.engine import submit
import numpy as np
import aiida
from aiida.orm import StructureData
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from ase.io import read
from aiida.orm import List

# Load AiiDA profile
aiida.load_profile()
#%%

atoms = read('FeO.scf.in', format='espresso-in')
# atoms = read('nbcl.relax.in', format='espresso-in')

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
        'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'U'
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

# Now you can set different Hubbard U for each Ni kind:
# hubbard_structure.initialize_onsites_hubbard(atom_name="Ni1", atom_manifold="3d", value=5.0)
# hubbard_structure.initialize_onsites_hubbard(atom_name="Ni2", atom_manifold="3d", value=5.0)

# === 2. Parameters ===
parameters = Dict(dict={
    'CONTROL': {
        'calculation': 'scf',
        'verbosity': 'high',
    },
    'SYSTEM': {
        'ibrav': 0,
        'ecutwfc': 60.0,
        'occupations': 'smearing',
        'nspin': 2,
        'smearing': 'gaussian',
        'degauss': 0.01,
        'ecutrho': 480.0,
    },
    'ELECTRONS': {
        'mixing_beta': 0.3,
        'electron_maxstep': 500,
        'conv_thr': 1.0e-8,
        'mixing_mode': 'local-TF',
    }
})

# === 3. K-points ===
kpoints = KpointsData()
kpoints.set_kpoints_mesh([4, 4, 4])
magnitude = 0.5
# === 5. Code ===
code = aiida.orm.load_code('pwx_dev_debug@daint-debug')
# code = aiida.orm.load_code('pw-docker@localhost')

#%%
#import aiida float
from aiida.orm import Float
from aiida.orm import Str


inputs = {
    'structure': hubbard_structure,
    'parameters': parameters,
    'kpoints': kpoints,
    'code': code,
    'tm_atoms': List(list=tm_atoms),
    'magnitude': Float(magnitude),
    # 'metadata': {
    #         'options': {
    #             'resources': {
    #                 'num_machines': 1,
    #                 'num_mpiprocs_per_machine': 1,
    #             },
                # optionally, add other options like wallclock_seconds, withmpi, etc.
            # },}
}

from lordcapulet.workflows import AFMScanWorkChain
from aiida.engine import submit

# wc = submit(AFMScanWorkChain, **inputs)

#print the workchain PK
# print(f"Submitted AFMScanWorkChain with PK = {wc.pk}")


#%%
from aiida.orm import load_node
from lordcapulet.functions import aiida_propose_occ_matrices_from_results


# Load workchain node
# Note: With the updated workchains, wc.outputs.all_occupation_matrices now contains
# PKs of OccupationMatrixAiidaData nodes (not calculation PKs)

# wc = load_node(48099)
# wc = load_node(59516) # UO2 
# wc = load_node(61435) # UO2 afm scan
wc = load_node(112030) # UO2 constrained scan



list_node = wc.outputs.all_constrained_matrices
# pk_list = list_node.get_list()
# list_node = load_node(62311).get_list() # UO2 constrained scan

# The aiida_propose_occ_matrices_from_results function can now handle:
# - OccupationMatrixAiidaData nodes (new format)
# - Dict nodes (legacy format) 
# - Calculation nodes (direct extraction)
aiida_list_proposals = aiida_propose_occ_matrices_from_results(
    pk_list = list_node.get_list(),  
    N=6,
    debug=True,
    mode='random_so_n',
    tm_atoms=tm_atoms,
    reporter_type='print',
    # mode='read',
    # readfile=Str('NiO_mixing_lTF_beta0.3_oscdft_data.json')
)

print(f"Created proposal nodes with PKs: " + str(aiida_list_proposals.get_list()))
#%%
# Load the first proposal node and examine the matrix
# Note: aiida_propose_occ_matrices_from_results still returns Dict nodes
# containing matrices in the format expected by ConstrainedScanWorkChain
first_node = load_node(aiida_list_proposals[4])
print(f"Loaded proposal node with PK: {first_node.pk}")

matrices = first_node.obj.data


with np.printoptions(precision=3, suppress=True):
    print("Proposed occupation matrix:")
    # first matrix is up, second is down
    for atom_idx, matrix in matrices.items():
        print(f"Atom index: {atom_idx}")
        print('Spin up:')
        print(np.array(matrix['occupation_matrix']['up']).real)
        print('Spin down:')
        print(np.array(matrix['occupation_matrix']['down']).real)
        # print sum of traces over up and down
        trace_up = np.trace(np.array(matrix['occupation_matrix']['up']))
        trace_down = np.trace(np.array(matrix['occupation_matrix']['down']))
        print(f"Trace up: {trace_up}, Trace down: {trace_down}")
        print(f"Total trace: {trace_up + trace_down}")
    

#%%
from lordcapulet.workflows import ConstrainedScanWorkChain

# Convert proposal Dict nodes to the format expected by ConstrainedScanWorkChain
# The proposals are already in the correct {'matrix': [...]} format
target_list = [ load_node(pk).obj for pk in aiida_list_proposals.get_list()]
code = aiida.orm.load_code('pwx_const_debug@daint-debug')


oscdft_card = Dict(dict={
    'oscdft_type': 2,
    # 'n_oscdft': 100,
    'n_oscdft': 100,
    'constraint_strength': 1.0,
    'constraint_conv_thr': 0.005,
    'constraint_maxstep': 200,
    'constraint_mixing_beta': 0.4,
})

inputs_constrained = {
    'structure': hubbard_structure,
    'parameters': parameters,
    'kpoints': kpoints,
    'code': code,
    'tm_atoms': List(list=tm_atoms),
    'oscdft_card': oscdft_card,
    'occupation_matrices_list': List(list=target_list),
}

wc2 = submit(ConstrainedScanWorkChain, **inputs_constrained)


print(f"Submitted ConstrainedScanWorkChain with PK: {wc2.pk}")

# %%
# start a few more
list_node2 = wc2.outputs.all_occupation_matrices

aiida_list_proposals = aiida_propose_occ_matrices_from_results(
    pk_list = List(list_node.get_list() + list_node2.get_list()),  
    N=9,
    debug=True,
    mode='read',
    readfile=Str('NiO_mixing_lTF_beta0.3_oscdft_data.json')
)

print(f"Created proposal Dict nodes with PKs: " + str(aiida_list_proposals.get_list()))
# %%


# target_list = [ pk for pk in aiida_list_proposals.get_list()]
target_list = aiida_list_proposals.get_list()
code = aiida.orm.load_code('pwx_const_debug@daint-debug')


inputs_constrained = {
    'structure': hubbard_structure,
    'parameters': parameters,
    'kpoints': kpoints,
    'code': code,
    'tm_atoms': List(list=tm_atoms),
    'oscdft_card': oscdft_card,
    'occupation_matrices_list': List(list=target_list),
}

wc3 = submit(ConstrainedScanWorkChain, **inputs_constrained)

print(f"Submitted ConstrainedScanWorkChain with PK: {wc3.pk}")
# %%
