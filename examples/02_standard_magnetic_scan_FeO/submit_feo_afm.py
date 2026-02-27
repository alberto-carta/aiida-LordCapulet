#%%
import aiida
from aiida.orm import Code, Dict, StructureData, KpointsData, List, Int, Str, Bool, Float, load_node
from aiida.engine import submit
from lordcapulet.workflows import AFMScanWorkChain
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

# Set up k-points
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
        'degauss': 0.02,
        'nspin': 2,
        # Add other system parameters as needed
    },
    'ELECTRONS': {
        'conv_thr': 1.0e-5,
        'mixing_beta': 0.1,
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



inputs = {
        'structure': hubbard_structure,  # or hubbard_structure
        'parameters': parameters,
        'kpoints': kpoints,
        'code': code,
        'tm_atoms': List(list=tm_atoms),
        'magnitude': Float(0.5),  # Magnetization magnitude for AFM
        'walltime_hours': Float(1.0),  # AFM calculations walltime (1 hour)
}

from lordcapulet.workflows import AFMScanWorkChain
from aiida.engine import submit


# Submit the workchain
workchain = submit(AFMScanWorkChain, **inputs)

print(f"Submitted AFMScanWorkChain with PK: {workchain.pk}")
print(f"Monitor progress with: verdi process status {workchain.pk}")

# create a file and save information about the workchain for postprocessing
# append to the file if it already exists, otherwise create a new one
with open('feo_afm_scan_info.txt', 'w') as f:
    f.write("="*40 + "\n")
    f.write(f"Workchain PK: {workchain.pk}\n")
    f.write(f"Material: FeO\n")
    f.write("="*40 + "\n")


# %%
from lordcapulet.utils.postprocessing.gather_workchain_data import WorkchainDataExtractor
# aiida profile load
aiida.load_profile()


material_name = "FeO"
workchain_pk = workchain.pk  

# Create extractor with SO(N) decomposition enabled
extractor = WorkchainDataExtractor(perform_so_n=True,
                            sanity_check_reconstruct=True,
                            debug=True)

# Extract data from workchain
data = extractor.extract_from_workchain(workchain_pk)

# Save to JSON
extractor.save_to_json(data, f"{material_name}_afm_scan.json")
