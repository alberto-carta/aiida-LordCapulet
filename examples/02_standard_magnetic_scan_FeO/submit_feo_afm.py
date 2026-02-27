#%%
import aiida
from aiida.orm import load_node
from aiida.engine import submit
from lordcapulet.workflows import AFMScanWorkChain
from lordcapulet.utils import prepare_tm_info, prepare_hubbard_structure
from ase.io import read

# Load AiiDA profile
aiida.load_profile()

# load scf file
atoms = read('../FeO.scf.in', format='espresso-in')  # Adjust path as needed

#%%
# tag transition metal atoms and get their manifolds and dimensions
tm_atoms, tm_manifolds, tm_dimensions = prepare_tm_info(atoms, table={'Fe'})

# print tags
print("Tagged transition atoms:", tm_atoms)
print("Corresponding manifolds:", tm_manifolds)
print("Corresponding dimensions:", tm_dimensions)
print("Total dimensions:", sum(tm_dimensions))

# u_values can be a single float (same U for all TM sites) or a per-atom
# list with one entry per site, e.g. u_values=[5.0, 4.0]
hubbard_structure = prepare_hubbard_structure(
    atoms, tm_atoms, tm_manifolds, U_values=5.0
)

code = aiida.orm.load_code('pwx_const@daint-general')  # Adjust to your code
# Use this if you want explicit control over every single input without relying
# on the YAML defaults.  You are responsible for setting every required field.
# Uncomment the block below and comment out Option B to use it.
#
# from aiida.orm import Dict, KpointsData, List, Float, Str
#
# kpoints = KpointsData()
# kpoints.set_kpoints_mesh([3, 3, 3])
#
# parameters = Dict(dict={
#     'CONTROL': {
#         'calculation': 'scf',
#         'restart_mode': 'from_scratch',
#         'verbosity': 'high',
#     },
#     'SYSTEM': {
#         'ecutwfc': 40.0,
#         'ecutrho': 480.0,
#         'occupations': 'smearing',
#         'smearing': 'cold',
#         'degauss': 0.02,
#         'nspin': 2,
#     },
#     'ELECTRONS': {
#         'conv_thr': 1.0e-5,
#         'mixing_beta': 0.1,
#         'electron_maxstep': 500,
#         'mixing_mode': 'local-TF',
#     },
# })
#
# builder = AFMScanWorkChain.get_builder()
# builder.code = code
# builder.structure = hubbard_structure
# builder.kpoints = kpoints
# builder.parameters = parameters
# builder.tm_atoms = List(list=tm_atoms)
# builder.magnitude = Float(0.5)          # magnetisation magnitude per site
# builder.walltime_hours = Float(1.0)     # walltime in hours
# builder.pseudo_family_string = Str('SSSP/1.3/PBEsol/efficiency')

# ── Option B: protocol-based builder (recommended) ───────────────────────────
# All defaults come from the YAML files in lordcapulet/workflows/protocols/.
# Pass `overrides` as a nested dict to change only what you need; everything
# else stays at the protocol default.  The k-point mesh is derived
# automatically from the structure's reciprocal-lattice density unless you
# explicitly pass 'kpoints_mesh' in overrides.
#
# Available overrides (non-exhaustive):
#   'kpoints_distance'          - spacing in 1/Å  (default 0.4)
#   'kpoints_mesh'              - explicit [nx,ny,nz] (bypasses density logic)
#   'magnitude'                 - magnetisation amplitude (default 0.5)
#   'walltime_hours'            - hours per calculation (default 2.0)
#   'pseudo_family'             - aiida-pseudo group label
#   'parameters'                - nested QE namelist overrides
builder = AFMScanWorkChain.get_builder_from_protocol(
    code=code,
    structure=hubbard_structure,
    tm_atoms=tm_atoms,
    overrides={
        # 'kpoints_mesh': [3, 3, 3],
        # 'kpoints_distance': 0.5,  # alternative to fixed mesh
        'walltime_hours': 1.0,
        'parameters': {
            'SYSTEM': {'ecutwfc': 40.0, 'ecutrho': 480.0, 'degauss': 0.02},
            'ELECTRONS': {'conv_thr': 1.0e-5},
        },
    },
)

# Submit the workchain
workchain = submit(builder)

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

