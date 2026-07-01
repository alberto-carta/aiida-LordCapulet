#%%
import json
import os
import aiida
from aiida.engine import submit
from lordcapulet.workflows import GlobalConstrainedSearchWorkChain
from lordcapulet.utils import prepare_tm_info, prepare_hubbard_structure
from ase.io import read

# Load AiiDA profile
aiida.load_profile()

material_name = "FeO"
JSON_FILE = f"{material_name}_scan_random.json"


def _read_pk_from_json(path):
    with open(path) as f:
        return json.load(f)['metadata']['pk']


#%%
if os.path.exists(JSON_FILE):
    workchain_pk = _read_pk_from_json(JSON_FILE)
    print(f"Found existing {JSON_FILE}, skipping submit. PK={workchain_pk}")
else:
    # load scf file
    atoms = read('../FeO.scf.in', format='espresso-in')  # Adjust path as needed

    # tag transition metal atoms and get their manifolds and dimensions
    hubbard_corr_atoms, hubbard_corr_manifolds, hubbard_corr_dimensions = prepare_tm_info(atoms, table={'Fe'})

    print("Tagged transition atoms:", hubbard_corr_atoms)
    print("Corresponding manifolds:", hubbard_corr_manifolds)
    print("Corresponding dimensions:", hubbard_corr_dimensions)
    print("Total dimensions:", sum(hubbard_corr_dimensions))

    # u_values can be a single float (same U for all TM sites) or a per-atom
    # list with one entry per site, e.g. u_values=[5.0, 4.0]
    hubbard_structure = prepare_hubbard_structure(
        atoms, hubbard_corr_atoms, hubbard_corr_manifolds, U_values=5.0
    )

    # code = aiida.orm.load_code('pwx_const@daint-general')  # Adjust to your code
    code = aiida.orm.load_code('pw-7.5-fix@prn')  # Adjust to your code

    # Build and submit using protocol defaults.
    # Use `overrides` to adjust DFT parameters or global search settings:
    #   overrides={
    #       'Nmax': 30, 'N': 8,
    #       'mag_scan':        {'kpoints_mesh': [4, 4, 4], 'walltime_hours': 2.0},
    #       'constrained': {'walltime_hours': 3.0},
    #   }
    parameters_qe = {
        'SYSTEM': {
            'ecutwfc': 40.0,
            'ecutrho': 480.0,
            'degauss': 0.05,
        },
    }

    builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
        code=code,
        structure=hubbard_structure,
        hubbard_corr_atoms=hubbard_corr_atoms,
        overrides={
            'Nmax': 20,
            'N': 10,
            'proposal_mode': 'random_so_n',
            'mag_scan': {
                'kpoints_mesh': [3, 3, 3],
                'walltime_hours': 1.0,
                'parameters': parameters_qe,
            },
            'constrained': {
                'kpoints_mesh': [3, 3, 3],
                'walltime_hours': 1.0,
                'parameters': parameters_qe,
            },
        },
    )

    # Submit the workchain
    workchain = submit(builder)
    workchain_pk = workchain.pk

    print(f"Submitted GlobalConstrainedSearchWorkChain with PK: {workchain_pk}")
    print(f"Monitor progress with: verdi process status {workchain_pk}")

    # Stub JSON so reruns find the PK before extraction completes.
    # Re-run the extractor cell below once the workchain finishes to overwrite
    # this stub with full results.
    with open(JSON_FILE, 'w') as f:
        json.dump(
            {'metadata': {'pk': workchain_pk, 'material': material_name, 'status': 'submitted'}},
            f, indent=2,
        )


# %%
from lordcapulet.utils.postprocessing.gather_workchain_data import WorkchainDataExtractor
aiida.load_profile()

extractor = WorkchainDataExtractor(perform_so_n=True,
                            sanity_check_reconstruct=True,
                            debug=True)

data = extractor.extract_from_workchain(workchain_pk)
extractor.save_to_json(data, JSON_FILE)
