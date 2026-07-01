#%%
"""
Standard magnetic scan for FeO.

Caching: result JSON ({material}_afm_scan.json) acts as the cache marker. On
first submit a stub JSON with just the workchain PK is written; the extractor
cell below overwrites it with full results. Re-running reloads the PK from the
JSON and skips submission. Delete the JSON to force a fresh run.
"""
import json
import os
import aiida
from aiida.engine import submit
from lordcapulet.workflows import StandardMagneticScanWorkChain
from lordcapulet.utils import prepare_tm_info, prepare_hubbard_structure
from ase.io import read

aiida.load_profile()

material_name = 'FeO'
JSON_FILE = f"{material_name}_afm_scan.json"


def _read_pk_from_json(path):
    with open(path) as f:
        return json.load(f)['metadata']['pk']


#%%
if os.path.exists(JSON_FILE):
    workchain_pk = _read_pk_from_json(JSON_FILE)
    print(f"Found existing {JSON_FILE}, skipping submit. PK={workchain_pk}")
else:
    atoms = read('../FeO.scf.in', format='espresso-in')

    hubbard_corr_atoms, hubbard_corr_manifolds, hubbard_corr_dimensions = prepare_tm_info(atoms, table={'Fe'})

    print("Tagged transition atoms:", hubbard_corr_atoms)
    print("Corresponding manifolds:", hubbard_corr_manifolds)
    print("Corresponding dimensions:", hubbard_corr_dimensions)
    print("Total dimensions:", sum(hubbard_corr_dimensions))

    hubbard_structure = prepare_hubbard_structure(
        atoms, hubbard_corr_atoms, hubbard_corr_manifolds, U_values=5.0
    )

    code = aiida.orm.load_code('pwx_const@daint-general')  # Adjust to your code

    # Protocol-based builder. All defaults come from the YAML files in
    # lordcapulet/workflows/protocols/. Pass `overrides` as a nested dict
    # to change only what you need.
    #
    # Available overrides (non-exhaustive):
    #   'kpoints_distance'  - spacing in 1/Å  (default 0.4)
    #   'kpoints_mesh'      - explicit [nx,ny,nz] (bypasses density logic)
    #   'magnitude'         - magnetisation amplitude (default 0.5)
    #   'walltime_hours'    - hours per calculation (default 2.0)
    #   'pseudo_family'     - aiida-pseudo group label
    #   'parameters'        - nested QE namelist overrides
    builder = StandardMagneticScanWorkChain.get_builder_from_protocol(
        code=code,
        structure=hubbard_structure,
        hubbard_corr_atoms=hubbard_corr_atoms,
        overrides={
            'kpoints_mesh': [3, 4, 3],
            'walltime_hours': 1.0,
            'parameters': {
                'SYSTEM': {'ecutwfc': 40.0, 'ecutrho': 480.0, 'degauss': 0.02},
                'ELECTRONS': {'conv_thr': 1.0e-5},
            },
        },
    )

    workchain = submit(builder)
    workchain_pk = workchain.pk

    print(f"Submitted StandardMagneticScanWorkChain with PK: {workchain_pk}")
    print(f"Monitor progress with: verdi process status {workchain_pk}")

    # Stub JSON so reruns find the PK before extraction completes.
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
