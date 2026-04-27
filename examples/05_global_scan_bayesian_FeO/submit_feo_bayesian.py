#%%
"""
Global constrained search for FeO using Bayesian-optimisation (Gaussian-process)
proposals.

Mirrors 04_global_scan_random_FeO but swaps the proposal mode from
`random_so_n` to `gaussian_process`. The first generation still falls back to
random proposals to seed the GP training set; subsequent generations use the
GP acquisition function to pick promising occupation matrices.

Caching: result JSON ({material}_scan_bayesian.json) acts as the cache marker.
On first submit a stub JSON with just the workchain PK is written; the
extractor cell below overwrites it with full results. Re-running reloads the PK
from the JSON and skips submission. Delete the JSON to force a fresh run.
"""
import json
import os
import aiida
from aiida.engine import submit
from lordcapulet.workflows import GlobalConstrainedSearchWorkChain
from lordcapulet.utils import prepare_tm_info, prepare_hubbard_structure
from ase.io import read

aiida.load_profile()

material_name = 'FeO'
JSON_FILE = f"{material_name}_scan_bayesian.json"


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

    code = aiida.orm.load_code('pw-7.5-fix@prn')  # Adjust to your code

    parameters_qe = {
        'SYSTEM': {
            'ecutwfc': 40.0,
            'ecutrho': 480.0,
            'degauss': 0.05,
        },
    }

    # Bayesian-optimisation run. Key change vs. example 04: proposal_mode.
    #   'gaussian_process' (alias 'gp') — train a GP on observed energies,
    #       select next batch via UCB-based acquisition.
    # Generation 0 is always random (seeds the GP training set); the GP
    # proposer kicks in from generation 1 onward.
    builder = GlobalConstrainedSearchWorkChain.get_builder_from_protocol(
        code=code,
        structure=hubbard_structure,
        hubbard_corr_atoms=hubbard_corr_atoms,
        overrides={
            'Nmax': 200,
            'N': 10,
            'proposal_mode': 'gaussian_process',
            # IMPORTANT: GP is refit from scratch every generation. The default
            # is Markovian (only the previous generation feeds the fit). For
            # real BO we want the GP to see all past converged calcs.
            'proposal_holistic': True,
            # Generation 0 is always a random warm-up (GP kicks in at gen 1).
            # `N_initial_random` sets the gen-0 batch size independently of
            # the per-generation `N`. Raise it to give the GP more seed data.
            # Total seed = mag_scan calcs (~4) + N_initial_random.
            'proposal_kwargs': {'N_initial_random': 100},
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

    workchain = submit(builder)
    workchain_pk = workchain.pk

    print(f"Submitted GlobalConstrainedSearchWorkChain (Bayesian) with PK: {workchain_pk}")
    print(f"Monitor progress with: verdi process status {workchain_pk}")

    # Stub JSON so reruns find the PK before extraction completes.
    with open(JSON_FILE, 'w') as f:
        json.dump(
            {'metadata': {
                'pk': workchain_pk,
                'material': material_name,
                'proposal_mode': 'gaussian_process',
                'status': 'submitted',
            }},
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
