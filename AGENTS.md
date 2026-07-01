# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LordCapulet is an AiiDA plugin for constrained DFT+U (OSCDFT) calculations with Quantum ESPRESSO. It exposes one custom `CalcJob` (`ConstrainedPWCalculation`) and three `WorkChain`s that orchestrate magnetic configuration scans and iterative occupation-matrix search, with optional Bayesian / GP-driven proposal modes (torch + botorch).

Current research direction: develop supercell occupation-matrix proposal methods based on cluster expansion. Near-term experiments may use local atomic/template-product heuristics as stepping stones, but the project goal is a cluster-expansion proposal model for supercells.

## Experiments

`experiments/` contains active run scripts, generated results, plots, and analysis notes for the research project. These files are important project records, but they are not intended to be included in the final packaged `aiida-lordcapulet` distribution.

- Preserve experiment outputs unless explicitly asked to clean them up; do not treat generated JSON/HTML/plots in `experiments/` as disposable package build artifacts.
- Keep the high-level run index in `experiments/summary.md`. Update it when adding a new run, refreshing results, or changing the interpretation of an experiment.
- Put detailed chronological notes, recovery commands, submission logs, and troubleshooting details in `log.md` inside the relevant experiment subfolder when the summary would become too long.
- Avoid importing experiment-only scripts from package modules under `lordcapulet/`. Shared code that should ship belongs in the package; exploratory orchestration can stay under `experiments/`.

## Python environment

This project uses **`uv`** for Python environment management. The project venv lives at `.venv/` with `uv.lock` committed.

- Run any Python/script/tool via `uv run ...` (e.g. `uv run python examples/plot_energy_vs_magnetization.py`, `uv run pytest tests/`). Do **not** call the system `python3` directly — it lacks project deps (plotly, aiida, torch, etc.).
- Add a dependency: `uv add <pkg>` (updates `pyproject.toml` + `uv.lock`). Dev dep: `uv add --dev <pkg>`.
- Sync env to lockfile: `uv sync`.

## Build / install

When iterating new feature, bug fixes, etc, build `verdi` binary and restart daemon workers after changing the core codes.

```bash
uv sync                        # install project + deps from uv.lock into .venv
uv pip install -e .[dev]       # or editable install with test deps if needed
verdi daemon restart           # required after install so AiiDA picks up new entry points
verdi plugin list aiida.calculations | grep lordcapulet
verdi plugin list aiida.workflows | grep lordcapulet
```

`pyproject.toml` is the source of truth for entry points. `setup.py` still exists but is **stale** (missing `global_constrained_search` and `standard_magnetic_scan`); prefer `pyproject.toml` when registering or renaming entry points.

## AiiDA profile setup

When adding or recreating an AiiDA profile for this project, install the SSSP pseudo families as well:

```bash
uv run aiida-pseudo install sssp
```

The workflows expect `SSSP/1.3/PBEsol/efficiency` to be available. If the pseudo family is missing, `StandardMagneticScanWorkChain` and `ConstrainedScanWorkChain` will fail in `run_all` when calling `load_group('SSSP/1.3/PBEsol/efficiency')`.

## Testing

```bash
uv run pytest tests/                                                 # full suite
uv run pytest tests/unit_tests/ -v                                   # unit only
uv run pytest tests/integration_tests/workflows/test_afm_scan.py -v  # single file
uv run pytest tests/ -m "not slow"                                   # skip slow marker (GP pipeline)
uv run pytest tests/unit_tests/utils/test_rotation_matrices.py::TestSphericalToCubicRotation::test_unitarity
uv run python pytest_runner.py                                       # equivalent wrapper
```

Tests use `aiida.tools.pytest_fixtures` (loaded in `tests/conftest.py` via `pytest_plugins`) to spin up an **ephemeral AiiDA profile** backed by `pgtest` — no persistent DB needed, nothing is retained between runs. Session-scoped `pseudo_family` fixture creates a stub `SSSP/1.3/PBEsol/efficiency` family because both `StandardMagneticScanWorkChain` and `ConstrainedScanWorkChain` hard-code `load_group('SSSP/1.3/PBEsol/efficiency')` in `run_all`.

Tests are **reordered by tier** in `pytest_collection_modifyitems` (`tests/conftest.py`): utils → functions → data_structures → bayesian(unit) → calculations → workflows → bayesian(integration non-slow) → bayesian(integration slow). Preserve this ordering when adding new test directories — update `_test_priority` in `conftest.py`.

## Architecture

Data flows through three layers:

### 1. Data structures (`lordcapulet/data_structures/`)
- `OccupationMatrixData` — canonical in-memory representation of a per-atom 5×5 (or n×n) up/down occupation matrix with `specie`/`shell` metadata. Round-trips JSON via `as_dict`/`from_dict`. Provides converters to/from three formats: AiiDA-QE parser output (`from_aiida_qe_occupations`), constrained-PW input file format (`to_constrained_matrix_format`), and legacy `Dict` (`from_legacy_dict` / `to_legacy_dict`).
- `DataBank` — immutable collection of calculation records (`pk`, `energy`, `converged`, `occ_data`, `metadata`). Functional API: operations return new instances. Used as the ground-truth oracle in `lordcapulet/simulator/` and as the training set for GP proposal mode.
- Persisted on AiiDA DB as `JsonableData(OccupationMatrixData)`. Legacy `Dict` nodes are still accepted as input but deprecated.

### 2. Calculation / WorkChains (`lordcapulet/calculations/`, `lordcapulet/workflows/`)
- `ConstrainedPWCalculation` (subclass of `aiida_quantumespresso.calculations.pw.PwCalculation`) — adds two extra inputs (`oscdft_card: Dict`, `target_matrix: Dict | JsonableData`) and writes the `TARGET_OCCUPATION_NUMBERS` block expected by OSCDFT-patched QE.
- `StandardMagneticScanWorkChain` (aliased as `AFMScanWorkChain` for back-compat) — runs a batch of ferro/antiferro `PwCalculation`s with different starting magnetizations to seed the search.
- `ConstrainedScanWorkChain` — runs a batch of `ConstrainedPWCalculation`s, one per target occupation matrix.
- `GlobalConstrainedSearchWorkChain` — top-level driver. Outline: `run_initial_mag_scan → process_mag_scan_results → while(should_continue_search){ run_constrained_batch → process_constrained_results → update_counters } → gather_final_results`. `expose_inputs` from both sub-workchains under `mag_scan` and `constrained` namespaces. Converges when `Nmax` total proposals have been evaluated, generating `N` proposals per generation.
- Seeded `GlobalConstrainedSearchWorkChain` runs import previous results as training/history. For seeded mode, `Nmax` counts only newly submitted constrained calculations in the new workchain, not imported seed calculations. The first proposal after seed import always uses only imported standard magnetic scan results plus imported random-warmup constrained results; `proposal_holistic` does not broaden that initial seed-training subset to later imported GP generations.

Node-count caveat: although `common.yaml` contains `metadata_options.resources.num_machines`, the submitted child calculations currently hard-code `resources: {'num_machines': 1}` inside `StandardMagneticScanWorkChain.run_all` and `ConstrainedScanWorkChain.run_all`. Do not assume an experiment-script `overrides` entry will tune Slurm node count for `GlobalConstrainedSearchWorkChain` today. To make node count configurable, first expose and merge scheduler resource options through the scan workchain inputs, then reinstall/restart the AiiDA daemon before submitting new jobs.

### 3. Proposal layer (`lordcapulet/functions/`)
- `aiida_propose_occ_matrices_from_results` — `@calcfunction` that wraps the pure-Python `propose_new_constraints`. **Strict rule**: all AiiDA-type handling (loading `JsonableData`/`Dict`/`CalcJobNode`, unwrapping `Int`/`Float`/`Str`/`List`) happens in the calcfunction; `propose_new_constraints` and everything under `proposal_modes/` operates only on native Python types + `OccupationMatrixData`. Do not leak AiiDA types inward.
- Modes dispatched in `propose_new_constraints`: `random`, `random_so_n`, `gaussian_process` (`gp`), `read` (NotImplemented). GP mode reads energies from calculation PKs, requires `current_generation` in kwargs, and falls back to random on any exception (see traceback handling).

### Protocol system (`lordcapulet/workflows/protocols/`)
Workchains inherit `ProtocolMixin` (defined in `protocols/utils.py`). `get_builder_from_protocol(code, structure, ..., protocol='default', overrides={...})` returns a fully-populated `ProcessBuilder`. The mixin merges four layers (each overriding the previous):
1. `common.yaml` `default_inputs` (shared DFT defaults)
2. Top-level keys in the workchain's own YAML (e.g. `standard_magnetic_scan.yaml`)
3. The selected `protocols.<name>` section in that YAML
4. Caller-supplied `overrides` dict (deep-merged via `recursive_merge`)

The mixin only reads YAML — it never builds AiiDA types. Each workchain's own `get_builder_from_protocol` does the `Dict`/`KpointsData`/pseudo wrapping. YAML files are shipped as package data (`[tool.setuptools.package-data]` in `pyproject.toml`) so editable installs pick them up.

### Simulator (`lordcapulet/simulator/`)
`SearchSimulator` runs the proposal loop against a pre-computed `DataBank` oracle — no DFT, no AiiDA — for fast iteration on proposal algorithms. Uses Euclidean distance in occupation-matrix space to match proposals to ground-truth entries.

**The simulator is a development tool for testing proposal quality only.** Its bundled oracle data (`FeO_scan_data_extractor_redone.json`) is NOT a comparable scientific result — it is an old 80 Ry / degauss 0.01 Ry FeO run and is not per-atom energy-comparable with the production 40 Ry runs. Never use this JSON for cross-run energy comparison or as a reported result; use the run JSONs under `experiments/` and `examples/` instead.

## Conventions specific to this codebase

- **AiiDA types stop at the calcfunction boundary.** The internal proposal/decomposition code (`proposal_modes/`, `utils/`) must accept and return plain Python / NumPy / `OccupationMatrixData`. The calcfunction `aiida_propose_occ_matrices_from_results` unwraps `Int`/`Float`/`Str`/`Dict`/`List` before calling into it.
- **`JsonableData(OccupationMatrixData)` is the preferred wire format**; `Dict` is accepted only for backward compatibility. When building `JsonableData` manually, `deepcopy` the dict first — `JsonableData.__init__` calls `as_dict()` and mutates the returned reference by adding `@class`/`@module` keys. See `tests/conftest.py::generate_inputs_constrained_pw` for the canonical workaround (`del target_matrix._obj` forces re-hydration via `from_dict`).
- **`hubbard_corr_atoms` is a `List` of atom labels** (not species) indicating which atoms carry Hubbard U. Kinds/labels matter — `tag_and_list_atoms` in `utils/preprocessing/submission.py` relabels equivalent atoms so AFM configurations can target them individually.
- **`StandardMagneticScanWorkChain` replaced `AFMScanWorkChain`**; the old name is an alias kept in both `lordcapulet/__init__.py` and `workflows/__init__.py`. Both names are registered as AiiDA workflow entry points (`lordcapulet.afm_scan` and `lordcapulet.standard_magnetic_scan`) — don't remove either without an entry-point migration plan.
