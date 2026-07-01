# Structure Occupation-Matrix Visualizer — Design

Date: 2026-06-28
Status: approved (pre-implementation)
Scope: `experiments/` only — not part of the packaged `aiida-lordcapulet` distribution.

## Goal

Interactive HTML browser where the user sees a 2D rendering of a structure's
atoms and, on click, sees the up/down occupation-matrix heatmaps for the
selected atom on a side panel. A dropdown switches between proposals/calculations.

This complements the existing `experiments/FeO_template_library/visualize_proposal_matrices.py`
(scatter of proposals → grid of all atom matrices). The new tool inverts the
interaction: structure layout → per-atom matrix.

## Decisions (locked)

- Geometry source: CIF / ASE-readable file.
- Input scope: many proposals/calcs, switchable via dropdown.
- Heatmaps: up and down side by side.
- Location: `experiments/` standalone script (approach A), self-contained HTML
  output following the sibling visualizer convention.
- Projection: auto — two longest lattice vectors.

## The atom-ordering problem (and the fix)

Atom order is **not** preserved end to end. Evidence from real data:

- 4x1x1 proposal JSON: `Atom_1`→specie `Fe8`, `Atom_2`→`Fe7`, …
- 2x1x1 scan JSON: `Atom_1`→`Fe4`, `Atom_2`→`Fe3`, …

Chain of custody:

1. `lordcapulet/utils/preprocessing/submission.py::tag_and_list_atoms` assigns
   Fe tags `Fe1, Fe2, …` in **original ASE atom order** (first Fe → `Fe1`).
2. AiiDA-QE gives each tagged Fe its own kind (`Fe1`..`FeN`) and reorders atoms
   by kind when writing the QE input.
3. The parser `atom_index` (`lordcapulet/data_structures/occupation_matrix.py::from_aiida_qe_occupations`)
   counts Hubbard atoms in **QE output order**. Hence `Atom_1` = `FeN` (reversed).

Therefore mapping by the `Atom_N` sequence index is **wrong**. The fix: the
**specie tag suffix is order-stable**.

> Map `specie = "Fe{k}"` → the k-th Fe atom in the CIF (Fe atoms taken in file
> order). `tag_and_list_atoms` defines `Fe{k}` as the k-th Fe in original
> structure order, so this is exact provided the CIF lists Fe in that order.

O atoms (no occupation entry) render as faded, non-clickable geometry points.

## Module & CLI

New file: `experiments/FeO_template_library/visualize_structure_matrices.py`.

```
uv run python experiments/FeO_template_library/visualize_structure_matrices.py \
  --proposals   FeO_4x1x1_template_proposals.json \
  --submissions FeO_4x1x1_template_proposal_submissions.json \   # optional → adds E/converged
  --structure   FeO_4x1x1.cif \                                  # ASE-readable
  --output      experiments/plots/FeO_4x1x1_structure_matrix_browser.html \
  --projection  auto                                             # default
```

## Architecture (decouple load from render)

The renderer is format-agnostic; loaders normalize each input shape into one
common record schema.

- `plot_structure_matrix_records(records, atoms2d, output_path, *, title)` —
  pure HTML renderer. Shared by all inputs.
- `build_proposal_records(proposal_json, *, submission_json=None)` —
  4x1x1 proposal + optional submission shape (proposals list; results joined by
  proposal index, mirroring the sibling's `_submission_result_index`).
- `build_scan_records(scan_json, *, pks=None, max_n=None)` —
  scan-JSON shape: top-level `calculations` dict keyed by pk; each entry carries
  `occupation_matrices`, `output_parameters`, `converged`. Used by the 2x1x1
  verification.

### Geometry (ASE)

`load_structure_2d(structure_path, projection="auto") -> Atoms2D`:

- `ase.io.read(structure_path)` → positions, cell, symbols.
- `auto` projection: rank the three lattice vectors by norm, take the top two,
  orthonormalize them into a 2D basis, project cartesian coords → (x, y).
- Partition atoms: Fe (Hubbard) vs other (O). Fe collected in **file order** →
  `fe_xy[k]` = k-th Fe position.

### Mapping + validation

`map_atoms_to_geometry(occ_atoms, atoms2d)`:

- For each occ atom, parse specie suffix `Fe{k}` → int k → `fe_xy[k-1]`.
- Validate: `#Fe in structure == #occ atoms`; suffixes are a unique permutation
  of `1..nFe`. On mismatch raise `ValueError` with a clear message (wrong CIF /
  wrong Fe order / wrong element).

### Common record schema

```json
{
  "proposal_index": 0,
  "label": "pk 10781",
  "target_total_magnetization": 0.0,
  "atoms": [
    {"atom_label": "Atom_1", "specie": "Fe4", "shell": "d",
     "moment": 3.5, "up": [[...]], "down": [[...]], "x": 1.2, "y": 0.0}
  ],
  "others": [{"specie": "O1", "x": 2.1, "y": 0.0}],
  "calc_pk": 10781, "energy": -123.4, "converged": true,
  "final_total_magnetization": 0.0
}
```

`atoms` = Fe only (clickable, carry matrices). `others` = O geometry points.
Result fields (`calc_pk`, `energy`, `converged`, `final_total_magnetization`)
present when the source provides them.

## HTML / interaction

Two-pane grid reusing the sibling's CSS skeleton.

- **Left** — `go.Scatter` of atom positions. Fe markers colored by `moment`
  (RdBu, cmid 0); O markers grey/faded. `customdata` = atom array index. Equal
  axis aspect (`scaleanchor`) so geometry is undistorted.
- **Right** — two heatmaps side by side (`up` | `down`), shared `coloraxis`,
  value-text overlay, y reversed — same heatmap styling as the sibling.
- **Top** — `<select>` dropdown switches proposal/calc → re-renders left scatter,
  clears right panel.
- **Click** a Fe atom → render its up/down heatmaps + metadata line
  (atom_label, specie, shell, moment). **Hover** → tooltip (specie, moment).
  O atoms ignored on click.

## Testing

Add `tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py`,
mirroring `test_feo_template_library_experiment.py`:

- `load_structure_2d` projects onto two longest axes; returns Fe in file order.
- `map_atoms_to_geometry` maps by suffix (reversal-correct), separates O.
- mismatch (wrong Fe count) raises `ValueError`.
- `build_proposal_records` and `build_scan_records` return the common schema on
  tiny synthetic inputs.
- HTML write smoke test (file created, non-empty), as the sibling does — no
  headless-browser test.

## Verification — FeO 2x1x1 (real data)

2x1x1 FeO rocksalt = 4 Fe + 4 O. Scan data:
`experiments/FeO_supercell_2x1x1_bayesian/FeO_2x1x1_scan_bayesian.json`
(`calculations` dict, 2012 entries; ground-state target `pk 10781`).

1. **Generate CIF** — `experiments/FeO_supercell_2x1x1_convergence/make_feo_2x1x1_cif.py`
   (ASE: rocksalt FeO conventional cell → `repeat((2,1,1))` → write
   `FeO_2x1x1.cif`).
   Caveat (documented in script + report): physical-site identity assumes the
   generated CIF's Fe order equals the run's structure order. This verification
   targets tool mechanics (render / suffix mapping / reversal handling), not
   site provenance.
2. **Run** the visualizer via `build_scan_records` restricted to `pk 10781` plus
   a few other PKs → `experiments/plots/FeO_2x1x1_structure_matrix_browser.html`.
3. **Manual checks**:
   - 4 Fe clickable, 4 O faded / non-clickable.
   - Reversal handled: the atom rendered as specie `Fe1` shows occ `Atom_4`'s
     matrix (not `Atom_1`).
   - up/down heatmaps for `pk 10781` match the existing
     `experiments/FeO_supercell_2x1x1_convergence/target_occupation_matrices.png`
     produced by `ground_state_target.py`.
   - feeding a wrong-Fe-count CIF raises the mapping `ValueError`.
4. **Unit test** extends the test file: `build_scan_records` returns 4 Fe records
   with suffix-correct coords on a tiny synthetic 2x1x1 fixture.

## Out of scope (YAGNI)

- Selectable-axes projection dropdown (auto only).
- 3D rendering.
- Spin-difference toggle (`_spin_difference` kept available, not primary).
- Any package install / shipping under `lordcapulet/`.
```

## Records / index updates

- Add a row to `experiments/outline.md` when the verification run is produced.
- Note the new browser HTML in the 2x1x1 convergence `report.md`.
