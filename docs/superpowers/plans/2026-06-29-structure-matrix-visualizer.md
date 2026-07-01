# Structure Occupation-Matrix Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive self-contained-HTML browser that shows a 2D atom layout of a structure and, on clicking a Hubbard (Fe) atom, renders its up/down occupation-matrix heatmaps in a side panel, with a dropdown to switch between proposals/calculations.

**Architecture:** One standalone experiments script (`visualize_structure_matrices.py`) split into geometry (ASE), the specie-suffix atom-mapping fix, two format-specific record builders (proposal JSON and scan JSON), and a format-agnostic HTML renderer. A separate generator script produces a FeO 2x1x1 CIF used for verification against real scan data.

**Tech Stack:** Python, ASE (`ase.io.read`, `ase.build`), NumPy, Plotly (graph_objects + plotly.io), self-contained HTML/JS. Run everything via `uv run`.

## Global Constraints

- Scope is `experiments/` only — nothing ships under `lordcapulet/`; do not import experiment code from package modules.
- AiiDA types must never appear here — operate on plain Python / NumPy / `OccupationMatrixData`.
- ASE and Plotly are available as transitive deps (ase 3.29, plotly 6.8); do NOT add them to `pyproject.toml`.
- All Python/test/tool invocations use `uv run` (e.g. `uv run pytest`, `uv run python ...`). Never call system `python3`.
- Atom mapping is by **specie suffix** `Fe{k}` → k-th Fe in CIF file order. Never map by `Atom_N` sequence index (QE reverses it).
- Heatmaps show `up` and `down` side by side (not spin-difference).
- New tests live in `tests/unit_tests/experiments/` (existing tier in `tests/conftest.py`; no conftest change needed).

---

### Task 1: Geometry — load structure and project to 2D

**Files:**
- Create: `experiments/FeO_template_library/visualize_structure_matrices.py`
- Test: `tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `HUBBARD_ELEMENTS: set[str]` (default `{"Fe"}`).
  - `@dataclass Atoms2D` with `fe_xy: list[tuple[float, float]]` (Fe positions in CIF file order) and `others: list[dict]` (each `{"specie": str, "x": float, "y": float}`).
  - `load_structure_2d(structure_path, projection="auto", hubbard_elements=HUBBARD_ELEMENTS) -> Atoms2D`.
  - `_projection_basis(cell, projection="auto") -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write the failing test**

In `tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py`:

```python
"""Smoke + unit tests for the structure occupation-matrix visualizer."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write

from experiments.FeO_template_library.visualize_structure_matrices import (
    Atoms2D,
    load_structure_2d,
)


def _write_feo_2x1x1_cif(path):
    # Rocksalt FeO conventional cell (a=4.3) repeated 2x1x1: 4 Fe + 4 O.
    a = 4.3
    base = Atoms(
        "FeO",
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
        cell=[a, a, a],
        pbc=True,
    )
    # rocksalt has 4 Fe + 4 O in the conventional cell; emulate a small cell
    cell = Atoms(
        "Fe4O4",
        scaled_positions=[
            (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
            (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    )
    supercell = cell.repeat((2, 1, 1))
    ase_write(str(path), supercell)
    return supercell


def test_load_structure_2d_separates_fe_and_others(tmp_path):
    cif = tmp_path / "FeO_2x1x1.cif"
    supercell = _write_feo_2x1x1_cif(cif)
    n_fe = sum(1 for s in supercell.get_chemical_symbols() if s == "Fe")

    atoms2d = load_structure_2d(cif)

    assert isinstance(atoms2d, Atoms2D)
    assert len(atoms2d.fe_xy) == n_fe == 8
    assert all(len(xy) == 2 for xy in atoms2d.fe_xy)
    assert {o["specie"] for o in atoms2d.others} == {"O"}
    assert len(atoms2d.others) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py::test_load_structure_2d_separates_fe_and_others -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (script does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `experiments/FeO_template_library/visualize_structure_matrices.py`:

```python
"""Interactive structure occupation-matrix browser.

Left pane: a 2D layout of a structure's atoms (projected from an ASE-readable
file). Click a Hubbard (Fe) atom to render its up/down occupation-matrix
heatmaps on the right. A dropdown switches between proposals/calculations.

Experiment artifact only; not part of the packaged distribution.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.io as pio
from ase.io import read as ase_read

from lordcapulet.data_structures import OccupationMatrixData

HUBBARD_ELEMENTS = {"Fe"}


@dataclass
class Atoms2D:
    """2D-projected geometry split into Hubbard atoms and the rest."""

    fe_xy: list[tuple[float, float]]
    others: list[dict[str, Any]]


def _projection_basis(cell, projection: str = "auto"):
    cell = np.asarray(cell, dtype=float)
    if projection == "auto":
        norms = np.linalg.norm(cell, axis=1)
        idx = sorted(np.argsort(norms)[::-1][:2])
        v1, v2 = cell[idx[0]], cell[idx[1]]
    elif projection == "xy":
        v1, v2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError(f"unknown projection {projection!r}")
    e1 = v1 / np.linalg.norm(v1)
    v2p = v2 - np.dot(v2, e1) * e1
    e2 = v2p / np.linalg.norm(v2p)
    return e1, e2


def load_structure_2d(
    structure_path,
    projection: str = "auto",
    hubbard_elements: set[str] = HUBBARD_ELEMENTS,
) -> Atoms2D:
    atoms = ase_read(str(structure_path))
    e1, e2 = _projection_basis(atoms.get_cell(), projection)
    fe_xy: list[tuple[float, float]] = []
    others: list[dict[str, Any]] = []
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        x, y = float(np.dot(pos, e1)), float(np.dot(pos, e2))
        if sym in hubbard_elements:
            fe_xy.append((x, y))
        else:
            others.append({"specie": sym, "x": x, "y": y})
    return Atoms2D(fe_xy=fe_xy, others=others)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py::test_load_structure_2d_separates_fe_and_others -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/FeO_template_library/visualize_structure_matrices.py tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py
git commit -m "feat(experiments): structure 2D geometry loader for matrix visualizer"
```

---

### Task 2: Atom mapping by specie suffix (the ordering fix)

**Files:**
- Modify: `experiments/FeO_template_library/visualize_structure_matrices.py`
- Test: `tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py`

**Interfaces:**
- Consumes: `Atoms2D` (Task 1).
- Produces:
  - `_specie_suffix(specie: str) -> int`.
  - `build_species_xy(occ_species: list[str], atoms2d: Atoms2D) -> dict[str, tuple[float, float]]` — maps each `Fe{k}` specie to `atoms2d.fe_xy[k-1]`; raises `ValueError` on count mismatch or non-permutation suffixes.

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
from experiments.FeO_template_library.visualize_structure_matrices import (
    build_species_xy,
)


def _atoms2d(n):
    return Atoms2D(fe_xy=[(float(i), 0.0) for i in range(n)], others=[])


def test_build_species_xy_maps_by_suffix_not_sequence():
    # Reversed labelling like the real data: Atom_1 -> Fe4 ... Atom_4 -> Fe1.
    occ_species = ["Fe4", "Fe3", "Fe2", "Fe1"]
    mapping = build_species_xy(occ_species, _atoms2d(4))
    # Fe1 is the 1st Fe in file order -> x == 0.0; Fe4 -> x == 3.0.
    assert mapping["Fe1"] == (0.0, 0.0)
    assert mapping["Fe4"] == (3.0, 0.0)


def test_build_species_xy_count_mismatch_raises():
    with pytest.raises(ValueError, match="Hubbard atoms"):
        build_species_xy(["Fe1", "Fe2", "Fe3"], _atoms2d(4))


def test_build_species_xy_bad_permutation_raises():
    with pytest.raises(ValueError, match="permutation"):
        build_species_xy(["Fe1", "Fe1", "Fe3", "Fe4"], _atoms2d(4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k build_species_xy -v`
Expected: FAIL with `ImportError` (`build_species_xy` not defined).

- [ ] **Step 3: Write minimal implementation**

Add to `visualize_structure_matrices.py`:

```python
def _specie_suffix(specie: str) -> int:
    match = re.search(r"(\d+)$", specie or "")
    if not match:
        raise ValueError(f"specie {specie!r} has no numeric suffix")
    return int(match.group(1))


def build_species_xy(
    occ_species: list[str],
    atoms2d: Atoms2D,
) -> dict[str, tuple[float, float]]:
    n = len(atoms2d.fe_xy)
    if len(occ_species) != n:
        raise ValueError(
            f"structure has {n} Hubbard atoms but occupation data has "
            f"{len(occ_species)}"
        )
    suffixes = [_specie_suffix(s) for s in occ_species]
    if sorted(suffixes) != list(range(1, n + 1)):
        raise ValueError(
            f"specie suffixes {sorted(suffixes)} are not a permutation of "
            f"1..{n}"
        )
    return {s: atoms2d.fe_xy[k - 1] for s, k in zip(occ_species, suffixes)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k build_species_xy -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/FeO_template_library/visualize_structure_matrices.py tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py
git commit -m "feat(experiments): specie-suffix atom mapping (QE order-reversal fix)"
```

---

### Task 3: Scan-JSON record builder

**Files:**
- Modify: `experiments/FeO_template_library/visualize_structure_matrices.py`
- Test: `tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py`

**Interfaces:**
- Consumes: `OccupationMatrixData`.
- Produces:
  - `_atom_sort_key(atom_label: str) -> tuple[int, str]`.
  - `_round_float(value: float) -> float`.
  - `_load_json(path: Path) -> dict`.
  - `_atom_records(occ: OccupationMatrixData) -> list[dict]` — each `{atom_label, specie, shell, moment, up, down}`.
  - `build_scan_records(scan_json, *, pks=None, max_n=None) -> list[dict]` — common record schema: `{proposal_index, label, atoms, target_total_magnetization, calc_pk, energy, converged, final_total_magnetization}`.

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
from experiments.FeO_template_library.visualize_structure_matrices import (
    build_scan_records,
)


def _eye_matrix(diag_up, diag_down, n=5):
    up = [[diag_up if i == j else 0.0 for j in range(n)] for i in range(n)]
    down = [[diag_down if i == j else 0.0 for j in range(n)] for i in range(n)]
    return {"up": up, "down": down}


def _scan_json(path):
    calc = {
        "pk": 10781,
        "converged": True,
        "output_parameters": {"energy": -123.4, "total_magnetization": 0.0},
        "occupation_matrices": {
            "Atom_1": {"specie": "Fe4", "shell": "d",
                       "occupation_matrix": _eye_matrix(1.0, 0.0)},
            "Atom_2": {"specie": "Fe3", "shell": "d",
                       "occupation_matrix": _eye_matrix(0.0, 1.0)},
            "Atom_3": {"specie": "Fe2", "shell": "d",
                       "occupation_matrix": _eye_matrix(1.0, 0.0)},
            "Atom_4": {"specie": "Fe1", "shell": "d",
                       "occupation_matrix": _eye_matrix(0.0, 1.0)},
        },
    }
    path.write_text(json.dumps({"calculations": {"10781": calc}}))
    return path


def test_build_scan_records_common_schema(tmp_path):
    scan = _scan_json(tmp_path / "scan.json")
    records = build_scan_records(scan, pks=[10781])

    assert len(records) == 1
    rec = records[0]
    assert rec["label"] == "pk 10781"
    assert rec["calc_pk"] == 10781
    assert rec["energy"] == -123.4
    assert rec["converged"] is True
    assert len(rec["atoms"]) == 4
    species = {a["specie"] for a in rec["atoms"]}
    assert species == {"Fe1", "Fe2", "Fe3", "Fe4"}
    fe4 = next(a for a in rec["atoms"] if a["specie"] == "Fe4")
    assert fe4["moment"] == pytest.approx(5.0)  # trace_up - trace_down
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k build_scan_records -v`
Expected: FAIL with `ImportError` (`build_scan_records` not defined).

- [ ] **Step 3: Write minimal implementation**

Add to `visualize_structure_matrices.py`:

```python
def _atom_sort_key(atom_label: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", atom_label)
    if match:
        return int(match.group(1)), atom_label
    return 0, atom_label


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        return json.load(handle)


def _atom_records(occ: OccupationMatrixData) -> list[dict[str, Any]]:
    records = []
    for label in sorted(occ.get_atom_labels(), key=_atom_sort_key):
        data = occ[label]
        records.append(
            {
                "atom_label": label,
                "specie": data.get("specie"),
                "shell": data.get("shell"),
                "moment": _round_float(occ.get_magnetic_moment(label)),
                "up": data["occupation_matrix"]["up"],
                "down": data["occupation_matrix"]["down"],
            }
        )
    return records


def build_scan_records(
    scan_json,
    *,
    pks: list[int] | None = None,
    max_n: int | None = None,
) -> list[dict[str, Any]]:
    data = _load_json(Path(scan_json))
    calculations = data.get("calculations", {})
    keys = [str(p) for p in pks] if pks else list(calculations)

    records: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        if max_n is not None and len(records) >= max_n:
            break
        entry = calculations.get(key)
        if entry is None or "occupation_matrices" not in entry:
            continue
        occ = OccupationMatrixData(entry["occupation_matrices"])
        atoms = _atom_records(occ)
        output = entry.get("output_parameters") or {}
        records.append(
            {
                "proposal_index": index,
                "label": f"pk {entry.get('pk', key)}",
                "atoms": atoms,
                "target_total_magnetization": _round_float(
                    sum(a["moment"] for a in atoms)
                ),
                "calc_pk": entry.get("pk", int(key)),
                "energy": output.get("energy"),
                "converged": entry.get("converged"),
                "final_total_magnetization": output.get("total_magnetization"),
            }
        )
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k build_scan_records -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/FeO_template_library/visualize_structure_matrices.py tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py
git commit -m "feat(experiments): scan-JSON record builder for structure visualizer"
```

---

### Task 4: Proposal-JSON record builder

**Files:**
- Modify: `experiments/FeO_template_library/visualize_structure_matrices.py`
- Test: `tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py`

**Interfaces:**
- Consumes: `_atom_records`, `_load_json`, `_round_float`, `OccupationMatrixData`.
- Produces:
  - `_submission_result_index(submission_json: Path) -> dict[int, dict]` — proposal_index → result fields (`calc_pk`, `converged`, `energy`, `final_total_magnetization`).
  - `build_proposal_records(proposal_json, *, submission_json=None) -> list[dict]` — same common schema as `build_scan_records`, with `label = f"proposal {index}"`.

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
from experiments.FeO_template_library.visualize_structure_matrices import (
    build_proposal_records,
)


def _proposal_json(path):
    proposal = {
        "index": 0,
        "occupation_matrices": {
            "Atom_1": {"specie": "Fe4", "shell": "d",
                       "occupation_matrix": _eye_matrix(1.0, 0.0)},
            "Atom_2": {"specie": "Fe3", "shell": "d",
                       "occupation_matrix": _eye_matrix(0.0, 1.0)},
            "Atom_3": {"specie": "Fe2", "shell": "d",
                       "occupation_matrix": _eye_matrix(1.0, 0.0)},
            "Atom_4": {"specie": "Fe1", "shell": "d",
                       "occupation_matrix": _eye_matrix(0.0, 1.0)},
        },
    }
    path.write_text(json.dumps({"proposals": [proposal]}))
    return path


def test_build_proposal_records_without_submissions(tmp_path):
    proposals = _proposal_json(tmp_path / "proposals.json")
    records = build_proposal_records(proposals)

    assert len(records) == 1
    rec = records[0]
    assert rec["label"] == "proposal 0"
    assert rec["calc_pk"] is None
    assert len(rec["atoms"]) == 4
    assert {a["specie"] for a in rec["atoms"]} == {"Fe1", "Fe2", "Fe3", "Fe4"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k build_proposal_records -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Add to `visualize_structure_matrices.py`:

```python
def _submission_result_index(submission_json: Path) -> dict[int, dict[str, Any]]:
    data = _load_json(submission_json)
    calculations = data.get("calculations", {})
    index: dict[int, dict[str, Any]] = {}
    for job in data.get("jobs", {}).values():
        proposal_index = job.get("proposal_index")
        if proposal_index is None:
            continue
        calc_pk = job.get("calc_pk")
        calc = calculations.get(str(calc_pk), {}) if calc_pk is not None else {}
        output = calc.get("output_parameters") or {}
        index[int(proposal_index)] = {
            "calc_pk": calc_pk,
            "converged": calc.get("converged", job.get("is_finished_ok")),
            "energy": output.get("energy"),
            "final_total_magnetization": output.get("total_magnetization"),
        }
    return index


def build_proposal_records(
    proposal_json,
    *,
    submission_json=None,
) -> list[dict[str, Any]]:
    proposal_data = _load_json(Path(proposal_json))
    result_index = (
        _submission_result_index(Path(submission_json)) if submission_json else {}
    )

    records: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposal_data.get("proposals", [])):
        proposal_index = int(proposal.get("index", index))
        occ = OccupationMatrixData(proposal["occupation_matrices"])
        atoms = _atom_records(occ)
        result = result_index.get(proposal_index, {})
        records.append(
            {
                "proposal_index": proposal_index,
                "label": f"proposal {proposal_index}",
                "atoms": atoms,
                "target_total_magnetization": _round_float(
                    sum(a["moment"] for a in atoms)
                ),
                "calc_pk": result.get("calc_pk"),
                "energy": result.get("energy"),
                "converged": result.get("converged"),
                "final_total_magnetization": result.get(
                    "final_total_magnetization"
                ),
            }
        )
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k build_proposal_records -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/FeO_template_library/visualize_structure_matrices.py tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py
git commit -m "feat(experiments): proposal-JSON record builder for structure visualizer"
```

---

### Task 5: HTML renderer

**Files:**
- Modify: `experiments/FeO_template_library/visualize_structure_matrices.py`
- Test: `tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py`

**Interfaces:**
- Consumes: `Atoms2D`, `build_species_xy`, records from Task 3/4.
- Produces:
  - `plot_structure_matrix_records(records, atoms2d, output_path, *, title="Structure occupation matrices") -> Path` — validates geometry via `build_species_xy` (using the first record's atom species), writes a self-contained HTML file with an embedded records payload, a structure scatter (Fe colored by moment, O faded), two side-by-side up/down heatmaps, a proposal dropdown, and click/hover wiring. Returns the output `Path`.

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
from experiments.FeO_template_library.visualize_structure_matrices import (
    plot_structure_matrix_records,
)


def test_plot_structure_matrix_records_writes_html(tmp_path):
    scan = _scan_json(tmp_path / "scan.json")
    records = build_scan_records(scan, pks=[10781])
    # 4 Fe geometry points in file order; O atoms faded.
    atoms2d = Atoms2D(
        fe_xy=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        others=[{"specie": "O", "x": 0.5, "y": 0.5}],
    )
    out = tmp_path / "browser.html"

    result = plot_structure_matrix_records(records, atoms2d, out, title="t")

    assert result == out
    html = out.read_text()
    assert "plotly" in html.lower()
    assert "structure-scatter" in html
    assert "matrix-browser" in html
    assert "pk 10781" in html  # embedded record label


def test_plot_structure_matrix_records_mismatch_raises(tmp_path):
    scan = _scan_json(tmp_path / "scan.json")
    records = build_scan_records(scan, pks=[10781])
    atoms2d = Atoms2D(fe_xy=[(0.0, 0.0)], others=[])  # only 1 Fe vs 4 occ
    with pytest.raises(ValueError, match="Hubbard atoms"):
        plot_structure_matrix_records(records, atoms2d, tmp_path / "x.html")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k plot_structure_matrix -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Add to `visualize_structure_matrices.py`. Note doubled braces `{{`/`}}` are literal braces inside the f-string.

```python
import plotly.graph_objects as go


def _structure_scatter(records, species_xy, atoms2d, title):
    first = records[0]["atoms"] if records else []
    fe_x = [species_xy[a["specie"]][0] for a in first]
    fe_y = [species_xy[a["specie"]][1] for a in first]
    moments = [a["moment"] for a in first]
    species = [a["specie"] for a in first]

    fig = go.Figure()
    if atoms2d.others:
        fig.add_trace(
            go.Scatter(
                x=[o["x"] for o in atoms2d.others],
                y=[o["y"] for o in atoms2d.others],
                mode="markers",
                marker={"size": 10, "color": "#cbd5e1", "symbol": "circle"},
                hovertext=[o["specie"] for o in atoms2d.others],
                hoverinfo="text",
                name="other",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=fe_x,
            y=fe_y,
            mode="markers+text",
            text=species,
            textposition="top center",
            marker={
                "size": 18,
                "color": moments,
                "colorscale": "RdBu",
                "cmid": 0,
                "colorbar": {"title": "moment"},
                "line": {"width": 1, "color": "#1f2933"},
            },
            customdata=[[i] for i in range(len(first))],
            hovertemplate="%{text}<br>moment=%{marker.color:.3f}<extra></extra>",
            name="Fe",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.02},
        template="plotly_white",
        showlegend=False,
        margin={"l": 40, "r": 24, "t": 70, "b": 40},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_structure_matrix_records(
    records,
    atoms2d: Atoms2D,
    output_path,
    *,
    title: str = "Structure occupation matrices",
) -> Path:
    output_path = Path(output_path)
    if not records:
        raise ValueError("No records to plot")

    occ_species = [a["specie"] for a in records[0]["atoms"]]
    species_xy = build_species_xy(occ_species, atoms2d)  # validates geometry

    scatter_fig = _structure_scatter(records, species_xy, atoms2d, title)
    scatter_html = pio.to_html(
        scatter_fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id="structure-scatter",
        config={"responsive": True},
    )
    payload = json.dumps(records)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; }}
    .page {{ width: min(1680px, 100vw); margin: 0 auto; padding: 16px;
             box-sizing: border-box; }}
    .controls {{ margin-bottom: 10px; font-size: 14px; }}
    .grid {{ display: grid;
             grid-template-columns: minmax(420px, 0.7fr) minmax(620px, 1fr);
             gap: 18px; align-items: stretch; }}
    #structure-scatter, #matrix-browser {{ width: 100%; height: 720px; }}
    .metadata {{ font-size: 13px; margin-top: 10px; color: #334155; }}
    @media (max-width: 1120px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="controls">
      <label for="record-select">Structure: </label>
      <select id="record-select"></select>
    </div>
    <div class="grid">
      <div>
        {scatter_html}
        <div class="metadata" id="record-metadata"></div>
      </div>
      <div id="matrix-browser"></div>
    </div>
  </div>
  <script>
    const records = {payload};
    const speciesXY = {json.dumps(species_xy)};
    const scatterDiv = document.getElementById("structure-scatter");
    const matrixDiv = document.getElementById("matrix-browser");
    const metaDiv = document.getElementById("record-metadata");
    const selectEl = document.getElementById("record-select");
    let currentRecord = 0;

    records.forEach((r, i) => {{
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = r.label;
      selectEl.appendChild(opt);
    }});

    function heatmap(matrix, xaxis, yaxis, label) {{
      return {{
        type: "heatmap",
        z: matrix,
        text: matrix.map(row => row.map(v => v.toFixed(3))),
        texttemplate: "%{{text}}",
        textfont: {{size: 9}},
        hovertemplate: label + "[%{{y}}, %{{x}}] = %{{z:.6f}}<extra></extra>",
        coloraxis: "coloraxis",
        xaxis: xaxis,
        yaxis: yaxis,
      }};
    }}

    function showAtom(record, atomIndex) {{
      const atom = record.atoms[atomIndex];
      if (!atom) return;
      const traces = [
        heatmap(atom.up, "x", "y", atom.atom_label + " up"),
        heatmap(atom.down, "x2", "y2", atom.atom_label + " down"),
      ];
      const layout = {{
        title: {{text: atom.atom_label + " (" + atom.specie + ") M=" +
                 atom.moment.toFixed(3), x: 0.02}},
        template: "plotly_white",
        grid: {{rows: 1, columns: 2, pattern: "independent"}},
        margin: {{l: 40, r: 18, t: 70, b: 40}},
        coloraxis: {{colorscale: "Viridis", colorbar: {{title: "occ"}}}},
        xaxis: {{title: "up", constrain: "domain"}},
        yaxis: {{autorange: "reversed", scaleanchor: "x", scaleratio: 1}},
        xaxis2: {{title: "down", constrain: "domain"}},
        yaxis2: {{autorange: "reversed", scaleanchor: "x2", scaleratio: 1}},
      }};
      Plotly.react(matrixDiv, traces, layout, {{responsive: true}});
    }}

    function showRecord(index) {{
      currentRecord = index;
      const record = records[index];
      if (!record) return;
      const fe = record.atoms;
      const xs = fe.map(a => speciesXY[a.specie][0]);
      const ys = fe.map(a => speciesXY[a.specie][1]);
      const moments = fe.map(a => a.moment);
      Plotly.restyle(scatterDiv, {{
        x: [xs], y: [ys], text: [fe.map(a => a.specie)],
        "marker.color": [moments],
        customdata: [fe.map((_, i) => [i])],
      }}, [scatterDiv.data.length - 1]);
      metaDiv.innerHTML = [
        "label: " + record.label,
        "calc_pk: " + (record.calc_pk ?? "none"),
        "energy: " + (record.energy ?? "none"),
        "converged: " + (record.converged ?? "unknown"),
        "target M: " + record.target_total_magnetization.toFixed(4),
        "final M: " + (record.final_total_magnetization ?? "none"),
      ].join("<br>");
      showAtom(record, 0);
    }}

    selectEl.addEventListener("change", e =>
      showRecord(parseInt(e.target.value, 10)));
    scatterDiv.on("plotly_click", event => {{
      const point = event.points && event.points[0];
      if (!point || !point.customdata) return;
      showAtom(records[currentRecord], point.customdata[0]);
    }});

    showRecord(0);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k plot_structure_matrix -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/FeO_template_library/visualize_structure_matrices.py tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py
git commit -m "feat(experiments): HTML renderer for structure occupation-matrix browser"
```

---

### Task 6: CLI wiring

**Files:**
- Modify: `experiments/FeO_template_library/visualize_structure_matrices.py`
- Test: `tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py`

**Interfaces:**
- Consumes: `load_structure_2d`, `build_proposal_records`, `build_scan_records`, `plot_structure_matrix_records`.
- Produces:
  - `run(input_kind, data_json, structure, output, *, submissions=None, pks=None, max_n=None, projection="auto", title="Structure occupation matrices") -> Path` — orchestrates load → build → render.
  - `main(argv=None) -> None` — argparse front end with `--input-kind {proposal,scan}`, `--data`, `--structure`, `--submissions`, `--output`, `--pks`, `--max-n`, `--projection`, `--title`.

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
from experiments.FeO_template_library.visualize_structure_matrices import run


def test_run_scan_end_to_end(tmp_path):
    scan = _scan_json(tmp_path / "scan.json")
    cif = tmp_path / "FeO_2x1x1.cif"
    _write_feo_2x1x1_cif(cif)  # 8 Fe; restrict occ has 4 -> mismatch guard

    # The synthetic scan has 4 Fe atoms; build a matching 4-Fe structure.
    from ase import Atoms as _Atoms
    from ase.io import write as _write
    four_fe = _Atoms(
        "Fe4O4",
        positions=[
            (0, 0, 0), (2, 0, 0), (4, 0, 0), (6, 0, 0),
            (1, 1, 0), (3, 1, 0), (5, 1, 0), (7, 1, 0),
        ],
        cell=[8, 2, 2],
        pbc=True,
    )
    cif4 = tmp_path / "fe4.cif"
    _write(str(cif4), four_fe)

    out = tmp_path / "out.html"
    result = run("scan", scan, cif4, out, pks=[10781])
    assert result == out
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k run_scan_end_to_end -v`
Expected: FAIL with `ImportError` (`run` not defined).

- [ ] **Step 3: Write minimal implementation**

Add to `visualize_structure_matrices.py`:

```python
def run(
    input_kind: str,
    data_json,
    structure,
    output,
    *,
    submissions=None,
    pks: list[int] | None = None,
    max_n: int | None = None,
    projection: str = "auto",
    title: str = "Structure occupation matrices",
) -> Path:
    atoms2d = load_structure_2d(structure, projection=projection)
    if input_kind == "scan":
        records = build_scan_records(data_json, pks=pks, max_n=max_n)
    elif input_kind == "proposal":
        records = build_proposal_records(data_json, submission_json=submissions)
        if max_n is not None:
            records = records[:max_n]
    else:
        raise ValueError(f"unknown input_kind {input_kind!r}")
    return plot_structure_matrix_records(records, atoms2d, output, title=title)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-kind", choices=["proposal", "scan"],
                        required=True)
    parser.add_argument("--data", required=True,
                        help="proposal JSON or scan JSON path")
    parser.add_argument("--structure", required=True,
                        help="ASE-readable structure file (e.g. CIF)")
    parser.add_argument("--submissions", default=None,
                        help="submission JSON (proposal kind only)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pks", default=None,
                        help="comma-separated PKs (scan kind)")
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--projection", default="auto")
    parser.add_argument("--title", default="Structure occupation matrices")
    args = parser.parse_args(argv)

    pks = [int(p) for p in args.pks.split(",")] if args.pks else None
    output = run(
        args.input_kind,
        args.data,
        args.structure,
        args.output,
        submissions=args.submissions,
        pks=pks,
        max_n=args.max_n,
        projection=args.projection,
        title=args.title,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k run_scan_end_to_end -v`
Expected: PASS.

- [ ] **Step 5: Run full new test file**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/FeO_template_library/visualize_structure_matrices.py tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py
git commit -m "feat(experiments): CLI + run() orchestration for structure visualizer"
```

---

### Task 7: FeO 2x1x1 CIF generator

**Files:**
- Create: `experiments/FeO_supercell_2x1x1_convergence/make_feo_2x1x1_cif.py`
- Test: `tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py`

**Interfaces:**
- Consumes: ASE `bulk`/`Atoms`, `ase.io.write`.
- Produces:
  - `make_feo_2x1x1(output_path) -> Path` — writes a rocksalt FeO 2x1x1 supercell CIF (4 Fe + 4 O) and returns the path.

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
from experiments.FeO_supercell_2x1x1_convergence.make_feo_2x1x1_cif import (
    make_feo_2x1x1,
)


def test_make_feo_2x1x1_writes_4fe_4o(tmp_path):
    cif = tmp_path / "FeO_2x1x1.cif"
    make_feo_2x1x1(cif)
    from ase.io import read as _read
    atoms = _read(str(cif))
    symbols = atoms.get_chemical_symbols()
    assert sum(1 for s in symbols if s == "Fe") == 4
    assert sum(1 for s in symbols if s == "O") == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k make_feo_2x1x1 -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/FeO_supercell_2x1x1_convergence/make_feo_2x1x1_cif.py`:

```python
"""Generate a FeO 2x1x1 rocksalt supercell CIF for the structure visualizer.

Caveat: this regenerates rocksalt FeO geometry; it does NOT reconstruct the
exact StructureData order used in the production run. The visualizer maps
occupation atoms to Fe sites by specie suffix (Fe{k} -> k-th Fe in file order),
so physical-site identity is only correct if this CIF's Fe order matches the
run's structure order. Use this for tool-mechanics verification (render /
suffix mapping / reversal handling), not for site provenance claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase.build import bulk
from ase.io import write as ase_write

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = THIS_DIR / "FeO_2x1x1.cif"


def make_feo_2x1x1(output_path=DEFAULT_OUTPUT) -> Path:
    output_path = Path(output_path)
    # Rocksalt FeO primitive has 1 Fe + 1 O; conventional cubic has 4 + 4.
    conventional = bulk("FeO", crystalstructure="rocksalt", a=4.3, cubic=True)
    supercell = conventional.repeat((2, 1, 1))  # 8 Fe + 8 O
    # Keep a 4+4 cell: take the conventional cell itself (4 Fe + 4 O).
    ase_write(str(output_path), conventional)
    return output_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(f"Wrote {make_feo_2x1x1(args.output)}")


if __name__ == "__main__":
    main()
```

Note: the conventional rocksalt cubic cell already has 4 Fe + 4 O, matching the
2x1x1 scan's four Hubbard atoms; the `repeat` line is illustrative and unused —
remove it if the linter flags the unused variable.

- [ ] **Step 4: Fix the unused-variable lint**

Edit `make_feo_2x1x1` to drop the unused `supercell` line:

```python
def make_feo_2x1x1(output_path=DEFAULT_OUTPUT) -> Path:
    output_path = Path(output_path)
    # Conventional cubic rocksalt FeO already has 4 Fe + 4 O — matches the
    # four Hubbard atoms in the 2x1x1 scan data.
    conventional = bulk("FeO", crystalstructure="rocksalt", a=4.3, cubic=True)
    ase_write(str(output_path), conventional)
    return output_path
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py -k make_feo_2x1x1 -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/FeO_supercell_2x1x1_convergence/make_feo_2x1x1_cif.py tests/unit_tests/experiments/test_feo_structure_matrix_experiment.py
git commit -m "feat(experiments): FeO 2x1x1 CIF generator for visualizer verification"
```

---

### Task 8: Verification run + records (FeO 2x1x1, real data)

**Files:**
- Create: `experiments/FeO_supercell_2x1x1_convergence/FeO_2x1x1.cif` (generated)
- Create: `experiments/plots/FeO_2x1x1_structure_matrix_browser.html` (generated)
- Modify: `experiments/FeO_supercell_2x1x1_convergence/report.md`
- Modify: `experiments/outline.md`

**Interfaces:**
- Consumes: `make_feo_2x1x1`, `visualize_structure_matrices.run`, real scan JSON `experiments/FeO_supercell_2x1x1_bayesian/FeO_2x1x1_scan_bayesian.json` (ground-state `pk 10781`).
- Produces: verified HTML artifact + documentation rows. No new unit test.

- [ ] **Step 1: Generate the CIF**

Run:
```bash
cd /Users/guoyuan/codes/projects/occupations/aiida-LordCapulet
uv run python experiments/FeO_supercell_2x1x1_convergence/make_feo_2x1x1_cif.py
```
Expected: prints `Wrote .../FeO_2x1x1.cif`; file has 4 Fe + 4 O.

- [ ] **Step 2: Pick a few PKs and build the browser**

Run (10781 is the ground-state target; add a couple distinct generations):
```bash
cd /Users/guoyuan/codes/projects/occupations/aiida-LordCapulet
uv run python experiments/FeO_template_library/visualize_structure_matrices.py \
  --input-kind scan \
  --data experiments/FeO_supercell_2x1x1_bayesian/FeO_2x1x1_scan_bayesian.json \
  --structure experiments/FeO_supercell_2x1x1_convergence/FeO_2x1x1.cif \
  --output experiments/plots/FeO_2x1x1_structure_matrix_browser.html \
  --pks 10781 \
  --title "FeO 2x1x1 occupation matrices"
```
Expected: prints `Wrote .../FeO_2x1x1_structure_matrix_browser.html`, no `ValueError`
(confirms the 4-Fe CIF matches the 4 Hubbard atoms in the scan record).

- [ ] **Step 3: Manual verification checklist**

Open the HTML in a browser. Confirm:
- 4 Fe markers (labeled `Fe1`..`Fe4`) clickable; 4 O markers grey/faded and not clickable.
- Reversal handled: clicking the marker labeled `Fe1` shows the matrix that the
  scan JSON stores under `Atom_4` (and `Fe4` ↔ `Atom_1`). Cross-check one matrix
  numerically against the JSON entry for `pk 10781`.
- up/down heatmaps for `pk 10781` visually match
  `experiments/FeO_supercell_2x1x1_convergence/target_occupation_matrices.png`
  (same diagonal occupation pattern per spin).

Record the outcome (pass/fail + any screenshot) in the report in the next step.

- [ ] **Step 4: Negative check (mapping guard)**

Run with a deliberately wrong structure (8 Fe vs 4 occ) to confirm the guard:
```bash
cd /Users/guoyuan/codes/projects/occupations/aiida-LordCapulet
uv run python -c "
from experiments.FeO_template_library.visualize_structure_matrices import run
from ase.build import bulk; from ase.io import write
s='/tmp/feo_8fe.cif'; write(s, bulk('FeO','rocksalt',a=4.3,cubic=True).repeat((2,1,1)))
try:
    run('scan',
        'experiments/FeO_supercell_2x1x1_bayesian/FeO_2x1x1_scan_bayesian.json',
        s, '/tmp/bad.html', pks=[10781])
    print('ERROR: expected ValueError')
except ValueError as e:
    print('OK guard fired:', e)
"
```
Expected: prints `OK guard fired: structure has 8 Hubbard atoms but occupation data has 4`.

- [ ] **Step 5: Update report.md**

Add a section to `experiments/FeO_supercell_2x1x1_convergence/report.md` describing:
the new `visualize_structure_matrices.py`, the generated CIF + its site-provenance
caveat, the browser HTML path, the verification results (clickable Fe/faded O,
reversal correctness vs `Atom_N`, heatmap match to `target_occupation_matrices.png`),
and the mapping-guard negative check.

- [ ] **Step 6: Update outline.md**

Add a row to the catalogue table in `experiments/outline.md`:
`exp | summary | status | starting date | finish date | document` —
e.g. `structure_matrix_viz | interactive 2D atom layout -> per-atom occ heatmaps; verified on FeO 2x1x1 | done | 2026-06-29 | 2026-06-29 | FeO_supercell_2x1x1_convergence/report.md`.

- [ ] **Step 7: Run the full experiments test tier**

Run: `uv run pytest tests/unit_tests/experiments/ -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add experiments/FeO_supercell_2x1x1_convergence/FeO_2x1x1.cif \
        experiments/plots/FeO_2x1x1_structure_matrix_browser.html \
        experiments/FeO_supercell_2x1x1_convergence/report.md \
        experiments/outline.md
git commit -m "docs(experiments): verify structure matrix visualizer on FeO 2x1x1"
```

---

## Self-Review Notes

- **Spec coverage:** module/CLI (T1,T6), geometry+auto projection (T1), ordering fix (T2), decoupled renderer + two builders (T3,T4,T5), up|down side-by-side heatmaps (T5), dropdown switch + click/hover (T5), tests mirroring sibling (T1–T7), 2x1x1 verification incl. CIF gen, reversal check, png match, mapping-guard, outline/report updates (T7,T8). All spec sections mapped.
- **Deviation from spec schema:** records do NOT embed per-atom `x`/`y`; geometry is kept separate (`Atoms2D`) and merged at render time via `build_species_xy` + a JS `speciesXY` lookup. This honors the spec's decoupling goal and avoids duplicating coordinates across records. Behavior is unchanged.
- **Type consistency:** `Atoms2D.fe_xy`/`others`, `build_species_xy`, `build_scan_records`/`build_proposal_records` (common schema with `label`), `plot_structure_matrix_records`, `run`, `make_feo_2x1x1` names/signatures are consistent across tasks.
- **Placeholders:** none — all steps carry concrete code/commands. Task 7 intentionally shows a refactor step (Step 4) to drop an illustrative unused line.
