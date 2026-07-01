# Proposal-Mode Metadata + GP Random Schedule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) Stamp every proposal with its origin (`random` / `random_so_n` / `gp` / `gp` warmup random / `gp` fallback random) so the final result reveals proposal source per calc. (2) Let users configure the GP warmup as `N_random_generations` of `N_random_per_generation` proposals each, instead of the current "always one warmup gen of size N".

**Architecture:**
- Add an optional top-level `metadata: dict` to `OccupationMatrixData`, embedded in `as_dict()` under a reserved key `__metadata__` (back-compat: empty metadata = current shape).
- Each proposer (`random`, `random_so_n`, `gp`) stamps `proposal_mode` on every proposal it returns. The dispatcher (`propose_new_constraints`) layers `current_generation` and `gp_warmup` / `gp_fallback` flags over that.
- The GP branch in `propose_new_constraints` treats the first `N_random_generations` generations as random warmup. Each warmup generation gets its own batch size via the workchain (which reads `N_random_per_generation`).
- The extractor surfaces proposal metadata by walking `calc.inputs.target_matrix.obj.metadata` for `ConstrainedPWCalculation` nodes.

**Tech Stack:** Python 3, AiiDA + JsonableData, pytest, `uv` for env management.

**Run all tests via:** `uv run pytest <path>` — never bare `python`/`pytest`. Project venv is in `.venv/`.

---

## File Structure

**Modify:**
- `lordcapulet/data_structures/occupation_matrix.py` — add `metadata` attribute, accessors, `as_dict`/`from_dict` round-trip via `__metadata__` reserved key.
- `lordcapulet/functions/proposal_modes/random_mode.py` — stamp `proposal_mode='random'` on every proposal.
- `lordcapulet/functions/proposal_modes/random_so_n_mode.py` — stamp `proposal_mode='random_so_n'`.
- `lordcapulet/functions/proposal_modes/gaussian_process.py` — stamp `proposal_mode='gp'`.
- `lordcapulet/functions/propose.py` — replace gen=0 special case with `N_random_generations` schedule, layer `current_generation` / `gp_warmup` / `gp_fallback` over per-proposer metadata.
- `lordcapulet/workflows/global_constrained_search.py` — let `run_constrained_batch` pick batch size from `N_random_generations` / `N_random_per_generation` kwargs.
- `lordcapulet/utils/postprocessing/gather_workchain_data.py` — new field `proposal_metadata` on each calc, populated from the input `target_matrix` for constrained calcs.

**Create:**
- `tests/unit_tests/data_structures/test_occupation_matrix_metadata.py`
- `tests/unit_tests/functions/test_proposal_metadata.py`
- `tests/unit_tests/functions/test_propose_dispatch_schedule.py`
- `tests/unit_tests/workflows/__init__.py` (if missing)
- `tests/unit_tests/workflows/test_global_search_random_schedule.py`

---

## Key Design Choices (locked, do not deviate)

1. **Reserved key for metadata**: `__metadata__`. Atom labels follow the `Atom_N` convention (see [occupation_matrix.py:82](lordcapulet/data_structures/occupation_matrix.py#L82)) and never start with `__`, so collision is impossible. Stored separately on `_metadata` attribute; only embedded in `as_dict()` when non-empty.
2. **Empty metadata = old shape**. `as_dict()` MUST return a plain `{atom_label: {...}}` dict when `metadata` is empty (or absent), so existing JsonableData nodes on the AiiDA DB and existing tests like `test_occupation_matrix.py:108` keep passing untouched.
3. **Each proposer owns its `proposal_mode` label**. The dispatcher only adds *additional* fields (e.g., `current_generation`, `gp_warmup=True`, `gp_fallback=True`); it does NOT overwrite `proposal_mode`.
4. **GP fallback path stamps `gp_fallback=True` AND `proposal_mode='random'`** (the random function ran and stamped its own label). The dispatcher layers `gp_fallback=True` so post-hoc analysis can tell "I asked for GP, got random".
5. **GP warmup vs fallback are distinct**:
   - Warmup: scheduled via `N_random_generations`. Flag: `gp_warmup=True`.
   - Fallback: GP raised an exception. Flag: `gp_fallback=True`.
6. **Workchain-side batch sizing**: only `run_constrained_batch` reads `N_random_generations` / `N_random_per_generation`. The proposer trusts whatever `N` the workchain passes.
7. **Backward compat**: if user does NOT pass `N_random_generations`, default is `1` (one warmup gen — matches current behavior). If user does NOT pass `N_random_per_generation`, default is `self.inputs.N.value` (matches current behavior). The legacy `N_initial_random` kwarg is removed; users migrate to the new keys.

---

## Task 1: OccupationMatrixData metadata field — round-trip

**Files:**
- Modify: `lordcapulet/data_structures/occupation_matrix.py:41-57` (`__init__`, `as_dict`, `from_dict`)
- Test: `tests/unit_tests/data_structures/test_occupation_matrix_metadata.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/unit_tests/data_structures/test_occupation_matrix_metadata.py`:

```python
"""Round-trip tests for the optional metadata attribute on OccupationMatrixData."""

import pytest

from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData


SIMPLE_DATA = {
    'Atom_1': {
        'specie': 'Fe',
        'shell': '3d',
        'occupation_matrix': {
            'up': [[1.0, 0.0], [0.0, 0.5]],
            'down': [[0.5, 0.0], [0.0, 0.5]],
        },
    }
}


class TestOccupationMatrixMetadata:
    def test_default_metadata_is_empty_dict(self):
        occ = OccupationMatrixData(SIMPLE_DATA)
        assert occ.metadata == {}

    def test_init_with_metadata(self):
        occ = OccupationMatrixData(SIMPLE_DATA, metadata={'proposal_mode': 'random'})
        assert occ.metadata == {'proposal_mode': 'random'}

    def test_set_metadata_replaces(self):
        occ = OccupationMatrixData(SIMPLE_DATA)
        occ.set_metadata({'proposal_mode': 'gp'})
        assert occ.metadata == {'proposal_mode': 'gp'}

    def test_update_metadata_merges(self):
        occ = OccupationMatrixData(SIMPLE_DATA, metadata={'proposal_mode': 'random'})
        occ.update_metadata(current_generation=2)
        assert occ.metadata == {'proposal_mode': 'random', 'current_generation': 2}

    def test_as_dict_omits_metadata_key_when_empty(self):
        occ = OccupationMatrixData(SIMPLE_DATA)
        assert occ.as_dict() == SIMPLE_DATA
        assert '__metadata__' not in occ.as_dict()

    def test_as_dict_embeds_metadata_when_non_empty(self):
        occ = OccupationMatrixData(SIMPLE_DATA, metadata={'proposal_mode': 'gp'})
        out = occ.as_dict()
        assert out['__metadata__'] == {'proposal_mode': 'gp'}
        # Atom data still present and untouched
        assert out['Atom_1'] == SIMPLE_DATA['Atom_1']

    def test_from_dict_pops_metadata(self):
        wire = {**SIMPLE_DATA, '__metadata__': {'proposal_mode': 'random_so_n'}}
        occ = OccupationMatrixData.from_dict(wire)
        assert occ.metadata == {'proposal_mode': 'random_so_n'}
        # Internal _data should NOT contain __metadata__
        assert '__metadata__' not in occ.data
        assert occ.data == SIMPLE_DATA

    def test_round_trip_preserves_metadata(self):
        meta = {'proposal_mode': 'gp', 'current_generation': 3, 'gp_warmup': False}
        occ1 = OccupationMatrixData(SIMPLE_DATA, metadata=meta)
        occ2 = OccupationMatrixData.from_dict(occ1.as_dict())
        assert occ2.metadata == meta
        assert occ2.data == occ1.data

    def test_round_trip_legacy_dict_no_metadata(self):
        """A dict with no __metadata__ key must round-trip cleanly to empty metadata."""
        occ = OccupationMatrixData.from_dict(SIMPLE_DATA)
        assert occ.metadata == {}
        assert occ.as_dict() == SIMPLE_DATA

    def test_iteration_does_not_yield_metadata_key(self):
        occ = OccupationMatrixData(SIMPLE_DATA, metadata={'proposal_mode': 'random'})
        assert list(iter(occ)) == ['Atom_1']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit_tests/data_structures/test_occupation_matrix_metadata.py -v`
Expected: 9 tests fail (no `metadata` attribute / `set_metadata` / `update_metadata` / `__metadata__` handling).

- [ ] **Step 3: Patch `OccupationMatrixData.__init__` and add metadata accessors**

In `lordcapulet/data_structures/occupation_matrix.py`, replace the `__init__`, `as_dict`, and `from_dict` methods (currently at [lines 41-57](lordcapulet/data_structures/occupation_matrix.py#L41-L57)) with:

```python
    METADATA_KEY = '__metadata__'

    def __init__(self, data: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """Initialize OccupationMatrixData with optional initial data and metadata.

        ``metadata`` is a free-form dict for tagging the matrix (e.g. proposal
        provenance). It is round-tripped through ``as_dict`` / ``from_dict`` via
        the reserved key ``__metadata__`` and is otherwise invisible to atom
        iteration / atom-keyed access.
        """
        self._data = data if data is not None else {}
        self._metadata = dict(metadata) if metadata else {}

    @property
    def data(self) -> Dict[str, Any]:
        """Get the internal atom-keyed data dictionary (no metadata)."""
        return self._data

    @property
    def metadata(self) -> Dict[str, Any]:
        """Free-form metadata dict (e.g. proposal provenance)."""
        return self._metadata

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Replace the metadata dict wholesale."""
        self._metadata = dict(metadata) if metadata else {}

    def update_metadata(self, **kwargs: Any) -> None:
        """Merge keyword args into metadata (in place)."""
        self._metadata.update(kwargs)

    def as_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary representation.

        If ``metadata`` is non-empty, embed it under the reserved key
        ``__metadata__``; otherwise return the atom-keyed dict unchanged
        (so existing JsonableData nodes stored on the AiiDA DB round-trip
        cleanly).
        """
        if self._metadata:
            return {**self._data, self.METADATA_KEY: dict(self._metadata)}
        return self._data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OccupationMatrixData':
        """Create instance from dictionary, popping ``__metadata__`` if present."""
        if data is None:
            return cls()
        # Don't mutate caller's dict
        atom_data = dict(data)
        metadata = atom_data.pop(cls.METADATA_KEY, None)
        return cls(atom_data, metadata=metadata)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit_tests/data_structures/test_occupation_matrix_metadata.py -v`
Expected: 9 passed.

- [ ] **Step 5: Run the existing OccupationMatrixData test suite to confirm no regression**

Run: `uv run pytest tests/unit_tests/data_structures/test_occupation_matrix.py -v`
Expected: all existing tests still pass (`as_dict() == simple_2x2_identity` etc. unchanged because metadata is empty).

- [ ] **Step 6: Commit**

```bash
git add lordcapulet/data_structures/occupation_matrix.py tests/unit_tests/data_structures/test_occupation_matrix_metadata.py
git commit -m "feat(occ_matrix): add optional metadata attribute with __metadata__ round-trip"
```

---

## Task 2: Stamp proposal_mode in random mode

**Files:**
- Modify: `lordcapulet/functions/proposal_modes/random_mode.py:103-105`
- Test: `tests/unit_tests/functions/test_proposal_metadata.py` (new)

- [ ] **Step 1: Write the failing test file**

Create `tests/unit_tests/functions/test_proposal_metadata.py`:

```python
"""Each proposer must stamp proposal_mode on every returned OccupationMatrixData."""

import numpy as np
import pytest

from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
from lordcapulet.functions.proposal_modes.random_mode import propose_random_constraints
from lordcapulet.functions.proposal_modes.random_so_n_mode import propose_random_so_n_constraints


def _sample_occ_list(n=3, dim=5):
    out = []
    for _ in range(n):
        data = {
            'Atom_1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': np.diag(np.random.rand(dim)).tolist(),
                    'down': np.diag(np.random.rand(dim)).tolist(),
                },
            }
        }
        out.append(OccupationMatrixData(data))
    return out


class TestRandomModeStampsMetadata:
    def test_random_mode_stamps_proposal_mode(self):
        proposals = propose_random_constraints(
            _sample_occ_list(), natoms=1, N=4, debug=False
        )
        assert len(proposals) == 4
        for p in proposals:
            assert p.metadata.get('proposal_mode') == 'random'


class TestRandomSoNModeStampsMetadata:
    def test_random_so_n_mode_stamps_proposal_mode(self):
        proposals = propose_random_so_n_constraints(
            _sample_occ_list(), natoms=1, N=4, debug=False
        )
        assert len(proposals) == 4
        for p in proposals:
            assert p.metadata.get('proposal_mode') == 'random_so_n'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit_tests/functions/test_proposal_metadata.py -v`
Expected: 2 failures (`metadata` is empty dict).

- [ ] **Step 3: Patch `propose_random_constraints` to stamp the mode**

In `lordcapulet/functions/proposal_modes/random_mode.py`, change [line 104-105](lordcapulet/functions/proposal_modes/random_mode.py#L104-L105) from:

```python
        # Create OccupationMatrixData from proposal
        proposal = OccupationMatrixData(proposal_data)
        proposals.append(proposal)
```

to:

```python
        # Create OccupationMatrixData from proposal, stamped with provenance
        proposal = OccupationMatrixData(
            proposal_data,
            metadata={'proposal_mode': 'random'},
        )
        proposals.append(proposal)
```

- [ ] **Step 4: Run the random-mode test to confirm it passes**

Run: `uv run pytest tests/unit_tests/functions/test_proposal_metadata.py::TestRandomModeStampsMetadata -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lordcapulet/functions/proposal_modes/random_mode.py tests/unit_tests/functions/test_proposal_metadata.py
git commit -m "feat(proposers): stamp proposal_mode='random' on random-mode proposals"
```

---

## Task 3: Stamp proposal_mode in random_so_n mode

**Files:**
- Modify: `lordcapulet/functions/proposal_modes/random_so_n_mode.py:133-135`

- [ ] **Step 1: Patch `propose_random_so_n_constraints`**

In `lordcapulet/functions/proposal_modes/random_so_n_mode.py`, change [lines 133-135](lordcapulet/functions/proposal_modes/random_so_n_mode.py#L133-L135) from:

```python
        # Create OccupationMatrixData from proposal
        proposal = OccupationMatrixData(proposal_data)
        proposals.append(proposal)
```

to:

```python
        # Create OccupationMatrixData from proposal, stamped with provenance
        proposal = OccupationMatrixData(
            proposal_data,
            metadata={'proposal_mode': 'random_so_n'},
        )
        proposals.append(proposal)
```

- [ ] **Step 2: Run the test to confirm it passes**

Run: `uv run pytest tests/unit_tests/functions/test_proposal_metadata.py::TestRandomSoNModeStampsMetadata -v`
Expected: PASS.

- [ ] **Step 3: Run the existing proposal-mode regression suite**

Run: `uv run pytest tests/unit_tests/functions/test_proposal_modes.py -v`
Expected: all existing tests still pass (`proposal.data['Atom_1']['specie'] == 'Fe1'` etc. — metadata is on a different attr, atom data is untouched).

- [ ] **Step 4: Commit**

```bash
git add lordcapulet/functions/proposal_modes/random_so_n_mode.py
git commit -m "feat(proposers): stamp proposal_mode='random_so_n' on so_n proposals"
```

---

## Task 4: Stamp proposal_mode in GP mode

**Files:**
- Modify: `lordcapulet/functions/proposal_modes/gaussian_process.py:425-432`
- Test: extend `tests/unit_tests/functions/test_proposal_metadata.py`

GP returns proposals via `databank.from_pytorch(...)` ([gaussian_process.py:426-429](lordcapulet/functions/proposal_modes/gaussian_process.py#L426-L429)) which constructs fresh `OccupationMatrixData` objects with no metadata. We stamp after that call.

- [ ] **Step 1: Add the failing GP test (skipped if torch missing)**

Append to `tests/unit_tests/functions/test_proposal_metadata.py`:

```python
class TestGaussianProcessModeStampsMetadata:
    def test_gp_mode_stamps_proposal_mode(self):
        torch = pytest.importorskip('torch')
        pytest.importorskip('botorch')
        from lordcapulet.functions.proposal_modes.gaussian_process import (
            propose_gaussian_process_constraints,
        )

        occ_list = _sample_occ_list(n=4, dim=5)
        energies = [-100.0, -100.5, -99.5, -101.0]
        proposals = propose_gaussian_process_constraints(
            occ_matr_list=occ_list,
            energies=energies,
            natoms=1,
            N=2,
            debug=False,
        )
        assert len(proposals) == 2
        for p in proposals:
            assert p.metadata.get('proposal_mode') == 'gp'
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit_tests/functions/test_proposal_metadata.py::TestGaussianProcessModeStampsMetadata -v`
Expected: FAIL (`metadata` is empty).

If torch/botorch are not installed locally, the test will skip — in that case run the dispatcher-level test in Task 5 to drive the change instead.

- [ ] **Step 3: Stamp metadata after GP candidates are converted**

In `lordcapulet/functions/proposal_modes/gaussian_process.py`, change [lines 425-432](lordcapulet/functions/proposal_modes/gaussian_process.py#L425-L432) from:

```python
        # convert candidates to OccupationMatrixData
        proposals = databank.from_pytorch( 
            matrices=candidates,
            atom_ids=atoms,
            spins=['up', 'down'])


    return proposals
```

to:

```python
        # convert candidates to OccupationMatrixData
        proposals = databank.from_pytorch(
            matrices=candidates,
            atom_ids=atoms,
            spins=['up', 'down'])

        # Stamp provenance so post-hoc analysis can tell GP from random.
        for proposal in proposals:
            proposal.update_metadata(proposal_mode='gp')

    return proposals
```

(Indentation matters: this stays inside the `if optim_strategy == "Boltzmann":` block, where `proposals` is defined.)

- [ ] **Step 4: Run the GP test to confirm it passes**

Run: `uv run pytest tests/unit_tests/functions/test_proposal_metadata.py::TestGaussianProcessModeStampsMetadata -v`
Expected: PASS (or SKIP if torch missing — that's acceptable; the dispatcher test in Task 5 will exercise the same line).

- [ ] **Step 5: Commit**

```bash
git add lordcapulet/functions/proposal_modes/gaussian_process.py tests/unit_tests/functions/test_proposal_metadata.py
git commit -m "feat(proposers): stamp proposal_mode='gp' on GP proposals"
```

---

## Task 5: Dispatcher schedules GP warmup, layers current_generation / gp_warmup / gp_fallback

**Files:**
- Modify: `lordcapulet/functions/propose.py:181-273` (`propose_new_constraints`)
- Test: `tests/unit_tests/functions/test_propose_dispatch_schedule.py` (new)

This is the central change for Goal 2. The current implementation ([propose.py:225-269](lordcapulet/functions/propose.py#L225-L269)) hard-codes "if `current_generation == 0` use random with `N_initial_random` count, else use GP". Replace with a generic `N_random_generations` schedule. The dispatcher also stamps `current_generation` on every proposal it returns and tags warmup/fallback.

- [ ] **Step 1: Write the failing test file**

Create `tests/unit_tests/functions/test_propose_dispatch_schedule.py`:

```python
"""Dispatcher (`propose_new_constraints`) schedules GP warmup and layers metadata."""

import numpy as np
import pytest

from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData
from lordcapulet.functions import propose as propose_module
from lordcapulet.functions.propose import propose_new_constraints


def _sample_occ_list(n=4, dim=5):
    out = []
    for _ in range(n):
        data = {
            'Atom_1': {
                'specie': 'Fe',
                'shell': '3d',
                'occupation_matrix': {
                    'up': np.diag(np.random.rand(dim)).tolist(),
                    'down': np.diag(np.random.rand(dim)).tolist(),
                },
            }
        }
        out.append(OccupationMatrixData(data))
    return out


class TestDispatcherStampsCurrentGeneration:
    def test_random_mode_stamps_current_generation(self):
        proposals = propose_new_constraints(
            _sample_occ_list(), N=3, mode='random', debug=False,
            current_generation=2,
        )
        for p in proposals:
            assert p.metadata.get('current_generation') == 2
            assert p.metadata.get('proposal_mode') == 'random'

    def test_random_mode_no_current_generation_kwarg_is_ok(self):
        """current_generation is optional for non-GP modes."""
        proposals = propose_new_constraints(
            _sample_occ_list(), N=3, mode='random_so_n', debug=False,
        )
        for p in proposals:
            assert p.metadata.get('proposal_mode') == 'random_so_n'
            # No current_generation kwarg -> not stamped
            assert 'current_generation' not in p.metadata


class TestGpWarmupSchedule:
    def test_default_warmup_is_one_generation(self):
        """No N_random_generations kwarg -> first generation (gen 0) random, then GP."""
        proposals = propose_new_constraints(
            _sample_occ_list(), N=3, mode='gp', debug=False,
            energies=[-100.0, -100.5, -99.5, -101.0],
            current_generation=0,
        )
        for p in proposals:
            assert p.metadata.get('proposal_mode') == 'random'
            assert p.metadata.get('gp_warmup') is True
            assert p.metadata.get('current_generation') == 0

    def test_warmup_three_generations_gen0_is_random(self):
        proposals = propose_new_constraints(
            _sample_occ_list(), N=3, mode='gp', debug=False,
            energies=[-100.0, -100.5, -99.5, -101.0],
            current_generation=0,
            N_random_generations=3,
        )
        for p in proposals:
            assert p.metadata.get('proposal_mode') == 'random'
            assert p.metadata.get('gp_warmup') is True

    def test_warmup_three_generations_gen2_is_random(self):
        proposals = propose_new_constraints(
            _sample_occ_list(), N=3, mode='gp', debug=False,
            energies=[-100.0, -100.5, -99.5, -101.0],
            current_generation=2,
            N_random_generations=3,
        )
        for p in proposals:
            assert p.metadata.get('proposal_mode') == 'random'
            assert p.metadata.get('gp_warmup') is True

    def test_warmup_three_generations_gen3_is_gp(self, monkeypatch):
        """At gen 3 with N_random_generations=3, GP branch is taken (gen >= warmup count)."""
        # Stub the GP proposer so we don't pull in torch in this test.
        called = {}

        def fake_gp(occ_matr_list, energies, natoms, N, gp_config=None,
                    debug=False, reporter=None, **kwargs):
            called['args'] = (len(occ_matr_list), len(energies), N)
            return [OccupationMatrixData(
                occ_matr_list[0].data,
                metadata={'proposal_mode': 'gp'},
            ) for _ in range(N)]

        monkeypatch.setattr(
            propose_module, 'propose_gaussian_process_constraints', fake_gp,
        )

        proposals = propose_new_constraints(
            _sample_occ_list(), N=2, mode='gp', debug=False,
            energies=[-100.0, -100.5, -99.5, -101.0],
            current_generation=3,
            N_random_generations=3,
        )
        assert called['args'] == (4, 4, 2)
        for p in proposals:
            assert p.metadata.get('proposal_mode') == 'gp'
            assert p.metadata.get('current_generation') == 3
            assert p.metadata.get('gp_warmup', False) is False
            assert p.metadata.get('gp_fallback', False) is False


class TestGpFallback:
    def test_gp_exception_falls_back_and_tags(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError('GP exploded')

        monkeypatch.setattr(
            propose_module, 'propose_gaussian_process_constraints', boom,
        )

        proposals = propose_new_constraints(
            _sample_occ_list(), N=3, mode='gp', debug=False,
            energies=[-100.0, -100.5, -99.5, -101.0],
            current_generation=5,
            N_random_generations=1,
        )
        assert len(proposals) == 3
        for p in proposals:
            assert p.metadata.get('proposal_mode') == 'random'
            assert p.metadata.get('gp_fallback') is True
            assert p.metadata.get('gp_warmup', False) is False
            assert p.metadata.get('current_generation') == 5


class TestGpModeRequiresCurrentGeneration:
    def test_gp_without_current_generation_raises(self):
        with pytest.raises(AssertionError):
            propose_new_constraints(
                _sample_occ_list(), N=3, mode='gp', debug=False,
                energies=[-100.0, -100.5, -99.5, -101.0],
            )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit_tests/functions/test_propose_dispatch_schedule.py -v`
Expected: most tests fail. Specifically:
- `test_random_mode_stamps_current_generation` → fails (dispatcher does not stamp `current_generation` for non-GP modes today)
- `test_default_warmup_is_one_generation` → may pass for proposal_mode but fails for `gp_warmup=True` flag (not currently stamped)
- `test_warmup_three_generations_*` → fail (`N_random_generations` is unsupported today)
- `test_gp_exception_falls_back_and_tags` → fails (`gp_fallback` flag not stamped)

- [ ] **Step 3: Replace the GP branch in `propose_new_constraints` with the new schedule**

In `lordcapulet/functions/propose.py`, replace the `match mode:` block ([lines 217-273](lordcapulet/functions/propose.py#L217-L273)) with:

```python
    # Pull out current_generation early so we can stamp it on every proposal,
    # regardless of mode. None for non-GP modes is fine; we just don't stamp.
    current_generation = kwargs.pop('current_generation', None)

    # implement case switch for mode
    match mode:

        case 'random':
            proposals = propose_random_constraints(
                occ_matr_list, natoms, N, debug=debug, **kwargs
            )

        case 'random_so_n':
            proposals = propose_random_so_n_constraints(
                occ_matr_list, natoms, N, debug=debug, **kwargs
            )

        case 'gaussian_process' | 'gp':
            # Pop energies (required by GP) and gp_config from kwargs before
            # forwarding the remaining kwargs to either branch.
            energies = kwargs.pop('energies', None)
            if energies is None:
                raise ValueError(
                    "Energies must be provided for Gaussian Process proposal mode"
                )
            gp_config = kwargs.pop('gp_config', None)

            assert current_generation is not None, (
                "current_generation must be provided in kwargs for GP mode"
            )

            # GP warmup schedule: first N_random_generations generations use
            # random proposals to seed the GP training set. Default = 1, which
            # matches the legacy behaviour (only generation 0 is random).
            N_random_generations = kwargs.pop('N_random_generations', 1)

            if debug:
                reporter(f"Energies provided: {energies}")
                reporter(
                    f"GP schedule: current_generation={current_generation}, "
                    f"N_random_generations={N_random_generations}"
                )

            if current_generation < N_random_generations:
                reporter(
                    f"Generation {current_generation} is in GP warmup "
                    f"({current_generation} < {N_random_generations}); "
                    f"using {N} random proposals."
                )
                proposals = propose_random_constraints(
                    occ_matr_list, natoms, N, debug=debug, **kwargs
                )
                # Tag this batch as GP-warmup random.
                for p in proposals:
                    p.update_metadata(gp_warmup=True)
            else:
                reporter(
                    f"Generation {current_generation} >= {N_random_generations}; "
                    f"proposing {N} candidates via Gaussian Process."
                )
                try:
                    proposals = propose_gaussian_process_constraints(
                        occ_matr_list, energies, natoms, N,
                        gp_config=gp_config, debug=debug, reporter=reporter,
                        **kwargs,
                    )
                except Exception as e:
                    reporter(f"Error in Gaussian Process proposal generation: {e}")
                    import traceback
                    reporter(traceback.format_exc())
                    reporter("Falling back to random constraint proposals")
                    proposals = propose_random_constraints(
                        occ_matr_list, natoms, N, debug=debug, **kwargs
                    )
                    for p in proposals:
                        p.update_metadata(gp_fallback=True)

        case 'read':
            raise NotImplementedError("The 'read' mode needs to be implemented")

        case _:
            raise ValueError(f"Unknown proposal mode: {mode!r}")

    # Layer current_generation on every proposal (for non-GP modes too, when
    # the caller chose to pass it). Per-proposer mode label is preserved.
    if current_generation is not None:
        for p in proposals:
            p.update_metadata(current_generation=current_generation)

    return proposals
```

Notes:
- The `case _:` arm is a small bonus that turns the `test_invalid_mode_raises` assertion (currently relying on `UnboundLocalError`) into an explicit `ValueError`. The existing test uses `pytest.raises(Exception)` so either passes.
- `propose_gaussian_process_constraints` is imported at the top of `propose.py`; the monkeypatch in tests replaces it on the `propose` module namespace, so do NOT change to a local import.

- [ ] **Step 4: Run the dispatcher tests**

Run: `uv run pytest tests/unit_tests/functions/test_propose_dispatch_schedule.py -v`
Expected: all pass.

- [ ] **Step 5: Run the existing dispatcher regression**

Run: `uv run pytest tests/unit_tests/functions/test_propose.py -v`
Expected: all pass. Specifically `test_gp_mode_generation_zero_falls_back_to_random` ([test_propose.py:95-108](tests/unit_tests/functions/test_propose.py#L95-L108)) keeps passing because the new default `N_random_generations=1` makes generation 0 a warmup-random batch (still 3 proposals).

- [ ] **Step 6: Commit**

```bash
git add lordcapulet/functions/propose.py tests/unit_tests/functions/test_propose_dispatch_schedule.py
git commit -m "feat(propose): N_random_generations schedule + current_generation/gp_warmup/gp_fallback metadata"
```

---

## Task 6: Workchain consumes N_random_generations / N_random_per_generation

**Files:**
- Modify: `lordcapulet/workflows/global_constrained_search.py:332-360` (`run_constrained_batch`)
- Test: `tests/unit_tests/workflows/test_global_search_random_schedule.py` (new)
- Test (parent dir init): `tests/unit_tests/workflows/__init__.py` (create if missing)

The workchain currently always passes `N=self.inputs.N` to the proposer ([global_constrained_search.py:316,446](lordcapulet/workflows/global_constrained_search.py#L316)). With this change, it picks the batch size based on the current generation.

- [ ] **Step 1: Ensure the test directory exists**

```bash
mkdir -p tests/unit_tests/workflows
```

If `tests/unit_tests/workflows/__init__.py` does not exist, create it as an empty file. Also update `tests/conftest.py` `_test_priority` if the new `workflows` unit-test path needs an explicit tier (see [conftest.py](tests/conftest.py)). Inspect `_test_priority` and append `'unit_tests/workflows'` next to the closest sibling (e.g. between `unit_tests/data_structures` and `unit_tests/calculations`); skip this step if the path already inherits a sensible default.

- [ ] **Step 2: Write the failing test (pure unit test of the batch-size helper)**

We're going to extract the batch-size logic into a tiny pure-Python helper on the workchain class so we can unit-test it without spinning up an AiiDA daemon.

Create `tests/unit_tests/workflows/test_global_search_random_schedule.py`:

```python
"""Unit tests for the batch-size helper used by GlobalConstrainedSearchWorkChain.

The helper computes how many proposals to run in a given generation, given
the workchain inputs (N, Nmax) and the GP warmup schedule
(N_random_generations, N_random_per_generation).
"""

import pytest

from lordcapulet.workflows.global_constrained_search import (
    GlobalConstrainedSearchWorkChain,
)


def _batch(generation, N, Nmax, cumulative,
           N_random_generations=1, N_random_per_generation=None):
    return GlobalConstrainedSearchWorkChain._compute_batch_size(
        generation=generation,
        N=N,
        Nmax=Nmax,
        cumulative=cumulative,
        N_random_generations=N_random_generations,
        N_random_per_generation=N_random_per_generation,
    )


class TestComputeBatchSize:
    def test_default_no_random_override_uses_N(self):
        # gen 1, default warmup=1, no override -> regular N
        assert _batch(generation=1, N=10, Nmax=200, cumulative=4) == 10

    def test_warmup_with_override_uses_random_per_generation(self):
        # gen 1, warmup=3 with per-gen override 25 -> use 25
        assert _batch(
            generation=1, N=10, Nmax=200, cumulative=4,
            N_random_generations=3, N_random_per_generation=25,
        ) == 25

    def test_warmup_without_override_uses_N(self):
        # gen 1, warmup=3, no override -> still N
        assert _batch(
            generation=1, N=10, Nmax=200, cumulative=4,
            N_random_generations=3,
        ) == 10

    def test_post_warmup_uses_N_even_with_override(self):
        # gen 5, warmup=3 -> override does not apply; use N
        assert _batch(
            generation=5, N=10, Nmax=200, cumulative=40,
            N_random_generations=3, N_random_per_generation=25,
        ) == 10

    def test_capped_by_remaining_budget(self):
        # Only 3 proposals left in budget -> batch capped at 3
        assert _batch(generation=2, N=10, Nmax=20, cumulative=17) == 3

    def test_override_capped_by_remaining_budget(self):
        # Override 25, but only 7 left
        assert _batch(
            generation=1, N=10, Nmax=20, cumulative=13,
            N_random_generations=3, N_random_per_generation=25,
        ) == 7

    def test_zero_remaining_budget_returns_zero(self):
        assert _batch(generation=1, N=10, Nmax=20, cumulative=20) == 0
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/unit_tests/workflows/test_global_search_random_schedule.py -v`
Expected: AttributeError — `GlobalConstrainedSearchWorkChain` has no `_compute_batch_size` yet.

- [ ] **Step 4: Add the helper as a static method**

In `lordcapulet/workflows/global_constrained_search.py`, add this method to `GlobalConstrainedSearchWorkChain` (e.g. right above `run_constrained_batch` near [line 332](lordcapulet/workflows/global_constrained_search.py#L332)):

```python
    @staticmethod
    def _compute_batch_size(
        generation: int,
        N: int,
        Nmax: int,
        cumulative: int,
        N_random_generations: int = 1,
        N_random_per_generation: int | None = None,
    ) -> int:
        """Pick the batch size for the given generation.

        - During GP warmup (``generation < N_random_generations``) and when
          ``N_random_per_generation`` is set, the warmup batch size is used.
        - Otherwise the regular ``N`` is used.
        - The result is capped by the remaining global budget (``Nmax - cumulative``).
        """
        if (
            N_random_per_generation is not None
            and generation < N_random_generations
        ):
            base = N_random_per_generation
        else:
            base = N
        remaining = max(0, Nmax - cumulative)
        return min(base, remaining)
```

- [ ] **Step 5: Run the helper test**

Run: `uv run pytest tests/unit_tests/workflows/test_global_search_random_schedule.py -v`
Expected: all pass.

- [ ] **Step 6: Wire the helper into `run_constrained_batch` and the proposal call sites**

In `lordcapulet/workflows/global_constrained_search.py`:

(a) Replace the body of [run_constrained_batch (lines 336-359)](lordcapulet/workflows/global_constrained_search.py#L336-L359) — specifically the `n_proposals = min(...)` line and the slice that follows — with a call to `_compute_batch_size`. From:

```python
    def run_constrained_batch(self):
        """
        Run a batch of constrained calculations with the current proposed matrices.
        """
        self.ctx.generation += 1
        n_proposals = min(self.inputs.N.value, 
                         self.inputs.Nmax.value - self.ctx.N_cumulative)
        
        self.report(f"Starting generation {self.ctx.generation} with {n_proposals} proposals")
        
        # Take only the number of proposals we need
        # THIS MIGHT NEED A CHANGE IF ONE NEEDS TO CHANGET THE NUMBER
        # OF PROPOSALS PER GENERATION, FOR INSTANCE FOR INITIAL GENERATION
        # OF GAUSSIAN PROCESS PROPOSALS
        
        current_proposals = self.ctx.current_proposals[:n_proposals]
```

to:

```python
    def run_constrained_batch(self):
        """
        Run a batch of constrained calculations with the current proposed matrices.
        """
        self.ctx.generation += 1
        proposal_kwargs_dict = (
            self.inputs.proposal_kwargs.get_dict()
            if 'proposal_kwargs' in self.inputs else {}
        )
        n_proposals = self._compute_batch_size(
            generation=self.ctx.generation - 1,  # batch produced for prev-gen index
            N=self.inputs.N.value,
            Nmax=self.inputs.Nmax.value,
            cumulative=self.ctx.N_cumulative,
            N_random_generations=proposal_kwargs_dict.get('N_random_generations', 1),
            N_random_per_generation=proposal_kwargs_dict.get('N_random_per_generation'),
        )

        self.report(f"Starting generation {self.ctx.generation} with {n_proposals} proposals")

        current_proposals = self.ctx.current_proposals[:n_proposals]
```

Note on the generation index: proposals for the constrained batch *N* were generated by the dispatcher's previous call (which had `current_generation = self.ctx.generation - 1` at submission time — see (b) below). The schedule decision is keyed on the dispatcher's `current_generation`, hence the `-1` offset here.

(b) The two `aiida_propose_occ_matrices_from_results` call sites ([line 313](lordcapulet/workflows/global_constrained_search.py#L313) in `process_mag_scan_results` and [line 443](lordcapulet/workflows/global_constrained_search.py#L443) in `process_constrained_results`) currently pass `N=self.inputs.N`. Update them to pass the same batch size the workchain will actually consume next, so the proposer doesn't generate too many. Compute it as:

```python
        next_generation = self.ctx.generation  # the upcoming run_constrained_batch will set generation = next_generation+1
        proposal_kwargs_dict = (
            self.inputs.proposal_kwargs.get_dict()
            if 'proposal_kwargs' in self.inputs else {}
        )
        next_batch = self._compute_batch_size(
            generation=next_generation,
            N=self.inputs.N.value,
            Nmax=self.inputs.Nmax.value,
            cumulative=self.ctx.N_cumulative,
            N_random_generations=proposal_kwargs_dict.get('N_random_generations', 1),
            N_random_per_generation=proposal_kwargs_dict.get('N_random_per_generation'),
        )
```

Then in BOTH proposal calls, replace:

```python
            N=self.inputs.N,
```

with:

```python
            N=Int(next_batch),
```

(The `Int` wrapper is needed because `aiida_propose_occ_matrices_from_results` is a calcfunction expecting an AiiDA `Int`.)

After the change, `process_mag_scan_results` passes `next_generation = self.ctx.generation = 0` (since `run_initial_mag_scan` does not increment) and `process_constrained_results` passes `next_generation = self.ctx.generation` (post-increment was done by `run_constrained_batch`, so this is still the next dispatcher `current_generation`).

(c) Both call sites also pass `current_generation` via `proposal_kwargs`. Verify and adjust: the existing code sets `proposal_kwargs['current_generation'] = Int(self.ctx.generation)` in both [line 294](lordcapulet/workflows/global_constrained_search.py#L294) and [line 412](lordcapulet/workflows/global_constrained_search.py#L412). Leave these as-is — they correctly reflect the dispatcher's `current_generation` for the upcoming batch.

- [ ] **Step 7: Run the workchain integration tests to confirm no regression**

Run: `uv run pytest tests/integration_tests/workflows/test_global_constrained_search.py -v`
Expected: all pass (random_so_n mode integration test still works because default `N_random_generations=1` with no override means batch size = N for every gen, identical to old behaviour).

- [ ] **Step 8: Run the new helper tests again to confirm they still pass**

Run: `uv run pytest tests/unit_tests/workflows/test_global_search_random_schedule.py -v`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add lordcapulet/workflows/global_constrained_search.py tests/unit_tests/workflows/
git commit -m "feat(workflow): N_random_generations + N_random_per_generation schedule for GP warmup"
```

---

## Task 7: Surface proposal_metadata in WorkchainDataExtractor

**Files:**
- Modify: `lordcapulet/utils/postprocessing/gather_workchain_data.py:315-388` (`_extract_calc_data`)
- Test: `tests/unit_tests/utils/test_postprocessing_proposal_metadata.py` (new)

The extractor currently surfaces only the OUTPUT occupation matrix (via `calc.base.extras['occupation_matrix_pk']` — see [gather_workchain_data.py:347-355](lordcapulet/utils/postprocessing/gather_workchain_data.py#L347-L355)). For constrained calcs, we add a new field `proposal_metadata` populated from the input `target_matrix` (a `JsonableData(OccupationMatrixData)` carrying `{'proposal_mode': ..., 'current_generation': ..., 'gp_warmup': ..., 'gp_fallback': ...}`).

For mag_scan calcs there is no `target_matrix` input → `proposal_metadata = None`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit_tests/utils/test_postprocessing_proposal_metadata.py`:

```python
"""WorkchainDataExtractor surfaces proposal metadata from input target_matrix."""

from types import SimpleNamespace

import pytest

from lordcapulet.utils.postprocessing.gather_workchain_data import (
    WorkchainDataExtractor,
)


class _FakeJsonableData:
    """Stand-in for JsonableData wrapping an OccupationMatrixData."""

    def __init__(self, metadata):
        self.obj = SimpleNamespace(metadata=metadata)


class _FakeCalc:
    """Stand-in for an AiiDA CalcJobNode with controllable inputs/outputs."""

    def __init__(self, pk, process_type, target_metadata=None,
                 has_target_input=True, exit_status=0):
        self.pk = pk
        self.process_type = process_type
        self.exit_status = exit_status
        self.is_finished_ok = exit_status == 0
        self.outputs = SimpleNamespace()
        # Minimal extras access used by extractor
        self.base = SimpleNamespace(extras=SimpleNamespace(
            get=lambda key, default=None: default,
            __contains__=lambda self_, key: False,
        ))
        # Wire up `'occupation_matrix_pk' in calc.base.extras` to be False
        self.base.extras = _NoExtras()

        inputs_dict = {}
        if has_target_input:
            inputs_dict['target_matrix'] = _FakeJsonableData(target_metadata or {})
        self.inputs = _Attrs(inputs_dict)


class _NoExtras:
    def __contains__(self, key):
        return False

    def get(self, key, default=None):
        return default


class _Attrs:
    def __init__(self, mapping):
        self._mapping = mapping

    def __contains__(self, key):
        return key in self._mapping

    def __getattr__(self, name):
        if name in self._mapping:
            return self._mapping[name]
        raise AttributeError(name)


class TestProposalMetadataSurface:
    def test_constrained_calc_surfaces_proposal_metadata(self):
        calc = _FakeCalc(
            pk=999, process_type='aiida.calculations:lordcapulet.constrained_pw',
            target_metadata={'proposal_mode': 'gp', 'current_generation': 5},
        )
        extractor = WorkchainDataExtractor(perform_so_n=False)
        data = extractor._extract_calc_data(calc, generation_number=5)

        assert data['proposal_metadata'] == {
            'proposal_mode': 'gp', 'current_generation': 5,
        }

    def test_constrained_calc_with_warmup_flag(self):
        calc = _FakeCalc(
            pk=1000, process_type='aiida.calculations:lordcapulet.constrained_pw',
            target_metadata={
                'proposal_mode': 'random',
                'current_generation': 1,
                'gp_warmup': True,
            },
        )
        extractor = WorkchainDataExtractor(perform_so_n=False)
        data = extractor._extract_calc_data(calc, generation_number=1)

        assert data['proposal_metadata']['proposal_mode'] == 'random'
        assert data['proposal_metadata']['gp_warmup'] is True

    def test_mag_scan_calc_has_no_proposal_metadata(self):
        calc = _FakeCalc(
            pk=1001, process_type='aiida.calculations:quantumespresso.pw',
            has_target_input=False,
        )
        extractor = WorkchainDataExtractor(perform_so_n=False)
        data = extractor._extract_calc_data(calc, generation_number=None)

        assert data['proposal_metadata'] is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit_tests/utils/test_postprocessing_proposal_metadata.py -v`
Expected: KeyError on `'proposal_metadata'` — the extractor does not produce that field yet.

- [ ] **Step 3: Add the field to `_extract_calc_data`**

In `lordcapulet/utils/postprocessing/gather_workchain_data.py`, modify `_extract_calc_data` ([starting at line 315](lordcapulet/utils/postprocessing/gather_workchain_data.py#L315)):

(a) After the `calc_data` dict is created (currently around [line 322-331](lordcapulet/utils/postprocessing/gather_workchain_data.py#L322-L331)), add `'proposal_metadata': None` so the field is always present:

```python
        calc_data = {
            'pk': calc.pk,
            'exit_status': calc.exit_status,
            'generation_number': generation_number,
            'converged': calc.exit_status == 0,
            'process_type': getattr(calc, 'process_type', 'unknown'),
            'calculation_source': self._determine_source(calc),
            'output_parameters': None,
            'occupation_matrices': None,
            'proposal_metadata': None,
        }
```

(b) Right after that block (still inside `_extract_calc_data`, before the `# Extract output parameters` block), populate the new field:

```python
        # Surface proposal provenance metadata from the input target_matrix
        # (only present on constrained calculations).
        try:
            if 'target_matrix' in calc.inputs:
                target = calc.inputs.target_matrix
                obj = getattr(target, 'obj', None)
                meta = getattr(obj, 'metadata', None) if obj is not None else None
                if meta:
                    calc_data['proposal_metadata'] = dict(meta)
        except Exception as e:
            if self.debug:
                print(f"Error reading proposal_metadata from {calc.pk}: {e}")
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `uv run pytest tests/unit_tests/utils/test_postprocessing_proposal_metadata.py -v`
Expected: all 3 tests pass.

- [ ] **Step 5: Run the existing extractor regression**

Run: `uv run pytest tests/unit_tests/utils/ -v` (or whichever tests cover `gather_workchain_data`).
Expected: no regression.

- [ ] **Step 6: Commit**

```bash
git add lordcapulet/utils/postprocessing/gather_workchain_data.py tests/unit_tests/utils/test_postprocessing_proposal_metadata.py
git commit -m "feat(extractor): surface proposal_metadata from input target_matrix"
```

---

## Task 8: Document the new knobs in the global-search protocol YAML

**Files:**
- Modify: `lordcapulet/workflows/protocols/global_search.yaml`
- Modify: `examples/05_global_scan_bayesian_FeO/submit_feo_bayesian.py` (only the comment block + override keys, to demonstrate the new API)

- [ ] **Step 1: Add new keys to the global protocol YAML with comments**

In `lordcapulet/workflows/protocols/global_search.yaml`, append after the `proposal_holistic` line:

```yaml
# GP-mode warmup schedule (only consulted when proposal_mode is 'gp' or
# 'gaussian_process'):
#   N_random_generations:    number of leading generations that use random
#                            proposals to seed the GP training set. Default 1.
#   N_random_per_generation: batch size for those warmup generations. If null,
#                            the workchain's `N` is used. Default null.
# proposal_kwargs:
#     N_random_generations: 3
#     N_random_per_generation: 25
```

(Leave the example commented so existing protocols are unchanged.)

- [ ] **Step 2: Update the bayesian example to use the new API**

In `examples/05_global_scan_bayesian_FeO/submit_feo_bayesian.py`, replace the `proposal_kwargs` block (currently [around line 84](examples/05_global_scan_bayesian_FeO/submit_feo_bayesian.py#L84)):

```python
            'proposal_kwargs': {'N_initial_random': 100},
```

with:

```python
            # Warm up the GP with 3 random generations of 25 proposals each
            # (= 75 random seeds spread across 3 batches), then switch to GP.
            'proposal_kwargs': {
                'N_random_generations': 3,
                'N_random_per_generation': 25,
            },
```

Update the surrounding comment block to match (the existing comment talks about `N_initial_random` which is removed). The block from [around line 76-83](examples/05_global_scan_bayesian_FeO/submit_feo_bayesian.py#L76-L83) should read:

```python
            # GP is refit from scratch every generation. The default is
            # Markovian (only the previous generation feeds the fit). For real
            # BO we want the GP to see all past converged calcs.
            'proposal_holistic': True,
            # GP warmup schedule:
            # - The first `N_random_generations` generations are random
            #   batches of `N_random_per_generation` proposals each, seeding
            #   the GP training set.
            # - Subsequent generations use GP acquisition with the regular
            #   batch size `N`.
```

- [ ] **Step 3: Run the full unit-test suite to confirm green**

Run: `uv run pytest tests/unit_tests/ -v`
Expected: all pass.

- [ ] **Step 4: Run the integration suite (skipping slow GP)**

Run: `uv run pytest tests/integration_tests/ -m "not slow" -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add lordcapulet/workflows/protocols/global_search.yaml examples/05_global_scan_bayesian_FeO/submit_feo_bayesian.py
git commit -m "docs(protocols): document N_random_generations / N_random_per_generation knobs"
```

---

## Task 9: End-to-end smoke check (manual, optional)

**Files:** none — verification only.

- [ ] **Step 1: Run the bayesian example with a small Nmax**

Edit `examples/05_global_scan_bayesian_FeO/submit_feo_bayesian.py` for a quick local check:
- `Nmax: 30`
- `N: 5`
- `N_random_generations: 2`
- `N_random_per_generation: 10`

Run: `uv run python examples/05_global_scan_bayesian_FeO/submit_feo_bayesian.py`

Expected outcome:
- `verdi process status <pk>` shows: 1 mag_scan, then 2 generations of 10 (warmup), then 2 generations of 5 (GP), totalling 30 constrained calcs.
- After completion, the resulting `FeO_scan_bayesian.json` contains `proposal_metadata` per calc:
  - mag_scan calcs → `null`
  - gen 1-2 constrained calcs → `{"proposal_mode": "random", "current_generation": 0|1, "gp_warmup": true}`
  - gen 3-4 constrained calcs → `{"proposal_mode": "gp", "current_generation": 2|3}`

Revert the example file changes after the smoke check (do NOT commit them).

- [ ] **Step 2: No commit — smoke check only.**

---

## Self-Review Notes

- All 9 tasks include exact code, exact test code, exact pytest commands, and the file paths to touch. No "TBD", no "implement appropriate validation".
- Method names are consistent across tasks: `metadata`, `set_metadata`, `update_metadata`, `_compute_batch_size`, `proposal_metadata` (extractor field).
- Goal 1 coverage:
  - Task 1 = data-structure plumbing.
  - Tasks 2-4 = each proposer stamps its own mode.
  - Task 5 = dispatcher layers `current_generation`, `gp_warmup`, `gp_fallback`.
  - Task 7 = extractor exposes the metadata in the final JSON.
- Goal 2 coverage:
  - Task 5 = dispatcher reads `N_random_generations`.
  - Task 6 = workchain reads `N_random_generations` and `N_random_per_generation`, sizes batches accordingly, and forwards the right batch size to the proposer.
  - Task 8 = protocol + example surface the new knobs to users.
- Backward compat verified by re-running `tests/unit_tests/data_structures/test_occupation_matrix.py`, `tests/unit_tests/functions/test_proposal_modes.py`, `tests/unit_tests/functions/test_propose.py`, `tests/integration_tests/workflows/test_global_constrained_search.py`.
- Test ordering: a new directory `tests/unit_tests/workflows/` is added; if `_test_priority` in `tests/conftest.py` does not already cover it, Task 6 Step 1 inserts it.
- The legacy `N_initial_random` kwarg is removed (Task 8 swaps it in the example). No callers use it elsewhere — verify with `grep -rn N_initial_random lordcapulet/ examples/` before Task 8 and remove any stragglers.
