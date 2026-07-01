# Rebasing `gp-proposal-core` onto `main` — resolution plan

Status: **not yet applied** — reviewed 2026-07-14, iterate later.

## Verdict: mergeable, no genuine logic collisions

Cause of conflicts: `main` performed **one rename** — `tm_*` → `hubbard_corr_*`
(`prepare_tm_info` → `prepare_hubbard_corr_info`, `tm_manifolds`/`tm_dimensions`/`tm_counts`
→ `hubbard_corr_*`, "TM" → "Hubbard-corrected"). Our branch only **added** features.
Every conflict is positional (rename vs old name, or add vs nothing), not two sides
changing the same logic.

Replaying 2 commits onto `main`: `36952e4` (feature commit — all conflicts here) then
`067513e` (agent-file removal — trivial).

## Per-file resolution

### Bucket 1 — rename (take main's `hubbard_corr_*` names)
- `lordcapulet/utils/__init__.py` — export `prepare_hubbard_corr_info`
- `lordcapulet/utils/preprocessing/submission.py` (10 hunks) — `prepare_hubbard_corr_info`,
  `hubbard_corr_manifolds`/`hubbard_corr_dimensions`, `hubbard_atom_counts`
- `lordcapulet/workflows/constrained_scan.py` — "Hubbard-corrected" comment
- `lordcapulet/workflows/global_constrained_search.py` — docstring wording
- `tests/unit_tests/calculations/test_hubbard_utils.py` (14 hunks) — class
  `TestPrepareTmInfo` → `TestPrepareHubbardCorrInfo` and all refs
- **After resolving: grep the whole branch for remaining `tm_` / `prepare_tm_info` in our
  added code and rename too** — those lines won't show as conflicts but will break at runtime.

### Bucket 2 — additive union (take ours; main side empty)
- `lordcapulet/data_structures/__init__.py` + `lordcapulet/functions/propose.py` —
  keep `clip_occupation_numbers` export/import
- `lordcapulet/workflows/standard_magnetic_scan.py` — keep `max_configurations`
  (Int import + spec.input + builder)
- `lordcapulet/workflows/global_constrained_search.py` — keep whole seeded-startup system
  (`import_seed_results`, `_build_proposal_kwargs` helper replacing inline kwargs, `if_()`
  outline, metadata helpers). main never touched this logic.
- `tests/conftest.py` — sort key `10` (ours), not `9`
- `tests/integration_tests/workflows/test_global_constrained_search.py` — keep new fixture
  params (`startup_mode`, `seed_global_workchain_pk`, `proposal_mode`, `proposal_holistic`)

### Bucket 3 — decisions / cleanup (OPEN)
- ⚠️ `lordcapulet/workflows/protocols/global_search.yaml`: `proposal_holistic` —
  **main=`true`, ours=`false`**. Pick one. **DECISION: _____**
- ⚠️ `examples/01_single_constrained_FeO/submit_single_constrained.py`: our committed
  version already contains stale `<<<<<<< Updated upstream` stash markers (pre-existing junk,
  unrelated to this rebase). Clean it while resolving.

## After resolving
- `pytest tests/unit_tests -m "not slow"` and `tests/integration_tests -m "not slow"`
  (baseline before rebase: 314 unit passed / 131 integration passed, 6 skipped).
- Force-push updates PR #5 (`alberto-carta:gp-proposal-core`, base `Testing_and_user_experience`).
