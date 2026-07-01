# TreeSHAP near-AFM focused constrained scan (FeO 4x1x1)

Date: 2026-06-28

## Motivation

The FeO 4x1x1 supercell GP/eta10 runs under-sample the compensated-AFM sector
(near-AFM coverage |M_tot|<2 is only ~12%, see
`experiments/FeO_supercell_4x1x1_bayesian_eta10/report.md` and the TreeSHAP
report `experiments/FeO_supercell_treeshap/report.md`). The TreeSHAP report's
"ground state" is an uncompensated ferrimagnet (M_tot=-7.24); the lowest
compensated-AFM config sits only 0.107 eV higher but is poorly sampled. Goal:
generate a focused batch of near-AFM 4x1x1 constrained targets, proposed by
combining the best naive-template local orbital orderings and ranked by the
fitted TreeSHAP energy model.

## Decisions (locked with user 2026-06-28)

- **Candidate building blocks:** top-16 naive Fe_d templates by `energy_rank`
  from `FeO_template_library/FeO_primitive_atomic_templates.json`. These carry
  richer local orbital ordering than the lossy K=16 KMeans centroids. Verified:
  the top 16 are perfectly spin-balanced (8 positive, 8 negative local moment,
  all high-spin |m|≈3.7 µB).
- **Ranker:** the fitted TreeSHAP XGBoost model (K=16 alphabet, seed=0),
  reproducing the report run (test R²=0.748, Spearman 0.86). Each site template
  is mapped to its nearest K=16 KMeans cluster, one-hot encoded, and scored by
  `model.predict`. Ties broken by `sum(template.energy_rank)`.
- **near-AFM cutoff:** |M_tot| < 2 µB. With the balanced top-16 alphabet this is
  exactly equivalent to a **4-positive / 4-negative** site assignment over the 8
  Fe sites (4×3.7 − 4×3.7 ≈ 0; any 3/5 split gives |M_tot|≈7.4, excluded).
- **Batch size N:** 100.
- **Action:** generate proposals + job spec + **dry-run only**. No real
  submission this round.

## Pipeline (`experiments/FeO_supercell_treeshap/propose_afm.py`)

1. Load top-16 Fe_d templates; split into positive- and negative-moment pools.
2. Refit the TreeSHAP model by reusing `run_treeshap` functions
   (`load_banks` → `build_alphabet` K=16 → `featurize` → `one_hot` →
   `train_model`, seed=0). Keep the fitted KMeans alphabet model and one-hot
   encoder.
3. Enumerate near-AFM combos by construction: choose which 4 of 8 sites are
   positive (C(8,4)=70), assign positive templates to those sites and negative
   templates to the rest. Full space ~1.2e9; because the model collapses
   templates to ≤16 clusters (many combos tie), **sample** a large pool
   (~2–5M) rather than exhaustively enumerate.
4. Score each combo: per-site template → nearest K=16 cluster (via
   `assign_site_templates`-equivalent prediction) → one-hot → `model.predict`.
5. Dedup by occupation matrix; sort by (predicted energy asc, template-rank
   sum asc); take top 100.
6. Emit `4x1x1/afm_proposals.json` in the template-library proposal shape:
   `{"metadata", "proposals": [{"index", "occupation_matrices": {Atom_1..Atom_8}}]}`.
   Each `Atom_k` block is the assigned template's `occupation_matrix`.
7. Emit `4x1x1/afm_job_spec.json`: `structure` (FeO primitive input, supercell
   [4,1,1], hubbard table/u_values), `scf` (code `pw-occ-fix@eiger-uenv-gnu-25.6`,
   `kpoints_mesh [1,3,3]`, `walltime_hours 4`, `label_prefix "treeshap afm"`),
   and `proposals_json` pointing at the proposals file.
8. Call `exp_util.submit_constrained_from_json(spec, cache, dry_run=True)` and
   print the plan.

## Diagnostics

- Alphabet moment table (id, energy_rank, moment).
- Pool size sampled, near-AFM survival (should be 100% by construction), number
  of distinct occupation-matrix configs and distinct model encodings (collision
  check — quantifies how many combos the K=16 model actually distinguishes).
- Predicted-energy spread of the top 100 and their M_tot distribution (must all
  be ≈0).

## Validation

- Model best predicted energy in the report is ≈ −40041.97 eV; top-100 near-AFM
  picks should cluster in that basin.
- Report's lowest compensated-AFM is 0.107 eV above the true (ferrimagnetic) GS;
  the picks' predicted energies are sanity-checked against that scale.

## Constraints / conventions

- Read-only on AiiDA for this round (dry-run). No daemon restart, no submit.
- AiiDA types stop at the calcfunction boundary — not relevant here; this is an
  experiment script that builds plain proposal JSON and defers all AiiDA wrapping
  to `exp_util.submit_constrained_from_json`.
- Experiment artifacts live under `experiments/FeO_supercell_treeshap/4x1x1/`;
  update `experiments/summary.md` and a `log.md` after the run.

## Out of scope

- Actual cluster submission (separate explicit step after dry-run review).
- 3-body terms, K=24 refit, longer-shell pair features.
- Cross-size transfer.
