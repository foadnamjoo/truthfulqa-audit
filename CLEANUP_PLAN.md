# Cleanup plan (aligned with the paper as written)

The repository supports the **full** surface-form audit paper: TruthfulQA (grouped CV, null, clean/confounded split, feature ablations), **nine-model** benchmark-impact analysis, **cross-dataset** audits (FEVER, FeverSymmetric, BoolQ, HaluEval, VitaminC), and the associated **figures, tables, and LaTeX** under `paper_assets/`. A **supplementary** track ships **feature-balanced** audited subsets (fixed sizes 300–650) with locked multi-seed verification under `results/truthfulqa_pruning_final_verification/`.

**Rule:** nothing that is required to reproduce paper figures, tables, or cited results should be deleted, renamed, or moved to `archive/`. When in doubt, **preserve** and tag below as **paper-used**.

---

## Paper-used — preserve (active repo)

| Area | Paths |
|------|--------|
| Audit features + benchmark-impact CSVs | `audits/*.csv` (including `truthfulqa_style_audit.csv`, `model_benchmark_impact_*.csv`, `seed_sweep_summary.csv`, `permutation_null_test_summary.csv`, `fever_audit_results.csv`, bridge/random-label CSVs as used) |
| Paper LaTeX + figures | `paper_assets/` (`fig/`, `figures/`, `tables/`) |
| Core + paper scripts | `scripts/truthfulqa_paper_audit.py`, `make_paper_assets.py`, `run_fever_audit.py`, `run_binary_choice_eval.py`, `make_final_tables.py`, `build_audit_notebook.py`, `import_chpc_predictions.py`, `make_example_predictions.py` |
| Main audit notebook | `notebooks/TruthfulQA_Style_Confound_Audit.ipynb` |
| Legacy audit figures (notebook-era) | `figures/*.pdf` (classifier vs null, feature importance, etc.) |
| Model predictions (if redistributed) | `data/predictions/` as applicable |
| Feature-balanced reference subsets + pruning verification (supplementary) | `data/subsets/feature_balanced_paper10/`, `results/feature_balanced_reference_subsets/`, `scripts/export_feature_balanced_subset_csvs.py`; locked run `results/truthfulqa_pruning_final_verification/`, `figures/truthfulqa_pruning_final_verification/`; `scripts/truthfulqa_pruning_final_verification.py`, `scripts/check_pruning_final_verification_repro.py`, `scripts/truthfulqa_pruning_utils.py`, `scripts/search_truthfulqa_pruned_improved.py`, `scripts/run_truthfulqa_pruning_improved.py`, `scripts/run_pruning_final_verification.py` |
| Dependencies | `requirements.txt` (core); `requirements-paper-full.txt` (HF, datasets, torch, etc.) |

---

## Safe to archive (exploratory / duplicate / superseded)

These are **not** required to reproduce the main paper PDF or the checked-in `paper_assets/` + `audits/` snapshots. They remain under `archive/` as **copies** (nothing is permanently deleted).

| Location | Contents |
|----------|----------|
| `archive/scripts/run_near_random_subset_refined.py` | Superseded grid before “better algorithms” |
| `archive/scripts/test_pruning_improved_pipeline.py` | Internal smoke test for pruning branch |
| `archive/results/near_random_subset_search/` | Early near-random sweeps |
| `archive/results/near_random_subset_search_refined/` | Refined grid outputs |
| `archive/results/near_random_subset_search_better_algorithms/` | Exploratory grid before final driver |
| `archive/results/truthfulqa_randomlike_subset/`, `truthfulqa_pruning_improved/`, `truthfulqa_pruning_final_verification/` | Pruning-replay branch artifacts |
| `archive/results/final_near_random_truthfulqa_subset_backup_before_repro_check/` | One-off backup |
| `archive/notebooks/TruthfulQA_Style_Confound_Audit.ipynb` | Duplicate of `notebooks/` (optional backup) |
| `archive/figures/` | Old exploratory figure trees (if any remain after restoring legacy PDFs to `figures/`) |

---

## Safe to remove

Avoid deleting **paper-used** paths listed above. Retired near-random-only drivers and their top-level `results/` / `figures/` trees were removed from `main` in favor of the feature-balanced track; older snapshots remain under `archive/results/` and in `git` history if you need them.

---

## Restored after the narrow cleanup (paper-critical)

The following were **moved back** from `archive/scripts/` (and related) so the public tree matches the paper:

- `make_paper_assets.py`, `run_fever_audit.py`, `run_binary_choice_eval.py`, `build_audit_notebook.py`, `make_final_tables.py`, `import_chpc_predictions.py`, `make_example_predictions.py`
- Notebook: `notebooks/TruthfulQA_Style_Confound_Audit.ipynb` (copy from `archive/notebooks/`)
- Legacy notebook figures: `figures/*.pdf` (from `archive/figures/legacy_pdfs/`)

---

## Verification

- `python3 scripts/make_paper_assets.py --root .` completes and refreshes `paper_assets/fig/` and key `paper_assets/tables/`.
- Pruning verification (long run): `python3 scripts/truthfulqa_pruning_final_verification.py --n-seeds 10 --base-seed 42`, then `python3 scripts/check_pruning_final_verification_repro.py`.

See **`PAPER_DEPENDENCY_MAP.md`** for section/table/figure-level mapping.
