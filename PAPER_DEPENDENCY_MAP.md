# Paper dependency map

This maps the **current paper** (“Judging by the Cover…”) to scripts, frozen outputs, and LaTeX assets in this repo. Table/figure numbering follows the paper’s structure (TruthfulQA audit → model impact → cross-dataset). **Keep/archive** indicates whether the path should stay in the active tree for reproduction.

Legend: **preserve** = required for paper as written or strongly implied supplementary artifact (feature-balanced subsets + pruning verification). **archive OK** = exploratory only; not cited by the main paper PDF.

---

## TruthfulQA: audit, null, clean/confounded split, feature ablations

| Paper artifact | Scripts / workflow | Inputs | Outputs / assets | Keep / archive |
|----------------|-------------------|--------|------------------|----------------|
| §2–3 Methods + TruthfulQA results; Tables 1–2 (feature ablation; random-label / controls as applicable) | `scripts/truthfulqa_paper_audit.py` (canonical **surface10** ten-feature CV + LR pipeline; legacy alias `paper10`); `notebooks/TruthfulQA_Style_Confound_Audit.ipynb` (full exploratory audit workflow); `scripts/build_audit_notebook.py` (regenerate notebook sources if needed) | `audits/truthfulqa_style_audit.csv` | Notebook logs; `paper_assets/tables/feature_ablation_table.tex`, `feature_ablation_paragraph.tex` (via `make_paper_assets.py`); legacy figures `figures/*.pdf` (classifier vs null, distributions, etc.) | **preserve** |
| Random-label control (BoolQ / VitaminC floor); `random_label_control_table.tex` | `scripts/run_fever_audit.py` (`--random-label-control-only` or combined run); uses same audit machinery | Optional local parquet paths; HF cache when downloading | `audits/random_label_control_101runs.csv` (if produced); `paper_assets/tables/random_label_control_table.tex` | **preserve** |
| Per-pair bridge (Spearman); `per_pair_bridge_table.tex` | Typically computed in audit notebook workflow; not a standalone `.py` in repo | `audits/per_pair_bridge_spearman.csv`; model predictions vs audit | `paper_assets/tables/per_pair_bridge_table.tex` | **preserve** (inputs + table) |

---

## Model-level benchmark impact (nine models)

| Paper artifact | Scripts / workflow | Inputs | Outputs / assets | Keep / archive |
|----------------|-------------------|--------|------------------|----------------|
| §4 + Tables 3–5; Figures 1–3 | `scripts/run_binary_choice_eval.py` (HF generations); `scripts/make_final_tables.py` (aggregate CSVs from predictions); **summary + permutation + seed sweep** cells in `notebooks/TruthfulQA_Style_Confound_Audit.ipynb` (also referenced in `build_audit_notebook.py`) | `audits/truthfulqa_style_audit.csv` (`style_violation`); `data/predictions/model_predictions*.csv` | `audits/model_benchmark_impact_summary.csv`, `model_benchmark_impact_by_file.csv`, `model_benchmark_impact_by_model.csv`, `seed_sweep_summary.csv`, `permutation_null_test_summary.csv` | **preserve** |
| Figures 1–3 + benchmark tables (LaTeX) | `scripts/make_paper_assets.py --root .` | CSVs above | `paper_assets/fig/impact_delta_bar.{pdf,png}`, `impact_acc_by_split.{pdf,png}`, `permutation_null_forest.{pdf,png}`; `paper_assets/tables/benchmark_impact_table.tex`, `seed_summary_table.tex`, `permutation_null_table.tex` | **preserve** |
| CHPC / batched predictions | `scripts/import_chpc_predictions.py` | CHPC output CSVs | Files under `data/predictions/` (layout as documented in script) | **preserve** |

---

## Cross-dataset audits (FEVER, FeverSymmetric, BoolQ, HaluEval, VitaminC)

| Paper artifact | Scripts / workflow | Inputs | Outputs / assets | Keep / archive |
|----------------|-------------------|--------|------------------|----------------|
| §5 + Tables 6–9; Figure 4 | `scripts/run_fever_audit.py` (FEVER/FeverSymmetric frozen in-script; BoolQ/HaluEval/VitaminC live) | Optional: `--boolq-data`, `--halueval-data`, `--vitaminc-data`; else HF `datasets` | `audits/fever_audit_results.csv`; `paper_assets/tables/cross_dataset_comparison_table.tex`, `boolq_feature_ablation_table.tex`, `halueval_feature_ablation_table.tex`, `vitaminc_feature_ablation_table.tex`; `paper_assets/fig/fever_audit_auc_comparison.pdf`. (`fever_feature_ablation_table.tex` is checked in for the paper; FEVER/FeverSymmetric rows are frozen constants in-script.) | **preserve** |

**Note:** `make_paper_assets.py` and `run_fever_audit.py` both write to `paper_assets/fig/`. LaTeX should `\includegraphics` from that single directory.

---

## Supplementary repo artifact (not the focus of the 11-page PDF body)

| Artifact | Scripts | Inputs | Outputs | Keep / archive |
|----------|---------|--------|---------|----------------|
| Feature-balanced audited subsets (300–650 pairs) + locked multi-seed verification | `scripts/export_feature_balanced_subset_csvs.py`; `scripts/truthfulqa_pruning_final_verification.py`; repro: `scripts/check_pruning_final_verification_repro.py`; search utilities: `scripts/search_truthfulqa_pruned_improved.py`, `scripts/truthfulqa_pruning_utils.py`, `scripts/run_truthfulqa_pruning_improved.py`, `scripts/run_pruning_final_verification.py` | `audits/truthfulqa_style_audit.csv`; verification reads prior JSON/CSVs as configured in-repo | `truthfulqaPro/` (CSVs, `subset_manifest.csv`, `pair_ids/`); `results/truthfulqa_pruning_final_verification/`, `figures/truthfulqa_pruning_final_verification/`; LaTeX note `paper_assets/tables/feature_balanced_subset_paragraph.tex` | **preserve** (supplementary; reproducibility) |

---

## Regeneration quick reference

1. **Plots/tables for model impact (from frozen audits):**  
   `python3 scripts/make_paper_assets.py --root .`

2. **Cross-dataset tables + figure:**  
   `pip install -r requirements.txt -r requirements-paper-full.txt`, then  
   `python3 scripts/run_fever_audit.py` (add local data flags if the Hub is unavailable).

3. **Benchmark impact CSVs from prediction files:**  
   `python3 scripts/make_final_tables.py --root .`

4. **Full model evaluation (expensive):**  
   `python3 scripts/run_binary_choice_eval.py` with appropriate flags and GPU.

5. **Pruning verification + feature-balanced exports (subset supplementary track):**  
   `python3 scripts/truthfulqa_pruning_final_verification.py --n-seeds 10 --base-seed 42` then `python3 scripts/check_pruning_final_verification_repro.py`; CSV export helper `python3 scripts/export_feature_balanced_subset_csvs.py`.

---

## Summary: keep vs archive

- **Keep in active repo:** everything in the tables above, including `audits/*.csv`, `paper_assets/**`, `truthfulqaPro/`, `data/predictions/` (if redistributable), `notebooks/`, and all listed `scripts/`.
- **Archive OK:** contents under `archive/results/`, `archive/scripts/run_near_random_subset_refined.py`, `archive/scripts/test_pruning_improved_pipeline.py`, duplicate exploratory figures, and old notebook copy under `archive/notebooks/` (retained as backup; active copy is `notebooks/`).
