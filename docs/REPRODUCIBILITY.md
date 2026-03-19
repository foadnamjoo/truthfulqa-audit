# Reproducibility

This document describes a compact end-to-end run order for reproducing the paper assets.

## Environment

```bash
pip install -r requirements.txt
```

## Data Source

- TruthfulQA repository: <https://github.com/sylinrl/TruthfulQA>
- CSV used by scripts: `TruthfulQA.csv` (binary-choice fields include `Best Answer` and `Best Incorrect Answer`)

For stable reproduction, keep a local pinned copy of `TruthfulQA.csv` and record its source commit/date in your run notes.

## Pipeline Order

1. (Optional) Generate model prediction CSV files:

```bash
python scripts/run_binary_choice_eval.py --help
```

2. Regenerate notebook scaffolding (if generator changed):

```bash
python scripts/build_audit_notebook.py
```

3. Run all notebook cells in `TruthfulQA_Style_Confound_Audit.ipynb`.
   - This produces audit summaries in `audits/`.

4. Build final tables and plots:

```bash
python scripts/make_final_tables.py
python scripts/make_paper_assets.py --root .
```

## Expected Key Artifacts

- `paper_assets/figures/impact_delta_bar.pdf`
- `paper_assets/figures/impact_acc_by_split.pdf`
- `paper_assets/figures/permutation_null_forest.pdf`
- `paper_assets/tables/benchmark_impact_table.tex`
- `paper_assets/tables/permutation_null_table.tex`
- `paper_assets/tables/seed_summary_table.tex`

## Notes on Determinism

- Classifier-side grouped CV/null tests use fixed seeds.
- Generative model outputs can vary by backend/hardware and decoder behavior.
- Treat small deltas near zero as potentially sensitive to inference/runtime configuration.
