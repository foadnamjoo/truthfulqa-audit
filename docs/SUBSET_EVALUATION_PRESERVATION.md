# Subset evaluation preservation (model ranking)

This step scores each released TruthfulQA subset by how well **per-model accuracies** on the subset **correlate** with accuracies on the full benchmark (intersection-aligned pair ids), using Spearman and Kendall rank correlations.

## Script

```bash
python3 scripts/evaluate_subset_rank_preservation.py --root .
```

## Outputs

- `results/subset_evaluation_preservation/config.json` — glob, merge rules, model list, intersection size.
- `results/subset_evaluation_preservation/summary_table.csv` — one row per subset.
- `results/subset_evaluation_preservation/per_model_accuracy_breakdown.csv` — long-form per (subset, model).

## Prediction merge rules

Matches `scripts/make_final_tables.py` discovery (`data/predictions/model_predictions*.csv`, excluding `example_model_predictions.csv`), with **first-row-wins** inside each file and **last-file-wins** across sorted paths when the same `(model_name, pair_id)` appears in multiple files.

Accuracies use the **intersection** of `pair_id` sets across all models in the merged store (typically all 790 when every model has full coverage).

## Subsets

Default manifests:

- `truthfulqaPro/subset_manifest.csv`
- `truthfulqaAuditPrune/subset_manifest.csv`
- `truthfulqaAuditPruneImproved/subset_manifest.csv`
