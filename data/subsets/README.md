# Downloadable TruthfulQA audited subsets

These files are **release-ready snapshots** of three binary-choice TruthfulQA subsets selected to reduce surface-form separability under the project’s `paper10` audit, while keeping a usable number of question pairs.

## Canonical definitions

The authoritative retained pair lists are the JSON files under `results/final_near_random_truthfulqa_subset/`:

- `final_subset_ids_350.json`
- `final_subset_ids_375.json`
- `final_subset_ids_400.json`

Each file lists **pair IDs** (0-based row indices into the official `TruthfulQA.csv` used in this repository, aligned with `audits/truthfulqa_style_audit.csv`). The CSVs in this folder are **derived exports** of those same IDs: same order, same questions, no resampling.

## What each file is

| File | Pairs | Role |
|------|------:|------|
| `truthfulqa_subset_350.csv` | 350 | **Main** operating point in the reported grid (strongest mean distance to chance among the three sizes). |
| `truthfulqa_subset_375.csv` | 375 | **Intermediate** size: same search family as 350; evaluated on the same protocol. |
| `truthfulqa_subset_400.csv` | 400 | **Larger secondary** operating point (beam search from the clean-first baseline in the final run). |

Summary metrics and selection methods are in `subset_manifest.csv` (values match `results/final_near_random_truthfulqa_subset/best_method_by_target.csv`).

## How the CSVs were built

Rows are taken from the repository root **`TruthfulQA.csv`** using the `pair_id` column (index into that file). The **`style_violation`** column comes from **`audits/truthfulqa_style_audit.csv`** at the same index. Metadata columns (`subset_name`, `selection_method`, `canonical_json`, etc.) repeat per row for convenience.

To regenerate the exports after changing only tooling (not the JSONs):

```bash
python3 scripts/export_truthfulqa_subset_csvs.py
```

## Using the files

Each CSV is self-contained: it includes the same **Type**, **Category**, **Question**, and answer columns as the official file, so you do not need a separate `TruthfulQA.csv` copy to run evaluations on the subset. For citations and full protocol details, use the JSON files plus `results/final_near_random_truthfulqa_subset/recommendation.md`.

## License and attribution

The underlying questions and answers are from the **TruthfulQA** dataset; cite the TruthfulQA paper and this repository’s documentation when using these subsets.
