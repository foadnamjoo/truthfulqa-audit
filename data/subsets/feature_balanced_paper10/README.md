# Feature-balanced pruning reference subsets (paper10)

These CSVs are **downloadable reference slices** for the `feature_balanced` fixed-kept-count protocol documented in `results/truthfulqa_pruning_final_verification/`.

## What is fixed

- **Examples** are original TruthfulQA rows; only membership changes.
- **Ordering** matches `scripts/truthfulqa_pruning_final_verification.py`: length-quartile stratified shuffle, then sort by negation/length gap/id, then keep the first *K* pairs (`feature_balanced_length_stratified_prefix`).
- **Reference split seed** (default `42`) fixes one concrete pair list per target size *K*. The paper’s held-out AUC entries are **means ± standard deviations over 10 GroupShuffleSplit seeds**; those means are summarized in `subset_manifest.csv`.

## Files

- `truthfulqa_feature_balanced_<K>.csv` — row export for target size *K* (300, 350, …, 650).
- `subset_manifest.csv` — *K*, paths, and verification means from the locked summary CSV.
- Canonical pair-ID JSON: `results/feature_balanced_reference_subsets/pair_ids_<K>_seed42.json`.

## Regenerate

```bash
python3 scripts/export_feature_balanced_subset_csvs.py
```
