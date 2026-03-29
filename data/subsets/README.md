# Downloadable TruthfulQA audited subsets (feature-balanced protocol)

Release-ready **binary-choice TruthfulQA** slices for the **`feature_balanced`** fixed-kept-count protocol: same `paper10` surface-form audit and grouped-CV setup as the main paper, with sizes **300–650** pairs (300, 350, 400, 450, 500, 550, 595, 650).

## Where to download

Full **GitHub raw URL table**, manifest, and regeneration instructions: **`data/subsets/feature_balanced_paper10/README.md`**.

Summary:

- **CSV exports:** `data/subsets/feature_balanced_paper10/truthfulqa_feature_balanced_<K>.csv`
- **Manifest (paths + verification means):** `data/subsets/feature_balanced_paper10/subset_manifest.csv`
- **Canonical `pair_id` JSON** (reference split seed 42): `results/feature_balanced_reference_subsets/pair_ids_<K>_seed42.json`
- **Locked multi-seed verification outputs:** `results/truthfulqa_pruning_final_verification/` (see root `README.md`)

Each CSV is self-contained (TruthfulQA-style columns plus audit metadata); you do not need a separate `TruthfulQA.csv` to evaluate the subset.

## Regenerate

```bash
python3 scripts/export_feature_balanced_subset_csvs.py
```

## License and attribution

Underlying text is from **TruthfulQA**; cite TruthfulQA and this repository when using these exports.
