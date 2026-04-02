# Downloadable TruthfulQA audited subsets (feature-balanced protocol)

Released **binary-choice TruthfulQA** slices for the **`feature_balanced`** fixed-kept-count protocol are shipped at the **repository root** under **`truthfulqaPro/`** (CSVs, `subset_manifest.csv`, and `pair_ids/*.json`).

## Where to download

Full **GitHub raw URL table**, manifest, and regeneration instructions: **`truthfulqaPro/README.md`**.

Summary:

- **CSV exports:** `truthfulqaPro/truthfulqaPro_<K>.csv`
- **Manifest (paths + verification means):** `truthfulqaPro/subset_manifest.csv`
- **Canonical `pair_id` JSON** (reference split seed 42): `truthfulqaPro/pair_ids/pair_ids_<K>_seed42.json`
- **Locked multi-seed verification outputs:** `results/truthfulqa_pruning_final_verification/` (see root `README.md`)

Each CSV is self-contained (TruthfulQA-style columns plus audit metadata); you do not need a separate `TruthfulQA.csv` to evaluate the subset.

## Regenerate

```bash
python3 scripts/export_feature_balanced_subset_csvs.py
```

## License and attribution

Underlying text is from **TruthfulQA**; cite TruthfulQA and this repository when using these exports.
