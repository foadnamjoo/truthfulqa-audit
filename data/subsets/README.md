# Downloadable TruthfulQA audited subsets (feature-balanced protocol)

Released **binary-choice TruthfulQA** slices for the **`feature_balanced`** fixed-kept-count protocol are shipped at the **repository root** under **`truthfulqaPro/`** (CSVs, `subset_manifest.csv`, and `pair_ids/*.json`). The same files are mirrored on Hugging Face as **`foadnamjoo/TruthfulQAPro`** (see `truthfulqaPro/README.md`).

## Where to download

Full **GitHub raw URL table**, manifest, and regeneration instructions: **`truthfulqaPro/README.md`**.

Summary:

- **CSV exports:** `truthfulqaPro/truthfulqaPro_<K>.csv`
- **Manifest (paths + verification means):** `truthfulqaPro/subset_manifest.csv`
- **Canonical `pair_id` JSON** (reference split seed 42): `truthfulqaPro/pair_ids/pair_ids_<K>_seed42.json`
- **Locked multi-seed verification outputs:** `results/truthfulqa_pruning_final_verification/` (see root `README.md`)

Each **subset CSV** has the MC fields and `style_violation` for every kept pair (no separate `TruthfulQA.csv` needed to run the benchmark). **Slice-level** metadata (*K*, selection method, seed, paths, verification means) is in **`subset_manifest.csv`** and **`pair_ids/*.json`**, not duplicated on every row.

## Regenerate

```bash
python3 scripts/export_feature_balanced_subset_csvs.py
```

## License and attribution

Underlying text is from **TruthfulQA**; cite TruthfulQA and this repository when using these exports.
