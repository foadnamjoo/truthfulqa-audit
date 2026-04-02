# Downloadable TruthfulQA audited subsets (feature-balanced protocol)

**On GitHub:** browse [`truthfulqaPro/`](https://github.com/foadnamjoo/truthfulqa-audit/tree/main/data/subsets/truthfulqaPro) (CSVs + `subset_manifest.csv`) and [`results/truthfulqaPro/pair_ids/`](https://github.com/foadnamjoo/truthfulqa-audit/tree/main/results/truthfulqaPro/pair_ids) (canonical JSON).

Release-ready **binary-choice TruthfulQA** slices for the **`feature_balanced`** fixed-kept-count protocol: same **surface10** surface-form audit (ten interpretable features; legacy alias `paper10`) and grouped-CV setup as the main paper, with sizes **300–650** pairs (300, 350, 400, 450, 500, 550, 595, 650).

## Where to download

Full **GitHub raw URL table**, manifest, and regeneration instructions: **`data/subsets/truthfulqaPro/README.md`**.

Summary:

- **CSV exports:** `data/subsets/truthfulqaPro/truthfulqaPro_<K>.csv`
- **Manifest (paths + verification means):** `data/subsets/truthfulqaPro/subset_manifest.csv`
- **Canonical `pair_id` JSON** (reference split seed 42): `results/truthfulqaPro/pair_ids/pair_ids_<K>_seed42.json`
- **Locked multi-seed verification outputs:** `results/truthfulqa_pruning_final_verification/` (see root `README.md`)

Each CSV is self-contained (TruthfulQA-style columns plus audit metadata); you do not need a separate `TruthfulQA.csv` to evaluate the subset.

## Regenerate

```bash
python3 scripts/export_feature_balanced_subset_csvs.py
```

## License and attribution

Underlying text is from **TruthfulQA**; cite TruthfulQA and this repository when using these exports.
