# TruthfulQAPro — feature-balanced reference subsets (GitHub)

These CSVs are **downloadable reference slices** for the `feature_balanced` fixed-kept-count protocol documented in `results/truthfulqa_pruning_final_verification/`.

**Naming:** the Hugging Face dataset is **`foadnamjoo/TruthfulQAPro`**. This repository keeps the directory and file prefix **`truthfulqaPro`** (e.g. `truthfulqaPro/truthfulqaPro_650.csv`) so paths stay stable in the manifest and scripts.

**Audit profile:** **surface10** — ten interpretable lexical and stylistic features (negation, hedging, length, punctuation, …) with grouped cross-validation, as in the main paper. Older releases called this `paper10`; that name is accepted as a legacy alias in scripts only.

## Direct download (GitHub)

Same pattern as the official TruthfulQA release ([`TruthfulQA.csv`](https://github.com/sylinrl/TruthfulQA/blob/main/TruthfulQA.csv)): stable paths in this repo, with **raw** URLs for scripts and notebooks.

**Repository:** [foadnamjoo/truthfulqa-audit](https://github.com/foadnamjoo/truthfulqa-audit)  
**Hugging Face (same files + dataset card):** [foadnamjoo/TruthfulQAPro](https://huggingface.co/datasets/foadnamjoo/TruthfulQAPro)  
**Default branch:** `main` (replace with a **tag or commit SHA** for a permanently frozen URL).

| Artifact | Browse on GitHub | Direct raw CSV / JSON |
|----------|------------------|------------------------|
| Manifest (metrics + paths) | [subset_manifest.csv](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/truthfulqaPro/subset_manifest.csv) | [raw](https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/subset_manifest.csv) |
| 300 pairs | [truthfulqaPro_300.csv](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/truthfulqaPro/truthfulqaPro_300.csv) | [raw](https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/truthfulqaPro_300.csv) |
| 350 pairs | […350.csv](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/truthfulqaPro/truthfulqaPro_350.csv) | [raw](https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/truthfulqaPro_350.csv) |
| 400 pairs | […400.csv](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/truthfulqaPro/truthfulqaPro_400.csv) | [raw](https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/truthfulqaPro_400.csv) |
| 450 pairs | […450.csv](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/truthfulqaPro/truthfulqaPro_450.csv) | [raw](https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/truthfulqaPro_450.csv) |
| 500 pairs | […500.csv](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/truthfulqaPro/truthfulqaPro_500.csv) | [raw](https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/truthfulqaPro_500.csv) |
| 550 pairs | […550.csv](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/truthfulqaPro/truthfulqaPro_550.csv) | [raw](https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/truthfulqaPro_550.csv) |
| 595 pairs | […595.csv](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/truthfulqaPro/truthfulqaPro_595.csv) | [raw](https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/truthfulqaPro_595.csv) |
| 650 pairs | […650.csv](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/truthfulqaPro/truthfulqaPro_650.csv) | [raw](https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/truthfulqaPro_650.csv) |

**Canonical pair-ID JSON** (same `K`; reference seed 42): under `truthfulqaPro/pair_ids/`, e.g. [pair_ids_650_seed42.json (browse)](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/truthfulqaPro/pair_ids/pair_ids_650_seed42.json) · [raw](https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/pair_ids/pair_ids_650_seed42.json).

**Python one-liner example:**

```python
import pandas as pd
url = "https://raw.githubusercontent.com/foadnamjoo/truthfulqa-audit/main/truthfulqaPro/truthfulqaPro_650.csv"
df = pd.read_csv(url)
```

## Hugging Face / `datasets`

The Hub dataset card lists **separate configs** (`manifest`, `subset_300`, …): one CSV schema each, so the viewer and `load_dataset` do not merge incompatible columns. Use `load_dataset("foadnamjoo/TruthfulQAPro", "manifest")` or `"subset_650"`, etc. Raw URLs still work in pandas (see table above).

## What is fixed

- **Examples** are original TruthfulQA rows; only membership changes.
- **Ordering** matches `scripts/truthfulqa_pruning_final_verification.py`: length-quartile stratified shuffle, then sort by negation/length gap/id, then keep the first *K* pairs (`feature_balanced_length_stratified_prefix`).
- **Reference split seed** (default `42`) fixes one concrete pair list per target size *K*. Held-out AUC entries in the manifest are **means ± standard deviations over 10 GroupShuffleSplit seeds** from the locked verification run.

## Files

- `truthfulqaPro_<K>.csv` — one row per selected MC pair. **Columns:** `pair_id` (index into upstream MC `TruthfulQA.csv`), TruthfulQA MC fields (`Type` … `Best Incorrect Answer`), `style_violation` (from `audits/truthfulqa_style_audit.csv`), `subset_name` (e.g. `truthfulqaPro_650`). **Not repeated on each row** (see manifest + JSON): target *K*, selection method, split seed, `TruthfulQA.csv` path, canonical JSON path, verification means.
- `subset_manifest.csv` — one row per *K*: paths, `reference_split_seed`, held-out AUC mean/std (multi-seed verification), confounded fraction, clean-pair count, `paper_role`.
- `pair_ids/pair_ids_<K>_seed42.json` — canonical ordered `pair_ids`, `selection_method`, `audit_profile`, split counts, verification-note text, `source_dataset`, etc.

## Regenerate

```bash
python3 scripts/export_feature_balanced_subset_csvs.py
```

The export script also refreshes this `README.md` body; if you extend the URL table, keep paths in sync with the top-level folder `truthfulqaPro/`.
