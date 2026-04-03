---
license: apache-2.0
language:
  - en
tags:
  - truthfulqa
  - multiple-choice
  - evaluation
  - llm
  - benchmark
pretty_name: TruthfulQAPro (feature-balanced subsets)
size_categories:
  - 1K<n<10K
---

# TruthfulQAPro

**TruthfulQAPro** is the Hugging Face dataset (`foadnamjoo/TruthfulQAPro`): **feature-balanced reference subsets** derived from [TruthfulQA](https://github.com/sylinrl/TruthfulQA)—fixed-size binary-choice slices (300–650 pairs), a manifest with verification metrics, and **canonical pair-ID JSON** files (seed 42) for exact reproduction.

**Naming:** the Hub repo id is **`TruthfulQAPro`**. Filenames on the Hub match the GitHub release: CSVs are still named **`truthfulqaPro_<K>.csv`** (historical prefix; same files as folder `truthfulqaPro/` in [truthfulqa-audit](https://github.com/foadnamjoo/truthfulqa-audit)).

- **Code & protocol:** [github.com/foadnamjoo/truthfulqa-audit](https://github.com/foadnamjoo/truthfulqa-audit)
- **Paper:** *Judging by the Cover: Auditing Surface-Form Shortcuts in Binary-Choice Truth Benchmarks* — canonical BibTeX: [paper_assets/references.bib](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/paper_assets/references.bib); GitHub metadata: [CITATION.cff](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/CITATION.cff).

## Dataset summary

| Item | Description |
|------|-------------|
| **Base data** | TruthfulQA multiple-choice rows (same examples as upstream; only subset membership differs). |
| **Audit profile** | **surface10** — ten interpretable lexical/stylistic features with grouped cross-validation (legacy alias `paper10` in scripts). |
| **CSVs** | `truthfulqaPro_<K>.csv` for *K* ∈ {300, 350, 400, 450, 500, 550, 595, 650}. |
| **Manifest** | `subset_manifest.csv` — *K*, paths, verification means from the locked summary. |
| **Pair lists** | `pair_ids/pair_ids_<K>_seed42.json` — canonical pair IDs for seed 42. |
| **Ordering** | Length-quartile stratified shuffle, then sort by negation/length gap/id, then keep the first *K* pairs (`feature_balanced_length_stratified_prefix`). |

## How to load

```python
from datasets import load_dataset

ds = load_dataset("foadnamjoo/TruthfulQAPro", data_files="truthfulqaPro_650.csv")
# Or load the manifest:
manifest = load_dataset("foadnamjoo/TruthfulQAPro", data_files="subset_manifest.csv")
```

JSON pair lists are plain files under `pair_ids/` (not tabular); download with `hf_hub` or the Hub file browser.

## Licenses

- **TruthfulQA** (underlying Q&A content and MC structure) is released under the **Apache License 2.0** by the original authors. See the [upstream LICENSE](https://github.com/sylinrl/TruthfulQA/blob/main/LICENSE).
- **Subset selection, manifest, pair-ID JSON, and documentation** in this release are provided by the audit authors under the **MIT License** (see [truthfulqa-audit LICENSE](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/LICENSE)). Redistribution of the CSVs remains subject to compliance with the TruthfulQA Apache-2.0 terms.

The Hub YAML `license: apache-2.0` reflects the **upstream** dataset license; see this section for the full picture.

## Citation

### TruthfulQA (please cite the original benchmark)

```bibtex
@article{lin2022truthfulqa,
  title     = {Truthful{QA}: Measuring How Models Mimic Human Falsehoods},
  author    = {Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  journal   = {arXiv preprint arXiv:2109.07958},
  year      = {2022}
}
```

### This audit / subsets

```bibtex
@misc{namjoo2026judging,
  title  = {Judging by the Cover: Auditing Surface-Form Shortcuts in Binary-Choice Truth Benchmarks},
  author = {Namjoo, Foad and Phillips, Jeff M.},
  year   = {2026},
  url    = {https://github.com/foadnamjoo/truthfulqa-audit},
  note   = {Manuscript in preparation.}
}
```

Keep this block in sync with [`paper_assets/references.bib`](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/paper_assets/references.bib) on GitHub. The repository also ships [`CITATION.cff`](https://github.com/foadnamjoo/truthfulqa-audit/blob/main/CITATION.cff) for GitHub’s citation button.

## Contact

- **Homepage:** [users.cs.utah.edu/~foad27/](https://users.cs.utah.edu/~foad27/)
- **GitHub:** [github.com/foadnamjoo](https://github.com/foadnamjoo)
