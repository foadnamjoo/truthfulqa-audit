---
license: apache-2.0
task_categories:
- question-answering
- multiple-choice
language:
- en
tags:
- truthfulqa
- benchmark
- evaluation
- llm
- surface-form-audit
- shortcut-learning
size_categories:
- n<1K
pretty_name: TruthfulQA-Audited
configs:
- config_name: surface_audited_tau052
  data_files: surface_audited/tqa_tau052.csv
- config_name: surface_audited_tau053
  data_files: surface_audited/tqa_tau053.csv
- config_name: surface_audited_tau054
  data_files: surface_audited/tqa_tau054.csv
- config_name: feature_balanced_K300
  data_files: feature_balanced/tqa_K300.csv
- config_name: feature_balanced_K400
  data_files: feature_balanced/tqa_K400.csv
- config_name: feature_balanced_K500
  data_files: feature_balanced/tqa_K500.csv
- config_name: feature_balanced_K650
  data_files: feature_balanced/tqa_K650.csv
---

# TruthfulQA-Audited

Surface-balanced and feature-balanced subsets of binary-choice TruthfulQA, released alongside the paper "Judging by the Cover: Auditing Surface-Form Shortcuts in Binary-Choice Truth Benchmarks" (Namjoo & Phillips, 2026).

All CSVs share the upstream TruthfulQA schema:
`pair_id, Type, Category, Question, Best Answer, Best Incorrect Answer, subset_name`

## Recommended subset

`surface_audited/tqa_tau052.csv` (528 pairs, leakage AUC = 0.513) is the primary recommendation. It reduces surface-form separability to near-random while preserving model rankings (Spearman rho = 0.980, Kendall tau = 0.922 across 14 open-weights models).

## surface_audited/ (audit-pruned subsets)

| File              | Pairs | Leakage AUC | Spearman rho | Kendall tau |
|-------------------|------:|------------:|-------------:|------------:|
| tqa_tau052.csv ⭐  |   528 |       0.513 |        0.980 |       0.922 |
| tqa_tau053.csv    |   536 |       0.530 |        0.971 |       0.900 |
| tqa_tau054.csv    |   536 |       0.530 |        0.979 |       0.922 |

Built with imbalance-based classifier-guided AUC-thresholded pruning. See `surface_audited_manifest.csv` for full metrics.

## feature_balanced/ (deterministic fixed-K baselines)

| File           | Pairs | Held-out AUC (mean +- std) |
|----------------|------:|---------------------------:|
| tqa_K300.csv   |   300 |             0.510 +- 0.009 |
| tqa_K350.csv   |   350 |             0.536 +- 0.020 |
| tqa_K400.csv   |   400 |             0.545 +- 0.016 |
| tqa_K450.csv   |   450 |             0.538 +- 0.037 |
| tqa_K500.csv   |   500 |             0.568 +- 0.045 |
| tqa_K550.csv   |   550 |             0.587 +- 0.045 |
| tqa_K595.csv   |   595 |             0.616 +- 0.038 |
| tqa_K650.csv   |   650 |             0.632 +- 0.039 |

Built with deterministic length-quartile-stratified prefix selection. See `feature_balanced_manifest.csv` for full metrics.

## Loading

```python
import pandas as pd
df = pd.read_csv("surface_audited/tqa_tau052.csv")
```

From Hugging Face directly:

```python
import pandas as pd
df = pd.read_csv(
    "hf://datasets/foadnamjoo/TruthfulQA-Audited/surface_audited/tqa_tau052.csv"
)
```

## Citation

```bibtex
@misc{namjoo2026judging,
  title  = {Judging by the Cover: Auditing Surface-Form Shortcuts in Binary-Choice Truth Benchmarks},
  author = {Namjoo, Foad and Phillips, Jeff M.},
  year   = {2026},
  note   = {Manuscript in preparation},
  url    = {https://github.com/foadnamjoo/truthfulqa-audit},
}
```

TruthfulQA itself is from Lin, Hilton & Evans (2022); please cite the original benchmark when using these subsets.
