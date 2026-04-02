# Public repository structure

High-level layout: **audits** (frozen CSV inputs), **paper_assets** (LaTeX + figures), **data/subsets** (released subset CSVs), **results** (verification + canonical pair-id JSON), **scripts** (reproducible drivers), **archive** (exploratory / superseded).

```
truthfulqa_audit/
├── README.md
├── requirements.txt
├── requirements-paper-full.txt
├── PUBLIC_REPO_STRUCTURE.md
├── PAPER_DEPENDENCY_MAP.md
├── audits/                              # Frozen audit + benchmark-impact CSVs
├── paper_assets/
│   └── fig/                             # Model-impact + cross-dataset figures
│   └── tables/                          # LaTeX fragments
├── figures/                             # Legacy audit PDFs + pruning-verification figures
├── notebooks/
│   └── TruthfulQA_Style_Confound_Audit.ipynb
├── scripts/                             # Python drivers (see PAPER_DEPENDENCY_MAP.md)
├── data/
│   ├── predictions/                     # Optional model outputs (may be gitignored)
│   └── subsets/
│       ├── README.md                    # Entry point for downloadable subsets
│       └── truthfulqaPro/               # Feature-balanced CSV exports + subset_manifest.csv
├── results/
│   ├── truthfulqaPro/
│   │   └── pair_ids/                    # Canonical pair_id JSON (reference seed 42)
│   └── truthfulqa_pruning_final_verification/  # Locked multi-seed verification tables
└── archive/                             # Exploratory grids, old scripts, notebook copy
```

**Naming:** **`surface10`** = default ten-feature surface audit in code (`scripts/truthfulqa_paper_audit.py`); legacy CLI alias `paper10` normalizes to `surface10`. Subset releases live under `data/subsets/truthfulqaPro/`; pair lists under `results/truthfulqaPro/pair_ids/`.

**Note:** `paper_assets/fig/` is the single canonical figure output directory for both model-impact and cross-dataset plots.
