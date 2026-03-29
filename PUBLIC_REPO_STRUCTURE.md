# Public repository structure

High-level layout: **audits** (frozen CSV inputs), **paper_assets** (LaTeX + figures), **data/subsets** (released subset CSVs), **results** (verification + canonical pair-id JSON), **scripts** (reproducible drivers), **archive** (exploratory / superseded).

```
truthfulqa_audit/
├── README.md
├── requirements.txt
├── requirements-paper-full.txt
├── CLEANUP_PLAN.md
├── PUBLIC_REPO_STRUCTURE.md
├── PAPER_DEPENDENCY_MAP.md
├── audits/                              # Frozen audit + benchmark-impact CSVs
├── paper_assets/
│   ├── fig/                             # Model-impact figures (make_paper_assets.py)
│   ├── figures/                         # Cross-dataset figure (run_fever_audit.py)
│   └── tables/                          # LaTeX fragments
├── figures/                             # Legacy audit PDFs + pruning-verification figures
├── notebooks/
│   └── TruthfulQA_Style_Confound_Audit.ipynb
├── scripts/                             # Python drivers (see PAPER_DEPENDENCY_MAP.md)
├── data/
│   ├── predictions/                     # Optional model outputs (may be gitignored)
│   └── subsets/
│       ├── README.md                    # Entry point for downloadable subsets
│       └── truthfulqa_feature_balanced/ # Feature-balanced CSV exports + subset_manifest.csv
├── results/
│   ├── truthfulqa_feature_balanced/
│   │   └── pair_ids/                    # Canonical pair_id JSON (reference seed 42)
│   └── truthfulqa_pruning_final_verification/  # Locked multi-seed verification tables
└── archive/                             # Exploratory grids, old scripts, notebook copy
```

**Naming:** **`surface10`** = default ten-feature surface audit in code (`scripts/truthfulqa_paper_audit.py`); legacy CLI alias `paper10` normalizes to `surface10`. Subset releases live under `data/subsets/truthfulqa_feature_balanced/`; pair lists under `results/truthfulqa_feature_balanced/pair_ids/`.

**Note:** `paper_assets/fig/` vs `paper_assets/figures/` is intentional (different generating scripts). Point LaTeX `\includegraphics` at the path you regenerate.
