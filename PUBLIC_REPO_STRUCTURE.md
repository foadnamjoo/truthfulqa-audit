# Public repository structure

This layout reflects the **full paper** (TruthfulQA audit + nine-model impact + cross-dataset audits + LaTeX assets) and the **supplementary** feature-balanced subset plus pruning-verification artifacts.

```
truthfulqa_audit/
├── README.md
├── requirements.txt              # Core: numpy, pandas, scipy, matplotlib, scikit-learn
├── requirements-paper-full.txt   # Optional: HF eval, datasets, torch, transformers, …
├── CLEANUP_PLAN.md
├── PUBLIC_REPO_STRUCTURE.md      # This file
├── PAPER_DEPENDENCY_MAP.md       # Section/figure/table → scripts → outputs
├── audits/                       # Frozen CSVs for audit + benchmark impact + FEVER run
├── paper_assets/
│   ├── fig/                      # Model-impact figures (from make_paper_assets.py)
│   ├── figures/                  # Cross-dataset bar plot (from run_fever_audit.py)
│   └── tables/                   # LaTeX fragments for the paper
├── figures/                      # Legacy audit PDFs + pruning-verification figures
├── notebooks/
│   └── TruthfulQA_Style_Confound_Audit.ipynb
├── scripts/                      # See listing in README.md / PAPER_DEPENDENCY_MAP.md
├── data/                         # Local datasets / predictions (optional; may be gitignored)
│   └── subsets/
│       ├── README.md             # Pointers to feature_balanced_paper10/
│       └── feature_balanced_paper10/
├── results/
│   ├── feature_balanced_reference_subsets/
│   └── truthfulqa_pruning_final_verification/
└── archive/                      # Exploratory runs, superseded scripts, duplicate notebook copy
    ├── scripts/
    ├── results/
    ├── figures/
    ├── notebooks/
    ├── requirements-full-legacy.txt
    └── README.md
```

**Naming note:** `paper_assets/fig/` vs `paper_assets/figures/` is intentional (different scripts: `make_paper_assets.py` vs `run_fever_audit.py`). Point LaTeX `\includegraphics` at the path you regenerate.
