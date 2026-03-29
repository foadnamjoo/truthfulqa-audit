# Public repository structure

This layout reflects the **full paper** (TruthfulQA audit + model impact + cross-dataset + LaTeX assets) and the **supplementary** near-random subset pipeline—not a single-track “subset only” repo.

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
├── figures/                      # Legacy audit PDFs + final near-random subset PDFs
├── notebooks/
│   └── TruthfulQA_Style_Confound_Audit.ipynb
├── scripts/
│   ├── truthfulqa_paper_audit.py
│   ├── make_paper_assets.py
│   ├── run_fever_audit.py
│   ├── run_binary_choice_eval.py
│   ├── make_final_tables.py
│   ├── build_audit_notebook.py
│   ├── import_chpc_predictions.py
│   ├── make_example_predictions.py
│   ├── run_truthfulqa_pruning_improved.py
│   ├── run_pruning_final_verification.py
│   ├── search_near_random_clean_subset.py
│   ├── run_near_random_better_algorithms.py
│   └── run_final_near_random_truthfulqa_subset.py
├── data/                         # Local datasets / predictions (optional; may be gitignored)
├── results/
│   ├── final_near_random_truthfulqa_subset/
│   └── final_near_random_truthfulqa_subset_repro_check/
└── archive/                      # Exploratory runs, superseded scripts, duplicate notebook copy
    ├── scripts/
    ├── results/
    ├── figures/
    ├── notebooks/
    ├── requirements-full-legacy.txt
    └── README.md
```

**Naming note:** `paper_assets/fig/` vs `paper_assets/figures/` is intentional (different scripts: `make_paper_assets.py` vs `run_fever_audit.py`). Point LaTeX `\includegraphics` at the path you regenerate.
