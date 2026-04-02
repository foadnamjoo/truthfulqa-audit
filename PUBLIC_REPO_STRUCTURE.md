# Public repository structure

High-level layout: **audits** (frozen CSV inputs), **paper_assets** (LaTeX + figures), **data/subsets** (released subset CSVs), **results** (verification + canonical pair-id JSON), **scripts** (reproducible drivers), **archive** (exploratory / superseded).

**Released audited subsets (paper supplementary):** top-level `truthfulqaPro/` (CSVs, manifest, and `pair_ids/`) — see root `README.md` callout and `truthfulqaPro/README.md`. `data/subsets/README.md` points here. Hugging Face mirror: **`foadnamjoo/TruthfulQAPro`** (same files; Hub repo name differs from the git folder prefix **`truthfulqaPro`**).

```
truthfulqa_audit/
├── README.md
├── requirements.txt
├── requirements-paper-full.txt
├── PUBLIC_REPO_STRUCTURE.md
├── PAPER_DEPENDENCY_MAP.md
├── CITATION.cff                         # GitHub “Cite this repository” + preferred paper citation
├── paper_assets/references.bib          # Canonical BibTeX (manuscript + TruthfulQA)
├── truthfulqaPro/                       # Released subset CSVs + manifest + pair_ids JSON
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
│       └── README.md                    # Pointer to truthfulqaPro/ at repo root
├── results/
│   └── truthfulqa_pruning_final_verification/  # Locked multi-seed verification tables
└── archive/                             # Exploratory grids, old scripts, notebook copy
```

**Naming:** **`surface10`** = default ten-feature surface audit in code (`scripts/truthfulqa_paper_audit.py`); legacy CLI alias `paper10` normalizes to `surface10`. Subset releases and canonical `pair_id` JSON live under top-level **`truthfulqaPro/`**.

**Note:** `paper_assets/fig/` is the single canonical figure output directory for both model-impact and cross-dataset plots.
