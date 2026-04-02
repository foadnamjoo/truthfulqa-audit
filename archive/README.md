# Archive

Exploratory and **superseded** artifacts (historical near-random search grids, pruning-replay outputs, internal tests, duplicate notebook copy). The **active** paper reproduction paths live in the repo root: see `README.md` and `PAPER_DEPENDENCY_MAP.md`.

Some `archive/scripts/*.py` files imported helpers that used to live under root `scripts/`; those drivers were retired from `main`. Recover deleted modules from `git` history if you need to rerun an archived script verbatim.

The older multi-method grid drivers **`run_pruning_final_verification.py`** and **`run_truthfulqa_pruning_improved.py`** were removed from active `scripts/`; the locked supplementary track uses **`truthfulqa_pruning_final_verification.py`** plus **`export_feature_balanced_subset_csvs.py`**. Archived scripts that still name the removed files expect you to recover them from git history if you rerun those paths.

Optional dependency pins: `requirements-full-legacy.txt` (same role as `requirements-paper-full.txt` at repo root).
