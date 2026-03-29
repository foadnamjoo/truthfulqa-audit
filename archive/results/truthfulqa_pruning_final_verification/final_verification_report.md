# Final verification report (corrected pruning-improved family)

- **Multi-seed methods:** `negation_first_constrained`, `feature_balanced`, `score_based_greedy`, `beam_or_multistart_greedy` — representative mode **`all_features`**.
- **Seeds:** 10, **GroupShuffleSplit** test_size=0.25, **paper10**.

> **Greedy multi-seed path:** All four defaults returned **identical** subsets each seed (**kept_count = 790** = full **N**). They **do not distinguish** here; stability of *held-out AUC across seeds* still reflects split noise (~**0.036** std).

> **Where methods differ:** Use the **fixed kept-count** sweep (`score_rank_fixed_global` vs `negation_rank_fixed_global`).

## 1. Stability across seeds/splits?
- Compare per-method **std** of `heldout_auc` in `method_stability_summary.csv` (typical scale **~0.03–0.05** on a **~25%** held-out fold).
- **Reported mean held-out (any default row):** **0.6918** (± **0.0360**), mean kept **790.0**.

## 2. Best method among the four defaults?
- **Achieved min held-out per seed (ties count):** `beam_or_multistart_greedy, feature_balanced, negation_first_constrained, score_based_greedy (tied)` — see `wins_best_heldout_per_seed`.
- **Interpretation:** Metrics match across methods each seed because **no default greedy run pruned** below full **N** (**790**); **feature_balanced does not win**—all are tied.

## 3. Held-out AUC vs fixed kept count?
- See **`fixed_kept_count_table.md`** and **`fixed_kept_count_summary.csv`** (best of `score_rank_fixed_global` vs `negation_rank_fixed_global` per target).

## 4. Is 595 a strong operating point?
- **595** (score_rank, mean ± std): **0.6846 ± 0.0240** — compare to **500/450** in the same table.

## 5. Better tradeoff at 650, 550, 500, 450, 400?
- **Lowest mean held-out** in this fixed grid (among reported rows): **0.6458** at target **500** (see CSV for method).

## 6. Strong enough for the paper?
- **Yes if** you report **held-out mean ± std over seeds** and contrast to full-set reference **0.7159**; **no** if you need a much lower AUC without accepting smaller **N** or different splits.

## 7. Numbers to cite
- **Full dataset (OOF):** **0.7159**, **N = 790**.
- **Greedy defaults (multi-seed):** mean held-out **0.6918** ± **0.0360**; **`all_four_defaults_identical (no pruning below full N)`** in `paper_summary_table.csv`.
- **Fixed retained sizes:** cite the row for your **target_kept_count** in `paper_summary_table.csv` (held-out mean ± std).

## 8. Caveats
- **Search-time AUC** is secondary; lead with **held-out AUC**.
- Default multi-seed path uses **`all_features`** only; full `modes=all` in the main script can differ.
- Fixed-kept rows may show **nearest feasible** global **N**; use **`actual_kept_count`** in per-seed CSV.
