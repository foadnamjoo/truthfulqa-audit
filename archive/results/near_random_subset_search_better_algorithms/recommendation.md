# Better algorithms — recommendation

## 1. Did any new algorithm beat `clean_first_then_low_score` on mean held-out AUC?
- **At 350:** best **search-only** mean held-out **0.5694** (`simulated_annealing_from_clean_first`) vs **clean_first** **0.5888**. **Beats** clean_first on mean held-out. Best **overall** **0.5694** (`simulated_annealing_from_clean_first`). Refined baseline ref **0.5888**.
- **At 400:** best **search-only** mean held-out **0.5749** (`simulated_annealing_from_clean_first`) vs **clean_first** **0.6023**. **Beats** clean_first on mean held-out. Best **overall** **0.5749** (`simulated_annealing_from_clean_first`). Refined baseline ref **0.6023**.

## 2. Best method at 350 / 400 (lowest mean held-out in grid)
- **350:** `simulated_annealing_from_clean_first`
- **400:** `simulated_annealing_from_clean_first`

## 3. Is 375 better than 350 or 400?
- Compare `mean_heldout_auc` / `mean_dist_to_chance` for your preferred method across 325–425 in `summary_by_target_and_method.csv`.

## 4. Practical limit
- Local search optimizes a **weighted proxy** during swaps; **reported** numbers are always **true held-out** OOF AUC. Margins over `clean_first` are often **small**; large gains are uncommon under paper10 + this split.

## 5. What to report
- **Strongest near-random:** smallest mean |AUC−0.5| you can defend (often 300–375).
- **Largest still-reasonable:** highest target with acceptable distance (often 375–400).

### Explicit sentences (same protocol, same seeds, vs refined `clean_first` baselines)
- At **350**, **`clean_first_then_low_score`** mean held-out is **0.5888** (replicated here). **`simulated_annealing_from_clean_first`** reaches **0.5694** — about **0.019** lower mean held-out (closer to 0.5).
- At **400**, **`clean_first_then_low_score`** is **0.6023** (replicated). **`simulated_annealing_from_clean_first`** reaches **0.5749** — about **0.027** lower.

### Is this branch still improving?
- **Yes, modestly:** with a small search budget (`eval_budget=8` full `evaluate_state` calls per run), **simulated annealing from `clean_first`** and **beam search from `clean_first`** sometimes find **lower held-out AUC** than one-shot `clean_first` / `score_rank`, especially at **350–400** and **375** (beam).
- **Caveat:** gains depend on **swap budget** and **RNG**; rerun with a larger `eval_budget` in `run_near_random_better_algorithms.py` if you need confirmation.