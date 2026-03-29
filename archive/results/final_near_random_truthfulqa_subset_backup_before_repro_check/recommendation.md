# Final near-random TruthfulQA subset — paper recommendation

## Locked scope
- **Branch:** near-random / better-algorithm only (not legacy pruning replay, not `feature_balanced` as primary).
- **Audit:** `paper10`, `audits/truthfulqa_style_audit.csv`, **no** label or text changes.
- **Held-out:** `GroupShuffleSplit(test_size=0.25)` per seed; held-out AUC = grouped OOF on **retained test** answer rows.

## Run configuration (this table)
- **Seeds:** 15 independent pair splits (`seed` = 0..14).
- **Search budget:** `22` full `evaluate_state` calls per **simulated annealing** / **beam** run (stronger than the exploratory `eval_budget=8` pass).
- **Targets:** [350, 375, 400].
- **Methods:** `clean_first_then_low_score`, `score_rank (low separability)`, `simulated_annealing_from_clean_first`, `beam_light_swaps_from_clean`.

## 1. Main paper operating point
- **Recommendation:** **350 pairs** using **`simulated_annealing_from_clean_first`**.
- **Rationale:** lowest **mean distance to chance** (mean |held-out AUC − 0.5|) among the three sizes, i.e. **strongest defensible “near-random” claim** in this grid.
- **Numbers:** mean held-out **0.5599** ± **0.0373**; mean |AUC−0.5| **0.0599**; mean clean pairs **229.3**; mean confounded fraction **0.3450**.

## 2. Secondary (larger) operating point
- **Recommendation:** **400 pairs** using **`beam_light_swaps_from_clean`**.
- **Rationale:** larger retained set for a **“still useful benchmark”** story while reporting held-out AUC honestly (typically **slightly farther** from 0.5 than the main point).
- **Numbers:** mean held-out **0.5547** ± **0.0492**; mean |AUC−0.5| **0.0632**; mean clean pairs **236.5**.

## 3. Per-target winners (350 / 375 / 400)
| Target | Best method | Mean held-out AUC | Std | Mean |AUC−0.5| |
|---:|---|---:|---:|---:|
| 350 | simulated_annealing_from_clean_first | 0.5599 | 0.0373 | 0.0599 |
| 375 | simulated_annealing_from_clean_first | 0.5726 | 0.0318 | 0.0726 |
| 400 | beam_light_swaps_from_clean | 0.5547 | 0.0492 | 0.0632 |

## 4. What to claim in the paper
- **Strongest near-random:** cite **350 pairs**, **`simulated_annealing_from_clean_first`**, held-out **0.5599 ± 0.0373** (15 splits), plus clean/confounded summary from `summary_by_target_and_method.csv`.
- **Larger still-useful benchmark:** cite **400 pairs**, **`beam_light_swaps_from_clean`**, held-out **0.5547 ± 0.0492**.

## 5. Algorithm name for the paper
**Simulated annealing from clean-first** (`simulated_annealing_from_clean_first`): start from `clean_first_then_low_score`, then random pair swaps optimizing a weighted held-out–centric loss until the evaluation budget is exhausted.

Generic paper name: **“clean-first constrained subset selection with fixed-size local search (simulated annealing / beam) under the paper10 audit.”**

## 6. Reproducible subset IDs
- `final_subset_ids_350.json`, `final_subset_ids_375.json`, `final_subset_ids_400.json` list **retained `pair_ids`** for the **per-target winning method** using **canonical** `GroupShuffleSplit` seed **42** and `search_rng_derived_seed` / `search_eval_budget` where applicable.

## 7. Honest comparison
- **375** is **not** the best on **mean |AUC−0.5|** in this strengthened run (it is **looser** than **350**).
- **400 + beam** actually has a **slightly lower mean held-out AUC** than **350 + SA** here (**0.5547** vs **0.5599**), but a **higher** mean cross-seed distance to chance (**0.0632** vs **0.0599**) because per-seed |AUC−0.5| averages differ from |mean AUC − 0.5|.
- **Primary vs secondary** in Sections 1–2 follows the pre-declared rule: **minimize mean distance to chance** first → **350 + SA** as main; **400 + beam** as the **larger** secondary. If the paper prioritizes **retained size** and **lower mean held-out**, authors may instead foreground **400 + beam** and cite **350 + SA** as the tighter-distance variant.
