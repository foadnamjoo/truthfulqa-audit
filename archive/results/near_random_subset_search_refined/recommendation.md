# Refined near-random subset branch — recommendations

Protocol: **paper10**, **GroupShuffleSplit** (`test_size=0.25`), held-out AUC = grouped OOF on retained **test** pairs only, `min_keep_global=200`.

## 1. Best overall if the priority is **as close to chance (0.5) as possible**
- In this grid, the **lowest mean |AUC−0.5|** is **0.0821** at **target 300**, method **`score_rank (low separability)`** (mean held-out **0.5821**, mean clean pairs **144.8**).
- **Tradeoff at 300:** **`clean_first_then_low_score`** is worse on distance (**~0.104**) but retains **~232** mean clean pairs vs **~145** for score_rank — choose it when the branch goal (cleaner benchmark) outweighs the tighter near-random margin.
- Smaller targets (300–375) typically move held-out AUC closer to 0.5; see `summary_by_target_and_method.csv`.

## 2. Best if the priority is the **largest still-reasonable** subset
- Among **per-target winners** with **mean |AUC−0.5| ≤ 0.12**, the largest target is **400** (**`clean_first_then_low_score`**, mean dist **0.1023**, mean held-out **0.6023**).
- If **no** target meets ≤0.12, the table falls back to the tightest available winner; **check `best_method_by_target.csv`**. 

## 3. Is **400** still acceptable vs **~370**?
- **clean_first_then_low_score** at **375**: mean held-out **0.5959** ± **0.0248**, mean |AUC−0.5| **0.0959**, mean clean **239.0**.
- **Same method** at **400**: mean held-out **0.6023** ± **0.0291**, mean |AUC−0.5| **0.1023**, mean clean **240.6**.
- **Honest read:** if your bar for “near-random” is **|AUC−0.5| ≲ 0.08–0.10**, **375 is usually closer** than **400**; at **400**, held-out AUC is often **materially higher** (weaker near-random claim) unless a different method wins at that grid point.

## 4. Targets **425–500** and closeness to chance
- Among **winners** at 425–500, mean |AUC−0.5| ranges up to **~0.155**. 
- **450–500** is typically **not** “close to 0.5” on held-out audit in this protocol: expect **~0.14+** distance to chance for the best method at those sizes unless the fold is unusually easy.

## 5. What to **report**
- **Stronger near-random claim:** cite **target 300–350**, **clean_first_then_low_score** (or whichever wins in `best_method_by_target.csv` there), with **mean ± std** held-out AUC over seeds from `per_seed_results.csv`.
- **Larger “still useful” benchmark:** cite a target **375–425** if |AUC−0.5| is acceptable for your narrative, again with **held-out** metrics and **mean clean pair count / confounded fraction**.

## Caveats
- One **held-out pair split** per seed; std reflects **split noise**, not full hierarchical CI.
- “Best method” uses **mean distance to chance**, then **cleaner**, then **larger actual** count — see `pick_best_per_target` in `run_near_random_subset_refined.py`.