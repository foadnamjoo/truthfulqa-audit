# Best-method consistency (multi-seed)

_Representative **`mode=all_features`** for all four default methods (see `run_pruning_final_verification.py`)._

> **Degenerate run:** For every seed, all four default methods produced the **same** `heldout_auc`, `search_time_auc`, `kept_count`, and objective (they did **not** reduce below full **N** under current hyperparameters). `wins_best_heldout_per_seed` therefore counts **ties** (each tied method gets +1 per seed).

- **Lowest mean held-out AUC:** `all methods (identical metrics)`
- **Lowest median held-out AUC:** `all methods (identical metrics)`
- **Most seed-wise wins** (among ties at min held-out): `beam_or_multistart_greedy, feature_balanced, negation_first_constrained, score_based_greedy (tied)`

```
                    method  mean_heldout_auc  std_heldout_auc  median_heldout_auc  mean_kept_count  std_kept_count  num_seed_runs  wins_best_heldout_per_seed
 beam_or_multistart_greedy          0.691755         0.036013            0.680587            790.0             0.0             10                          10
          feature_balanced          0.691755         0.036013            0.680587            790.0             0.0             10                          10
negation_first_constrained          0.691755         0.036013            0.680587            790.0             0.0             10                          10
        score_based_greedy          0.691755         0.036013            0.680587            790.0             0.0             10                          10
```

_Where methods actually prune differently, compare `mean_heldout_auc` and `wins_*`; see fixed-kept sweep._