# Feature-balanced subsets: random-label null (held-out AUC)

`summary.csv` is produced by:

```bash
python3 scripts/feature_balanced_random_label_null.py --n-permutations 200
```

**Protocol:** Same as `truthfulqa_pruning_final_verification` for `feature_balanced` fixed \(K\): each seed defines a GroupShuffleSplit train/hold split and the length-stratified sort; **held-out** pairs from the prefix of size \(K\) get GroupKFold OOF AUC with the same logistic regression features as `search_truthfulqa_pruned_improved._auc_pairs`.

**Null:** labels are permuted **within** each `pair_id` (pair-structured), then OOF AUC is recomputed. **Empirical \(p\)** = fraction of null draws with AUC \(\ge\) the **mean** observed AUC across seeds (one-sided, “how often does shuffled noise look this separable?”).

Increase `--n-permutations` (e.g. 500–1000) for smoother null std in the paper.
