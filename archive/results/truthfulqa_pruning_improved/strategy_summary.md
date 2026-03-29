# TruthfulQA pruning (improved search) — summary

- **Audit profile (default paper path):** `paper10`
- **Feature columns:** `['neg_lead', 'neg_cnt', 'hedge_rate', 'auth_rate', 'len_gap', 'word_count', 'sent_count', 'avg_token_len', 'type_token', 'punc_rate']`
- **Train/test pair split:** 0.75 / 0.25

Default run uses only: negation_first_constrained, feature_balanced, score_based_greedy, beam_or_multistart_greedy.

- **`best_subset.csv` / `best_subset_ids.json`:** lowest **composite objective** across all (method, mode) runs in this invocation (`selection=lowest_composite_objective`). Report **held-out AUC** as the primary generalization metric.

## Methods tried (this run)

negation_first_constrained, feature_balanced, score_based_greedy, beam_or_multistart_greedy

## Row matching exported best subset

```
method                          negation_first_constrained
mode                                         negation_only
audit_profile                                      paper10
search_time_auc                                   0.716611
heldout_auc                                       0.690197
optimism_gap                                      0.026414
retained_count                                         790
dropped_count                                            0
retained_confounded_fraction                      0.674684
retained_negation_rate                            0.506329
retained_length_gap_mean                          0.236555
retained_label_balance                                 0.5
objective_score                                   0.140395
```


## Best composite objective row

```
method                          negation_first_constrained
mode                                         negation_only
audit_profile                                      paper10
search_time_auc                                   0.716611
heldout_auc                                       0.690197
optimism_gap                                      0.026414
retained_count                                         790
dropped_count                                            0
retained_confounded_fraction                      0.674684
retained_negation_rate                            0.506329
retained_length_gap_mean                          0.236555
retained_label_balance                                 0.5
objective_score                                   0.140395
```

## Interpretation

Report **held-out AUC** as the primary generalization metric; **search-time AUC** is optimistic.

### Paper readiness
If held-out AUC cannot get clearly below ~0.65 while keeping 500+ pairs, the bottleneck is likely that surface-form signal is diffuse across many pairs (not a few outliers), so greedy pair-dropping underperforms. Say so honestly in the paper and show Pareto tradeoffs.