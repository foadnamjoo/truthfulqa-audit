# Near-random held-out AUC subset search

- Protocol: paper10, GroupShuffleSplit test_size=0.25, min_keep_global=200, band=|AUC-0.5|≤0.06
- Pair counts: target global retained pairs in [300, 500] (step 5 for selection).

## Method summary (mean ± std over seeds)

| method | mean actual pairs | std | mean held-out AUC | std | mean |AUC-0.5| | seeds in band |
|---|---:|---:|---:|---:|---:|---:|
| clean_first_then_low_score | 370.2 | 40.99 | 0.5579 | 0.0056 | 0.0579 | 7/10 |
| score_rank (low separability) | 351.3 | 56.07 | 0.5580 | 0.0278 | 0.0580 | 4/10 |
| len_gap_rank (symmetric length) | 337.6 | 37.62 | 0.5660 | 0.0217 | 0.0660 | 5/10 |
| negation_rank (low asymmetry) | 429.4 | 67.06 | 0.5902 | 0.0506 | 0.0902 | 3/10 |
| uniform_random_train_pairs | 370.7 | 60.35 | 0.5937 | 0.0464 | 0.0937 | 2/10 |

## Recommendation
- **Best tradeoff (lowest mean distance to 0.5):** `clean_first_then_low_score` — see per_seed_best.csv and sweep CSV.