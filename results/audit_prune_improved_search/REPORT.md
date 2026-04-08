# Improved Search Report

Primary objective: maximize retained pairs subject to grouped CV ACCURACY threshold.
Secondary metric: grouped CV AUC.

## Best subset by threshold
- acc <= 0.60: strategy=imbalance seed=915 retained=653 acc=0.5980 auc=0.6411
- acc <= 0.57: strategy=hybrid_a50 seed=2109 retained=613 acc=0.5693 auc=0.5773
- acc <= 0.55: strategy=hybrid_a50 seed=2109 retained=580 acc=0.5500 auc=0.5571
- acc <= 0.53: strategy=hybrid_a25 seed=2109 retained=565 acc=0.5265 auc=0.5482

## Single strongest overall candidate
- strategy=hybrid_a50 threshold=0.55 retained=580 acc=0.5500 auc=0.5571

## Did we beat prior near-0.55 best?
- Prior: retained=568, acc~0.5528, auc~0.5493; New strongest retained=580, acc=0.5500, auc=0.5571. Beat prior: True

## Accuracy-vs-AUC optimization impact
- Compared with AUC-thresholded current method, this run prioritizes accuracy constraints first; see comparison_table.csv for per-threshold retained-size tradeoffs.

## 600-700 pairs near 55% classifier performance
- Check acc<=0.55 row in summary_table.csv; this is the direct test for whether retained count approaches 600-700.
