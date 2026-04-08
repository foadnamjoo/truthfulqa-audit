# Improved Search (Accuracy-Thresholded)

This experiment adds a stronger search path on top of existing methods without changing:
- fixed-prefix `truthfulqaPro` baseline
- current `audit_prune_thresholded` implementation

## Objective

For each threshold `tau_acc`, maximize retained pair count subject to:

- grouped CV accuracy <= `tau_acc` (primary constraint)
- grouped CV AUC reported as secondary metric

Thresholds explored:
- 0.60, 0.57, 0.55, 0.53

## Search variants

Implemented in `scripts/run_audit_prune_improved_search.py`:

- multiple random restarts
- optional beam kickoff (small-width perturbations)
- scoring strategies:
  - confidence
  - imbalance
  - hybrid (`alpha` in {0.25, 0.50, 0.75})
- greedy pruning trajectory (remove one pair at a time)
- add-back refinement
- optional swap refinement

## Reused audit pipeline

Uses the same answer-level frame and grouped CV logistic-regression path as existing audit pruning:
- `search_truthfulqa_pruned_improved._ans_frame`
- grouped `GroupKFold` CV
- `StandardScaler + LogisticRegression`
- OOF probabilities to compute grouped CV accuracy and AUC

## Outputs

- `results/audit_prune_improved_search/config.json`
- `results/audit_prune_improved_search/exploratory_runs.csv`
- `results/audit_prune_improved_search/summary_table.csv`
- `results/audit_prune_improved_search/comparison_table.csv`
- `results/audit_prune_improved_search/REPORT.md`
- `truthfulqaAuditPruneImproved/*.csv`
- `truthfulqaAuditPruneImproved/pair_ids/*.json`
- `truthfulqaAuditPruneImproved/subset_manifest.csv`

