# Classifier-guided audit pruning (thresholded)

This document describes the **audit-prune-thresholded** subset-selection method implemented in `scripts/run_audit_prune_thresholded.py` and `scripts/audit_subset_evaluator.py`.

## Objective

**Maximize** the number of retained TruthfulQA question pairs (equivalently, maximize retained fraction) **subject to** the existing surface-form audit classifier achieving **grouped 5-fold cross-validated ROC–AUC** at most a target \(\tau\) on the retained set.

Formally: find a large set \(S\) of `pair_id` / `example_id` values such that

\[
\mathrm{AUC}_{\mathrm{grouped\text{-}CV}}(S) \le \tau,
\]

where \(\mathrm{AUC}_{\mathrm{grouped\text{-}CV}}\) is computed exactly like the pruning/search audit in `search_truthfulqa_pruned_improved._auc_pairs`: answer-level rows (two per pair), **StandardScaler + LogisticRegression**, **GroupKFold** by pair, out-of-fold `predict_proba`, then ROC–AUC between correct (1) and incorrect (0) answer rows.

The implementation uses the same `_ans_frame` feature construction as that module (surface10-style ten features).

## Algorithm (exact)

**Inputs:** Full merged candidate frame (790 pairs), target thresholds \(\tau \in \{0.60, 0.57, 0.55, 0.53\}\), removal scoring mode (`confidence` or `imbalance`), random seed for the logistic regression / CV.

1. **Greedy removal**  
   Start from \(S\) = all pairs. Repeat until \(\mathrm{AUC}(S) \le \tau\) (numerical tolerance \(10^{-9}\)):
   - Compute grouped CV OOF probabilities on \(S\) (same pipeline as evaluation).
   - **Confidence:** per pair, score = \(|p(\text{correct}) - p(\text{incorrect})|\) using OOF probabilities (higher = more confidently separated → remove first).
   - **Imbalance:** on the current answer-level matrix, compute global \(\mu_{\text{pos}} - \mu_{\text{neg}}\) per feature; for each pair, \(\Delta\) = feature vector on correct minus incorrect; score = \(\sum_f |\Delta_f|\) over features where \(\Delta_f\) has the same sign as the global gap (reinforces class-conditional shift).
   - Remove the pair with **maximum** score; ties broken by **smallest** `example_id`.
   - Stop if \(|S|\) would drop below `CV_SPLITS` (5) pairs without meeting \(\tau\) → error.

2. **Add-back refinement**  
   Let \(R\) be the set of removed pairs. Repeatedly scan \(R\) in ascending `example_id` order: if \(S \cup \{p\}\) still satisfies \(\mathrm{AUC} \le \tau\), add \(p\) back to \(S\) and remove it from \(R\). Repeat full passes until a pass adds nothing (local maximum under single-pair additions).

**Output:** Final retained set \(S\), with final grouped CV AUC and accuracy on \(S\).

## Difference from the fixed-\(K\) prefix baseline

The **released baseline** (`feature_balanced` / `feature_balanced_length_stratified_prefix`) is:

- `qcut` on `length_gap` → shuffle within quartiles with `seed + bin` → sort by `(negation_flag, length_gap, example_id)` → **keep the first \(K\) pairs**.
- It does **not** optimize the audit AUC; \(K\) is chosen **externally**.

The **new method** chooses **which** pairs to keep so that the **audit objective** (grouped CV AUC) is **explicitly** driven below \(\tau\) while **retaining as many pairs as possible** (greedy + add-back heuristic).

## Why this is closer to “max retained size under low leakage”

- **Leakage** is measured by the **same** audit you care about, not by a proxy ordering.
- The procedure **directly enforces** \(\mathrm{AUC} \le \tau\) instead of post-hoc measuring a fixed prefix.
- **Add-back** attempts to **increase** \(|S|\) without violating \(\tau\), approximating a **large** feasible set rather than a single nested prefix chain.

**Limitations:** Greedy removal is not guaranteed to find the **global** maximum \(|S|\); non-monotone AUC under nested sets is possible; cost is high (many repeated CV fits). The **baseline comparison** row uses the **largest prefix length** \(K\) on the seed-42 feature-balanced ordering such that the **same** grouped CV AUC on that prefix is \(\le \tau\) (linear scan over \(K\)).

## Artifacts

- `truthfulqaAuditPrune/*.csv` — subset rows (same column pattern as `truthfulqaPro` exports).
- `truthfulqaAuditPrune/pair_ids/pair_ids_<tauSlug>_<strategy>_seed<seed>.json`
- `truthfulqaAuditPrune/subset_manifest.csv`
- `results/audit_prune_thresholded/config.json`
- `results/audit_prune_thresholded/summary_table.csv`
- `results/audit_prune_thresholded/comparison_vs_baseline.csv`

## How to run

From the repository root:

```bash
python scripts/run_audit_prune_thresholded.py \
  --truthfulqa-csv TruthfulQA.csv \
  --audit-csv audits/truthfulqa_style_audit.csv \
  --seed 42 \
  --baseline-prefix-seed 42 \
  --thresholds 0.60 0.57 0.55 0.53 \
  --strategies confidence imbalance
```

Use `--skip-artifacts` to compute metrics only (no CSV/JSON/manifest).
