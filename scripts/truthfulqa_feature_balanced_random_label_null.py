#!/usr/bin/env python3
"""
Pair-structured random-label null for feature-balanced fixed-K subsets.

For each target K and each GroupShuffleSplit seed (same policy as
``truthfulqa_pruning_final_verification.py`` / ``export_feature_balanced_subset_csvs.py``):

1. **Observed** held-out AUC: grouped OOF ROC-AUC on *holdout* pairs in the length-
   stratified prefix of size K (same code path as ``_evaluate_train_hold`` → ``_auc_pairs``).
2. **Null** AUC: repeat ``n_null`` times with labels permuted **within pair_id**
   (pair-structured null), same features and same GroupKFold CV layout.

Writes ``summary_by_target.csv`` and ``report.md`` under
``results/truthfulqa_feature_balanced_random_label_null/`` by default.

Example::

    python3 scripts/truthfulqa_feature_balanced_random_label_null.py --n-null 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import truthfulqa_paper_audit as tpa
from export_feature_balanced_subset_csvs import sorted_df_feature_balanced
from search_truthfulqa_pruned_improved import _apply_prefix_keep, _ans_frame, _evaluate_train_hold
from truthfulqa_pruning_utils import CV_SPLITS, load_candidates_with_features, repo_root

FEAT_COLS = [
    "neg_lead",
    "neg_cnt",
    "hedge_rate",
    "auth_rate",
    "len_gap",
    "word_count",
    "sent_count",
    "avg_token_len",
    "type_token",
    "punc_rate",
]


def oof_auc_with_labels(kh: pd.DataFrame, seed: int, y: np.ndarray) -> float:
    """Grouped OOF AUC on holdout pair rows; y length 2 * len(kh), order matches _ans_frame."""
    if len(kh) < CV_SPLITS:
        return float("nan")
    ans = _ans_frame(kh)
    if len(y) != len(ans):
        raise ValueError("label length mismatch")
    X = ans[FEAT_COLS].to_numpy()
    g = ans["pair_id"].to_numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    proba = cross_val_predict(pipe, X, y, cv=GroupKFold(n_splits=CV_SPLITS), groups=g, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def oof_auc_observed(kh: pd.DataFrame, seed: int) -> float:
    """Same as ``_auc_pairs(kh, seed)`` — real labels."""
    if len(kh) < CV_SPLITS:
        return float("nan")
    ans = _ans_frame(kh)
    y = ans["label"].to_numpy()
    return oof_auc_with_labels(kh, seed, y)


def oof_auc_one_null_draw(kh: pd.DataFrame, seed: int, rng: np.random.Generator) -> float:
    """One pair-structured random-label draw (permute within pair_id)."""
    if len(kh) < CV_SPLITS:
        return float("nan")
    ans = _ans_frame(kh)
    y = ans["label"].to_numpy()
    g = ans["pair_id"].to_numpy()
    y_perm = tpa.shuffle_labels_within_groups(y, g, rng)
    return oof_auc_with_labels(kh, seed, y_perm)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truthfulqa-csv", type=str, default="TruthfulQA.csv")
    p.add_argument("--audit-csv", type=str, default="audits/truthfulqa_style_audit.csv")
    p.add_argument("--holdout-fraction", type=float, default=0.25)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=10, help="Match pruning verification (10 GroupShuffleSplit seeds).")
    p.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[300, 350, 400, 450, 500, 550],
        help="Fixed kept counts K (default: 300–550 step 50; add more sizes explicitly if needed).",
    )
    p.add_argument("--n-null", type=int, default=200, help="Random-label draws per (seed, K) cell.")
    p.add_argument("--rng-seed", type=int, default=42, help="Base RNG for null permutations (independent of split seeds).")
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/truthfulqa_feature_balanced_random_label_null",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    tq = (root / args.truthfulqa_csv).resolve()
    audit_path = (root / args.audit_csv).resolve()
    if not tq.is_file() or not audit_path.is_file():
        print("Missing TruthfulQA.csv or audit CSV.", file=sys.stderr)
        return 1

    df = load_candidates_with_features(tq, audit_path)
    groups = df["example_id"].astype(int).to_numpy()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell_rows: list[dict[str, float | int]] = []
    all_null_by_target: dict[int, list[float]] = {k: [] for k in args.targets}

    for si in range(args.n_seeds):
        split_seed = args.base_seed + si
        gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_fraction, random_state=split_seed)
        tr_idx, ho_idx = next(gss.split(np.arange(len(df)), groups=groups))
        train_ids = set(df.iloc[tr_idx]["example_id"].astype(int).tolist())
        hold_ids = set(df.iloc[ho_idx]["example_id"].astype(int).tolist())
        d_sorted = sorted_df_feature_balanced(df, split_seed)

        # Independent RNG stream per (seed, target) so results are stable when only one target changes
        for target in args.targets:
            rng = np.random.default_rng(args.rng_seed + 100_003 * split_seed + target)
            _, h_obs = _evaluate_train_hold(d_sorted, target, train_ids, hold_ids, "surface10", split_seed)

            kept = _apply_prefix_keep(d_sorted, target)
            kh = kept[kept["example_id"].isin(hold_ids)]
            n_ho = int(len(kh))

            null_aucs: list[float] = []
            for _ in range(args.n_null):
                null_aucs.append(oof_auc_one_null_draw(kh, split_seed, rng))

            for v in null_aucs:
                if np.isfinite(v):
                    all_null_by_target[target].append(float(v))

            arr = np.array(null_aucs, dtype=float)
            per_cell_rows.append(
                {
                    "split_seed": split_seed,
                    "target_k": target,
                    "n_holdout_pairs": n_ho,
                    "observed_heldout_auc": float(h_obs),
                    "null_mean": float(np.nanmean(arr)),
                    "null_std": float(np.nanstd(arr, ddof=1)) if np.sum(np.isfinite(arr)) > 1 else 0.0,
                    "null_q05": float(np.nanquantile(arr, 0.05)),
                    "null_q95": float(np.nanquantile(arr, 0.95)),
                    "n_null": args.n_null,
                }
            )

    per_cell = pd.DataFrame(per_cell_rows)
    per_cell_path = out_dir / "per_seed_target.csv"
    per_cell.to_csv(per_cell_path, index=False)

    summary_rows: list[dict[str, float | int | str]] = []
    for target in sorted(args.targets, reverse=True):
        sub = per_cell[per_cell["target_k"] == target]
        obs_mean = float(sub["observed_heldout_auc"].mean())
        obs_std = float(sub["observed_heldout_auc"].std(ddof=1)) if len(sub) > 1 else 0.0
        null_pool = np.array(all_null_by_target[target], dtype=float)
        null_pool = null_pool[np.isfinite(null_pool)]
        nm = float(null_pool.mean()) if len(null_pool) else float("nan")
        ns = float(null_pool.std(ddof=1)) if len(null_pool) > 1 else 0.0
        q05 = float(np.quantile(null_pool, 0.05)) if len(null_pool) else float("nan")
        q50 = float(np.quantile(null_pool, 0.50)) if len(null_pool) else float("nan")
        q95 = float(np.quantile(null_pool, 0.95)) if len(null_pool) else float("nan")
        # Empirical one-sided p: how often is null AUC >= observed (excess separability vs label-randomized)
        frac_ge = float(np.mean(null_pool >= obs_mean)) if len(null_pool) else float("nan")
        inside = bool(q05 <= obs_mean <= q95) if len(null_pool) else False

        summary_rows.append(
            {
                "target_k": target,
                "observed_mean_heldout_auc": obs_mean,
                "observed_std_across_seeds": obs_std,
                "null_mean_pooled": nm,
                "null_std_pooled": ns,
                "null_q05_pooled": q05,
                "null_q50_pooled": q50,
                "null_q95_pooled": q95,
                "empirical_p_null_ge_observed": frac_ge,
                "observed_inside_null_q05_q95": inside,
                "n_seeds": args.n_seeds,
                "n_null_per_cell": args.n_null,
                "n_null_pooled": int(len(null_pool)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_by_target.csv"
    summary.to_csv(summary_path, index=False)

    table_txt = summary.to_string(index=False)
    lines = [
        "# Random-label null vs observed (feature-balanced fixed K)",
        "",
        "Same **GroupShuffleSplit** train/hold and **feature_balanced** ordering as pruning verification.",
        "Held-out metric: grouped **OOF AUC** on holdout pairs only (`_evaluate_train_hold` / `_auc_pairs`).",
        "Null: **pair-structured** label permutations (swap correct/incorrect within each pair), repeated `n_null` times per (seed, K); table pools nulls across seeds.",
        "",
        "```",
        table_txt,
        "```",
        "",
        f"Machine-readable: `{summary_path.relative_to(root)}`",
        f"Per-seed detail: `{per_cell_path.relative_to(root)}`",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {summary_path.relative_to(root)}")
    print(f"Wrote {per_cell_path.relative_to(root)}")
    print(f"Wrote {(out_dir / 'report.md').relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
