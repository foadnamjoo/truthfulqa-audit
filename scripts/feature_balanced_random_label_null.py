#!/usr/bin/env python3
"""
Pair-structured random-label null for feature-balanced fixed-K *held-out* audit AUC.

Mirrors ``truthfulqa_pruning_final_verification.run_fixed_kept`` for method
``feature_balanced``: for each GroupShuffleSplit seed, build the length-stratified
sorted prefix of size K, take **holdout** pairs only, then run the same
GroupKFold OOF logistic regression as ``search_truthfulqa_pruned_improved._auc_pairs``.

**Observed:** real correct/incorrect labels (one AUC per seed).
**Null:** for each (seed, permutation), shuffle labels **within** each pair_id
(destroy alignment of features to correctness while keeping one positive /
one negative row per pair), then recompute OOF AUC with the same pipeline.

Writes a CSV and prints LaTeX-friendly rows for the paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from search_truthfulqa_pruned_improved import _apply_prefix_keep, _ans_frame
from truthfulqa_pruning_final_verification import _sorted_df_feature_balanced
from truthfulqa_pruning_utils import CV_SPLITS, load_candidates_with_features, repo_root
import truthfulqa_paper_audit as tpa

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


def oof_auc_answer_rows(ans: pd.DataFrame, lr_seed: int) -> float:
    """Same as ``_auc_pairs`` but on a pre-built answer-level frame."""
    if ans["pair_id"].nunique() < CV_SPLITS:
        return float("nan")
    X = ans[FEAT_COLS].to_numpy()
    y = ans["label"].to_numpy()
    g = ans["pair_id"].to_numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=lr_seed))
    try:
        proba = cross_val_predict(pipe, X, y, cv=GroupKFold(n_splits=CV_SPLITS), groups=g, method="predict_proba")[:, 1]
    except ValueError:
        return float("nan")
    return float(roc_auc_score(y, proba))


def holdout_pair_df_for_seed(
    df: pd.DataFrame,
    target_keep: int,
    split_seed: int,
    holdout_fraction: float,
) -> pd.DataFrame | None:
    """Holdout-only pair rows (same K and split as verification). May bump K slightly for CV."""
    groups = df["example_id"].astype(int).to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=split_seed)
    tr_idx, ho_idx = next(gss.split(np.arange(len(df)), groups=groups))
    train_ids = set(df.iloc[tr_idx]["example_id"].astype(int).tolist())
    hold_ids = set(df.iloc[ho_idx]["example_id"].astype(int).tolist())
    d_sorted = _sorted_df_feature_balanced(df, split_seed)
    n_full = len(d_sorted)
    k = int(min(max(target_keep, CV_SPLITS + 1), n_full))
    for _ in range(n_full):
        kept = _apply_prefix_keep(d_sorted, k)
        kt = kept[kept["example_id"].isin(train_ids)]
        kh = kept[kept["example_id"].isin(hold_ids)]
        if len(kt) >= CV_SPLITS and len(kh) >= CV_SPLITS:
            return kh
        k += 1
        if k > n_full:
            break
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truthfulqa-csv", type=str, default="TruthfulQA.csv")
    p.add_argument("--audit-csv", type=str, default="audits/truthfulqa_style_audit.csv")
    p.add_argument("--holdout-fraction", type=float, default=0.25)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=10, help="Match locked verification (42..51).")
    p.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[300, 350, 400, 450, 500, 550],
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=200,
        help="Random-label draws per split seed (increase e.g. 500–1000 for smoother null std).",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default="results/feature_balanced_random_label_null/summary.csv",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for label permutations.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    tq = (root / args.truthfulqa_csv).resolve()
    audit_path = (root / args.audit_csv).resolve()
    if not tq.is_file() or not audit_path.is_file():
        print("Missing TruthfulQA.csv or audit CSV", file=sys.stderr)
        return 1

    df = load_candidates_with_features(tq, audit_path)
    rng = np.random.default_rng(args.seed)

    out_rows: list[dict[str, float | int | str]] = []
    out_path = (root / args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "target_K | obs_mean | obs_std | null_mean | null_std | null_p95 | "
        "emp_p_ge_obs_mean | n_null"
    )

    for K in args.targets:
        obs_list: list[float] = []
        null_list: list[float] = []

        for si in range(args.n_seeds):
            split_seed = args.base_seed + si
            kh = holdout_pair_df_for_seed(df, K, split_seed, args.holdout_fraction)
            if kh is None or len(kh) < CV_SPLITS:
                continue
            ans = _ans_frame(kh)
            obs = oof_auc_answer_rows(ans, split_seed)
            if np.isfinite(obs):
                obs_list.append(obs)

            y_base = ans["label"].to_numpy()
            g = ans["pair_id"].to_numpy()
            for _ in range(args.n_permutations):
                y_perm = tpa.shuffle_labels_within_groups(y_base, g, rng)
                a2 = ans.copy()
                a2["label"] = y_perm
                nu = oof_auc_answer_rows(a2, split_seed)
                if np.isfinite(nu):
                    null_list.append(nu)

        if not obs_list or not null_list:
            print(f"{K} | (skip: insufficient data)", file=sys.stderr)
            continue

        obs_mean = float(np.mean(obs_list))
        obs_std = float(np.std(obs_list, ddof=1)) if len(obs_list) > 1 else 0.0
        null_arr = np.array(null_list, dtype=float)
        null_mean = float(np.mean(null_arr))
        null_std = float(np.std(null_arr, ddof=1)) if len(null_arr) > 1 else 0.0
        null_p95 = float(np.percentile(null_arr, 95))
        emp_p = float(np.mean(null_arr >= obs_mean))

        out_rows.append(
            {
                "target_kept_count": K,
                "observed_mean_heldout_auc": obs_mean,
                "observed_std_heldout_auc": obs_std,
                "null_mean_auc": null_mean,
                "null_std_auc": null_std,
                "null_p95_auc": null_p95,
                "empirical_p_value_ge_observed_mean": emp_p,
                "n_seeds_used": len(obs_list),
                "n_null_draws": len(null_list),
            }
        )

        print(
            f"{K} | {obs_mean:.4f} | {obs_std:.4f} | {null_mean:.4f} | {null_std:.4f} | "
            f"{null_p95:.4f} | {emp_p:.4f} | {len(null_list)}"
        )

    if out_rows:
        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")

        print("\n% LaTeX table fragment (paste and adjust caption):")
        print(r"\begin{tabular}{@{}lcccc@{}}")
        print(r"\toprule")
        print(
            r"$K$ & Obs.\ held-out AUC & Null mean AUC & Null std & $P(\mathrm{null}\ge \mathrm{obs.\ mean})$ \\"
        )
        print(r"\midrule")
        for r in out_rows:
            print(
                f"{int(r['target_kept_count'])} & "
                f"${r['observed_mean_heldout_auc']:.4f} \\pm {r['observed_std_heldout_auc']:.4f}$ & "
                f"${r['null_mean_auc']:.4f}$ & "
                f"${r['null_std_auc']:.4f}$ & "
                f"${r['empirical_p_value_ge_observed_mean']:.3f}$ \\\\"
            )
        print(r"\bottomrule")
        print(r"\end{tabular}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
