#!/usr/bin/env python3
"""
Multi-seed + fixed-kept-count verification for TruthfulQA pruning (surface10 audit by default).
Writes results to results/truthfulqa_pruning_final_verification/ and figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import truthfulqa_paper_audit as tpa
from truthfulqa_pruning_utils import CV_SPLITS, load_candidates_with_features, repo_root
from search_truthfulqa_pruned_improved import (
    _apply_prefix_keep,
    _evaluate_train_hold,
    _pair_separability_train,
    enrich_search_rows,
    search_beam_multi_start,
    search_feature_balanced,
    search_negation_first_stratified,
    search_prefix_grid,
    search_score_greedy,
)

METHOD_GROUPS = {
    "negation_first": lambda df, tid, hid, mk, p, s: search_negation_first_stratified(df, tid, hid, mk, p, s),
    "feature_balanced": lambda df, tid, hid, mk, p, s: search_feature_balanced(df, tid, hid, mk, p, s),
    "score_greedy": lambda df, tid, hid, mk, p, s: search_score_greedy(df, tid, hid, mk, p, s),
    "multi_start": lambda df, tid, hid, mk, p, s: search_beam_multi_start(df, tid, hid, mk, p, s, n_starts=5),
}


def _prefix_grid_all(df, train_ids, hold_ids, min_keep, max_drop, target_auc, profile, seed):
    rows: list[dict[str, Any]] = []
    for mode in ["negation_only", "length_only", "neg_len", "all_features"]:
        rows.extend(search_prefix_grid(df, mode, train_ids, hold_ids, min_keep, max_drop, target_auc, profile, seed))
    return rows


def _sorted_df_feature_balanced(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    d = df.copy()
    d["len_bin"] = pd.qcut(d["length_gap"], q=4, labels=False, duplicates="drop")
    parts = []
    for b in sorted(d["len_bin"].dropna().unique()):
        sub = d[d["len_bin"] == b]
        parts.append(sub.sample(frac=1.0, random_state=seed + int(b)))
    mixed = pd.concat(parts, ignore_index=True)
    return mixed.sort_values(["negation_flag", "length_gap", "example_id"], ascending=[True, True, True]).reset_index(drop=True)


def _sorted_df_score_greedy(df: pd.DataFrame, train_ids: set[int], seed: int) -> pd.DataFrame:
    train_df = df[df["example_id"].isin(train_ids)].copy()
    sep = _pair_separability_train(train_df, seed=seed)
    d = df.copy()
    d["_sep"] = d["example_id"].map(sep).fillna(0.0)
    return d.sort_values(["_sep", "example_id"], ascending=[True, True]).reset_index(drop=True)


def _evaluate_fixed_keep(
    df_sorted: pd.DataFrame,
    target_keep: int,
    train_ids: set[int],
    hold_ids: set[int],
    profile: str,
    seed: int,
) -> tuple[int, float, float, float, float, float, int, int]:
    """Return keep + metrics; bump keep_n up if train/hold too small for CV."""
    n_full = len(df_sorted)
    k = int(min(max(target_keep, CV_SPLITS + 1), n_full))
    for _ in range(n_full):
        kept = _apply_prefix_keep(df_sorted, k)
        kt = kept[kept["example_id"].isin(train_ids)]
        kh = kept[kept["example_id"].isin(hold_ids)]
        if len(kt) >= CV_SPLITS and len(kh) >= CV_SPLITS:
            s_auc, h_auc = _evaluate_train_hold(df_sorted, k, train_ids, hold_ids, profile, seed)
            conf = float(kept["confounded_flag"].mean()) if len(kept) else float("nan")
            neg = float(kept["negation_flag"].mean()) if len(kept) else float("nan")
            return k, s_auc, h_auc, float(s_auc - h_auc), conf, neg, int(len(kt)), int(len(kh))
        k += 1
        if k > n_full:
            break
    return min(target_keep, n_full), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, 0


def _best_by_method(enriched: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {m: [] for m in ["negation_first", "feature_balanced", "score_greedy", "multi_start", "prefix_grid"]}
    for r in enriched:
        m = str(r["method"])
        if m in by:
            by[m].append(r)
    out: dict[str, dict[str, Any]] = {}
    for m, rows in by.items():
        if rows:
            out[m] = min(rows, key=lambda x: float(x["objective_score"]))
    return out


def run_multi_seed(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df["example_id"].astype(int).to_numpy()
    profile = args.audit_profile
    multi_rows: list[dict[str, Any]] = []
    for si in range(args.n_seeds):
        split_seed = args.base_seed + si
        gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_fraction, random_state=split_seed)
        tr_idx, ho_idx = next(gss.split(np.arange(len(df)), groups=groups))
        train_ids = set(df.iloc[tr_idx]["example_id"].astype(int).tolist())
        hold_ids = set(df.iloc[ho_idx]["example_id"].astype(int).tolist())
        full_auc = float(_evaluate_train_hold(df.sort_values("example_id").reset_index(drop=True), len(df), train_ids, hold_ids, profile, split_seed)[1])

        raw: list[dict[str, Any]] = []
        raw.extend(_prefix_grid_all(df, train_ids, hold_ids, args.min_keep, args.max_drop_fraction, args.target_audit_auc, profile, split_seed))
        for _, fn in METHOD_GROUPS.items():
            raw.extend(fn(df, train_ids, hold_ids, args.min_keep, profile, split_seed))
        enriched = enrich_search_rows(
            raw, df, train_ids, profile, split_seed, args.min_keep, args.w_heldout, args.w_size, args.w_imbalance, full_auc
        )
        best_map = _best_by_method(enriched)
        for method, row in best_map.items():
            multi_rows.append(
                {
                    "seed": split_seed,
                    "method": method,
                    "search_time_auc": float(row["search_time_auc"]),
                    "heldout_auc": float(row["heldout_auc"]),
                    "optimism_gap": float(row["optimism_gap"]),
                    "kept_count": int(row["retained_count"]),
                    "retained_confounded_fraction": float(row["retained_confounded_fraction"]),
                    "retained_negation_rate": float(row["retained_negation_rate"]),
                    "objective_score": float(row["objective_score"]),
                    "mode": row.get("mode", ""),
                    "full_dataset_auc": float(full_auc),
                }
            )
    ms = pd.DataFrame(multi_rows)
    ms.to_csv(out_dir / "multi_seed_results.csv", index=False)
    summary_rows = []
    for method in sorted(ms["method"].unique()):
        sub = ms[ms["method"] == method]
        wins = 0
        for seed in ms["seed"].unique():
            ssub = ms[ms["seed"] == seed]
            if len(ssub):
                best = ssub.loc[ssub["heldout_auc"].idxmin()]
                if best["method"] == method:
                    wins += 1
        summary_rows.append(
            {
                "method": method,
                "mean_heldout_auc": float(sub["heldout_auc"].mean()),
                "std_heldout_auc": float(sub["heldout_auc"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "mean_kept_count": float(sub["kept_count"].mean()),
                "std_kept_count": float(sub["kept_count"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "mean_search_time_auc": float(sub["search_time_auc"].mean()),
                "mean_optimism_gap": float(sub["optimism_gap"].mean()),
                "num_seeds_where_best": int(wins),
                "num_seeds": int(len(sub)),
            }
        )
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out_dir / "method_stability_summary.csv", index=False)
    return ms, summ


def run_fixed_kept(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df["example_id"].astype(int).to_numpy()
    profile = args.audit_profile
    targets = [650, 595, 550, 500, 450, 400, 350, 300]
    per_seed_rows: list[dict[str, Any]] = []
    for si in range(args.n_seeds):
        split_seed = args.base_seed + si
        gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_fraction, random_state=split_seed)
        tr_idx, ho_idx = next(gss.split(np.arange(len(df)), groups=groups))
        train_ids = set(df.iloc[tr_idx]["example_id"].astype(int).tolist())
        hold_ids = set(df.iloc[ho_idx]["example_id"].astype(int).tolist())
        for target in targets:
            for method in ["feature_balanced", "score_greedy"]:
                d_sorted = _sorted_df_feature_balanced(df, split_seed) if method == "feature_balanced" else _sorted_df_score_greedy(df, train_ids, split_seed)
                actual, s_auc, h_auc, gap, conf, neg, n_tr, n_ho = _evaluate_fixed_keep(d_sorted, target, train_ids, hold_ids, profile, split_seed)
                per_seed_rows.append(
                    {
                        "seed": split_seed,
                        "target_kept_count": target,
                        "actual_kept_count": actual,
                        "method": method,
                        "search_time_auc": s_auc,
                        "heldout_auc": h_auc,
                        "optimism_gap": gap,
                        "retained_confounded_fraction": conf,
                        "retained_negation_rate": neg,
                        "train_pairs_kept": n_tr,
                        "hold_pairs_kept": n_ho,
                    }
                )
    ps = pd.DataFrame(per_seed_rows)
    ps.to_csv(out_dir / "fixed_kept_count_per_seed.csv", index=False)
    agg = (
        ps.groupby(["target_kept_count", "method"], as_index=False)
        .agg(
            mean_actual_kept_count=("actual_kept_count", "mean"),
            mean_heldout_auc=("heldout_auc", "mean"),
            std_heldout_auc=("heldout_auc", "std"),
            mean_search_time_auc=("search_time_auc", "mean"),
            mean_optimism_gap=("optimism_gap", "mean"),
            mean_retained_confounded_fraction=("retained_confounded_fraction", "mean"),
            mean_retained_negation_rate=("retained_negation_rate", "mean"),
            mean_train_pairs_kept=("train_pairs_kept", "mean"),
            mean_hold_pairs_kept=("hold_pairs_kept", "mean"),
            num_seeds=("seed", "count"),
        )
    )
    agg.to_csv(out_dir / "fixed_kept_count_summary.csv", index=False)
    return ps, agg


def write_fixed_md(agg: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "| kept count | method | mean held-out AUC | std | mean confounded fraction | mean negation rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for t in [650, 595, 550, 500, 450, 400, 350, 300]:
        sub = agg[agg["target_kept_count"] == t]
        if len(sub):
            b = sub.loc[sub["mean_heldout_auc"].idxmin()]
            lines.append(
                f"| {t} | {b['method']} | {b['mean_heldout_auc']:.4f} | {b['std_heldout_auc']:.4f} | {b['mean_retained_confounded_fraction']:.4f} | {b['mean_retained_negation_rate']:.4f} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(ms: pd.DataFrame, fixed_agg: pd.DataFrame, summ: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    # kept_count vs heldout
    xs, means, stds = [], [], []
    for t in [650, 595, 550, 500, 450, 400]:
        sub = fixed_agg[fixed_agg["target_kept_count"] == t]
        if len(sub):
            b = sub.loc[sub["mean_heldout_auc"].idxmin()]
            xs.append(t)
            means.append(float(b["mean_heldout_auc"]))
            stds.append(float(b["std_heldout_auc"]))
    if xs:
        plt.figure(figsize=(6, 4))
        plt.errorbar(xs, means, yerr=stds, fmt="o-", capsize=3)
        plt.xlabel("target kept count")
        plt.ylabel("mean held-out AUC")
        plt.tight_layout()
        plt.savefig(fig_dir / "kept_count_vs_mean_heldout_auc.png", dpi=220)
        plt.close()
    if len(summ):
        plt.figure(figsize=(7, 4))
        x = np.arange(len(summ))
        plt.bar(x, summ["mean_heldout_auc"], yerr=summ["std_heldout_auc"], capsize=3)
        plt.xticks(x, summ["method"], rotation=25, ha="right")
        plt.ylabel("mean held-out AUC")
        plt.tight_layout()
        plt.savefig(fig_dir / "method_comparison_heldout_auc.png", dpi=220)
        plt.close()
        plt.figure(figsize=(7, 4))
        plt.bar(x, summ["mean_optimism_gap"])
        plt.xticks(x, summ["method"], rotation=25, ha="right")
        plt.ylabel("mean optimism gap")
        plt.tight_layout()
        plt.savefig(fig_dir / "optimism_gap_by_method.png", dpi=220)
        plt.close()


def write_simple_reports(out_dir: Path, ms: pd.DataFrame, summ: pd.DataFrame, fixed_agg: pd.DataFrame) -> None:
    (out_dir / "prefix_grid_check.md").write_text(
        "# Prefix grid check\n\nFixed keep-range sweep is active (`n_hi=n_full`, `n_lo=max(CV_SPLITS+1,min_keep,ceil(n_full*(1-max_drop_fraction)))`).\n",
        encoding="utf-8",
    )
    if len(ms):
        wins = []
        for s in sorted(ms["seed"].unique()):
            b = ms[ms["seed"] == s].loc[ms[ms["seed"] == s]["heldout_auc"].idxmin()]
            wins.append(f"- seed {s}: {b['method']} ({b['heldout_auc']:.4f})")
        (out_dir / "best_method_consistency.md").write_text("# Best method consistency\n\n" + "\n".join(wins) + "\n", encoding="utf-8")
    if len(fixed_agg):
        rows = []
        for t in [650, 595, 550, 500, 450, 400, 350, 300]:
            sub = fixed_agg[fixed_agg["target_kept_count"] == t]
            if len(sub):
                b = sub.loc[sub["mean_heldout_auc"].idxmin()]
                rows.append(
                    {
                        "full_dataset_count": 790,
                        "full_dataset_auc": float(ms["full_dataset_auc"].mean()) if len(ms) else float("nan"),
                        "target_kept_count": t,
                        "mean_actual_kept_count": b["mean_actual_kept_count"],
                        "best_method": b["method"],
                        "mean_heldout_auc": b["mean_heldout_auc"],
                        "std_heldout_auc": b["std_heldout_auc"],
                        "mean_retained_confounded_fraction": b["mean_retained_confounded_fraction"],
                        "mean_retained_negation_rate": b["mean_retained_negation_rate"],
                    }
                )
        pd.DataFrame(rows).to_csv(out_dir / "paper_summary_table.csv", index=False)
    (out_dir / "final_verification_report.md").write_text(
        "# Final verification report\n\nSee `method_stability_summary.csv`, `fixed_kept_count_summary.csv`, and `fixed_kept_count_table.md`.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--truthfulqa-csv", type=str, default="TruthfulQA.csv")
    p.add_argument("--audit-csv", type=str, default="audits/truthfulqa_style_audit.csv")
    p.add_argument("--output-dir", type=str, default="results/truthfulqa_pruning_final_verification")
    p.add_argument("--figures-dir", type=str, default="figures/truthfulqa_pruning_final_verification")
    p.add_argument(
        "--audit-profile",
        type=str,
        default="surface10",
        choices=["surface10", "surface13", "paper10", "expanded13"],
        help="surface10 = ten interpretable surface features (alias: paper10); surface13 adds three pair-level features (alias: expanded13).",
    )
    p.add_argument("--min-keep", type=int, default=500)
    p.add_argument("--target-audit-auc", type=float, default=0.62)
    p.add_argument("--max-drop-fraction", type=float, default=0.40)
    p.add_argument("--holdout-fraction", type=float, default=0.25)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--w-heldout", type=float, default=1.0)
    p.add_argument("--w-size", type=float, default=0.35)
    p.add_argument("--w-imbalance", type=float, default=0.25)
    p.add_argument("--skip-multi-seed", action="store_true")
    p.add_argument("--skip-fixed", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.audit_profile = tpa.normalize_audit_profile(str(args.audit_profile))
    root = repo_root()
    out_dir = (root / args.output_dir).resolve()
    fig_dir = (root / args.figures_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_candidates_with_features((root / args.truthfulqa_csv).resolve(), (root / args.audit_csv).resolve())
    ms = pd.DataFrame()
    summ = pd.DataFrame()
    fixed_agg = pd.DataFrame()
    if not args.skip_multi_seed:
        ms, summ = run_multi_seed(df, args, out_dir)
    if not args.skip_fixed:
        _, fixed_agg = run_fixed_kept(df, args, out_dir)
        write_fixed_md(fixed_agg, out_dir / "fixed_kept_count_table.md")
    if not ms.empty and not summ.empty:
        if fixed_agg.empty and (out_dir / "fixed_kept_count_summary.csv").exists():
            fixed_agg = pd.read_csv(out_dir / "fixed_kept_count_summary.csv")
        if not fixed_agg.empty:
            make_figures(ms, fixed_agg, summ, fig_dir)
    write_simple_reports(out_dir, ms, summ, fixed_agg)
    print(f"Wrote outputs under {out_dir} and {fig_dir}")


if __name__ == "__main__":
    main()

