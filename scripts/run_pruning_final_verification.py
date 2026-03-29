#!/usr/bin/env python3
"""
Final verification: multi-seed pair splits, fixed global kept counts, stability summaries.

Multi-seed stage runs **only** the corrected pruning-improved default family:
negation_first_constrained, feature_balanced, score_based_greedy, beam_or_multistart_greedy
(representative mode: `all_features`, matching the main search path for comparability).

Fixed-kept-count sweep uses score_rank_fixed_global vs negation_rank_fixed_global (targeted sizes).

Outputs: results/truthfulqa_pruning_final_verification/ and figures/truthfulqa_pruning_final_verification/
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import truthfulqa_paper_audit as tpa


def _load_pruning_module():
    path = _SCRIPT_DIR / "run_truthfulqa_pruning_improved.py"
    name = "tq_prune_improved"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PR = _load_pruning_module()


def split_train_test_pairs_gss(
    n_pairs: int, test_size: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """One GroupShuffleSplit partition: each pair_id is its own group."""
    X = np.zeros((n_pairs, 1))
    groups = np.arange(n_pairs, dtype=np.int64)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, groups=groups))
    return np.sort(train_idx.astype(np.int64)), np.sort(test_idx.astype(np.int64))


def global_count_from_train_kept(
    keep_train: np.ndarray,
    train_sorted: np.ndarray,
    test_pairs: np.ndarray,
    df_full: pd.DataFrame,
    audit: pd.DataFrame,
    profile: tpa.AuditProfile,
    seed: int,
) -> int:
    kg, _ = PR.fit_apply_retain_fraction(
        df_full, audit, train_sorted, test_pairs, keep_train, profile, seed
    )
    return int(kg.sum())


def predicted_global_retained(
    k_tr: int, m_train: int, n_test_pairs: int
) -> int:
    """Match `fit_apply_retain_fraction` test keep rule (incl. min 1 test pair when split nonempty)."""
    frac = k_tr / max(m_train, 1)
    k_te = int(round(frac * n_test_pairs))
    k_te = max(0, min(k_te, n_test_pairs))
    if n_test_pairs > 0 and k_te == 0:
        k_te = min(1, n_test_pairs)
    return int(k_tr + k_te)


def solve_train_keep_for_target_global(
    target_global: int,
    m_train: int,
    n_test_pairs: int,
    mk_train: int,
    n_full: int,
) -> Tuple[int, int, str]:
    """
    Choose k_tr in [mk_train, m_train] so predicted global G ≈ target_global
    (same rounding as fit_apply_retain_fraction).
    """
    best_kt, best_g, best_d = mk_train, -1, 10**9
    for kt in range(mk_train, m_train + 1):
        g = predicted_global_retained(kt, m_train, n_test_pairs)
        d = abs(g - target_global)
        if d < best_d or (d == best_d and abs(g - n_full) < abs(best_g - n_full)):
            best_d, best_kt, best_g = d, kt, g
    note = "exact" if best_g == target_global else f"nearest_G={best_g}_abs_diff={best_d}"
    return best_kt, best_g, note


def keep_mask_lowest_score_train_pairs(
    train_sorted: np.ndarray,
    st_scores_on_sorted: np.ndarray,
    k_keep_train: int,
) -> np.ndarray:
    """Retain k_keep_train train pairs with **lowest** separability scores (drop highest)."""
    m = len(train_sorted)
    assert st_scores_on_sorted.shape == (m,)
    order = np.argsort(st_scores_on_sorted)
    keep = np.zeros(m, dtype=bool)
    keep[order[:k_keep_train]] = True
    return keep


def evaluate_fixed_global_score_rank(
    audit: pd.DataFrame,
    df_full: pd.DataFrame,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
    min_keep_global: int,
    max_drop_fraction: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
    target_global_keep: int,
) -> PR.SearchState:
    """score_rank_quantile-style: fixed ~global kept count via train k_tr search."""
    train_sorted = np.sort(train_pairs)
    m = len(train_sorted)
    mk = PR._min_keep_train(min_keep_global, len(audit), m)
    n_te = len(test_pairs)
    k_tr, g_pred, _ = solve_train_keep_for_target_global(
        target_global_keep, m, n_te, mk, len(audit)
    )
    df_tr = PR.subset_df_ans(df_full, train_sorted)
    sc_all = PR.pair_separability_scores(df_tr, profile, seed)
    st_scores = sc_all[train_sorted]
    keep = keep_mask_lowest_score_train_pairs(train_sorted, st_scores, k_tr)
    return PR.evaluate_state(
        df_full,
        audit,
        train_pairs,
        test_pairs,
        keep,
        profile,
        seed,
        n_splits,
        w_auc,
        w_size,
        w_imb,
        target_auc,
    )


def evaluate_fixed_global_negation_rank(
    audit: pd.DataFrame,
    df_full: pd.DataFrame,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
    min_keep_global: int,
    max_drop_fraction: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
    target_global_keep: int,
) -> PR.SearchState:
    """Comparator: keep train pairs with lowest negation asymmetry."""
    train_sorted = np.sort(train_pairs)
    m = len(train_sorted)
    mk = PR._min_keep_train(min_keep_global, len(audit), m)
    n_te = len(test_pairs)
    k_tr, _, _ = solve_train_keep_for_target_global(
        target_global_keep, m, n_te, mk, len(audit)
    )
    _, nasym = PR.negation_pair_features(audit)
    asym = nasym[train_sorted].astype(np.float64)
    order = np.argsort(asym)
    keep = np.zeros(m, dtype=bool)
    keep[order[:k_tr]] = True
    return PR.evaluate_state(
        df_full,
        audit,
        train_pairs,
        test_pairs,
        keep,
        profile,
        seed,
        n_splits,
        w_auc,
        w_size,
        w_imb,
        target_auc,
    )


def state_to_metrics(
    st: PR.SearchState,
    kg: np.ndarray,
    audit: pd.DataFrame,
    method: str,
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    opt_gap = (
        float(st.search_auc - st.heldout_auc)
        if np.isfinite(st.search_auc) and np.isfinite(st.heldout_auc)
        else float("nan")
    )
    sub = audit.iloc[np.where(kg)[0]]
    n_any_neg, _ = PR.negation_pair_features(sub)
    row: Dict[str, Any] = {
        "seed": seed,
        "method": method,
        "search_time_auc": st.search_auc,
        "heldout_auc": st.heldout_auc,
        "optimism_gap": opt_gap,
        "kept_count": int(kg.sum()),
        "retained_confounded_fraction": float(sub["style_violation"].mean()) if len(sub) else float("nan"),
        "retained_negation_rate": float(np.mean(n_any_neg)) if len(sub) else float("nan"),
        "objective_score": st.objective,
    }
    if extra:
        row.update(extra)
    return row


def run_multi_seed(
    audit: pd.DataFrame,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    seeds: List[int],
    test_size: float,
    n_splits: int,
    min_keep: int,
    max_drop_fraction: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
    beam_width: int,
    n_multistarts: int,
    *,
    mode: str = "all_features",
) -> pd.DataFrame:
    n_pairs = len(audit)
    rows: List[Dict] = []
    for seed in seeds:
        train_pairs, test_pairs = split_train_test_pairs_gss(n_pairs, test_size, seed)
        runners: List[Tuple[str, Callable[[], PR.SearchState]]] = [
            (
                "negation_first_constrained",
                lambda s=seed, tp=train_pairs, tsp=test_pairs: PR.method_negation_first(
                    audit,
                    tp,
                    tsp,
                    df_full,
                    profile,
                    s,
                    n_splits,
                    min_keep,
                    max_drop_fraction,
                    target_auc,
                    w_auc,
                    w_size,
                    w_imb,
                    mode,
                ),
            ),
            (
                "feature_balanced",
                lambda s=seed, tp=train_pairs, tsp=test_pairs: PR.method_feature_balanced(
                    audit,
                    tp,
                    tsp,
                    df_full,
                    profile,
                    s,
                    n_splits,
                    min_keep,
                    max_drop_fraction,
                    target_auc,
                    w_auc,
                    w_size,
                    w_imb,
                ),
            ),
            (
                "score_based_greedy",
                lambda s=seed, tp=train_pairs, tsp=test_pairs: PR.method_backward_elimination(
                    audit,
                    tp,
                    tsp,
                    df_full,
                    profile,
                    s,
                    n_splits,
                    min_keep,
                    max_drop_fraction,
                    target_auc,
                    w_auc,
                    w_size,
                    w_imb,
                ),
            ),
            (
                "beam_or_multistart_greedy",
                lambda s=seed, tp=train_pairs, tsp=test_pairs: PR.method_beam_multistart(
                    audit,
                    tp,
                    tsp,
                    df_full,
                    profile,
                    s,
                    n_splits,
                    min_keep,
                    max_drop_fraction,
                    target_auc,
                    w_auc,
                    w_size,
                    w_imb,
                    beam_width,
                    n_multistarts,
                ),
            ),
        ]
        for name, fn in runners:
            st = fn()
            kg, _ = PR.fit_apply_retain_fraction(
                df_full, audit, train_pairs, test_pairs, st.keep_mask_train, profile, seed
            )
            rows.append(state_to_metrics(st, kg, audit, name, seed))
    return pd.DataFrame(rows)


def run_fixed_kept(
    audit: pd.DataFrame,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    seeds: List[int],
    test_size: float,
    n_splits: int,
    min_keep: int,
    max_drop_fraction: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
    targets: List[int],
) -> pd.DataFrame:
    rows: List[Dict] = []
    n_pairs = len(audit)
    for seed in seeds:
        train_pairs, test_pairs = split_train_test_pairs_gss(n_pairs, test_size, seed)
        train_sorted = np.sort(train_pairs)
        m = len(train_sorted)
        mk = PR._min_keep_train(min_keep, n_pairs, m)
        for K in targets:
            for method_name, evaluator in (
                (
                    "score_rank_fixed_global",
                    evaluate_fixed_global_score_rank,
                ),
                (
                    "negation_rank_fixed_global",
                    evaluate_fixed_global_negation_rank,
                ),
            ):
                st = evaluator(
                    audit,
                    df_full,
                    train_pairs,
                    test_pairs,
                    profile,
                    seed,
                    n_splits,
                    min_keep,
                    max_drop_fraction,
                    target_auc,
                    w_auc,
                    w_size,
                    w_imb,
                    K,
                )
                kg, _ = PR.fit_apply_retain_fraction(
                    df_full,
                    audit,
                    train_pairs,
                    test_pairs,
                    st.keep_mask_train,
                    profile,
                    seed,
                )
                n_te = len(test_pairs)
                k_tr = int(st.keep_mask_train.sum())
                _, g_pred, note = solve_train_keep_for_target_global(K, m, n_te, mk, n_pairs)
                opt_gap = (
                    float(st.search_auc - st.heldout_auc)
                    if np.isfinite(st.search_auc) and np.isfinite(st.heldout_auc)
                    else float("nan")
                )
                sub = audit.iloc[np.where(kg)[0]]
                n_any_neg, _ = PR.negation_pair_features(sub)
                rows.append(
                    {
                        "seed": seed,
                        "target_kept_count": K,
                        "actual_kept_count": int(kg.sum()),
                        "predicted_global_from_ktr": g_pred,
                        "k_train_kept": k_tr,
                        "sizing_note": note,
                        "method": method_name,
                        "search_time_auc": st.search_auc,
                        "heldout_auc": st.heldout_auc,
                        "optimism_gap": opt_gap,
                        "retained_confounded_fraction": float(sub["style_violation"].mean())
                        if len(sub)
                        else float("nan"),
                        "retained_negation_rate": float(np.mean(n_any_neg)) if len(sub) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_fixed_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    gcols = ["target_kept_count", "method"]
    agg = (
        per_seed.groupby(list(gcols), as_index=False)
        .agg(
            mean_actual_kept_count=("actual_kept_count", "mean"),
            mean_heldout_auc=("heldout_auc", "mean"),
            std_heldout_auc=("heldout_auc", "std"),
            mean_search_time_auc=("search_time_auc", "mean"),
            mean_optimism_gap=("optimism_gap", "mean"),
            mean_retained_confounded_fraction=("retained_confounded_fraction", "mean"),
            mean_retained_negation_rate=("retained_negation_rate", "mean"),
            num_seeds=("seed", "count"),
        )
    )
    return agg


def greedy_multi_seed_is_degenerate(multi: pd.DataFrame) -> bool:
    """True if, for every seed, all methods share the same held-out AUC and kept count."""
    for _, g in multi.groupby("seed", sort=False):
        if len(g) < 2:
            continue
        if g["heldout_auc"].nunique(dropna=False) > 1 or g["kept_count"].nunique() > 1:
            return False
    return len(multi) > 0


def method_stability_from_multi(multi: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for method, grp in multi.groupby("method"):
        parts.append(
            {
                "method": method,
                "mean_heldout_auc": grp["heldout_auc"].mean(),
                "std_heldout_auc": grp["heldout_auc"].std(),
                "median_heldout_auc": grp["heldout_auc"].median(),
                "mean_kept_count": grp["kept_count"].mean(),
                "std_kept_count": grp["kept_count"].std(),
                "num_seed_runs": len(grp),
            }
        )
    stab = pd.DataFrame(parts)
    methods_u = sorted(multi["method"].unique())
    win_counts: Dict[str, int] = {m: 0 for m in methods_u}
    atol = 1e-9
    for _, grp in multi.groupby("seed", sort=False):
        if grp.empty:
            continue
        valid = grp[np.isfinite(grp["heldout_auc"])]
        if valid.empty:
            continue
        min_h = float(valid["heldout_auc"].min())
        winners = valid.loc[
            np.isfinite(valid["heldout_auc"])
            & (valid["heldout_auc"] <= min_h + atol),
            "method",
        ].unique()
        for m in winners:
            win_counts[str(m)] += 1
    stab["wins_best_heldout_per_seed"] = stab["method"].map(win_counts)
    return stab.sort_values("mean_heldout_auc")


def write_fixed_md(summary: pd.DataFrame, path: Path) -> None:
    # One row per (target, best method by mean heldout) for paper simplicity
    lines = [
        "| kept count | method | mean held-out AUC | std | mean confounded fraction | mean negation rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for K in sorted(summary["target_kept_count"].unique(), reverse=True):
        sub = summary[summary["target_kept_count"] == K]
        if sub.empty:
            continue
        j = sub["mean_heldout_auc"].idxmin()
        r = sub.loc[j]
        lines.append(
            f"| {int(K)} | {r['method']} | {r['mean_heldout_auc']:.4f} | {r['std_heldout_auc']:.4f} | "
            f"{r['mean_retained_confounded_fraction']:.4f} | {r['mean_retained_negation_rate']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--audit-csv", type=str, default="audits/truthfulqa_style_audit.csv")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--min-keep", type=int, default=400)
    ap.add_argument("--max-drop-fraction", type=float, default=0.45)
    ap.add_argument("--target-audit-auc", type=float, default=0.62)
    ap.add_argument("--w-auc", type=float, default=2.0)
    ap.add_argument("--w-size", type=float, default=1.0)
    ap.add_argument("--w-imb", type=float, default=0.5)
    ap.add_argument("--beam-width", type=int, default=4)
    ap.add_argument("--n-multistarts", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = root / "results" / "truthfulqa_pruning_final_verification"
    fig_out = root / "figures" / "truthfulqa_pruning_final_verification"
    out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    profile: tpa.AuditProfile = "paper10"
    audit_path = (root / args.audit_csv).resolve()
    audit = pd.read_csv(audit_path)
    df_full = tpa.build_answer_level_audit_frame(audit, profile=profile, copy_audit_meta=True)
    n_pairs = len(audit)

    full_auc = PR.audit_auc_safe(df_full, profile, 42, args.cv_splits)

    seeds = [args.seed_start + i for i in range(args.n_seeds)]
    targets = [650, 595, 550, 500, 450, 400]

    (out / "prefix_grid_check.md").write_text(
        "# Pruning-improved verification note\n\n"
        "This **final verification** run evaluates only the **corrected default method family** "
        "(see `run_truthfulqa_pruning_improved.py` defaults). Optional `quantile_score` / quantile grids "
        "are **out of scope** here.\n",
        encoding="utf-8",
    )

    print(f"[final_verification] full_dataset OOF AUC ≈ {full_auc:.4f} (paper10, all pairs)")

    multi = run_multi_seed(
        audit,
        df_full,
        profile,
        seeds,
        args.test_size,
        args.cv_splits,
        args.min_keep,
        args.max_drop_fraction,
        args.target_audit_auc,
        args.w_auc,
        args.w_size,
        args.w_imb,
        args.beam_width,
        args.n_multistarts,
        mode="all_features",
    )
    multi.to_csv(out / "multi_seed_results.csv", index=False)

    stab = method_stability_from_multi(multi)
    stab.to_csv(out / "method_stability_summary.csv", index=False)

    fixed_ps = run_fixed_kept(
        audit,
        df_full,
        profile,
        seeds,
        args.test_size,
        args.cv_splits,
        args.min_keep,
        args.max_drop_fraction,
        args.target_audit_auc,
        args.w_auc,
        args.w_size,
        args.w_imb,
        targets,
    )
    fixed_ps.to_csv(out / "fixed_kept_count_per_seed.csv", index=False)
    fixed_sum = aggregate_fixed_summary(fixed_ps)
    fixed_sum.to_csv(out / "fixed_kept_count_summary.csv", index=False)
    write_fixed_md(fixed_sum, out / "fixed_kept_count_table.md")

    # Best method consistency narrative
    degenerate = greedy_multi_seed_is_degenerate(multi)
    best_mean_m = ""
    best_median_m = ""
    win_leader = ""
    if len(stab):
        mean_unc = stab["mean_heldout_auc"].nunique()
        best_mean_m = (
            "all methods (identical metrics)"
            if mean_unc <= 1
            else str(stab.loc[stab["mean_heldout_auc"].idxmin(), "method"])
        )
        med = multi.groupby("method")["heldout_auc"].median().sort_values()
        if len(med):
            med_unc = med.nunique()
            best_median_m = (
                "all methods (identical metrics)"
                if med_unc <= 1
                else str(med.index[0])
            )
        wmax = int(stab["wins_best_heldout_per_seed"].max())
        leaders = stab.loc[stab["wins_best_heldout_per_seed"] == wmax, "method"].astype(str).tolist()
        win_leader = ", ".join(sorted(leaders)) + (" (tied)" if len(leaders) > 1 else "")
    cons_lines = [
        "# Best-method consistency (multi-seed)",
        "",
        "_Representative **`mode=all_features`** for all four default methods (see `run_pruning_final_verification.py`)._",
        "",
    ]
    if degenerate:
        cons_lines.extend(
            [
                "> **Degenerate run:** For every seed, all four default methods produced the **same** "
                "`heldout_auc`, `search_time_auc`, `kept_count`, and objective (they did **not** reduce "
                "below full **N** under current hyperparameters). `wins_best_heldout_per_seed` therefore counts "
                "**ties** (each tied method gets +1 per seed).",
                "",
            ]
        )
    cons_lines.extend(
        [
            f"- **Lowest mean held-out AUC:** `{best_mean_m}`",
            f"- **Lowest median held-out AUC:** `{best_median_m}`",
            f"- **Most seed-wise wins** (among ties at min held-out): `{win_leader}`",
            "",
            "```",
            stab.to_string(index=False),
            "```",
            "",
            "_Where methods actually prune differently, compare `mean_heldout_auc` and `wins_*`; see fixed-kept sweep._",
        ]
    )
    (out / "best_method_consistency.md").write_text("\n".join(cons_lines), encoding="utf-8")

    # Figures
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, grp in fixed_sum.groupby("method"):
        sub = grp.sort_values("target_kept_count")
        ax.errorbar(
            sub["target_kept_count"],
            sub["mean_heldout_auc"],
            yerr=sub["std_heldout_auc"],
            marker="o",
            capsize=3,
            label=method,
        )
    ax.set_xlabel("Target kept count")
    ax.set_ylabel("Mean held-out AUC ± std")
    ax.legend(fontsize=8)
    ax.set_title("Fixed kept-count sweep (multi-seed)")
    fig.tight_layout()
    fig.savefig(fig_out / "fixed_kept_vs_mean_heldout.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    msub = method_stability_from_multi(multi).sort_values("mean_heldout_auc")
    x = np.arange(len(msub))
    ax.bar(x, msub["mean_heldout_auc"], yerr=msub["std_heldout_auc"], capsize=3, color="#4477AA")
    ax.set_xticks(x)
    ax.set_xticklabels(msub["method"], rotation=25, ha="right")
    ax.set_ylabel("Mean held-out AUC")
    ax.set_title("Method comparison (multi-seed)")
    fig.tight_layout()
    fig.savefig(fig_out / "method_comparison_mean_heldout.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    mm = multi.groupby("method")["optimism_gap"].agg(["mean", "std"]).reset_index()
    mm = mm.sort_values("mean")
    x = np.arange(len(mm))
    ax.bar(x, mm["mean"], yerr=mm["std"], capsize=3, color="#CC6677")
    ax.set_xticks(x)
    ax.set_xticklabels(mm["method"], rotation=25, ha="right")
    ax.set_ylabel("Mean optimism gap")
    fig.tight_layout()
    fig.savefig(fig_out / "mean_optimism_gap_by_method.pdf")
    plt.close(fig)

    # paper_summary_table: pick better of two methods per target by mean heldout
    paper_rows = []
    full_cnt = n_pairs
    for K in targets:
        sub = fixed_sum[fixed_sum["target_kept_count"] == K]
        if sub.empty:
            continue
        j = sub["mean_heldout_auc"].idxmin()
        r = sub.loc[j]
        paper_rows.append(
            {
                "full_dataset_count": full_cnt,
                "full_dataset_auc": full_auc,
                "target_kept_count": int(K),
                "mean_actual_kept_count": r["mean_actual_kept_count"],
                "best_method": r["method"],
                "mean_heldout_auc": r["mean_heldout_auc"],
                "std_heldout_auc": r["std_heldout_auc"],
                "mean_retained_confounded_fraction": r["mean_retained_confounded_fraction"],
                "mean_retained_negation_rate": r["mean_retained_negation_rate"],
            }
        )
    paper_df = pd.DataFrame(paper_rows)
    best_mean_row = stab.loc[stab["mean_heldout_auc"].idxmin()]
    mrow = best_mean_row
    multi_best_method = (
        "all_four_defaults_identical (no pruning below full N)"
        if degenerate
        else str(mrow["method"])
    )
    paper_df2 = pd.concat(
        [
            paper_df,
            pd.DataFrame(
                [
                    {
                        "full_dataset_count": full_cnt,
                        "full_dataset_auc": full_auc,
                        "target_kept_count": "multi_seed_four_defaults_mode_all_features",
                        "mean_actual_kept_count": mrow["mean_kept_count"],
                        "best_method": multi_best_method,
                        "mean_heldout_auc": mrow["mean_heldout_auc"],
                        "std_heldout_auc": mrow["std_heldout_auc"],
                        "mean_retained_confounded_fraction": float("nan"),
                        "mean_retained_negation_rate": float("nan"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    paper_df2.to_csv(out / "paper_summary_table.csv", index=False)

    # final_verification_report.md (held-out focused; corrected default methods only)
    leader = win_leader
    br = best_mean_row
    held_min = float(fixed_sum["mean_heldout_auc"].min())
    held_min_rows = fixed_sum[fixed_sum["mean_heldout_auc"] == held_min]
    best_k_fixed = (
        int(held_min_rows.iloc[0]["target_kept_count"]) if len(held_min_rows) else None
    )
    m595 = fixed_sum[
        (fixed_sum["target_kept_count"] == 595)
        & (fixed_sum["method"] == "score_rank_fixed_global")
    ]
    s595 = (
        f"{m595['mean_heldout_auc'].iloc[0]:.4f} ± {m595['std_heldout_auc'].iloc[0]:.4f}"
        if len(m595)
        else "n/a"
    )
    rep = [
        "# Final verification report (corrected pruning-improved family)",
        "",
        f"- **Multi-seed methods:** `negation_first_constrained`, `feature_balanced`, "
        f"`score_based_greedy`, `beam_or_multistart_greedy` — representative mode **`all_features`**.",
        f"- **Seeds:** {args.n_seeds}, **GroupShuffleSplit** test_size={args.test_size}, **paper10**.",
        "",
    ]
    if degenerate:
        rep.extend(
            [
                "> **Greedy multi-seed path:** All four defaults returned **identical** subsets each seed "
                f"(**kept_count = {int(mrow['mean_kept_count'])}** = full **N**). They **do not distinguish** "
                "here; stability of *held-out AUC across seeds* still reflects split noise (~**{:.3f}** std).".format(
                    float(br["std_heldout_auc"])
                ),
                "",
                "> **Where methods differ:** Use the **fixed kept-count** sweep (`score_rank_fixed_global` vs "
                "`negation_rank_fixed_global`).",
                "",
            ]
        )
    rep.extend(
        [
            "## 1. Stability across seeds/splits?",
            f"- Compare per-method **std** of `heldout_auc` in `method_stability_summary.csv` "
            f"(typical scale **~0.03–0.05** on a **~25%** held-out fold).",
            f"- **Reported mean held-out (any default row):** **{float(br['mean_heldout_auc']):.4f}** "
            f"(± **{float(br['std_heldout_auc']):.4f}**), mean kept **{float(br['mean_kept_count']):.1f}**.",
            "",
            "## 2. Best method among the four defaults?",
            f"- **Achieved min held-out per seed (ties count):** `{leader}` — see `wins_best_heldout_per_seed`.",
        ]
    )
    if degenerate:
        rep.extend(
            [
                "- **Interpretation:** Metrics match across methods each seed because **no default greedy run pruned** "
                f"below full **N** (**{int(mrow['mean_kept_count'])}**); **feature_balanced does not win**—all are tied.",
                "",
            ]
        )
    else:
        rep.extend(
            [
                f"- **Lowest mean held-out:** **`{br['method']}`** (see `method_stability_summary.csv`).",
                "",
            ]
        )
    rep.extend(
        [
        "## 3. Held-out AUC vs fixed kept count?",
        "- See **`fixed_kept_count_table.md`** and **`fixed_kept_count_summary.csv`** "
        "(best of `score_rank_fixed_global` vs `negation_rank_fixed_global` per target).",
        "",
        "## 4. Is 595 a strong operating point?",
        f"- **595** (score_rank, mean ± std): **{s595}** — compare to **500/450** in the same table.",
        "",
        "## 5. Better tradeoff at 650, 550, 500, 450, 400?",
        f"- **Lowest mean held-out** in this fixed grid (among reported rows): "
        f"**{held_min:.4f}** at target **{best_k_fixed}** (see CSV for method).",
        "",
        "## 6. Strong enough for the paper?",
        "- **Yes if** you report **held-out mean ± std over seeds** and contrast to full-set reference "
        f"**{full_auc:.4f}**; **no** if you need a much lower AUC without accepting smaller **N** or different splits.",
        "",
        "## 7. Numbers to cite",
        f"- **Full dataset (OOF):** **{full_auc:.4f}**, **N = {full_cnt}**.",
        f"- **Greedy defaults (multi-seed):** mean held-out **{float(br['mean_heldout_auc']):.4f}** "
        f"± **{float(br['std_heldout_auc']):.4f}**; **`{multi_best_method}`** in `paper_summary_table.csv`.",
        "- **Fixed retained sizes:** cite the row for your **target_kept_count** in `paper_summary_table.csv` (held-out mean ± std).",
        "",
        "## 8. Caveats",
        "- **Search-time AUC** is secondary; lead with **held-out AUC**.",
        "- Default multi-seed path uses **`all_features`** only; full `modes=all` in the main script can differ.",
        "- Fixed-kept rows may show **nearest feasible** global **N**; use **`actual_kept_count`** in per-seed CSV.",
        "",
        ]
    )
    (out / "final_verification_report.md").write_text("\n".join(rep), encoding="utf-8")

    (out / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote outputs under {out}")


if __name__ == "__main__":
    main()
