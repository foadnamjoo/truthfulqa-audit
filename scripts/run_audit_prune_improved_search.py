#!/usr/bin/env python3
"""
Experimental stronger search for TruthfulQA subset selection under ACCURACY thresholds.

Keeps existing baseline/current methods untouched. Writes outputs to:
  - results/audit_prune_improved_search/
  - truthfulqaAuditPruneImproved/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from audit_subset_evaluator import (
    FEATURE_COLS,
    SubsetAuditDetailed,
    SubsetAuditMetrics,
    df_pairs_from_ids,
    evaluate_subset_grouped_cv,
    evaluate_subset_grouped_cv_detailed,
)
from search_truthfulqa_pruned_improved import _apply_prefix_keep
from truthfulqa_pruning_utils import CV_SPLITS, load_candidates_with_features

StrategyName = Literal["confidence", "imbalance", "hybrid"]
BASE_COLS = ["Type", "Category", "Question", "Best Answer", "Best Incorrect Answer"]
EPS = 1e-9


@dataclass
class Candidate:
    pair_ids: Set[int]
    metrics: SubsetAuditMetrics
    method: str
    strategy: str
    threshold_type: str
    threshold_value: float
    seed: int
    metadata: Dict[str, Any]


def sorted_df_feature_balanced_baseline(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    d = df.copy()
    d["len_bin"] = pd.qcut(d["length_gap"], q=4, labels=False, duplicates="drop")
    parts = []
    for b in sorted(d["len_bin"].dropna().unique()):
        sub = d[d["len_bin"] == b]
        parts.append(sub.sample(frac=1.0, random_state=seed + int(b)))
    mixed = pd.concat(parts, ignore_index=True)
    return mixed.sort_values(["negation_flag", "length_gap", "example_id"], ascending=[True, True, True]).reset_index(
        drop=True
    )


def _pair_scores_conf_from_ans(ans: pd.DataFrame) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for pid in ans["pair_id"].unique():
        sub = ans[ans["pair_id"] == int(pid)]
        p_pos = float(sub[sub["label"] == 1]["proba"].iloc[0])
        p_neg = float(sub[sub["label"] == 0]["proba"].iloc[0])
        out[int(pid)] = abs(p_pos - p_neg)
    return out


def _pair_scores_imbalance_from_ans(ans: pd.DataFrame) -> Dict[int, float]:
    X = ans[FEATURE_COLS].to_numpy()
    y = ans["label"].to_numpy()
    mu_pos = X[y == 1].mean(axis=0)
    mu_neg = X[y == 0].mean(axis=0)
    gap = mu_pos - mu_neg
    out: Dict[int, float] = {}
    for pid in ans["pair_id"].unique():
        sub = ans[ans["pair_id"] == int(pid)]
        xt = sub[sub["label"] == 1][FEATURE_COLS].to_numpy().ravel()
        xf = sub[sub["label"] == 0][FEATURE_COLS].to_numpy().ravel()
        v = xt - xf
        s = 0.0
        for j in range(len(FEATURE_COLS)):
            if gap[j] == 0.0:
                continue
            if v[j] * gap[j] > 0:
                s += abs(float(v[j]))
        out[int(pid)] = s
    return out


def _normalize_scores(s: Dict[int, float]) -> Dict[int, float]:
    if not s:
        return {}
    vals = np.array(list(s.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo + 1e-15:
        return {k: 0.0 for k in s}
    return {k: (float(v) - lo) / (hi - lo) for k, v in s.items()}


def build_pair_scores(detailed: SubsetAuditDetailed, strategy: StrategyName, alpha: float) -> Dict[int, float]:
    ans = detailed.answer_frame_with_oof
    conf = _pair_scores_conf_from_ans(ans)
    if strategy == "confidence":
        return conf
    imb = _pair_scores_imbalance_from_ans(ans)
    if strategy == "imbalance":
        return imb
    conf_n = _normalize_scores(conf)
    imb_n = _normalize_scores(imb)
    all_ids = set(conf_n) | set(imb_n)
    return {pid: alpha * conf_n.get(pid, 0.0) + (1.0 - alpha) * imb_n.get(pid, 0.0) for pid in all_ids}


def pick_remove_id(scores: Dict[int, float], retained: Set[int], rng: random.Random, top_k: int) -> int:
    ranked = sorted(((pid, float(scores.get(pid, 0.0))) for pid in retained), key=lambda x: (-x[1], x[0]))
    if not ranked:
        raise RuntimeError("No retained pairs available for removal.")
    k = max(1, min(top_k, len(ranked)))
    return int(rng.choice(ranked[:k])[0])


def remove_trajectory(
    full_df: pd.DataFrame,
    strategy: StrategyName,
    alpha: float,
    seed: int,
    min_acc_target: float,
    top_k_remove: int,
    beam_kick_ids: Optional[Iterable[int]] = None,
) -> List[Tuple[Set[int], SubsetAuditMetrics]]:
    """Generate one pruning trajectory (all thresholds will reuse it)."""
    rng = random.Random(seed)
    retained: Set[int] = set(int(x) for x in full_df["example_id"].tolist())
    if beam_kick_ids:
        for pid in beam_kick_ids:
            retained.discard(int(pid))
    traj: List[Tuple[Set[int], SubsetAuditMetrics]] = []
    while len(retained) >= CV_SPLITS:
        detailed = evaluate_subset_grouped_cv_detailed(df_pairs_from_ids(full_df, retained), seed=seed)
        m = detailed.metrics
        traj.append((set(retained), m))
        if m.accuracy <= min_acc_target + EPS:
            break
        scores = build_pair_scores(detailed, strategy, alpha)
        pid = pick_remove_id(scores, retained, rng, top_k_remove)
        retained.remove(pid)
    return traj


def first_feasible_from_trajectory(
    trajectory: List[Tuple[Set[int], SubsetAuditMetrics]],
    tau_acc: float,
) -> Optional[Tuple[Set[int], SubsetAuditMetrics]]:
    # trajectory is in descending size order
    for ids, m in trajectory:
        if m.accuracy <= tau_acc + EPS:
            return set(ids), m
    return None


def add_back_refine_accuracy(
    full_df: pd.DataFrame,
    retained: Set[int],
    tau_acc: float,
    seed: int,
    max_passes: int = 3,
) -> Set[int]:
    all_ids = set(int(x) for x in full_df["example_id"].tolist())
    ret = set(retained)
    for _ in range(max_passes):
        changed = False
        removed = sorted(all_ids - ret)
        for pid in removed:
            trial = set(ret)
            trial.add(pid)
            m = evaluate_subset_grouped_cv(df_pairs_from_ids(full_df, trial), seed=seed)
            if np.isfinite(m.accuracy) and m.accuracy <= tau_acc + EPS:
                ret.add(pid)
                changed = True
        if not changed:
            break
    return ret


def swap_refine_accuracy(
    full_df: pd.DataFrame,
    retained: Set[int],
    tau_acc: float,
    strategy: StrategyName,
    alpha: float,
    seed: int,
    max_rounds: int = 2,
    removed_probe: int = 24,
    retained_probe: int = 24,
) -> Set[int]:
    """Try 1-for-1 swaps that reduce AUC while keeping accuracy feasible."""
    all_ids = set(int(x) for x in full_df["example_id"].tolist())
    ret = set(retained)
    best_m = evaluate_subset_grouped_cv(df_pairs_from_ids(full_df, ret), seed=seed)
    for _ in range(max_rounds):
        improved = False
        rem = sorted(all_ids - ret)
        if not rem:
            break
        # candidate add list: lower score first (less harmful)
        det_ret = evaluate_subset_grouped_cv_detailed(df_pairs_from_ids(full_df, ret), seed=seed)
        scores_ret = build_pair_scores(det_ret, strategy, alpha)
        rem_sorted = rem[:removed_probe]
        kept_ranked = sorted(ret, key=lambda pid: (-scores_ret.get(pid, 0.0), pid))[:retained_probe]
        for r in rem_sorted:
            for q in kept_ranked:
                trial = set(ret)
                trial.remove(q)
                trial.add(r)
                m = evaluate_subset_grouped_cv(df_pairs_from_ids(full_df, trial), seed=seed)
                if not np.isfinite(m.accuracy):
                    continue
                if m.accuracy <= tau_acc + EPS and m.auc + 1e-12 < best_m.auc:
                    ret = trial
                    best_m = m
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return ret


def baseline_max_prefix_under_acc(d_sorted: pd.DataFrame, tau_acc: float, seed: int) -> Tuple[Optional[int], SubsetAuditMetrics]:
    n_full = len(d_sorted)
    best_k: Optional[int] = None
    best_m = SubsetAuditMetrics(auc=float("nan"), accuracy=float("nan"), n_pairs=0, n_answer_rows=0)
    for k in range(CV_SPLITS, n_full + 1):
        sub = _apply_prefix_keep(d_sorted, k)
        m = evaluate_subset_grouped_cv(sub, seed=seed)
        if m.accuracy <= tau_acc + EPS:
            best_k = k
            best_m = m
    return best_k, best_m


def load_current_audit_prune_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def best_current_under_acc(current_df: pd.DataFrame, tau_acc: float) -> Optional[pd.Series]:
    if current_df.empty:
        return None
    feasible = current_df[current_df["pruned_accuracy"] <= tau_acc + EPS]
    if feasible.empty:
        return None
    return feasible.sort_values(["pruned_pair_count", "pruned_auc"], ascending=[False, True]).iloc[0]


def _slug_tau_acc(tau: float) -> str:
    return f"acc{int(round(tau * 1000)):04d}"


def _strategy_slug(strategy: StrategyName, alpha: float) -> str:
    if strategy != "hybrid":
        return strategy
    return f"hybrid_a{int(round(alpha * 100)):02d}"


def write_subset_outputs(
    root: Path,
    subset_name: str,
    pair_ids: List[int],
    df_t: pd.DataFrame,
    style: np.ndarray,
    meta: Dict[str, Any],
) -> Tuple[str, str]:
    out_dir = root / "truthfulqaAuditPruneImproved"
    json_dir = out_dir / "pair_ids"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"{subset_name}.csv"
    json_name = f"{subset_name}.json"
    cpath = out_dir / csv_name
    jpath = json_dir / json_name
    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", *BASE_COLS, "style_violation", "subset_name"])
        w.writeheader()
        for pid in pair_ids:
            row = df_t.iloc[pid]
            w.writerow(
                {
                    "pair_id": pid,
                    **{c: row[c] for c in BASE_COLS},
                    "style_violation": int(style[pid]),
                    "subset_name": subset_name,
                }
            )
    jpath.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return f"truthfulqaAuditPruneImproved/{csv_name}", f"truthfulqaAuditPruneImproved/pair_ids/{json_name}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truthfulqa-csv", type=str, default="TruthfulQA.csv")
    p.add_argument("--audit-csv", type=str, default="audits/truthfulqa_style_audit.csv")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-prefix-seed", type=int, default=42)
    p.add_argument("--thresholds-acc", type=float, nargs="+", default=[0.60, 0.57, 0.55, 0.53])
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--top-k-remove", type=int, default=5)
    p.add_argument("--hybrid-alphas", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    p.add_argument("--enable-swap", action="store_true")
    p.add_argument("--beam-width", type=int, default=2)
    p.add_argument("--beam-depth", type=int, default=1)
    p.add_argument(
        "--results-subdir",
        type=str,
        default="audit_prune_improved_search",
        help="Subdirectory under results/ for summary CSVs and config.json",
    )
    p.add_argument(
        "--current-summary-path",
        type=str,
        default="results/audit_prune_thresholded/summary_table.csv",
        help="Path relative to --output-root for audit_prune thresholded summary (current_audit_prune rows)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).resolve()
    tq_path = (root / args.truthfulqa_csv).resolve()
    audit_path = (root / args.audit_csv).resolve()
    results_dir = root / "results" / args.results_subdir
    out_dir = root / "truthfulqaAuditPruneImproved"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_candidates_with_features(tq_path, audit_path)
    df_t = pd.read_csv(tq_path)
    audit = pd.read_csv(audit_path)
    style = audit["style_violation"].to_numpy()
    all_ids = set(int(x) for x in full_df["example_id"].tolist())
    n_full = len(full_df)

    d_sorted = sorted_df_feature_balanced_baseline(full_df, args.baseline_prefix_seed)
    current_summary_path = (root / args.current_summary_path).resolve()
    current_df = load_current_audit_prune_rows(current_summary_path)

    strategies: List[Tuple[StrategyName, float]] = [("confidence", 0.0), ("imbalance", 0.0)]
    for a in args.hybrid_alphas:
        strategies.append(("hybrid", float(a)))

    config = {
        "truthfulqa_csv": args.truthfulqa_csv,
        "audit_csv": args.audit_csv,
        "seed": args.seed,
        "baseline_prefix_seed": args.baseline_prefix_seed,
        "thresholds_acc": args.thresholds_acc,
        "restarts": args.restarts,
        "top_k_remove": args.top_k_remove,
        "hybrid_alphas": args.hybrid_alphas,
        "enable_swap": bool(args.enable_swap),
        "beam_width": args.beam_width,
        "beam_depth": args.beam_depth,
        "evaluator": "grouped CV accuracy primary; AUC secondary",
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    min_tau = min(args.thresholds_acc)
    experiment_rows: List[Dict[str, Any]] = []
    candidates_by_tau: Dict[float, List[Candidate]] = {float(t): [] for t in args.thresholds_acc}

    # Optional beam kickoff from full set (single scored expansion)
    beam_kicks: List[List[int]] = [[]]
    if args.beam_depth > 0 and args.beam_width > 1:
        det0 = evaluate_subset_grouped_cv_detailed(full_df, seed=args.seed)
        s0 = build_pair_scores(det0, "imbalance", alpha=0.0)
        ranked = sorted(s0.items(), key=lambda x: (-x[1], x[0]))
        beam_kicks = [[int(pid)] for pid, _ in ranked[: args.beam_width]]
        beam_kicks.insert(0, [])

    for strategy, alpha in strategies:
        sslug = _strategy_slug(strategy, alpha)
        for restart in range(args.restarts):
            rs = int(args.seed + 1000 * restart + 97 * (len(sslug) + restart))
            kick = beam_kicks[restart % len(beam_kicks)] if beam_kicks else []
            print(f"\n[trajectory] strategy={sslug} restart={restart} seed={rs} kick={kick}")
            traj = remove_trajectory(
                full_df=full_df,
                strategy=strategy,
                alpha=alpha,
                seed=rs,
                min_acc_target=min_tau,
                top_k_remove=args.top_k_remove,
                beam_kick_ids=kick,
            )
            for tau in args.thresholds_acc:
                feas = first_feasible_from_trajectory(traj, tau_acc=float(tau))
                if feas is None:
                    continue
                ids, m0 = feas
                ids_add = add_back_refine_accuracy(full_df, ids, tau_acc=float(tau), seed=rs, max_passes=3)
                if args.enable_swap:
                    ids_add = swap_refine_accuracy(
                        full_df=full_df,
                        retained=ids_add,
                        tau_acc=float(tau),
                        strategy=strategy,
                        alpha=alpha,
                        seed=rs,
                    )
                    ids_add = add_back_refine_accuracy(full_df, ids_add, tau_acc=float(tau), seed=rs, max_passes=2)
                m = evaluate_subset_grouped_cv(df_pairs_from_ids(full_df, ids_add), seed=rs)
                cand = Candidate(
                    pair_ids=set(ids_add),
                    metrics=m,
                    method="improved_search",
                    strategy=sslug,
                    threshold_type="accuracy",
                    threshold_value=float(tau),
                    seed=rs,
                    metadata={"restart": restart, "beam_kick": kick},
                )
                candidates_by_tau[float(tau)].append(cand)
                experiment_rows.append(
                    {
                        "threshold_type": "accuracy",
                        "threshold_value": float(tau),
                        "method": "improved_search",
                        "strategy": sslug,
                        "seed": rs,
                        "restart": restart,
                        "retained_pairs": len(ids_add),
                        "retained_fraction": len(ids_add) / n_full,
                        "grouped_cv_accuracy": m.accuracy,
                        "grouped_cv_auc": m.auc,
                    }
                )

    # Save exploratory rows
    exp_path = results_dir / "exploratory_runs.csv"
    if experiment_rows:
        pd.DataFrame(experiment_rows).to_csv(exp_path, index=False)

    summary_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    best_print_rows: List[Dict[str, Any]] = []

    for tau in args.thresholds_acc:
        tau = float(tau)
        bk, bm = baseline_max_prefix_under_acc(d_sorted, tau_acc=tau, seed=args.seed)
        baseline_pair_ids = set(d_sorted.iloc[:bk]["example_id"].astype(int).tolist()) if bk is not None else set()
        bdom = False
        # baseline row
        summary_rows.append(
            {
                "threshold_type": "accuracy",
                "threshold_value": tau,
                "method": "fixed_prefix_baseline",
                "strategy": "feature_balanced_prefix",
                "retained_pairs": int(bk) if bk is not None else 0,
                "retained_fraction": (int(bk) / n_full) if bk is not None else float("nan"),
                "grouped_cv_accuracy": bm.accuracy,
                "grouped_cv_auc": bm.auc,
                "dominates_current_best": bdom,
                "dominates_fixed_prefix": False,
            }
        )

        # current method best feasible under acc cap
        curr = best_current_under_acc(current_df, tau)
        curr_pairs = 0
        curr_acc = float("nan")
        curr_auc = float("nan")
        curr_strategy = "none"
        if curr is not None:
            curr_pairs = int(curr["pruned_pair_count"])
            curr_acc = float(curr["pruned_accuracy"])
            curr_auc = float(curr["pruned_auc"])
            curr_strategy = str(curr["strategy"])
        summary_rows.append(
            {
                "threshold_type": "accuracy",
                "threshold_value": tau,
                "method": "current_audit_prune",
                "strategy": curr_strategy,
                "retained_pairs": curr_pairs,
                "retained_fraction": (curr_pairs / n_full) if curr_pairs else float("nan"),
                "grouped_cv_accuracy": curr_acc,
                "grouped_cv_auc": curr_auc,
                "dominates_current_best": False,
                "dominates_fixed_prefix": curr_pairs > (bk or 0) and np.isfinite(curr_acc) and curr_acc <= tau + EPS,
            }
        )

        # best improved candidate for this tau
        cands = candidates_by_tau.get(tau, [])
        if cands:
            best = sorted(
                cands,
                key=lambda c: (-len(c.pair_ids), abs(c.metrics.accuracy - tau), c.metrics.auc),
            )[0]
            dominates_current_best = bool(curr_pairs and len(best.pair_ids) > curr_pairs and best.metrics.accuracy <= tau + EPS)
            dominates_fixed = bool(bk is not None and len(best.pair_ids) > bk and best.metrics.accuracy <= tau + EPS)
            summary_rows.append(
                {
                    "threshold_type": "accuracy",
                    "threshold_value": tau,
                    "method": "improved_search_best",
                    "strategy": best.strategy,
                    "retained_pairs": len(best.pair_ids),
                    "retained_fraction": len(best.pair_ids) / n_full,
                    "grouped_cv_accuracy": best.metrics.accuracy,
                    "grouped_cv_auc": best.metrics.auc,
                    "dominates_current_best": dominates_current_best,
                    "dominates_fixed_prefix": dominates_fixed,
                }
            )
            subset_name = f"truthfulqaAuditPruneImproved_{_slug_tau_acc(tau)}_{best.strategy}_seed{best.seed}"
            pair_ids_sorted = sorted(best.pair_ids)
            meta = {
                "selection_method": "audit_prune_improved_search_accuracy_threshold",
                "threshold_type": "accuracy",
                "threshold_value": tau,
                "strategy": best.strategy,
                "seed": best.seed,
                "n_pairs": len(pair_ids_sorted),
                "grouped_cv_oof_accuracy": best.metrics.accuracy,
                "grouped_cv_oof_auc": best.metrics.auc,
                "dominates_current_best": dominates_current_best,
                "dominates_fixed_prefix": dominates_fixed,
                "pair_ids": pair_ids_sorted,
                "metadata": best.metadata,
            }
            csv_rel, json_rel = write_subset_outputs(
                root=root,
                subset_name=subset_name,
                pair_ids=pair_ids_sorted,
                df_t=df_t,
                style=style,
                meta=meta,
            )
            manifest_rows.append(
                {
                    "subset_name": subset_name,
                    "threshold_type": "accuracy",
                    "threshold_value": tau,
                    "strategy": best.strategy,
                    "csv_path": csv_rel,
                    "canonical_json": json_rel,
                    "retained_pairs": len(pair_ids_sorted),
                    "retained_fraction": len(pair_ids_sorted) / n_full,
                    "grouped_cv_accuracy": best.metrics.accuracy,
                    "grouped_cv_auc": best.metrics.auc,
                    "dominates_current_best": dominates_current_best,
                    "dominates_fixed_prefix": dominates_fixed,
                    "paper_role": "audit_prune_improved_search_primary",
                }
            )
            best_print_rows.append(
                {
                    "threshold": tau,
                    "strategy": best.strategy,
                    "seed": best.seed,
                    "retained_pairs": len(pair_ids_sorted),
                    "acc": best.metrics.accuracy,
                    "auc": best.metrics.auc,
                }
            )

    summ_df = pd.DataFrame(summary_rows)
    summ_df.to_csv(results_dir / "summary_table.csv", index=False)
    cmp_df = summ_df[
        [
            "threshold_type",
            "threshold_value",
            "method",
            "strategy",
            "retained_pairs",
            "retained_fraction",
            "grouped_cv_accuracy",
            "grouped_cv_auc",
            "dominates_current_best",
            "dominates_fixed_prefix",
        ]
    ].copy()
    cmp_df.to_csv(results_dir / "comparison_table.csv", index=False)

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(out_dir / "subset_manifest.csv", index=False)

    # Strongest overall candidate near tau=0.55
    near55 = [r for r in best_print_rows if abs(float(r["threshold"]) - 0.55) < 1e-9]
    strongest = near55[0] if near55 else (best_print_rows[0] if best_print_rows else None)

    lines = [
        "# Improved Search Report",
        "",
        "Primary objective: maximize retained pairs subject to grouped CV ACCURACY threshold.",
        "Secondary metric: grouped CV AUC.",
        "",
        "## Best subset by threshold",
    ]
    for r in best_print_rows:
        lines.append(
            f"- acc <= {r['threshold']:.2f}: strategy={r['strategy']} seed={r['seed']} "
            f"retained={r['retained_pairs']} acc={r['acc']:.4f} auc={r['auc']:.4f}"
        )
    lines.append("")
    if strongest is not None:
        lines.append("## Single strongest overall candidate")
        lines.append(
            f"- strategy={strongest['strategy']} threshold={strongest['threshold']:.2f} "
            f"retained={strongest['retained_pairs']} acc={strongest['acc']:.4f} auc={strongest['auc']:.4f}"
        )
        lines.append("")
    # Compare against old best near 0.55
    old_best_pairs = 568
    old_best_acc = 0.5528
    old_best_auc = 0.5493
    if strongest is not None:
        beat = strongest["retained_pairs"] > old_best_pairs and strongest["acc"] <= old_best_acc + 1e-6
        lines.append("## Did we beat prior near-0.55 best?")
        lines.append(
            f"- Prior: retained={old_best_pairs}, acc~{old_best_acc:.4f}, auc~{old_best_auc:.4f}; "
            f"New strongest retained={strongest['retained_pairs']}, acc={strongest['acc']:.4f}, auc={strongest['auc']:.4f}. "
            f"Beat prior: {beat}"
        )
    lines.append("")
    lines.append("## Accuracy-vs-AUC optimization impact")
    lines.append(
        "- Compared with AUC-thresholded current method, this run prioritizes accuracy constraints first; "
        "see comparison_table.csv for per-threshold retained-size tradeoffs."
    )
    lines.append("")
    lines.append("## 600-700 pairs near 55% classifier performance")
    lines.append(
        "- Check acc<=0.55 row in summary_table.csv; this is the direct test for whether retained count approaches 600-700."
    )
    (results_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nBest subsets by threshold (accuracy caps):")
    for r in best_print_rows:
        print(
            f"  tau_acc={r['threshold']:.2f} strategy={r['strategy']} seed={r['seed']} "
            f"retained={r['retained_pairs']} acc={r['acc']:.4f} auc={r['auc']:.4f}"
        )
    if strongest is not None:
        print(
            f"\nSingle strongest candidate: tau_acc={strongest['threshold']:.2f}, "
            f"strategy={strongest['strategy']}, retained={strongest['retained_pairs']}, "
            f"acc={strongest['acc']:.4f}, auc={strongest['auc']:.4f}"
        )

    print(f"\nWrote outputs under {results_dir} and {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

