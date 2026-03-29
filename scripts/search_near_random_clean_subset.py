#!/usr/bin/env python3
"""
Search for a large TruthfulQA pair subset (target ~300–500 pairs) whose held-out audit
AUC is close to chance (~0.5) under the paper-compatible pruning protocol.

Uses the same train/test pair split + global retain rule as run_truthfulqa_pruning_improved.py
(via evaluate_state / fit_apply_retain_fraction).

Outputs a CSV table and prints a Markdown table.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import truthfulqa_paper_audit as tpa


def _load_pruning():
    path = _SCRIPT_DIR / "run_truthfulqa_pruning_improved.py"
    name = "tq_prune_improved"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PR = _load_pruning()

# Reuse verification helpers (same protocol)
import run_pruning_final_verification as vrf  # noqa: E402


def evaluate_fixed_len_gap_rank(
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
    """Keep train pairs with smallest length gap (most symmetric lengths)."""
    train_sorted = np.sort(train_pairs)
    m = len(train_sorted)
    mk = PR._min_keep_train(min_keep_global, len(audit), m)
    n_te = len(test_pairs)
    k_tr, _, _ = vrf.solve_train_keep_for_target_global(
        target_global_keep, m, n_te, mk, len(audit)
    )
    lg = audit["len_gap"].to_numpy(dtype=np.float64)[train_sorted]
    order = np.argsort(lg)
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


def evaluate_fixed_confound_free_train(
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
    """
    Prefer clean pairs (style_violation==0) on train, then lowest separability among the rest.
    """
    train_sorted = np.sort(train_pairs)
    m = len(train_sorted)
    mk = PR._min_keep_train(min_keep_global, len(audit), m)
    n_te = len(test_pairs)
    k_tr, _, _ = vrf.solve_train_keep_for_target_global(
        target_global_keep, m, n_te, mk, len(audit)
    )
    clean = audit["style_violation"].to_numpy() == 0
    df_tr = PR.subset_df_ans(df_full, train_sorted)
    sc_all = PR.pair_separability_scores(df_tr, profile, seed)
    st = sc_all[train_sorted]
    idx_clean = [i for i in range(m) if clean[int(train_sorted[i])]]
    idx_rest = [i for i in range(m) if not clean[int(train_sorted[i])]]
    idx_clean.sort(key=lambda i: float(st[i]))
    idx_rest.sort(key=lambda i: float(st[i]))
    order = idx_clean + idx_rest
    keep = np.zeros(m, dtype=bool)
    for i in order[:k_tr]:
        keep[i] = True
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


def make_random_fixed_eval(rng_base: int) -> Callable[..., PR.SearchState]:
    """Uniform random train subset of size k_tr (rng fixed per seed+k for reproducibility)."""

    def evaluate_fixed_random(
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
        train_sorted = np.sort(train_pairs)
        m = len(train_sorted)
        mk = PR._min_keep_train(min_keep_global, len(audit), m)
        n_te = len(test_pairs)
        k_tr, _, _ = vrf.solve_train_keep_for_target_global(
            target_global_keep, m, n_te, mk, len(audit)
        )
        rs = (rng_base + seed * 100003 + target_global_keep) & 0xFFFFFFFF
        rng = np.random.default_rng(rs)
        pick = rng.choice(m, size=k_tr, replace=False)
        keep = np.zeros(m, dtype=bool)
        keep[pick] = True
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

    return evaluate_fixed_random


def split_gss(
    n_pairs: int, test_size: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    X = np.zeros((n_pairs, 1))
    groups = np.arange(n_pairs, dtype=np.int64)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, groups=groups))
    return np.sort(train_idx.astype(np.int64)), np.sort(test_idx.astype(np.int64))


def run_one(
    name: str,
    fn: Callable[..., PR.SearchState],
    audit: pd.DataFrame,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    seed: int,
    n_splits: int,
    min_keep: int,
    max_drop: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
    k_target: int,
) -> Dict[str, float]:
    st = fn(
        audit,
        df_full,
        train_pairs,
        test_pairs,
        profile,
        seed,
        n_splits,
        min_keep,
        max_drop,
        target_auc,
        w_auc,
        w_size,
        w_imb,
        k_target,
    )
    kg, _ = PR.fit_apply_retain_fraction(
        df_full, audit, train_pairs, test_pairs, st.keep_mask_train, profile, seed
    )
    return {
        "method": name,
        "seed": seed,
        "target_global_pairs": k_target,
        "actual_global_pairs": float(kg.sum()),
        "heldout_auc": float(st.heldout_auc),
        "search_time_auc": float(st.search_auc),
        "gap": float(st.search_auc - st.heldout_auc)
        if np.isfinite(st.search_auc) and np.isfinite(st.heldout_auc)
        else float("nan"),
        "dist_to_chance": abs(float(st.heldout_auc) - 0.5),
    }


def best_k_for_seed(
    method_name: str,
    fn: Callable[..., PR.SearchState],
    *,
    audit: pd.DataFrame,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    seed: int,
    k_candidates: List[int],
    min_keep: int,
    band: float,
) -> Tuple[int, Dict[str, float]]:
    """
    Return (best_k, row) where we maximize actual global pair count in band [k_min,k_max]
    subject to |heldout_auc - 0.5| <= band; if impossible, minimize dist_to_chance
    then prefer larger K.
    """
    w_auc, w_size, w_imb = 2.0, 1.0, 0.5
    target_auc = 0.62
    max_drop = 0.45
    n_splits = 5
    rows = []
    for k in sorted(k_candidates, reverse=True):
        r = run_one(
            method_name,
            fn,
            audit,
            df_full,
            profile,
            train_pairs,
            test_pairs,
            seed,
            n_splits,
            min_keep,
            max_drop,
            target_auc,
            w_auc,
            w_size,
            w_imb,
            k,
        )
        rows.append(r)
    df = pd.DataFrame(rows)
    df_feas = df[df["dist_to_chance"] <= band]
    if len(df_feas):
        j = int(df_feas.sort_values("actual_global_pairs", ascending=False).index[0])
        best = df.loc[j].to_dict()
        return int(best["target_global_pairs"]), best
    j = int(df["dist_to_chance"].idxmin())
    sub = df[df["dist_to_chance"] == df.loc[j, "dist_to_chance"]]
    jj = int(sub.sort_values("actual_global_pairs", ascending=False).index[0])
    return int(df.loc[jj, "target_global_pairs"]), df.loc[jj].to_dict()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "near_random_subset_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = root / "audits" / "truthfulqa_style_audit.csv"
    audit = pd.read_csv(audit_path)
    profile: tpa.AuditProfile = "paper10"
    df_full = tpa.build_answer_level_audit_frame(audit, profile=profile, copy_audit_meta=True)
    n_pairs = len(audit)

    # Allow retaining ~300 global pairs (paper verification used min_keep=400 which blocks small sets)
    min_keep_global = 200
    k_min, k_max = 300, 500
    step = 5
    k_candidates = list(range(k_max, k_min - 1, -step))
    band = 0.06  # |AUC - 0.5| <= 0.06  (~null SD scale on held-out fold)
    seeds = list(range(10))

    methods: List[Tuple[str, Callable[..., PR.SearchState]]] = [
        ("score_rank (low separability)", vrf.evaluate_fixed_global_score_rank),
        ("negation_rank (low asymmetry)", vrf.evaluate_fixed_global_negation_rank),
        ("len_gap_rank (symmetric length)", evaluate_fixed_len_gap_rank),
        ("clean_first_then_low_score", evaluate_fixed_confound_free_train),
        ("uniform_random_train_pairs", make_random_fixed_eval(2026)),
    ]

    summary_rows = []
    all_rows = []

    for mname, fn in methods:
        per_seed_best = []
        for seed in seeds:
            train_pairs, test_pairs = split_gss(n_pairs, 0.25, seed)
            bk, row = best_k_for_seed(
                mname,
                fn,
                audit=audit,
                df_full=df_full,
                profile=profile,
                train_pairs=train_pairs,
                test_pairs=test_pairs,
                seed=seed,
                k_candidates=k_candidates,
                min_keep=min_keep_global,
                band=band,
            )
            row["best_target_k"] = bk
            per_seed_best.append(row)
            all_rows.append(row)

        ps = pd.DataFrame(per_seed_best)
        summary_rows.append(
            {
                "method": mname,
                "mean_actual_pairs": ps["actual_global_pairs"].mean(),
                "std_actual_pairs": ps["actual_global_pairs"].std(),
                "mean_heldout_auc": ps["heldout_auc"].mean(),
                "std_heldout_auc": ps["heldout_auc"].std(),
                "mean_dist_to_chance": ps["dist_to_chance"].mean(),
                "feasible_seeds": int((ps["dist_to_chance"] <= band).sum()),
                "n_seeds": len(seeds),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("mean_dist_to_chance", ascending=True)
    pd.DataFrame(all_rows).to_csv(out_dir / "per_seed_best.csv", index=False)
    summary.to_csv(out_dir / "method_summary.csv", index=False)

    # Full sweep for winning method only (fine grid) — pick method with lowest mean dist
    winner = summary.iloc[0]["method"]
    winner_fn = dict(methods)[winner] if winner in dict(methods) else None
    # methods list has display names - map back
    name_to_fn = {a: b for a, b in methods}
    winner_fn = name_to_fn[winner]

    fine = list(range(k_max, k_min - 1, -2))
    fine_rows = []
    for seed in seeds:
        train_pairs, test_pairs = split_gss(n_pairs, 0.25, seed)
        for k in fine:
            fine_rows.append(
                run_one(
                    winner,
                    winner_fn,
                    audit,
                    df_full,
                    profile,
                    train_pairs,
                    test_pairs,
                    seed,
                    5,
                    min_keep_global,
                    0.45,
                    0.62,
                    2.0,
                    1.0,
                    0.5,
                    k,
                )
            )
    safe = "".join(c if c.isalnum() else "_" for c in winner)[:40]
    pd.DataFrame(fine_rows).to_csv(out_dir / f"sweep_winner_{safe}.csv", index=False)

    # Print tables
    print("\n### Method ranking (10 seeds, target K in [300,500] step 5, band |AUC-0.5|≤%.2f)\n" % band)
    print(summary.to_string(index=False))
    print("\n### Best method:", winner)
    print("### Artifacts:", out_dir)

    md = [
        "# Near-random held-out AUC subset search",
        "",
        f"- Protocol: paper10, GroupShuffleSplit test_size=0.25, min_keep_global={min_keep_global}, band=|AUC-0.5|≤{band}",
        f"- Pair counts: target global retained pairs in [{k_min}, {k_max}] (step {step} for selection).",
        "",
        "## Method summary (mean ± std over seeds)",
        "",
        "| method | mean actual pairs | std | mean held-out AUC | std | mean |AUC-0.5| | seeds in band |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        md.append(
            f"| {r['method']} | {r['mean_actual_pairs']:.1f} | {r['std_actual_pairs']:.2f} | "
            f"{r['mean_heldout_auc']:.4f} | {r['std_heldout_auc']:.4f} | {r['mean_dist_to_chance']:.4f} | "
            f"{int(r['feasible_seeds'])}/{int(r['n_seeds'])} |"
        )
    md.append("")
    md.append("## Recommendation")
    md.append(
        f"- **Best tradeoff (lowest mean distance to 0.5):** `{winner}` — see per_seed_best.csv and sweep CSV."
    )
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
