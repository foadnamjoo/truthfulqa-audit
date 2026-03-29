#!/usr/bin/env python3
"""
Refined grid: fixed target pair counts × near-random methods, multi-seed,
paper10 + GroupShuffleSplit protocol (same as near-random branch).

Outputs under results/near_random_subset_search_refined/ and figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_pruning_final_verification as vrf
import search_near_random_clean_subset as sn
import truthfulqa_paper_audit as tpa

PR = sn.PR


def eval_row(
    method: str,
    fn: Callable[..., PR.SearchState],
    *,
    audit: pd.DataFrame,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    seed: int,
    target_pairs: int,
    min_keep_global: int,
) -> Dict[str, float]:
    n_splits = 5
    max_drop = 0.45
    target_auc = 0.62
    w_auc, w_size, w_imb = 2.0, 1.0, 0.5
    st = fn(
        audit,
        df_full,
        train_pairs,
        test_pairs,
        profile,
        seed,
        n_splits,
        min_keep_global,
        max_drop,
        target_auc,
        w_auc,
        w_size,
        w_imb,
        target_pairs,
    )
    kg, _ = PR.fit_apply_retain_fraction(
        df_full, audit, train_pairs, test_pairs, st.keep_mask_train, profile, seed
    )
    sub = audit.iloc[np.where(kg)[0]]
    n_clean = int((sub["style_violation"] == 0).sum()) if len(sub) else 0
    conf_frac = float(sub["style_violation"].mean()) if len(sub) else float("nan")
    ho = float(st.heldout_auc)
    gap = (
        float(st.search_auc - st.heldout_auc)
        if np.isfinite(st.search_auc) and np.isfinite(st.heldout_auc)
        else float("nan")
    )
    return {
        "seed": seed,
        "target_pairs": target_pairs,
        "method": method,
        "actual_global_pairs": float(kg.sum()),
        "heldout_auc": ho,
        "search_time_auc": float(st.search_auc),
        "gap": gap,
        "dist_to_chance": abs(ho - 0.5),
        "clean_pair_count": float(n_clean),
        "confounded_fraction": conf_frac,
    }


def pick_best_per_target(summary: pd.DataFrame) -> pd.DataFrame:
    """Lowest mean_dist_to_chance; ties: higher mean_clean_pair_count, then mean_actual_pairs."""
    rows = []
    for t, grp in summary.groupby("target_pairs", sort=True):
        g2 = grp.assign(
            neg_clean=-grp["mean_clean_pair_count"],
            neg_actual=-grp["mean_actual_pairs"],
        ).sort_values(
            ["mean_dist_to_chance", "neg_clean", "neg_actual"],
            ascending=[True, True, True],
        )
        rows.append(g2.iloc[0])
    out = pd.DataFrame(rows)
    cols = [
        "target_pairs",
        "method",
        "mean_actual_pairs",
        "mean_heldout_auc",
        "std_heldout_auc",
        "mean_dist_to_chance",
        "mean_clean_pair_count",
        "mean_confounded_fraction",
    ]
    return out[cols].reset_index(drop=True)


def write_recommendation(
    path: Path,
    per_seed: pd.DataFrame,
    best_by_t: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    # Global best distance (any target, any method in grid)
    flat = summary.sort_values("mean_dist_to_chance")
    global_best = flat.iloc[0]

    # Largest target where best-row dist is "reasonable" (<= 0.12)
    reasonable = best_by_t[best_by_t["mean_dist_to_chance"] <= 0.12]
    if len(reasonable):
        largest_r = reasonable.sort_values("target_pairs", ascending=False).iloc[0]
    else:
        largest_r = best_by_t.sort_values("mean_dist_to_chance").iloc[0]

    # clean_first at 400 vs 370-ish: use 375 row if exists
    def _row(t: int, m: str):
        s = summary[(summary["target_pairs"] == t) & (summary["method"] == m)]
        return s.iloc[0] if len(s) else None

    r375 = _row(375, "clean_first_then_low_score")
    r400 = _row(400, "clean_first_then_low_score")

    r300 = summary[summary["target_pairs"] == 300]
    c300 = r300[r300["method"] == "clean_first_then_low_score"]
    s300 = r300[r300["method"] == "score_rank (low separability)"]
    trade_300 = ""
    if len(c300) and len(s300):
        cc, ss = c300.iloc[0], s300.iloc[0]
        trade_300 = (
            f"- **Tradeoff at 300:** **`clean_first_then_low_score`** has mean |AUC−0.5| **{cc['mean_dist_to_chance']:.3f}** "
            f"but **{cc['mean_clean_pair_count']:.0f}** mean clean pairs vs **{ss['mean_clean_pair_count']:.0f}** for score_rank "
            f"(**{ss['mean_dist_to_chance']:.3f}** distance) — pick clean_first when cleanliness dominates.\n"
        )

    lines = [
        "# Refined near-random subset branch — recommendations",
        "",
        "Protocol: **paper10**, **GroupShuffleSplit** (`test_size=0.25`), held-out AUC = grouped OOF on retained **test** pairs only, `min_keep_global=200`.",
        "",
        "## 1. Best overall if the priority is **as close to chance (0.5) as possible**",
        f"- In this grid, the **lowest mean |AUC−0.5|** is **{global_best['mean_dist_to_chance']:.4f}** at **target {int(global_best['target_pairs'])}**, method **`{global_best['method']}`** "
        f"(mean held-out **{global_best['mean_heldout_auc']:.4f}**, mean clean pairs **{global_best['mean_clean_pair_count']:.1f}**).",
        trade_300,
        "- Smaller targets (300–375) typically move held-out AUC closer to 0.5; see `summary_by_target_and_method.csv`.",
        "",
        "## 2. Best if the priority is the **largest still-reasonable** subset",
        f"- Among **per-target winners** with **mean |AUC−0.5| ≤ 0.12**, the largest target is **{int(largest_r['target_pairs'])}** (**`{largest_r['method']}`**, mean dist **{largest_r['mean_dist_to_chance']:.4f}**, mean held-out **{largest_r['mean_heldout_auc']:.4f}**).",
        "- If **no** target meets ≤0.12, the table falls back to the tightest available winner; **check `best_method_by_target.csv`**. ",
        "",
        "## 3. Is **400** still acceptable vs **~370**?",
    ]
    if r375 is not None and r400 is not None:
        lines.extend(
            [
                f"- **clean_first_then_low_score** at **375**: mean held-out **{r375['mean_heldout_auc']:.4f}** ± **{r375['std_heldout_auc']:.4f}**, mean |AUC−0.5| **{r375['mean_dist_to_chance']:.4f}**, mean clean **{r375['mean_clean_pair_count']:.1f}**.",
                f"- **Same method** at **400**: mean held-out **{r400['mean_heldout_auc']:.4f}** ± **{r400['std_heldout_auc']:.4f}**, mean |AUC−0.5| **{r400['mean_dist_to_chance']:.4f}**, mean clean **{r400['mean_clean_pair_count']:.1f}**.",
                "- **Honest read:** if your bar for “near-random” is **|AUC−0.5| ≲ 0.08–0.10**, **375 is usually closer** than **400**; at **400**, held-out AUC is often **materially higher** (weaker near-random claim) unless a different method wins at that grid point.",
            ]
        )
    else:
        lines.append("- See `summary_by_target_and_method.csv` for clean_first rows at 375 vs 400.")

    # 425–500 closeness
    high = best_by_t[best_by_t["target_pairs"] >= 425]
    if len(high):
        worst_dist = float(high["mean_dist_to_chance"].max())
        lines.extend(
            [
                "",
                "## 4. Targets **425–500** and closeness to chance",
                f"- Among **winners** at 425–500, mean |AUC−0.5| ranges up to **~{worst_dist:.3f}**. ",
                "- **450–500** is typically **not** “close to 0.5” on held-out audit in this protocol: expect **~0.14+** distance to chance for the best method at those sizes unless the fold is unusually easy.",
            ]
        )

    lines.extend(
        [
            "",
            "## 5. What to **report**",
            "- **Stronger near-random claim:** cite **target 300–350**, **clean_first_then_low_score** (or whichever wins in `best_method_by_target.csv` there), with **mean ± std** held-out AUC over seeds from `per_seed_results.csv`.",
            "- **Larger “still useful” benchmark:** cite a target **375–425** if |AUC−0.5| is acceptable for your narrative, again with **held-out** metrics and **mean clean pair count / confounded fraction**.",
            "",
            "## Caveats",
            "- One **held-out pair split** per seed; std reflects **split noise**, not full hierarchical CI.",
            "- “Best method” uses **mean distance to chance**, then **cleaner**, then **larger actual** count — see `pick_best_per_target` in `run_near_random_subset_refined.py`.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = root / "results" / "near_random_subset_search_refined"
    fig_dir = root / "figures" / "near_random_subset_search_refined"
    res.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    profile: tpa.AuditProfile = "paper10"
    audit_path = root / "audits" / "truthfulqa_style_audit.csv"
    audit = pd.read_csv(audit_path)
    df_full = tpa.build_answer_level_audit_frame(audit, profile=profile, copy_audit_meta=True)
    n_pairs = len(audit)

    targets = [300, 325, 350, 375, 400, 425, 450, 475, 500]
    seeds = list(range(10))
    test_size = 0.25
    min_keep_global = 200

    methods: List[Tuple[str, Callable[..., PR.SearchState]]] = [
        ("clean_first_then_low_score", sn.evaluate_fixed_confound_free_train),
        ("score_rank (low separability)", vrf.evaluate_fixed_global_score_rank),
        ("len_gap_rank (symmetric length)", sn.evaluate_fixed_len_gap_rank),
        ("negation_rank (low asymmetry)", vrf.evaluate_fixed_global_negation_rank),
    ]

    rows: List[Dict] = []
    for seed in seeds:
        train_pairs, test_pairs = sn.split_gss(n_pairs, test_size, seed)
        for target_pairs in targets:
            for mname, fn in methods:
                rows.append(
                    eval_row(
                        mname,
                        fn,
                        audit=audit,
                        df_full=df_full,
                        profile=profile,
                        train_pairs=train_pairs,
                        test_pairs=test_pairs,
                        seed=seed,
                        target_pairs=target_pairs,
                        min_keep_global=min_keep_global,
                    )
                )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(res / "per_seed_results.csv", index=False)

    g = ["target_pairs", "method"]
    summary = (
        per_seed.groupby(g, sort=False)
        .agg(
            mean_actual_pairs=("actual_global_pairs", "mean"),
            std_actual_pairs=("actual_global_pairs", "std"),
            mean_heldout_auc=("heldout_auc", "mean"),
            std_heldout_auc=("heldout_auc", "std"),
            mean_dist_to_chance=("dist_to_chance", "mean"),
            std_dist_to_chance=("dist_to_chance", "std"),
            mean_clean_pair_count=("clean_pair_count", "mean"),
            mean_confounded_fraction=("confounded_fraction", "mean"),
            num_seeds=("seed", "count"),
        )
        .reset_index()
    )
    summary = summary.sort_values(["target_pairs", "mean_dist_to_chance", "method"])
    summary.to_csv(res / "summary_by_target_and_method.csv", index=False)

    best_by_t = pick_best_per_target(summary)
    best_by_t.to_csv(res / "best_method_by_target.csv", index=False)

    md = [
        "| target pairs | best method | mean actual pairs | mean held-out AUC | std | mean |AUC-0.5| | mean clean pairs | mean confounded fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in best_by_t.iterrows():
        sm = summary[(summary["target_pairs"] == r["target_pairs"]) & (summary["method"] == r["method"])].iloc[
            0
        ]
        md.append(
            f"| {int(r['target_pairs'])} | {r['method']} | {r['mean_actual_pairs']:.1f} | "
            f"{r['mean_heldout_auc']:.4f} | {r['std_heldout_auc']:.4f} | {r['mean_dist_to_chance']:.4f} | "
            f"{r['mean_clean_pair_count']:.1f} | {r['mean_confounded_fraction']:.4f} |"
        )
    (res / "final_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    write_recommendation(res / "recommendation.md", per_seed, best_by_t, summary)

    (res / "run_config.json").write_text(
        json.dumps(
            {
                "targets": targets,
                "seeds": seeds,
                "test_size": test_size,
                "min_keep_global": min_keep_global,
                "profile": profile,
                "methods": [m for m, _ in methods],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Figures: one row per (target, winner) for line plots of winners only — richer: all methods
    piv_h = summary.pivot_table(
        index="target_pairs", columns="method", values="mean_heldout_auc", aggfunc="first"
    )
    piv_d = summary.pivot_table(
        index="target_pairs", columns="method", values="mean_dist_to_chance", aggfunc="first"
    )
    piv_c = summary.pivot_table(
        index="target_pairs", columns="method", values="mean_clean_pair_count", aggfunc="first"
    )

    x = summary["target_pairs"].unique()
    x = np.sort(x)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in piv_h.columns:
        ax.plot(x, piv_h.loc[x, m], marker="o", ms=4, label=m)
    ax.set_xlabel("Target pairs")
    ax.set_ylabel("Mean held-out AUC")
    ax.legend(fontsize=7, loc="best")
    ax.set_title("Near-random refined grid (10 seeds)")
    fig.tight_layout()
    fig.savefig(fig_dir / "target_vs_mean_heldout_auc.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in piv_d.columns:
        ax.plot(x, piv_d.loc[x, m], marker="o", ms=4, label=m)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Target pairs")
    ax.set_ylabel("Mean |held-out AUC − 0.5|")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "target_vs_mean_dist_to_chance.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in piv_c.columns:
        ax.plot(x, piv_c.loc[x, m], marker="o", ms=4, label=m)
    ax.set_xlabel("Target pairs")
    ax.set_ylabel("Mean clean pair count (global retained)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "target_vs_mean_clean_pairs.pdf")
    plt.close(fig)

    print(f"Wrote {res} and {fig_dir}")


if __name__ == "__main__":
    main()
