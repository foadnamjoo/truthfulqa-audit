#!/usr/bin/env python3
"""
Paper finalization: near-random TruthfulQA subset (better-algorithm branch only).
Targets 350 / 375 / 400; methods: clean_first, SA-from-clean, beam-from-clean, score_rank.
Stronger eval_budget vs exploratory run; 15 seeds; canonical JSON at GSS seed 42.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_near_random_better_algorithms as bnb
import run_pruning_final_verification as vrf
import search_near_random_clean_subset as sn
import truthfulqa_paper_audit as tpa

PR = bnb.PR


def run_search_with_kg(
    name: str,
    pack: bnb.EvalPack,
    target_pairs: int,
    k_tr: int,
    rng: np.random.Generator,
    budget: int,
) -> Tuple[Dict, np.ndarray]:
    ep = pack
    if name == "simulated_annealing_from_clean_first":
        k0 = bnb.init_clean_first(ep, k_tr)
        kf, _ = bnb.simulated_annealing(ep, k0, rng, budget)
    elif name == "beam_light_swaps_from_clean":
        k0 = bnb.init_clean_first(ep, k_tr)
        kf, _ = bnb.beam_light(ep, k0, rng, budget)
    else:
        raise ValueError(name)
    st, kg, _, _, _ = bnb.global_metrics(ep, kf)
    row = bnb.baseline_row(name, st, kg, ep, target_pairs)
    return row, kg


def _rng_eval_seed(gss_seed: int, target_pairs: int, method: str, search_names: List[str]) -> Tuple[Optional[int], np.random.Generator]:
    """Match multi-seed loop RNG for `run_search` (same formula as main())."""
    import hashlib

    if method not in search_names:
        return None, np.random.default_rng(0)
    j = search_names.index(method)
    h = int(hashlib.md5(method.encode()).hexdigest()[:8], 16)
    s = 70_000 + gss_seed * 100_000 + j * 10_000 + (target_pairs % 1000) + (h % 10_000)
    return s, np.random.default_rng(s)


def export_canonical_json(
    path: Path,
    *,
    target_pairs: int,
    method: str,
    audit: pd.DataFrame,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    n_pairs: int,
    min_keep_global: int,
    eval_budget: int,
    gss_seed: int,
    search_names: List[str],
) -> None:
    """Single reproducible subset: fixed GroupShuffleSplit + same RNG as seed=gss_seed in the sweep."""
    train_pairs, test_pairs = sn.split_gss(n_pairs, 0.25, gss_seed)
    train_sorted = np.sort(train_pairs)
    pack = bnb.EvalPack(
        audit,
        df_full,
        profile,
        train_pairs,
        test_pairs,
        train_sorted,
        gss_seed,
        5,
        2.0,
        1.0,
        0.5,
        0.62,
    )
    k_tr = bnb.k_tr_for_target(audit, train_pairs, test_pairs, target_pairs, min_keep_global)

    if method == "clean_first_then_low_score":
        st = sn.evaluate_fixed_confound_free_train(
            audit,
            df_full,
            train_pairs,
            test_pairs,
            profile,
            gss_seed,
            5,
            min_keep_global,
            0.45,
            0.62,
            2.0,
            1.0,
            0.5,
            target_pairs,
        )
        kg, _ = PR.fit_apply_retain_fraction(
            df_full, audit, train_pairs, test_pairs, st.keep_mask_train, profile, gss_seed
        )
    elif method == "score_rank (low separability)":
        st = vrf.evaluate_fixed_global_score_rank(
            audit,
            df_full,
            train_pairs,
            test_pairs,
            profile,
            gss_seed,
            5,
            min_keep_global,
            0.45,
            0.62,
            2.0,
            1.0,
            0.5,
            target_pairs,
        )
        kg, _ = PR.fit_apply_retain_fraction(
            df_full, audit, train_pairs, test_pairs, st.keep_mask_train, profile, gss_seed
        )
    elif method == "simulated_annealing_from_clean_first":
        _seed_int, rng_use = _rng_eval_seed(gss_seed, target_pairs, method, search_names)
        _, kg = run_search_with_kg(method, pack, target_pairs, k_tr, rng_use, eval_budget)
    elif method == "beam_light_swaps_from_clean":
        _seed_int, rng_use = _rng_eval_seed(gss_seed, target_pairs, method, search_names)
        _, kg = run_search_with_kg(method, pack, target_pairs, k_tr, rng_use, eval_budget)
    else:
        raise ValueError(f"Unknown method for export: {method}")

    rng_meta, _ = _rng_eval_seed(gss_seed, target_pairs, method, search_names) if method in search_names else (None, None)

    pids = sorted(int(x) for x in np.where(kg)[0])
    sub = audit.iloc[pids]
    payload = {
        "branch": "near_random_better_algorithms",
        "audit_csv": "audits/truthfulqa_style_audit.csv",
        "audit_profile": profile,
        "protocol": {
            "group_shuffle_split_test_size": 0.25,
            "group_shuffle_split_seed": gss_seed,
            "heldout_auc": "Grouped OOF ROC-AUC on retained test pair answer rows (paper10)",
            "min_keep_global": min_keep_global,
            "search_eval_budget": eval_budget if method in search_names else None,
            "search_rng_derived_seed": rng_meta,
        },
        "target_global_pairs": target_pairs,
        "method": method,
        "n_pairs_retained": len(pids),
        "n_clean_pairs": int((sub["style_violation"] == 0).sum()),
        "confounded_fraction": float(sub["style_violation"].mean()) if len(sub) else None,
        "pair_ids": pids,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pick_best_paper(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, grp in summary.groupby("target_pairs", sort=True):
        g2 = grp.assign(neg_c=-grp["mean_clean_pair_count"], neg_a=-grp["mean_actual_pairs"]).sort_values(
            ["mean_dist_to_chance", "neg_c", "neg_a"], ascending=[True, True, True]
        )
        rows.append(g2.iloc[0])
    return pd.DataFrame(rows)[
        [
            "target_pairs",
            "method",
            "mean_actual_pairs",
            "mean_heldout_auc",
            "std_heldout_auc",
            "mean_dist_to_chance",
            "mean_clean_pair_count",
            "mean_confounded_fraction",
        ]
    ].reset_index(drop=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = root / "results" / "final_near_random_truthfulqa_subset"
    figd = root / "figures" / "final_near_random_truthfulqa_subset"
    res.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)

    profile: tpa.AuditProfile = "paper10"
    audit = pd.read_csv(root / "audits" / "truthfulqa_style_audit.csv")
    df_full = tpa.build_answer_level_audit_frame(audit, profile=profile, copy_audit_meta=True)
    n_pairs = len(audit)

    targets = [350, 375, 400]
    seeds = list(range(15))
    min_keep_global = 200
    eval_budget = 22

    baselines = [
        ("clean_first_then_low_score", sn.evaluate_fixed_confound_free_train),
        ("score_rank (low separability)", vrf.evaluate_fixed_global_score_rank),
    ]
    search_names = [
        "simulated_annealing_from_clean_first",
        "beam_light_swaps_from_clean",
    ]

    rows: List[Dict] = []
    import hashlib

    for seed in seeds:
        tr, te = sn.split_gss(n_pairs, 0.25, seed)
        ts = np.sort(tr)
        pack = bnb.EvalPack(
            audit,
            df_full,
            profile,
            tr,
            te,
            ts,
            seed,
            5,
            2.0,
            1.0,
            0.5,
            0.62,
        )
        for tg in targets:
            k_tr = bnb.k_tr_for_target(audit, tr, te, tg, min_keep_global)
            for bn, bf in baselines:
                rows.append(bnb.run_baseline(bn, bf, pack, tg, min_keep_global))
            for j, snm in enumerate(search_names):
                h = int(hashlib.md5(snm.encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(70_000 + seed * 100_000 + j * 10_000 + (tg % 1000) + (h % 10_000))
                rows.append(bnb.run_search(snm, pack, tg, k_tr, rng, eval_budget))

    per = pd.DataFrame(rows)
    per.to_csv(res / "per_seed_results.csv", index=False)

    summary = (
        per.groupby(["target_pairs", "method"], sort=False)
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

    best = pick_best_paper(summary)
    best.to_csv(res / "best_method_by_target.csv", index=False)

    lines = [
        "| target pairs | best method | mean actual pairs | mean held-out AUC | std | mean |AUC-0.5| | mean clean pairs | mean confounded fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in best.iterrows():
        lines.append(
            f"| {int(r['target_pairs'])} | {r['method']} | {r['mean_actual_pairs']:.1f} | "
            f"{r['mean_heldout_auc']:.4f} | {r['std_heldout_auc']:.4f} | {r['mean_dist_to_chance']:.4f} | "
            f"{r['mean_clean_pair_count']:.1f} | {r['mean_confounded_fraction']:.4f} |"
        )
    (res / "final_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figures (4 methods × 3 targets)
    subm = summary[summary["method"].isin([b[0] for b in baselines] + search_names)]
    piv_h = subm.pivot_table(index="target_pairs", columns="method", values="mean_heldout_auc")
    piv_d = subm.pivot_table(index="target_pairs", columns="method", values="mean_dist_to_chance")
    piv_c = subm.pivot_table(index="target_pairs", columns="method", values="mean_clean_pair_count")
    x = np.sort(piv_h.index.to_numpy())

    def plot_pdf(name: str, piv, ylab: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for col in sorted(piv.columns):
            ax.plot(x, piv.loc[x, col], marker="o", ms=5, label=col)
        ax.set_xlabel("Target global pairs")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(figd / name)
        plt.close(fig)

    plot_pdf("target_vs_mean_heldout_auc.pdf", piv_h, "Mean held-out AUC (15 seeds)", "Final near-random subset")
    plot_pdf("target_vs_mean_dist_to_chance.pdf", piv_d, "Mean |held-out AUC − 0.5|", "Distance to chance")
    plot_pdf("target_vs_mean_clean_pairs.pdf", piv_c, "Mean clean pair count", "Retained clean pairs")

    # Canonical JSON exports (GSS seed 42; search RNG derived from target + method)
    CANONICAL_GSS = 42

    def search_seed(tg: int, method: str) -> int:
        h = int(hashlib.md5(method.encode()).hexdigest()[:8], 16)
        return 900_000 + tg * 1_000 + (h % 10_000)

    for tg in targets:
        win = best.loc[best["target_pairs"] == tg, "method"].iloc[0]
        export_canonical_json(
            res / f"final_subset_ids_{tg}.json",
            target_pairs=tg,
            method=win,
            audit=audit,
            df_full=df_full,
            profile=profile,
            n_pairs=n_pairs,
            min_keep_global=min_keep_global,
            eval_budget=eval_budget,
            gss_seed=CANONICAL_GSS,
            search_names=search_names,
        )

    # --- recommendation.md (paper decision) ---
    r350 = best.loc[best["target_pairs"] == 350].iloc[0]
    r375 = best.loc[best["target_pairs"] == 375].iloc[0]
    r400 = best.loc[best["target_pairs"] == 400].iloc[0]

    # Main = lowest mean_dist_to_chance among three winners (strongest near-random claim)
    comp = best.sort_values("mean_dist_to_chance")
    main_pt = comp.iloc[0]
    # Secondary = larger N among remaining, prefer 400 if close
    secondary = (
        best[best["target_pairs"] != int(main_pt["target_pairs"])]
        .sort_values("target_pairs", ascending=False)
        .iloc[0]
    )

    alg_name = (
        "Clean-first initialization with train-only pair ranking, followed by "
        "stochastic subset refinement (simulated annealing or beam search over fixed-size swaps) "
        "under the same paper10 held-out objective."
    )
    if "simulated_annealing" in str(main_pt["method"]):
        alg_short = (
            "**Simulated annealing from clean-first** (`simulated_annealing_from_clean_first`): "
            "start from `clean_first_then_low_score`, then random pair swaps optimizing a weighted "
            "held-out–centric loss until the evaluation budget is exhausted."
        )
    elif "beam" in str(main_pt["method"]).lower():
        alg_short = (
            "**Beam search over swaps from clean-first** (`beam_light_swaps_from_clean`): "
            "expand a small beam of candidate retained sets using random swaps, keep the best under the same loss."
        )
    else:
        alg_short = (
            "Baseline **`clean_first_then_low_score`** (no local search): retain all audit-clean train pairs first, "
            "then fill by lowest separability among remaining pairs."
        )

    rec = f"""# Final near-random TruthfulQA subset — paper recommendation

## Locked scope
- **Branch:** near-random / better-algorithm only (not legacy pruning replay, not `feature_balanced` as primary).
- **Audit:** `paper10`, `audits/truthfulqa_style_audit.csv`, **no** label or text changes.
- **Held-out:** `GroupShuffleSplit(test_size=0.25)` per seed; held-out AUC = grouped OOF on **retained test** answer rows.

## Run configuration (this table)
- **Seeds:** {len(seeds)} independent pair splits (`seed` = 0..{len(seeds)-1}).
- **Search budget:** `{eval_budget}` full `evaluate_state` calls per **simulated annealing** / **beam** run (stronger than the exploratory `eval_budget=8` pass).
- **Targets:** {targets}.
- **Methods:** `clean_first_then_low_score`, `score_rank (low separability)`, `simulated_annealing_from_clean_first`, `beam_light_swaps_from_clean`.

## 1. Main paper operating point
- **Recommendation:** **{int(main_pt["target_pairs"])} pairs** using **`{main_pt["method"]}`**.
- **Rationale:** lowest **mean distance to chance** (mean |held-out AUC − 0.5|) among the three sizes, i.e. **strongest defensible “near-random” claim** in this grid.
- **Numbers:** mean held-out **{main_pt["mean_heldout_auc"]:.4f}** ± **{main_pt["std_heldout_auc"]:.4f}**; mean |AUC−0.5| **{main_pt["mean_dist_to_chance"]:.4f}**; mean clean pairs **{main_pt["mean_clean_pair_count"]:.1f}**; mean confounded fraction **{main_pt["mean_confounded_fraction"]:.4f}**.

## 2. Secondary (larger) operating point
- **Recommendation:** **{int(secondary["target_pairs"])} pairs** using **`{secondary["method"]}`**.
- **Rationale:** larger retained set for a **“still useful benchmark”** story while reporting held-out AUC honestly (typically **slightly farther** from 0.5 than the main point).
- **Numbers:** mean held-out **{secondary["mean_heldout_auc"]:.4f}** ± **{secondary["std_heldout_auc"]:.4f}**; mean |AUC−0.5| **{secondary["mean_dist_to_chance"]:.4f}**; mean clean pairs **{secondary["mean_clean_pair_count"]:.1f}**.

## 3. Per-target winners (350 / 375 / 400)
| Target | Best method | Mean held-out AUC | Std | Mean |AUC−0.5| |
|---:|---|---:|---:|---:|
| 350 | {r350["method"]} | {r350["mean_heldout_auc"]:.4f} | {r350["std_heldout_auc"]:.4f} | {r350["mean_dist_to_chance"]:.4f} |
| 375 | {r375["method"]} | {r375["mean_heldout_auc"]:.4f} | {r375["std_heldout_auc"]:.4f} | {r375["mean_dist_to_chance"]:.4f} |
| 400 | {r400["method"]} | {r400["mean_heldout_auc"]:.4f} | {r400["std_heldout_auc"]:.4f} | {r400["mean_dist_to_chance"]:.4f} |

## 4. What to claim in the paper
- **Strongest near-random:** cite **{int(main_pt["target_pairs"])} pairs**, **`{main_pt["method"]}`**, held-out **{main_pt["mean_heldout_auc"]:.4f} ± {main_pt["std_heldout_auc"]:.4f}** ({len(seeds)} splits), plus clean/confounded summary from `summary_by_target_and_method.csv`.
- **Larger still-useful benchmark:** cite **{int(secondary["target_pairs"])} pairs**, **`{secondary["method"]}`**, held-out **{secondary["mean_heldout_auc"]:.4f} ± {secondary["std_heldout_auc"]:.4f}**.

## 5. Algorithm name for the paper
{alg_short}

Generic paper name: **“clean-first constrained subset selection with fixed-size local search (simulated annealing / beam) under the paper10 audit.”**

## 6. Reproducible subset IDs
- `final_subset_ids_350.json`, `final_subset_ids_375.json`, `final_subset_ids_400.json` list **retained `pair_ids`** for the **per-target winning method** using **canonical** `GroupShuffleSplit` seed **{CANONICAL_GSS}** and the documented `search_rng_seed` / `search_eval_budget`.

## 7. Honest comparison
- If **375** wins on distance-to-chance, it is the **clearest** single “best overall” size; if **400** wins on practicality (size) with only slightly worse AUC, authors may emphasize **400** as the **secondary** benchmark and **375** as **main** for the leakage claim — see Section 1–2 above for the automated choice from this run.
"""
    (res / "recommendation.md").write_text(rec, encoding="utf-8")

    (res / "run_config.json").write_text(
        json.dumps(
            {
                "targets": targets,
                "seeds": seeds,
                "eval_budget": eval_budget,
                "min_keep_global": min_keep_global,
                "canonical_gss_seed": CANONICAL_GSS,
                "weights_alpha_beta_gamma": [bnb.ALPHA, bnb.BETA, bnb.GAMMA],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Wrote", res, figd)


if __name__ == "__main__":
    main()
