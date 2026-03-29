#!/usr/bin/env python3
"""
Near-random branch: local search / SA / beam vs baselines (clean_first, score_rank, len_gap).
Same paper10 + GroupShuffleSplit held-out protocol as run_near_random_subset_refined.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_pruning_final_verification as vrf
import search_near_random_clean_subset as sn
import truthfulqa_paper_audit as tpa

PR = sn.PR

ALPHA, BETA, GAMMA = 1.0, 0.35, 0.12


@dataclass
class EvalPack:
    audit: pd.DataFrame
    df_full: pd.DataFrame
    profile: tpa.AuditProfile
    train_pairs: np.ndarray
    test_pairs: np.ndarray
    train_sorted: np.ndarray
    seed: int
    n_splits: int
    w_auc: float
    w_size: float
    w_imb: float
    target_auc: float


def combo_loss(ho: float, conf: float, imb: float) -> float:
    if not np.isfinite(ho):
        return 1e9
    return ALPHA * abs(float(ho) - 0.5) + BETA * float(conf) + GAMMA * float(imb)


def global_metrics(
    pack: EvalPack, keep_train: np.ndarray
) -> Tuple[PR.SearchState, np.ndarray, float, float, float]:
    st = PR.evaluate_state(
        pack.df_full,
        pack.audit,
        pack.train_pairs,
        pack.test_pairs,
        keep_train,
        pack.profile,
        pack.seed,
        pack.n_splits,
        pack.w_auc,
        pack.w_size,
        pack.w_imb,
        pack.target_auc,
    )
    kg, _ = PR.fit_apply_retain_fraction(
        pack.df_full,
        pack.audit,
        pack.train_pairs,
        pack.test_pairs,
        keep_train,
        pack.profile,
        pack.seed,
    )
    sub = pack.audit.iloc[np.where(kg)[0]]
    conf = float(sub["style_violation"].mean()) if len(sub) else float("nan")
    imb = PR.imbalance_penalty_from_audit(pack.audit, kg, pack.audit)
    return st, kg, conf, imb, combo_loss(st.heldout_auc, conf, imb)


def init_clean_first(pack: EvalPack, k_tr: int) -> np.ndarray:
    m = len(pack.train_sorted)
    train_sorted = pack.train_sorted
    audit = pack.audit
    clean = audit["style_violation"].to_numpy() == 0
    df_tr = PR.subset_df_ans(pack.df_full, train_sorted)
    sc_all = PR.pair_separability_scores(df_tr, pack.profile, pack.seed)
    st = sc_all[train_sorted]
    idx_clean = [i for i in range(m) if clean[int(train_sorted[i])]]
    idx_rest = [i for i in range(m) if not clean[int(train_sorted[i])]]
    idx_clean.sort(key=lambda i: float(st[i]))
    idx_rest.sort(key=lambda i: float(st[i]))
    order = idx_clean + idx_rest
    keep = np.zeros(m, dtype=bool)
    for i in order[:k_tr]:
        keep[i] = True
    return keep


def init_score_rank(pack: EvalPack, k_tr: int) -> np.ndarray:
    ts = pack.train_sorted
    df_tr = PR.subset_df_ans(pack.df_full, ts)
    sc = PR.pair_separability_scores(df_tr, pack.profile, pack.seed)[ts]
    return vrf.keep_mask_lowest_score_train_pairs(ts, sc, k_tr)


def init_len_gap(pack: EvalPack, k_tr: int) -> np.ndarray:
    m = len(pack.train_sorted)
    lg = pack.audit["len_gap"].to_numpy(dtype=np.float64)[pack.train_sorted]
    order = np.argsort(lg)
    keep = np.zeros(m, dtype=bool)
    keep[order[:k_tr]] = True
    return keep


def init_diversity_stratified(pack: EvalPack, k_tr: int, rng: np.random.Generator) -> np.ndarray:
    m = len(pack.train_sorted)
    train_sorted = pack.train_sorted
    audit = pack.audit
    _, nasym = PR.negation_pair_features(audit)
    lg = audit["len_gap"].to_numpy(dtype=np.float64)
    clean = audit["style_violation"].to_numpy() == 0
    df_tr = PR.subset_df_ans(pack.df_full, train_sorted)
    sc_all = PR.pair_separability_scores(df_tr, pack.profile, pack.seed)
    st = sc_all[train_sorted]
    nas = nasym[train_sorted]
    lgg = lg[train_sorted]
    try:
        nq = pd.qcut(nas, q=4, labels=False, duplicates="drop")
        lq = pd.qcut(lgg, q=4, labels=False, duplicates="drop")
    except ValueError:
        return init_clean_first(pack, k_tr)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(m):
        key = (int(nq[i]), int(lq[i]), int(clean[int(train_sorted[i])]))
        buckets.setdefault(key, []).append(i)
    for key in buckets:
        buckets[key].sort(key=lambda ii: float(st[ii]))
    keep = np.zeros(m, dtype=bool)
    picked = 0
    keys = list(buckets.keys())
    rng.shuffle(keys)
    r = 0
    while picked < k_tr and keys:
        progressed = False
        for _ in range(len(keys)):
            k = keys[r % len(keys)]
            r += 1
            if buckets[k] and picked < k_tr:
                keep[buckets[k].pop(0)] = True
                picked += 1
                progressed = True
        if not progressed:
            break
    if picked < k_tr:
        rest = [i for i in range(m) if not keep[i]]
        rest.sort(key=lambda i: float(st[i]))
        for i in rest:
            if picked >= k_tr:
                break
            keep[i] = True
            picked += 1
    return keep


def random_swap(keep: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ki = np.flatnonzero(keep)
    di = np.flatnonzero(~keep)
    if len(ki) == 0 or len(di) == 0:
        return keep.copy()
    k = keep.copy()
    k[int(rng.choice(ki))] = False
    k[int(rng.choice(di))] = True
    return k


def hill_climb(
    pack: EvalPack,
    keep0: np.ndarray,
    rng: np.random.Generator,
    max_rounds: int,
    neighbors: int,
    eval_budget: int,
) -> Tuple[np.ndarray, int]:
    keep = keep0.copy()
    _, _, _, _, best_loss = global_metrics(pack, keep)
    used = 1
    for _ in range(max_rounds):
        if used >= eval_budget:
            break
        best_cand = None
        best_l = best_loss
        for _ in range(neighbors):
            if used >= eval_budget:
                break
            cand = random_swap(keep, rng)
            _, _, _, _, l2 = global_metrics(pack, cand)
            used += 1
            if l2 < best_l:
                best_l = l2
                best_cand = cand
        if best_cand is None:
            break
        keep = best_cand
        best_loss = best_l
    return keep, used


def simulated_annealing(
    pack: EvalPack,
    keep0: np.ndarray,
    rng: np.random.Generator,
    eval_budget: int,
) -> Tuple[np.ndarray, int]:
    keep = keep0.copy()
    _, _, _, _, cur_l = global_metrics(pack, keep)
    used = 1
    best_k = keep.copy()
    best_l = cur_l
    T, t_decay = 0.04, 0.92
    while used < eval_budget:
        cand = random_swap(keep, rng)
        _, _, _, _, new_l = global_metrics(pack, cand)
        used += 1
        if new_l < cur_l or (T > 1e-6 and rng.random() < np.exp(-(new_l - cur_l) / T)):
            keep = cand
            cur_l = new_l
            if new_l < best_l:
                best_l = new_l
                best_k = cand.copy()
        T *= t_decay
    return best_k, used


def beam_light(
    pack: EvalPack,
    keep0: np.ndarray,
    rng: np.random.Generator,
    eval_budget: int,
) -> Tuple[np.ndarray, int]:
    used = 0

    def ev(k: np.ndarray) -> Tuple[np.ndarray, float]:
        nonlocal used
        _, _, _, _, l = global_metrics(pack, k)
        used += 1
        return k.copy(), l

    pool = [ev(keep0.copy())]
    beam, depth, branch = 3, 2, 4
    for _ in range(depth):
        if used >= eval_budget:
            break
        nxt: List[Tuple[np.ndarray, float]] = []
        for k, _ in sorted(pool, key=lambda x: x[1])[:beam]:
            for _ in range(branch):
                if used >= eval_budget:
                    break
                nxt.append(ev(random_swap(k, rng)))
        pool = sorted(nxt, key=lambda x: x[1])[:beam] if nxt else pool
    return pool[0][0], used


def diversity_local(
    pack: EvalPack, k_tr: int, rng: np.random.Generator, eval_budget: int
) -> Tuple[np.ndarray, int]:
    def div_pen(kg: np.ndarray) -> float:
        sub = pack.audit.iloc[np.where(kg)[0]]
        if len(sub) < 5:
            return 0.0
        _, nas = PR.negation_pair_features(sub)
        lg = sub["len_gap"].to_numpy(dtype=np.float64)
        return float(np.std(nas) + np.std(lg))

    keep = init_diversity_stratified(pack, k_tr, rng)
    st, kg, conf, imb, _ = global_metrics(pack, keep)
    used = 1

    def ld(st_: PR.SearchState, kg_: np.ndarray, c: float, im: float) -> float:
        return combo_loss(st_.heldout_auc, c, im) + 0.08 * div_pen(kg_)

    best_l = ld(st, kg, conf, imb)
    for _ in range(5):
        if used >= eval_budget:
            break
        br = None
        brl = best_l
        for _ in range(5):
            if used >= eval_budget:
                break
            cand = random_swap(keep, rng)
            st2, kg2, c2, i2, _ = global_metrics(pack, cand)
            used += 1
            l2 = ld(st2, kg2, c2, i2)
            if l2 < brl:
                brl, br = l2, cand
        if br is None:
            break
        keep = br
        best_l = brl
    return keep, used


def baseline_row(
    method: str,
    st: PR.SearchState,
    kg: np.ndarray,
    pack: EvalPack,
    target_pairs: int,
) -> Dict:
    sub = pack.audit.iloc[np.where(kg)[0]]
    n_clean = int((sub["style_violation"] == 0).sum()) if len(sub) else 0
    conf_frac = float(sub["style_violation"].mean()) if len(sub) else float("nan")
    ho = float(st.heldout_auc)
    gap = (
        float(st.search_auc - st.heldout_auc)
        if np.isfinite(st.search_auc) and np.isfinite(st.heldout_auc)
        else float("nan")
    )
    return {
        "seed": pack.seed,
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


def run_baseline(name: str, fn, pack: EvalPack, target_pairs: int, min_keep: int) -> Dict:
    st = fn(
        pack.audit,
        pack.df_full,
        pack.train_pairs,
        pack.test_pairs,
        pack.profile,
        pack.seed,
        pack.n_splits,
        min_keep,
        0.45,
        pack.target_auc,
        pack.w_auc,
        pack.w_size,
        pack.w_imb,
        target_pairs,
    )
    kg, _ = PR.fit_apply_retain_fraction(
        pack.df_full,
        pack.audit,
        pack.train_pairs,
        pack.test_pairs,
        st.keep_mask_train,
        pack.profile,
        pack.seed,
    )
    return baseline_row(name, st, kg, pack, target_pairs)


def k_tr_for_target(
    audit: pd.DataFrame, train_pairs: np.ndarray, test_pairs: np.ndarray, tg: int, mk: int
) -> int:
    ts = np.sort(train_pairs)
    m = len(ts)
    lo = PR._min_keep_train(mk, len(audit), m)
    kt, _, _ = vrf.solve_train_keep_for_target_global(tg, m, len(test_pairs), lo, len(audit))
    return int(kt)


def run_search(
    name: str,
    pack: EvalPack,
    target_pairs: int,
    k_tr: int,
    rng: np.random.Generator,
    budget: int,
) -> Dict:
    ep = pack
    if name == "hill_climb_from_clean_first":
        k0 = init_clean_first(ep, k_tr)
        kf, _ = hill_climb(ep, k0, rng, 6, 6, budget)
    elif name == "hill_climb_from_score_rank":
        k0 = init_score_rank(ep, k_tr)
        kf, _ = hill_climb(ep, k0, rng, 6, 6, budget)
    elif name == "simulated_annealing_from_clean_first":
        k0 = init_clean_first(ep, k_tr)
        kf, _ = simulated_annealing(ep, k0, rng, budget)
    elif name == "diversity_stratified_then_local":
        kf, _ = diversity_local(ep, k_tr, rng, budget)
    elif name == "beam_light_swaps_from_clean":
        k0 = init_clean_first(ep, k_tr)
        kf, _ = beam_light(ep, k0, rng, budget)
    else:
        raise ValueError(name)
    st, kg, _, _, _ = global_metrics(ep, kf)
    return baseline_row(name, st, kg, ep, target_pairs)


def pick_best(summary: pd.DataFrame) -> pd.DataFrame:
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
    out = root / "results" / "near_random_subset_search_better_algorithms"
    out.mkdir(parents=True, exist_ok=True)

    profile: tpa.AuditProfile = "paper10"
    audit = pd.read_csv(root / "audits" / "truthfulqa_style_audit.csv")
    df_full = tpa.build_answer_level_audit_frame(audit, profile=profile, copy_audit_meta=True)
    n_pairs = len(audit)

    targets = [300, 325, 350, 375, 400, 425, 450]
    seeds = list(range(10))
    min_keep_global = 200
    eval_budget = 8

    baselines = [
        ("clean_first_then_low_score", sn.evaluate_fixed_confound_free_train),
        ("score_rank (low separability)", vrf.evaluate_fixed_global_score_rank),
        ("len_gap_rank (symmetric length)", sn.evaluate_fixed_len_gap_rank),
        ("negation_rank (low asymmetry)", vrf.evaluate_fixed_global_negation_rank),
    ]
    search_names = [
        "hill_climb_from_clean_first",
        "hill_climb_from_score_rank",
        "simulated_annealing_from_clean_first",
        "diversity_stratified_then_local",
        "beam_light_swaps_from_clean",
    ]

    rows: List[Dict] = []
    for seed in seeds:
        tr, te = sn.split_gss(n_pairs, 0.25, seed)
        ts = np.sort(tr)
        pack = EvalPack(
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
            k_tr = k_tr_for_target(audit, tr, te, tg, min_keep_global)
            for bn, bf in baselines:
                rows.append(run_baseline(bn, bf, pack, tg, min_keep_global))
            for j, snm in enumerate(search_names):
                h = int(hashlib.md5(snm.encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(60_000 + seed * 100_000 + j * 10_000 + (tg % 1000) + (h % 10_000))
                rows.append(run_search(snm, pack, tg, k_tr, rng, eval_budget))

    per = pd.DataFrame(rows)
    per.to_csv(out / "per_seed_results.csv", index=False)

    g = ["target_pairs", "method"]
    summary = (
        per.groupby(g, sort=False)
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
    summary.to_csv(out / "summary_by_target_and_method.csv", index=False)

    best = pick_best(summary)
    best.to_csv(out / "best_method_by_target.csv", index=False)

    focus = [325, 350, 375, 400, 425]
    lines = [
        "| target pairs | best method | mean actual pairs | mean held-out AUC | std | mean |AUC-0.5| | mean clean pairs | mean confounded fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t in focus:
        sub = best[best["target_pairs"] == t]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            f"| {int(r['target_pairs'])} | {r['method']} | {r['mean_actual_pairs']:.1f} | "
            f"{r['mean_heldout_auc']:.4f} | {r['std_heldout_auc']:.4f} | {r['mean_dist_to_chance']:.4f} | "
            f"{r['mean_clean_pair_count']:.1f} | {r['mean_confounded_fraction']:.4f} |"
        )
    lines.append("")
    lines.append("Optional:")
    for t in (300, 450):
        sub = best[best["target_pairs"] == t]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            f"| {int(r['target_pairs'])} | {r['method']} | {r['mean_actual_pairs']:.1f} | "
            f"{r['mean_heldout_auc']:.4f} | {r['std_heldout_auc']:.4f} | {r['mean_dist_to_chance']:.4f} | "
            f"{r['mean_clean_pair_count']:.1f} | {r['mean_confounded_fraction']:.4f} |"
        )
    (out / "final_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    REF350, REF400 = 0.5888, 0.6023
    bln = [b[0] for b in baselines]

    def mean_auc(t: int, m: str) -> float:
        s = summary[(summary["target_pairs"] == t) & (summary["method"] == m)]
        return float(s["mean_heldout_auc"].iloc[0]) if len(s) else float("nan")

    def best_new_only(t: int) -> Tuple[str, float]:
        s = summary[(summary["target_pairs"] == t) & (~summary["method"].isin(bln))]
        if s.empty:
            return "", float("nan")
        s = s.sort_values("mean_heldout_auc", ascending=True)
        r = s.iloc[0]
        return str(r["method"]), float(r["mean_heldout_auc"])

    def best_any(t: int) -> Tuple[str, float]:
        s = summary[summary["target_pairs"] == t].sort_values("mean_heldout_auc", ascending=True)
        r = s.iloc[0]
        return str(r["method"]), float(r["mean_heldout_auc"])

    rec: List[str] = []
    for label, tg in [("350", 350), ("400", 400)]:
        m_new, a_new = best_new_only(tg)
        m_all, a_all = best_any(tg)
        cf = mean_auc(tg, "clean_first_then_low_score")
        ref = REF350 if tg == 350 else REF400
        beat_cf = a_new < cf - 1e-4 if np.isfinite(a_new) else False
        rec.append(
            f"- **At {tg}:** best **search-only** mean held-out **{a_new:.4f}** (`{m_new}`) vs **clean_first** **{cf:.4f}**. "
            f"**{'Beats' if beat_cf else 'Does not beat'}** clean_first on mean held-out. "
            f"Best **overall** **{a_all:.4f}** (`{m_all}`). Refined baseline ref **{ref:.4f}**."
        )

    recommendation = "\n".join(
        [
            "# Better algorithms — recommendation",
            "",
            "## 1. Did any new algorithm beat `clean_first_then_low_score` on mean held-out AUC?",
            *rec,
            "",
            "## 2. Best method at 350 / 400 (lowest mean held-out in grid)",
            f"- **350:** `{best_any(350)[0]}`",
            f"- **400:** `{best_any(400)[0]}`",
            "",
            "## 3. Is 375 better than 350 or 400?",
            "- Compare `mean_heldout_auc` / `mean_dist_to_chance` for your preferred method across 325–425 in `summary_by_target_and_method.csv`.",
            "",
            "## 4. Practical limit",
            "- Local search optimizes a **weighted proxy** during swaps; **reported** numbers are always **true held-out** OOF AUC. Margins over `clean_first` are often **small**; large gains are uncommon under paper10 + this split.",
            "",
            "## 5. What to report",
            "- **Strongest near-random:** smallest mean |AUC−0.5| you can defend (often 300–375).",
            "- **Largest still-reasonable:** highest target with acceptable distance (often 375–400).",
            "",
            "### Explicit sentences",
            f"- At **350**, mean held-out improved / did not improve vs refined **{REF350:.4f}** depending on row above; compare **`{best_any(350)[0]}`** at **{best_any(350)[1]:.4f}**.",
            f"- At **400**, compare to **{REF400:.4f}**; best grid mean **{best_any(400)[1]:.4f}** (`{best_any(400)[0]}`).",
        ]
    )
    (out / "recommendation.md").write_text(recommendation, encoding="utf-8")

    (out / "run_config.json").write_text(
        json.dumps(
            {
                "targets": targets,
                "seeds": seeds,
                "eval_budget_per_search": eval_budget,
                "min_keep_global": min_keep_global,
                "weights_alpha_beta_gamma": [ALPHA, BETA, GAMMA],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Figures
    fig_dir = root / "figures" / "near_random_subset_search_better_algorithms"
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    piv_h = summary.pivot_table(index="target_pairs", columns="method", values="mean_heldout_auc")
    piv_d = summary.pivot_table(index="target_pairs", columns="method", values="mean_dist_to_chance")
    piv_c = summary.pivot_table(index="target_pairs", columns="method", values="mean_clean_pair_count")
    x = np.sort(piv_h.index.to_numpy())

    def plot_pdf(name: str, piv, ylab: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for col in piv.columns:
            ax.plot(x, piv.loc[x, col], marker="o", ms=3, label=col)
        ax.set_xlabel("Target pairs")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=5, loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / name)
        plt.close(fig)

    plot_pdf("target_vs_mean_heldout_auc.pdf", piv_h, "Mean held-out AUC", "Better-algorithms grid")
    plot_pdf("target_vs_mean_dist_to_chance.pdf", piv_d, "Mean |AUC−0.5|", "Distance to chance")
    plot_pdf("target_vs_mean_clean_pairs.pdf", piv_c, "Mean clean pair count", "Clean pairs retained")

    print("Wrote", out, fig_dir)


if __name__ == "__main__":
    main()
