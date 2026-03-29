#!/usr/bin/env python3
"""
Improved TruthfulQA audit subset search (pruning) under the paper-compatible pipeline.

Default audit: 10-feature answer-level grouped CV (see truthfulqa_paper_audit.py).

Evaluation protocol (honest reporting):
- Pair-level train/test split (default 75/25). Pruning decisions use **train pairs only**
  for greedy / threshold tuning.
- A **global** retained set is formed by applying the same **score ranking** (from a
  logistic model fit on train answer-rows only) to test pairs with the **same retain
  fraction** as on train, so test labels are not used to choose which test pairs drop.
- **search_time_auc**: OOF audit AUC on retained **train** pairs only.
- **heldout_auc**: OOF audit AUC on retained **test** pairs only.
- **optimism_gap**: search_time_auc - heldout_auc (reported explicitly; do not cite
  search-time alone as the final number).

Outputs under results/truthfulqa_pruning_improved/ and figures/truthfulqa_pruning_improved/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt
import truthfulqa_paper_audit as tpa
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Pair-level helpers (pruning operates on whole pairs)
# ---------------------------------------------------------------------------


def _pair_ids_from_mask(n_pairs: int, keep: np.ndarray) -> np.ndarray:
    return np.where(keep)[0]


def subset_df_ans(df_ans: pd.DataFrame, keep_pairs: np.ndarray) -> pd.DataFrame:
    s = set(int(x) for x in keep_pairs)
    return df_ans[df_ans["pair_id"].isin(s)].copy()


def audit_auc_safe(
    df_sub: pd.DataFrame,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
) -> float:
    if df_sub.empty or df_sub["pair_id"].nunique() < n_splits:
        return float("nan")
    r = tpa.paper_compatible_audit_oof_auc(
        df_sub, profile=profile, seed=seed, n_splits=n_splits
    )
    return r.auc_oof


def pair_separability_scores(
    df_ans: pd.DataFrame,
    profile: tpa.AuditProfile,
    seed: int,
) -> np.ndarray:
    """
    Higher score => model (fit on this df) is more confident on that pair => stronger
    surface-form separability cue (approximate removal priority for backward elimination).
    """
    groups = df_ans["pair_id"].to_numpy(dtype=np.int64)
    n_pairs = int(groups.max()) + 1 if len(groups) else 0
    cols = tpa.feature_columns_for_profile(profile)
    X = df_ans[cols].fillna(0).to_numpy()
    y = df_ans["label"].to_numpy()
    pipe = tpa.make_lr_pipeline(seed)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    scores = np.zeros(max(n_pairs, 1), dtype=np.float64)
    counts = np.zeros(max(n_pairs, 1), dtype=np.int32)
    for i, g in enumerate(groups):
        gi = int(g)
        scores[gi] += abs(float(proba[i]) - 0.5)
        counts[gi] += 1
    counts = np.maximum(counts, 1)
    return scores / counts


def pair_scores_with_pipe(df_full: pd.DataFrame, profile: tpa.AuditProfile, pipe) -> np.ndarray:
    """Mean |p-0.5| per pair_id using a pre-fit sklearn pipeline (train on subset, score all rows)."""
    cols = tpa.feature_columns_for_profile(profile)
    X = df_full[cols].fillna(0).to_numpy()
    proba = pipe.predict_proba(X)[:, 1]
    groups = df_full["pair_id"].to_numpy(dtype=np.int64)
    n_pairs = int(groups.max()) + 1 if len(groups) else 0
    scores = np.zeros(max(n_pairs, 1), dtype=np.float64)
    counts = np.zeros(max(n_pairs, 1), dtype=np.int32)
    for i in range(len(groups)):
        g = int(groups[i])
        scores[g] += abs(float(proba[i]) - 0.5)
        counts[g] += 1
    counts = np.maximum(counts, 1)
    return scores / counts


def negation_pair_features(audit: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (neg_any_rate, neg_asymmetry) per pair for negation-first modes."""
    nl_t = audit["neg_lead_true"].to_numpy()
    nl_f = audit["neg_lead_false"].to_numpy()
    nc_t = audit["neg_cnt_true"].to_numpy()
    nc_f = audit["neg_cnt_false"].to_numpy()
    any_neg = ((nl_t + nl_f + nc_t + nc_f) > 0).astype(np.float64)
    asym = (np.abs(nl_t.astype(float) - nl_f) + np.abs(nc_t.astype(float) - nc_f)).astype(
        np.float64
    )
    return any_neg, asym


def length_pair_feature(audit: pd.DataFrame) -> np.ndarray:
    return audit["len_gap"].to_numpy(dtype=np.float64)


def bucketize(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 2:
        return np.zeros(len(x), dtype=np.int64)
    return np.digitize(x, edges[1:-1], right=False).astype(np.int64)


def pair_bucket_signature(audit: pd.DataFrame) -> np.ndarray:
    _, nasym = negation_pair_features(audit)
    neg_b = bucketize(nasym, 4)
    len_b = bucketize(audit["len_gap"].to_numpy(), 4)
    hd = np.abs(audit["hedge_rate_true"] - audit["hedge_rate_false"]).to_numpy()
    ad = np.abs(audit["auth_rate_true"] - audit["auth_rate_false"]).to_numpy()
    hb = bucketize(hd, 4)
    ab = bucketize(ad, 4)
    return neg_b + 100 * len_b + 10000 * hb + 1000000 * ab


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def imbalance_penalty_from_audit(
    audit: pd.DataFrame, keep: np.ndarray, full_audit: pd.DataFrame
) -> float:
    """L1 distance of retained marginal bucket histogram vs full dataset (per feature proxy)."""
    if not keep.any():
        return 10.0
    sub = audit.iloc[np.where(keep)[0]]
    full = full_audit
    pen = 0.0
    for col in ["len_gap"]:
        h, _ = np.histogram(sub[col], bins=10, range=(full[col].min(), full[col].max()))
        hf, _ = np.histogram(full[col], bins=10, range=(full[col].min(), full[col].max()))
        hf = np.maximum(hf.astype(float), 1e-6)
        h = h.astype(float) / max(h.sum(), 1e-6)
        hf = hf / hf.sum()
        pen += float(np.sum(np.abs(h - hf)))
    na_sub = negation_pair_features(sub)[0].mean()
    na_full = negation_pair_features(full)[0].mean()
    pen += abs(na_sub - na_full)
    if "Category" in full_audit.columns and len(sub):
        vc = sub["Category"].value_counts(normalize=True)
        vf = full_audit["Category"].value_counts(normalize=True)
        for c in vf.index:
            pen += abs(float(vc.get(c, 0.0)) - float(vf.get(c, 0.0)))
    return pen


def composite_objective(
    heldout_auc: float,
    n_retained: int,
    n_total: int,
    imb_pen: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
    target_auc: float,
) -> float:
    if not np.isfinite(heldout_auc):
        return 1e9
    auc_pen = max(0.0, heldout_auc - target_auc)
    size_pen = 1.0 - (n_retained / max(n_total, 1))
    return w_auc * auc_pen + w_size * size_pen + w_imb * imb_pen


# ---------------------------------------------------------------------------
# Apply global mask via train-fitted scores (test generalization)
# ---------------------------------------------------------------------------


def fit_apply_retain_fraction(
    df_full: pd.DataFrame,
    audit: pd.DataFrame,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    keep_train: np.ndarray,
    profile: tpa.AuditProfile,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    keep_train: bool vector aligned with sorted(train_pairs); True = retain that train pair.

    Fit logistic regression on **retained train** answer rows only, score every pair in the
    dataset by mean |p-0.5| on its two answer rows. Retain test pairs with the **lowest**
    scores until the test count matches round(frac * |test|) where frac = retained_train/|train|.
    """
    n_pairs = len(audit)
    keep_global = np.zeros(n_pairs, dtype=bool)
    train_sorted = np.sort(train_pairs)
    assert len(keep_train) == len(train_sorted)
    kt = np.zeros(n_pairs, dtype=bool)
    for i, pid in enumerate(train_sorted):
        if keep_train[i]:
            kt[int(pid)] = True

    kept_train_pids = train_sorted[keep_train]
    df_fit = subset_df_ans(df_full, kept_train_pids)
    scores = np.zeros(n_pairs, dtype=np.float64)
    if df_fit["pair_id"].nunique() >= 5:
        cols = tpa.feature_columns_for_profile(profile)
        pipe = tpa.make_lr_pipeline(seed)
        pipe.fit(df_fit[cols].fillna(0).to_numpy(), df_fit["label"].to_numpy())
        scores = pair_scores_with_pipe(df_full, profile, pipe)[:n_pairs]
    else:
        df_tr_all = subset_df_ans(df_full, train_sorted)
        ps = pair_separability_scores(df_tr_all, profile, seed)
        scores = ps[:n_pairs]

    for pid in train_sorted:
        if kt[int(pid)]:
            keep_global[int(pid)] = True

    test_sorted = np.sort(test_pairs)
    n_keep_tr = int(keep_train.sum())
    frac = n_keep_tr / max(len(train_sorted), 1)
    n_keep_te = int(round(frac * len(test_sorted)))
    n_keep_te = max(0, min(n_keep_te, len(test_sorted)))
    if n_keep_te == 0 and len(test_sorted) > 0:
        n_keep_te = min(1, len(test_sorted))

    te_scores = [(float(scores[int(p)]), int(p)) for p in test_sorted]
    te_scores.sort(key=lambda x: x[0])
    kept_test = {p for _, p in te_scores[:n_keep_te]}
    for p in kept_test:
        keep_global[int(p)] = True

    return keep_global, scores


def refine_keep_train_only(
    keep_global: np.ndarray,
    train_pairs: np.ndarray,
) -> np.ndarray:
    """Restrict mask to train decisions only (for search-time AUC)."""
    n = len(keep_global)
    kt = np.zeros(n, dtype=bool)
    for p in train_pairs:
        if keep_global[int(p)]:
            kt[int(p)] = True
    return kt


# ---------------------------------------------------------------------------
# Search methods (train pairs only; indices into sorted(train_pairs))
# ---------------------------------------------------------------------------


@dataclass
class SearchState:
    keep_mask_train: np.ndarray  # bool, len = len(train_pairs)
    search_auc: float
    heldout_auc: float
    objective: float
    meta: Dict = field(default_factory=dict)


def evaluate_state(
    df_full: pd.DataFrame,
    audit: pd.DataFrame,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    keep_train_vec: np.ndarray,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
    w_auc: float,
    w_size: float,
    w_imb: float,
    target_auc: float,
) -> SearchState:
    n_pairs = len(audit)
    keep_global, _ = fit_apply_retain_fraction(
        df_full,
        audit,
        train_pairs,
        test_pairs,
        keep_train_vec,
        profile,
        seed,
    )
    tr_kept = refine_keep_train_only(keep_global, train_pairs)
    te_only = np.zeros(n_pairs, dtype=bool)
    for p in test_pairs:
        te_only[int(p)] = keep_global[int(p)]

    df_tr = subset_df_ans(df_full, np.where(tr_kept)[0])
    df_te = subset_df_ans(df_full, np.where(te_only)[0])
    s_auc = audit_auc_safe(df_tr, profile, seed, n_splits)
    h_auc = audit_auc_safe(df_te, profile, seed, n_splits)
    imb = imbalance_penalty_from_audit(audit, keep_global, audit)
    n_ret = int(keep_global.sum())
    obj = composite_objective(
        h_auc, n_ret, n_pairs, imb, w_auc, w_size, w_imb, target_auc
    )
    return SearchState(
        keep_mask_train=keep_train_vec,
        search_auc=s_auc,
        heldout_auc=h_auc,
        objective=obj,
        meta={},
    )


def _min_keep_train(min_keep_global: int, n_pairs: int, m_train: int) -> int:
    return max(5, int(np.ceil(min_keep_global * (m_train / max(n_pairs, 1)))))


def method_quantile_score(
    audit: pd.DataFrame,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
    min_keep: int,
    max_drop_fraction: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
    drop_quantiles: Sequence[float],
) -> SearchState:
    """
    Remove a **fraction** of train pairs with the highest train-fitted separability scores
    (batch removal). Tries several quantile cutoffs and keeps the best composite objective.
    """
    train_sorted = np.sort(train_pairs)
    m = len(train_sorted)
    mk = _min_keep_train(min_keep, len(audit), m)
    df_tr = subset_df_ans(df_full, train_sorted)
    sc_all = pair_separability_scores(df_tr, profile, seed)
    st_scores = sc_all[train_sorted]

    baseline = evaluate_state(
        df_full,
        audit,
        train_pairs,
        test_pairs,
        np.ones(m, dtype=bool),
        profile,
        seed,
        n_splits,
        w_auc,
        w_size,
        w_imb,
        target_auc,
    )
    best_obj = baseline
    max_drop = int(max_drop_fraction * len(audit))
    best_heldout_pruned: Optional[SearchState] = None
    # Unique train drop counts only: several quantiles can map to the same drop_n after rounding,
    # which collapses the effective search grid (redundant evals, not different keep sizes).
    drop_ns: Set[int] = set()
    for q in drop_quantiles:
        dn = int(round(float(q) * m))
        dn = max(0, min(dn, max_drop, m - mk))
        if dn <= 0:
            continue
        if m - dn >= mk:
            drop_ns.add(dn)
    for drop_n in sorted(drop_ns):
        order = np.argsort(-st_scores)
        keep = np.ones(m, dtype=bool)
        keep[order[:drop_n]] = False
        st = evaluate_state(
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
        if st.objective < best_obj.objective and np.isfinite(st.heldout_auc):
            best_obj = st
        if np.isfinite(st.heldout_auc) and int(keep.sum()) < m:
            if best_heldout_pruned is None or st.heldout_auc < best_heldout_pruned.heldout_auc:
                best_heldout_pruned = st
    # Prefer a real pruning run that minimizes held-out leakage; composite alone often keeps full data.
    if best_heldout_pruned is not None:
        return best_heldout_pruned
    return best_obj


def method_negation_first(
    audit: pd.DataFrame,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
    min_keep: int,
    max_drop_fraction: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
    mode: str,
) -> SearchState:
    n_pairs = len(audit)
    train_sorted = np.sort(train_pairs)
    m = len(train_sorted)
    mk = _min_keep_train(min_keep, n_pairs, m)
    _, nasym = negation_pair_features(audit)
    order = np.argsort(-nasym[train_sorted])

    keep = np.ones(m, dtype=bool)
    best = evaluate_state(
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

    max_drop = int(max_drop_fraction * n_pairs)
    dropped = 0
    no_improve = 0
    for j in order:
        if int(keep.sum()) <= mk:
            break
        if dropped >= max_drop:
            break
        if mode in ("length_only",):
            break
        keep_try = keep.copy()
        keep_try[int(j)] = False
        st = evaluate_state(
            df_full,
            audit,
            train_pairs,
            test_pairs,
            keep_try,
            profile,
            seed,
            n_splits,
            w_auc,
            w_size,
            w_imb,
            target_auc,
        )
        if st.objective < best.objective and np.isfinite(st.heldout_auc):
            best = st
            keep = keep_try
            dropped += 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 25:
                break
    return best


def method_length_focus(
    audit: pd.DataFrame,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
    min_keep: int,
    max_drop_fraction: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
    initial_keep: Optional[np.ndarray] = None,
) -> SearchState:
    train_sorted = np.sort(train_pairs)
    m = len(train_sorted)
    mk = _min_keep_train(min_keep, len(audit), m)
    lg = length_pair_feature(audit)
    order = np.argsort(-lg[train_sorted])
    keep = initial_keep.copy() if initial_keep is not None else np.ones(m, dtype=bool)
    if keep.shape[0] != m:
        keep = np.ones(m, dtype=bool)
    best = evaluate_state(
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
    max_drop = int(max_drop_fraction * len(audit))
    dropped = 0
    no_improve = 0
    for j in order:
        if int(keep.sum()) <= mk or dropped >= max_drop:
            break
        if not keep[int(j)]:
            continue
        keep_try = keep.copy()
        keep_try[int(j)] = False
        st = evaluate_state(
            df_full,
            audit,
            train_pairs,
            test_pairs,
            keep_try,
            profile,
            seed,
            n_splits,
            w_auc,
            w_size,
            w_imb,
            target_auc,
        )
        if st.objective < best.objective and np.isfinite(st.heldout_auc):
            best = st
            keep = keep_try
            dropped += 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 35:
                break
    return best


def method_feature_balanced(
    audit: pd.DataFrame,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
    min_keep: int,
    max_drop_fraction: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
) -> SearchState:
    """Greedily drop pairs in over-represented bucket cells vs global prior."""
    train_sorted = np.sort(train_pairs)
    m = len(train_sorted)
    mk = _min_keep_train(min_keep, len(audit), m)
    sig_full = pair_bucket_signature(audit)
    full_hist: Dict[int, float] = {}
    for p in range(len(audit)):
        full_hist[int(sig_full[p])] = full_hist.get(int(sig_full[p]), 0) + 1
    total = float(len(audit))
    for k in list(full_hist):
        full_hist[k] /= total

    keep = np.ones(m, dtype=bool)
    best = evaluate_state(
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
    max_drop = int(max_drop_fraction * len(audit))
    dropped = 0
    while dropped < max_drop and int(keep.sum()) > mk:
        active = np.where(keep)[0]
        if len(active) == 0:
            break
        sub_hist: Dict[int, int] = {}
        for j in active:
            pid = int(train_sorted[j])
            s = int(sig_full[pid])
            sub_hist[s] = sub_hist.get(s, 0) + 1
        sub_total = float(len(active))
        worst_j = None
        worst_excess = -1.0
        for j in active:
            pid = int(train_sorted[j])
            s = int(sig_full[pid])
            obs = sub_hist[s] / sub_total
            exp = full_hist.get(s, 1e-6)
            excess = obs / max(exp, 1e-6)
            if excess > worst_excess:
                worst_excess = excess
                worst_j = j
        if worst_j is None:
            break
        keep_try = keep.copy()
        keep_try[worst_j] = False
        st = evaluate_state(
            df_full,
            audit,
            train_pairs,
            test_pairs,
            keep_try,
            profile,
            seed,
            n_splits,
            w_auc,
            w_size,
            w_imb,
            target_auc,
        )
        if st.objective < best.objective:
            best = st
            keep = keep_try
            dropped += 1
        else:
            break
    return best


def method_backward_elimination(
    audit: pd.DataFrame,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
    min_keep: int,
    max_drop_fraction: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
) -> SearchState:
    train_sorted = np.sort(train_pairs)
    m = len(train_sorted)
    mk = _min_keep_train(min_keep, len(audit), m)
    df_tr_all = subset_df_ans(df_full, train_sorted)
    scores = pair_separability_scores(df_tr_all, profile, seed)
    order = np.argsort(-scores[train_sorted])

    keep = np.ones(m, dtype=bool)
    best = evaluate_state(
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
    max_drop = int(max_drop_fraction * len(audit))
    dropped = 0
    no_improve = 0
    for j in order:
        if int(keep.sum()) <= mk or dropped >= max_drop:
            break
        if not keep[int(j)]:
            continue
        keep_try = keep.copy()
        keep_try[int(j)] = False
        st = evaluate_state(
            df_full,
            audit,
            train_pairs,
            test_pairs,
            keep_try,
            profile,
            seed,
            n_splits,
            w_auc,
            w_size,
            w_imb,
            target_auc,
        )
        if st.objective < best.objective and np.isfinite(st.heldout_auc):
            best = st
            keep = keep_try
            dropped += 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 35:
                break
    return best


def method_beam_multistart(
    audit: pd.DataFrame,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    df_full: pd.DataFrame,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
    min_keep: int,
    max_drop_fraction: float,
    target_auc: float,
    w_auc: float,
    w_size: float,
    w_imb: float,
    beam_width: int,
    n_starts: int,
) -> SearchState:
    rng = np.random.default_rng(seed)
    train_sorted = np.sort(train_pairs)
    m = len(train_sorted)
    mk = _min_keep_train(min_keep, len(audit), m)
    df_tr_all = subset_df_ans(df_full, train_sorted)
    base_scores = pair_separability_scores(df_tr_all, profile, seed)

    best_global = evaluate_state(
        df_full,
        audit,
        train_pairs,
        test_pairs,
        np.ones(m, dtype=bool),
        profile,
        seed,
        n_splits,
        w_auc,
        w_size,
        w_imb,
        target_auc,
    )
    max_drop = int(max_drop_fraction * len(audit))

    # Multi-start greedy; optional beam is approximated by keeping the best few
    # trajectories when tie-breaking on objective (stable, low complexity).
    for s in range(n_starts):
        noise = rng.standard_normal(m)
        order = np.argsort(-(base_scores[train_sorted] + 0.05 * noise))
        keep = np.ones(m, dtype=bool)
        cur = evaluate_state(
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
        dropped = 0
        no_improve = 0
        for j in order:
            if int(keep.sum()) <= mk or dropped >= max_drop:
                break
            if not keep[int(j)]:
                continue
            keep_try = keep.copy()
            keep_try[int(j)] = False
            st = evaluate_state(
                df_full,
                audit,
                train_pairs,
                test_pairs,
                keep_try,
                profile,
                seed,
                n_splits,
                w_auc,
                w_size,
                w_imb,
                target_auc,
            )
            if st.objective < cur.objective and np.isfinite(st.heldout_auc):
                cur = st
                keep = keep_try
                dropped += 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 30:
                    break
        if cur.objective < best_global.objective:
            best_global = cur

    _ = beam_width  # reserved for future wider beam search
    return best_global


def method_pareto_collection(
    states: List[SearchState],
    audit: pd.DataFrame,
    df_full: pd.DataFrame,
    keep_masks: List[np.ndarray],
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    profile: tpa.AuditProfile,
    seed: int,
    n_splits: int,
) -> pd.DataFrame:
    rows = []
    n_pairs = len(audit)
    for st, km in zip(states, keep_masks):
        kg, _ = fit_apply_retain_fraction(
            df_full, audit, train_pairs, test_pairs, km, profile, seed
        )
        conf_frac = float(audit.loc[kg, "style_violation"].mean()) if kg.any() else float("nan")
        rows.append(
            {
                "heldout_auc": st.heldout_auc,
                "retained_count": int(kg.sum()),
                "confounded_fraction": conf_frac,
                "search_auc": st.search_auc,
                "objective": st.objective,
            }
        )
    pdf = pd.DataFrame(rows)
    if pdf.empty:
        return pdf
    pdf = pdf[np.isfinite(pdf["heldout_auc"])]
    pareto = []
    for i, r in pdf.iterrows():
        dominated = False
        for j, r2 in pdf.iterrows():
            if i == j:
                continue
            if (
                r2["heldout_auc"] <= r["heldout_auc"]
                and r2["retained_count"] >= r["retained_count"]
                and r2["confounded_fraction"] <= r["confounded_fraction"]
                and (
                    r2["heldout_auc"] < r["heldout_auc"]
                    or r2["retained_count"] > r["retained_count"]
                    or r2["confounded_fraction"] < r["confounded_fraction"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pdf.loc[pareto].sort_values(["heldout_auc", "retained_count"])


# ---------------------------------------------------------------------------
# Diagnostics & IO
# ---------------------------------------------------------------------------


def retained_diagnostics(
    audit: pd.DataFrame,
    keep_global: np.ndarray,
    method: str,
    mode: str,
    profile: str,
) -> Dict[str, object]:
    sub = audit.iloc[np.where(keep_global)[0]].copy()
    dropped = int((~keep_global).sum())
    n_any_neg, _ = negation_pair_features(sub)
    neg_rate = float(np.mean(n_any_neg)) if len(sub) else float("nan")
    return {
        "retained_count": int(keep_global.sum()),
        "dropped_count": dropped,
        "retained_confounded_fraction": float(sub["style_violation"].mean())
        if len(sub)
        else float("nan"),
        "retained_negation_rate": neg_rate,
        "retained_length_gap_mean": float(sub["len_gap"].mean()) if len(sub) else float("nan"),
        "retained_hedge_diff_mean": float(
            np.abs(sub["hedge_rate_true"] - sub["hedge_rate_false"]).mean()
        )
        if len(sub)
        else float("nan"),
        "retained_authority_diff_mean": float(
            np.abs(sub["auth_rate_true"] - sub["auth_rate_false"]).mean()
        )
        if len(sub)
        else float("nan"),
        "retained_label_balance": 0.5,
        "method": method,
        "mode": mode,
        "audit_profile": profile,
    }


def build_dropped_explanations(
    audit: pd.DataFrame,
    keep_global: np.ndarray,
    scores: np.ndarray,
    method: str,
    profile: tpa.AuditProfile,
) -> pd.DataFrame:
    dropped_ids = np.where(~keep_global)[0]
    rows = []
    sig = pair_bucket_signature(audit)
    for rank, pid in enumerate(sorted(dropped_ids, key=lambda p: -scores[p])):
        r = audit.iloc[int(pid)]
        rows.append(
            {
                "example_id": int(pid),
                "pair_id": int(pid),
                "drop_rank": rank,
                "method": method,
                "estimated_removal_contribution": float(scores[int(pid)]),
                "neg_asymmetry": float(
                    abs(r["neg_lead_true"] - r["neg_lead_false"])
                    + abs(r["neg_cnt_true"] - r["neg_cnt_false"])
                ),
                "len_gap": float(r["len_gap"]),
                "style_violation": int(r["style_violation"]),
                "bucket_sig": int(sig[int(pid)]),
                "drop_reason": f"pruned_by_{method}_train_score_rank",
            }
        )
    return pd.DataFrame(rows)


def run_model_eval_hook(
    root: Path,
    best_ids: List[int],
    predictions_glob: str,
) -> Dict[str, object]:
    pred_dir = root / "data" / "predictions"
    paths = list(pred_dir.glob(predictions_glob)) if pred_dir.exists() else []
    missing_summary: Dict[str, object] = {"n_prediction_files": len(paths), "per_file": []}
    best_set = set(int(x) for x in best_ids)
    for p in paths[:20]:
        try:
            df = pd.read_csv(p)
        except Exception as e:  # pragma: no cover
            missing_summary["per_file"].append({"file": str(p), "error": str(e)})
            continue
        if "pair_id" not in df.columns:
            continue
        ids = set(df["pair_id"].astype(int).unique())
        miss = sorted(best_set - ids)
        missing_summary["per_file"].append(
            {"file": str(p), "missing_best_subset_ids_count": len(miss), "sample_missing": miss[:5]}
        )
    return missing_summary


# ---------------------------------------------------------------------------
# Default pruning methods (public names for CSV / summaries)
# ---------------------------------------------------------------------------

DEFAULT_PRUNING_METHODS: Tuple[str, ...] = (
    "negation_first_constrained",
    "feature_balanced",
    "score_based_greedy",
    "beam_or_multistart_greedy",
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Improved TruthfulQA audit pruning search")
    ap.add_argument("--root", type=str, default=".", help="Repo root")
    ap.add_argument(
        "--audit-csv",
        type=str,
        default="audits/truthfulqa_style_audit.csv",
        help="Audit features CSV",
    )
    ap.add_argument(
        "--audit-profile",
        type=str,
        default="surface10",
        choices=["surface10", "surface13", "paper10", "expanded13"],
        help="surface10 = default ten-feature surface audit (alias paper10); surface13 adds pair-level extras (alias expanded13).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-fraction", type=float, default=0.75)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--min-keep", type=int, default=500)
    ap.add_argument("--target-audit-auc", type=float, default=0.62)
    ap.add_argument("--max-drop-fraction", type=float, default=0.40)
    ap.add_argument("--w-auc", type=float, default=2.0)
    ap.add_argument("--w-size", type=float, default=1.0)
    ap.add_argument("--w-imb", type=float, default=0.5)
    ap.add_argument("--beam-width", type=int, default=4)
    ap.add_argument("--n-multistarts", type=int, default=5)
    ap.add_argument(
        "--methods",
        type=str,
        default="all",
        help=(
            "Comma-separated list or 'all'. Default 'all' runs only: "
            + ", ".join(DEFAULT_PRUNING_METHODS)
            + ". Use --include-quantile-score to add quantile_score when methods=all."
        ),
    )
    ap.add_argument(
        "--include-quantile-score",
        action="store_true",
        help="Opt-in only: add quantile_score to the run when --methods all (never in default).",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="all",
        help="Comma list: negation_only,length_only,neg_length,all_features — or 'all'",
    )
    ap.add_argument(
        "--run-model-eval",
        action="store_true",
        help="Check best_subset_ids alignment against data/predictions/**/*.csv",
    )
    ap.add_argument(
        "--predictions-glob",
        type=str,
        default="**/*.csv",
        help="Glob under data/predictions for --run-model-eval",
    )
    ap.add_argument(
        "--quantile-grid",
        type=str,
        default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5",
        help="Comma-separated drop fractions for quantile_score method (train pairs)",
    )
    args = ap.parse_args()
    profile = tpa.normalize_audit_profile(str(args.audit_profile))
    quantile_grid = tuple(float(x.strip()) for x in args.quantile_grid.split(",") if x.strip())
    if args.include_quantile_score and args.methods.strip().lower() != "all":
        raise SystemExit("--include-quantile-score is only supported with --methods all")
    root = Path(args.root).resolve()
    audit_path = (root / args.audit_csv).resolve()
    if not audit_path.exists():
        raise SystemExit(f"Missing audit CSV: {audit_path}")

    audit = pd.read_csv(audit_path)
    df_full = tpa.build_answer_level_audit_frame(audit, profile=profile, copy_audit_meta=True)
    feat_cols = tpa.feature_columns_for_profile(profile)

    n_pairs = len(audit)
    rng = np.random.default_rng(args.seed)
    all_pairs = np.arange(n_pairs, dtype=np.int64)
    rng.shuffle(all_pairs)
    n_tr = int(args.train_fraction * n_pairs)
    train_pairs = np.sort(all_pairs[:n_tr])
    test_pairs = np.sort(all_pairs[n_tr:])

    out_dir = root / "results" / "truthfulqa_pruning_improved"
    fig_dir = root / "figures" / "truthfulqa_pruning_improved"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[truthfulqa_pruning_improved] audit_profile={profile} "
        f"feature_columns={feat_cols} n_pairs={n_pairs} "
        f"train_pairs={len(train_pairs)} test_pairs={len(test_pairs)}"
    )

    if args.methods.strip().lower() == "all":
        methods = list(DEFAULT_PRUNING_METHODS)
        if args.include_quantile_score:
            methods = methods + ["quantile_score"]
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    modes_all = ["negation_only", "length_only", "neg_length", "all_features"]
    if args.modes.strip().lower() == "all":
        modes = modes_all
    else:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    results_rows: List[Dict] = []
    pareto_inputs: List[Tuple[SearchState, np.ndarray]] = []
    best_obj = float("inf")
    best_pack: Optional[Tuple[SearchState, np.ndarray, str, str, np.ndarray]] = None

    def run_for_mode(
        mode: str,
        method: str,
    ) -> Optional[Tuple[SearchState, np.ndarray, np.ndarray]]:
        if method == "quantile_score":
            if mode in ("negation_only", "length_only"):
                return None
            st = method_quantile_score(
                audit,
                train_pairs,
                test_pairs,
                df_full,
                profile,
                args.seed,
                args.cv_splits,
                args.min_keep,
                args.max_drop_fraction,
                args.target_audit_auc,
                args.w_auc,
                args.w_size,
                args.w_imb,
                quantile_grid,
            )
        elif method == "negation_first_constrained":
            if mode in ("length_only",):
                return None
            st = method_negation_first(
                audit,
                train_pairs,
                test_pairs,
                df_full,
                profile,
                args.seed,
                args.cv_splits,
                args.min_keep,
                args.max_drop_fraction,
                args.target_audit_auc,
                args.w_auc,
                args.w_size,
                args.w_imb,
                mode,
            )
        elif method == "feature_balanced":
            if mode not in ("all_features", "neg_length"):
                return None
            st = method_feature_balanced(
                audit,
                train_pairs,
                test_pairs,
                df_full,
                profile,
                args.seed,
                args.cv_splits,
                args.min_keep,
                args.max_drop_fraction,
                args.target_audit_auc,
                args.w_auc,
                args.w_size,
                args.w_imb,
            )
        elif method == "score_based_greedy":
            if mode not in ("all_features", "neg_length"):
                return None
            st = method_backward_elimination(
                audit,
                train_pairs,
                test_pairs,
                df_full,
                profile,
                args.seed,
                args.cv_splits,
                args.min_keep,
                args.max_drop_fraction,
                args.target_audit_auc,
                args.w_auc,
                args.w_size,
                args.w_imb,
            )
        elif method == "beam_or_multistart_greedy":
            if mode not in ("all_features", "neg_length"):
                return None
            st = method_beam_multistart(
                audit,
                train_pairs,
                test_pairs,
                df_full,
                profile,
                args.seed,
                args.cv_splits,
                args.min_keep,
                args.max_drop_fraction,
                args.target_audit_auc,
                args.w_auc,
                args.w_size,
                args.w_imb,
                args.beam_width,
                args.n_multistarts,
            )
        else:
            return None

        kg, sc = fit_apply_retain_fraction(
            df_full, audit, train_pairs, test_pairs, st.keep_mask_train, profile, args.seed
        )
        return st, kg, sc

    for mode in modes:
        for method in methods:
            pack = run_for_mode(mode, method)
            if pack is None:
                continue
            st, kg, sc = pack
            diag = retained_diagnostics(audit, kg, method, mode, profile)
            opt_gap = (
                float(st.search_auc - st.heldout_auc)
                if np.isfinite(st.search_auc) and np.isfinite(st.heldout_auc)
                else float("nan")
            )
            row = {
                "method": method,
                "mode": mode,
                "audit_profile": profile,
                "search_time_auc": st.search_auc,
                "heldout_auc": st.heldout_auc,
                "optimism_gap": opt_gap,
                "retained_count": diag["retained_count"],
                "dropped_count": diag["dropped_count"],
                "retained_confounded_fraction": diag["retained_confounded_fraction"],
                "retained_negation_rate": diag["retained_negation_rate"],
                "retained_length_gap_mean": diag["retained_length_gap_mean"],
                "retained_label_balance": diag["retained_label_balance"],
                "objective_score": st.objective,
            }
            results_rows.append(row)
            pareto_inputs.append((st, st.keep_mask_train.copy()))
            if st.objective < best_obj:
                best_obj = st.objective
                best_pack = (st, kg, method, mode, sc)

    res_df = pd.DataFrame(results_rows)
    res_path = out_dir / "improved_search_results.csv"
    res_df.to_csv(res_path, index=False)
    print(f"Wrote {res_path}")

    # Pareto (from unique global masks)
    seen: Set[Tuple] = set()
    states_p: List[SearchState] = []
    masks_p: List[np.ndarray] = []
    for st, km in pareto_inputs:
        kg, _ = fit_apply_retain_fraction(
            df_full, audit, train_pairs, test_pairs, km, profile, args.seed
        )
        key = tuple(kg.tolist())
        if key in seen:
            continue
        seen.add(key)
        states_p.append(st)
        masks_p.append(km.copy())
    pareto_df = method_pareto_collection(
        states_p, audit, df_full, masks_p, train_pairs, test_pairs, profile, args.seed, args.cv_splits
    )
    pareto_path = out_dir / "pareto_front.csv"
    pareto_df.to_csv(pareto_path, index=False)
    print(f"Wrote {pareto_path}")

    if best_pack is None:
        raise SystemExit("No successful search runs (check methods/modes).")

    st_c, kg_c, meth_c, mode_c, sc_c = best_pack
    selection = "lowest_composite_objective"

    best_ids = [int(x) for x in np.where(kg_c)[0]]
    (out_dir / "best_subset_ids.json").write_text(
        json.dumps(
            {
                "pair_ids": best_ids,
                "example_ids": best_ids,
                "method": meth_c,
                "mode": mode_c,
                "search_time_auc": st_c.search_auc,
                "heldout_auc": st_c.heldout_auc,
                "optimism_gap": float(st_c.search_auc - st_c.heldout_auc)
                if np.isfinite(st_c.search_auc) and np.isfinite(st_c.heldout_auc)
                else None,
                "selection": selection,
                "meets_cli_min_keep": int(kg_c.sum()) >= args.min_keep,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sub_audit = audit.iloc[np.where(kg_c)[0]].copy()
    sub_audit.insert(0, "example_id", np.where(kg_c)[0])
    sub_audit.to_csv(out_dir / "best_subset.csv", index=False)
    print(
        f"Best subset (lowest composite objective): method={meth_c} mode={mode_c} "
        f"retained={kg_c.sum()} search_auc={st_c.search_auc:.4f} heldout_auc={st_c.heldout_auc:.4f}"
    )

    expl = build_dropped_explanations(audit, kg_c, sc_c, meth_c, profile)
    expl.to_csv(out_dir / "dropped_example_explanations.csv", index=False)
    print(f"Wrote {out_dir / 'dropped_example_explanations.csv'}")

    # Figures
    if not res_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            res_df["retained_count"],
            res_df["heldout_auc"],
            c=res_df["search_time_auc"],
            cmap="viridis",
            alpha=0.7,
        )
        ax.set_xlabel("Retained count")
        ax.set_ylabel("Held-out audit AUC")
        ax.set_title("Retained size vs held-out AUC (color=search-time AUC)")
        fig.tight_layout()
        fig.savefig(fig_dir / "retained_vs_heldout_auc.pdf")
        fig.savefig(fig_dir / "retained_vs_heldout_auc.png")
        plt.close(fig)

        full_auc = audit_auc_safe(df_full, profile, args.seed, args.cv_splits)
        best_full = audit_auc_safe(subset_df_ans(df_full, np.where(kg_c)[0]), profile, args.seed, args.cv_splits)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["full dataset", "best retained"], [full_auc, best_full], color=["#888", "#117733"])
        ax.set_ylabel("OOF audit AUC (all pairs, biased)")
        ax.set_title("Full vs best subset (whole-dataset OOF — exploratory)")
        fig.tight_layout()
        fig.savefig(fig_dir / "full_vs_best_auc.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(res_df["retained_negation_rate"], res_df["heldout_auc"], alpha=0.6)
        ax.set_xlabel("Retained negation rate")
        ax.set_ylabel("Held-out AUC")
        fig.tight_layout()
        fig.savefig(fig_dir / "negation_rate_vs_heldout_auc.pdf")
        plt.close(fig)

        if not pareto_df.empty:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.scatter(
                pareto_df["retained_count"],
                pareto_df["heldout_auc"],
                c=pareto_df["confounded_fraction"],
                cmap="magma",
            )
            ax.set_xlabel("Retained count")
            ax.set_ylabel("Held-out AUC")
            ax.set_title("Pareto front (approx)")
            fig.tight_layout()
            fig.savefig(fig_dir / "pareto_front.pdf")
            plt.close(fig)

    # strategy_summary.md
    res_valid = res_df.dropna(subset=["objective_score"]) if not res_df.empty else res_df
    if not res_valid.empty:
        best_row = res_valid.loc[res_valid["objective_score"].idxmin()]
    else:
        best_row = None
    export_row = None
    if not res_df.empty:
        hit = res_df[(res_df["method"] == meth_c) & (res_df["mode"] == mode_c)]
        if len(hit) == 1:
            export_row = hit.iloc[0]

    default_run = args.methods.strip().lower() == "all" and not args.include_quantile_score
    methods_note = (
        "Default run uses only: " + ", ".join(DEFAULT_PRUNING_METHODS) + "."
        if default_run
        else ("Methods this run: " + ", ".join(methods) + ".")
    )
    md_lines = [
        "# TruthfulQA pruning (improved search) — summary",
        "",
        f"- **Audit profile:** `{profile}` (surface10 = ten interpretable surface features)",
        f"- **Feature columns:** `{feat_cols}`",
        f"- **Train/test pair split:** {args.train_fraction:.2f} / {1 - args.train_fraction:.2f}",
        "",
        methods_note,
        "",
        f"- **`best_subset.csv` / `best_subset_ids.json`:** lowest **composite objective** across "
        f"all (method, mode) runs in this invocation (`selection={selection}`). "
        f"Report **held-out AUC** as the primary generalization metric.",
        "",
        "## Methods tried (this run)",
        "",
        ", ".join(methods),
        "",
        "## Row matching exported best subset",
        "",
    ]
    if export_row is not None:
        md_lines.append("```\n" + export_row.to_string() + "\n```\n")
    else:
        md_lines.append("_See `best_subset_ids.json`._\n")

    md_lines.extend(
        [
            "",
            "## Best composite objective row",
            "",
        ]
    )
    if best_row is not None:
        md_lines.append("```\n" + best_row.to_string() + "\n```")
        md_lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "Report **held-out AUC** as the primary generalization metric; **search-time AUC** is optimistic.",
                "",
            ]
        )
    below_65 = (
        best_row is not None
        and float(best_row["heldout_auc"]) > 0.65
        and int(best_row["retained_count"]) >= 500
    )
    md_lines.append(
        "### Paper readiness\n"
        + (
            "If held-out AUC cannot get clearly below ~0.65 while keeping 500+ pairs, the bottleneck is likely "
            "that surface-form signal is diffuse across many pairs (not a few outliers), so greedy pair-dropping "
            "underperforms. Say so honestly in the paper and show Pareto tradeoffs."
            if below_65 or best_row is None
            else "Tradeoff looks improved; validate with nested runs and model-eval on `best_subset.csv`."
        )
    )
    (out_dir / "strategy_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    if args.run_model_eval:
        info = run_model_eval_hook(root, best_ids, args.predictions_glob)
        (out_dir / "model_eval_alignment.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )
        print(f"Wrote {out_dir / 'model_eval_alignment.json'}")


if __name__ == "__main__":
    main()
