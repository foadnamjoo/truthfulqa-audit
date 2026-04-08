#!/usr/bin/env python3
"""
Reusable grouped CV audit metrics on a retained pair subset.

Uses the same answer-level frame and classifier pipeline as
``search_truthfulqa_pruned_improved._auc_pairs`` (surface10-style features).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from truthfulqa_pruning_utils import CV_SPLITS
from search_truthfulqa_pruned_improved import _ans_frame

# Must match _auc_pairs / _ans_frame stacking in search_truthfulqa_pruned_improved.py
FEATURE_COLS: List[str] = [
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


@dataclass(frozen=True)
class SubsetAuditMetrics:
    """Grouped CV OOF metrics on answer-level rows (two per pair)."""

    auc: float
    accuracy: float
    n_pairs: int
    n_answer_rows: int


@dataclass(frozen=True)
class SubsetAuditDetailed:
    """Grouped CV metrics plus answer-level OOF probabilities."""

    metrics: SubsetAuditMetrics
    answer_frame_with_oof: pd.DataFrame


def df_pairs_from_ids(full_df: pd.DataFrame, pair_ids: Set[int] | List[int]) -> pd.DataFrame:
    """Return rows of ``full_df`` for the given ``example_id`` values."""
    if not pair_ids:
        return full_df.iloc[0:0].copy()
    return full_df[full_df["example_id"].isin(set(pair_ids))].copy()


def evaluate_subset_grouped_cv(df_pairs: pd.DataFrame, seed: int) -> SubsetAuditMetrics:
    """
    Grouped 5-fold CV with StandardScaler + LogisticRegression; OOF proba for AUC and accuracy.

    ``df_pairs`` must contain the columns required by ``_ans_frame`` (one row per question pair).
    """
    n_pairs = len(df_pairs)
    if n_pairs < CV_SPLITS:
        return SubsetAuditMetrics(
            auc=float("nan"),
            accuracy=float("nan"),
            n_pairs=n_pairs,
            n_answer_rows=2 * n_pairs,
        )
    ans = _ans_frame(df_pairs)
    X = ans[FEATURE_COLS].to_numpy()
    y = ans["label"].to_numpy()
    g = ans["pair_id"].to_numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    proba = cross_val_predict(pipe, X, y, cv=GroupKFold(n_splits=CV_SPLITS), groups=g, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    acc = float(np.mean((proba >= 0.5) == y))
    return SubsetAuditMetrics(
        auc=auc,
        accuracy=acc,
        n_pairs=n_pairs,
        n_answer_rows=int(len(y)),
    )


def evaluate_subset_grouped_cv_detailed(df_pairs: pd.DataFrame, seed: int) -> SubsetAuditDetailed:
    """Same grouped CV evaluation as evaluate_subset_grouped_cv with OOF proba attached."""
    n_pairs = len(df_pairs)
    if n_pairs < CV_SPLITS:
        empty_ans = _ans_frame(df_pairs).copy()
        empty_ans["proba"] = np.nan
        return SubsetAuditDetailed(
            metrics=SubsetAuditMetrics(
                auc=float("nan"),
                accuracy=float("nan"),
                n_pairs=n_pairs,
                n_answer_rows=2 * n_pairs,
            ),
            answer_frame_with_oof=empty_ans,
        )
    ans = _ans_frame(df_pairs).copy()
    X = ans[FEATURE_COLS].to_numpy()
    y = ans["label"].to_numpy()
    g = ans["pair_id"].to_numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    proba = cross_val_predict(pipe, X, y, cv=GroupKFold(n_splits=CV_SPLITS), groups=g, method="predict_proba")[:, 1]
    ans["proba"] = proba
    auc = float(roc_auc_score(y, proba))
    acc = float(np.mean((proba >= 0.5) == y))
    return SubsetAuditDetailed(
        metrics=SubsetAuditMetrics(
            auc=auc,
            accuracy=acc,
            n_pairs=n_pairs,
            n_answer_rows=int(len(y)),
        ),
        answer_frame_with_oof=ans,
    )


def oof_pair_confidence_scores(df_pairs: pd.DataFrame, seed: int) -> Dict[int, float]:
    """
    Per-pair score = |p(correct) - p(incorrect)| from OOF probabilities (higher = more separable).
    """
    if len(df_pairs) < CV_SPLITS:
        return {}
    ans = _ans_frame(df_pairs)
    X = ans[FEATURE_COLS].to_numpy()
    y = ans["label"].to_numpy()
    g = ans["pair_id"].to_numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    proba = cross_val_predict(pipe, X, y, cv=GroupKFold(n_splits=CV_SPLITS), groups=g, method="predict_proba")[:, 1]
    ans = ans.copy()
    ans["proba"] = proba
    scores: Dict[int, float] = {}
    for pid in df_pairs["example_id"].unique():
        sub = ans[ans["pair_id"] == int(pid)]
        p_pos = float(sub[sub["label"] == 1]["proba"].iloc[0])
        p_neg = float(sub[sub["label"] == 0]["proba"].iloc[0])
        scores[int(pid)] = abs(p_pos - p_neg)
    return scores


def oof_pair_imbalance_scores(df_pairs: pd.DataFrame, seed: int) -> Dict[int, float]:
    """
    Per-pair score = sum_f |Δ_f| * 1[Δ_f aligns with global (μ_pos - μ_neg)_f].

    ``Δ_f`` is (feature on correct answer - feature on incorrect answer) for the pair.
    """
    _ = seed  # API parity with confidence scorer; deterministic given retained pairs
    if len(df_pairs) < CV_SPLITS:
        return {}
    ans = _ans_frame(df_pairs)
    X = ans[FEATURE_COLS].to_numpy()
    y = ans["label"].to_numpy()
    mu_pos = X[y == 1].mean(axis=0)
    mu_neg = X[y == 0].mean(axis=0)
    gap = mu_pos - mu_neg
    scores: Dict[int, float] = {}
    for pid in df_pairs["example_id"].unique():
        sub = ans[ans["pair_id"] == int(pid)]
        x_t = sub[sub["label"] == 1][FEATURE_COLS].to_numpy().ravel()
        x_f = sub[sub["label"] == 0][FEATURE_COLS].to_numpy().ravel()
        v = x_t - x_f
        s = 0.0
        for j in range(len(FEATURE_COLS)):
            if gap[j] == 0.0:
                continue
            if v[j] * gap[j] > 0:
                s += abs(float(v[j]))
        scores[int(pid)] = s
    return scores


def pick_pair_to_remove(scores: Dict[int, float], retained: set[int]) -> int:
    """Remove pair with maximum score; tie-break by smallest example_id."""
    best = -1.0
    candidates: List[int] = []
    for k in retained:
        sk = float(scores.get(k, 0.0))
        if sk > best + 1e-15:
            best = sk
            candidates = [k]
        elif abs(sk - best) <= 1e-15:
            candidates.append(k)
    return min(candidates)
