#!/usr/bin/env python3
"""
Canonical TruthfulQA surface-form audit (paper-compatible).

Default profile **paper10** matches `scripts/make_paper_assets.py` and the validated
notebook pipeline: answer-level rows, 10 unified feature names, GroupKFold by pair_id,
StandardScaler + LogisticRegression, OOF ROC-AUC from predict_proba.

**expanded13** adds three pair-level columns (same value on both answer rows of a pair):
temporal_fragile, hedge_pair_diff, auth_pair_diff. Use only when explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AuditProfile = Literal["paper10", "expanded13"]

# Unified answer-row feature names (paper / notebook).
FEAT_COLS_PAPER10: List[str] = [
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

COLS_TRUE_PAPER10: List[str] = [
    "neg_lead_true",
    "neg_cnt_true",
    "hedge_rate_true",
    "auth_rate_true",
    "len_gap",
    "word_count_true",
    "sent_count_true",
    "avg_token_len_true",
    "type_token_true",
    "punc_rate_true",
]

COLS_FALSE_PAPER10: List[str] = [
    "neg_lead_false",
    "neg_cnt_false",
    "hedge_rate_false",
    "auth_rate_false",
    "len_gap",
    "word_count_false",
    "sent_count_false",
    "avg_token_len_false",
    "type_token_false",
    "punc_rate_false",
]

FEAT_COLS_EXPANDED13: List[str] = FEAT_COLS_PAPER10 + [
    "temporal_fragile",
    "hedge_pair_diff",
    "auth_pair_diff",
]

DEFAULT_CV_SPLITS = 5
DEFAULT_LR_MAX_ITER = 1000


@dataclass(frozen=True)
class AuditScorerResult:
    auc_oof: float
    feature_columns: Tuple[str, ...]
    audit_profile: str
    n_answer_rows: int
    n_pairs: int


def default_group_kfold(n_splits: int = DEFAULT_CV_SPLITS) -> GroupKFold:
    return GroupKFold(n_splits=n_splits)


def make_lr_pipeline(random_state: int, max_iter: int = DEFAULT_LR_MAX_ITER):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=random_state),
    )


def build_answer_level_audit_frame(
    audit: pd.DataFrame,
    profile: AuditProfile = "paper10",
    copy_audit_meta: bool = True,
) -> pd.DataFrame:
    """
    Stack correct (label=1) and incorrect (label=0) answer rows with pair_id = row index.

    If copy_audit_meta, pair-level columns from `audit` are duplicated onto both rows
    (first len(audit) rows correspond to label=1, next block to label=0), keyed by pair_id.
    """
    rows_true = audit[COLS_TRUE_PAPER10].copy()
    rows_true.columns = FEAT_COLS_PAPER10
    rows_true["label"] = 1
    rows_true["pair_id"] = np.arange(len(audit), dtype=np.int64)

    rows_false = audit[COLS_FALSE_PAPER10].copy()
    rows_false.columns = FEAT_COLS_PAPER10
    rows_false["label"] = 0
    rows_false["pair_id"] = np.arange(len(audit), dtype=np.int64)

    df_ans = pd.concat([rows_true, rows_false], ignore_index=True)

    if profile == "expanded13":
        n = len(audit)
        hf = (audit["hedge_rate_true"] - audit["hedge_rate_false"]).to_numpy()
        af = (audit["auth_rate_true"] - audit["auth_rate_false"]).to_numpy()
        tf = audit["temporal_fragile"].to_numpy() if "temporal_fragile" in audit.columns else np.zeros(n)
        # Broadcast pair-level scalars to both answer rows in concat order: [all true, all false]
        dup = np.concatenate([tf, tf])
        df_ans["temporal_fragile"] = dup
        df_ans["hedge_pair_diff"] = np.concatenate([hf, hf])
        df_ans["auth_pair_diff"] = np.concatenate([af, af])

    if copy_audit_meta:
        meta_cols = [
            c
            for c in (
                "Type",
                "Category",
                "Question",
                "Best Answer",
                "Best Incorrect Answer",
                "style_violation",
                "neg_lead_true",
                "neg_lead_false",
                "neg_cnt_true",
                "neg_cnt_false",
                "hedge_rate_true",
                "hedge_rate_false",
                "auth_rate_true",
                "auth_rate_false",
                "len_gap",
            )
            if c in audit.columns
        ]
        for c in meta_cols:
            v = audit[c].to_numpy()
            df_ans[c] = np.concatenate([v, v])

    return df_ans


def feature_columns_for_profile(profile: AuditProfile) -> List[str]:
    if profile == "paper10":
        return list(FEAT_COLS_PAPER10)
    if profile == "expanded13":
        return list(FEAT_COLS_EXPANDED13)
    raise ValueError(f"Unknown audit profile: {profile}")


def paper_compatible_audit_oof_auc(
    df_ans: pd.DataFrame,
    *,
    profile: AuditProfile = "paper10",
    feature_columns: Sequence[str] | None = None,
    seed: int = 42,
    n_splits: int = DEFAULT_CV_SPLITS,
    pair_id_column: str = "pair_id",
) -> AuditScorerResult:
    """
    Canonical audit scorer for pruning/search: grouped CV OOF ROC-AUC (correct vs incorrect).
    """
    cols = list(feature_columns) if feature_columns is not None else feature_columns_for_profile(profile)
    missing = [c for c in cols if c not in df_ans.columns]
    if missing:
        raise KeyError(f"df_ans missing feature columns: {missing}")

    X = df_ans[cols].fillna(0).to_numpy()
    y = df_ans["label"].to_numpy()
    groups = df_ans[pair_id_column].to_numpy()
    cv = GroupKFold(n_splits=n_splits)
    pipe = make_lr_pipeline(seed)
    proba = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    n_pairs = int(len(np.unique(groups)))
    return AuditScorerResult(
        auc_oof=auc,
        feature_columns=tuple(cols),
        audit_profile=profile,
        n_answer_rows=int(len(df_ans)),
        n_pairs=n_pairs,
    )


def shuffle_labels_within_groups(
    y: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    y_perm = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y_perm[idx] = rng.permutation(y_perm[idx])
    return y_perm
