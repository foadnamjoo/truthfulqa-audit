#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import truthfulqa_paper_audit as tpa

CV_SPLITS = 5


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_candidates_with_features(truthfulqa_csv: Path, audit_csv: Path) -> pd.DataFrame:
    tq = pd.read_csv(truthfulqa_csv)
    audit = pd.read_csv(audit_csv)
    if len(tq) != len(audit):
        raise ValueError(f"Row mismatch: TruthfulQA={len(tq)} audit={len(audit)}")
    df = pd.DataFrame()
    df["example_id"] = np.arange(len(tq), dtype=int)
    df["question"] = tq["Question"]
    df["answer_a"] = tq["Best Answer"]
    df["answer_b"] = tq["Best Incorrect Answer"]
    df["correct_label"] = "A"
    df["confounded_flag"] = audit["style_violation"].astype(int)
    df["neg_lead_true"] = audit["neg_lead_true"].astype(float)
    df["neg_lead_false"] = audit["neg_lead_false"].astype(float)
    df["neg_cnt_true"] = audit["neg_cnt_true"].astype(float)
    df["neg_cnt_false"] = audit["neg_cnt_false"].astype(float)
    df["negation_flag"] = (
        (df["neg_lead_true"] + df["neg_lead_false"] + df["neg_cnt_true"] + df["neg_cnt_false"]) > 0
    ).astype(int)
    df["length_gap"] = audit["len_gap"].astype(float)
    df["hedge_diff"] = (audit["hedge_rate_true"] - audit["hedge_rate_false"]).abs().astype(float)
    df["authority_diff"] = (audit["auth_rate_true"] - audit["auth_rate_false"]).abs().astype(float)
    df["Type"] = tq.get("Type", "")
    df["Category"] = tq.get("Category", "")
    return df


def audit_auc_subset(df: pd.DataFrame, profile: str, seed: int) -> float:
    # Rebuild answer-level frame from original audit columns when available.
    cols = [
        "neg_lead_true",
        "neg_lead_false",
        "neg_cnt_true",
        "neg_cnt_false",
        "length_gap",
        "hedge_diff",
        "authority_diff",
    ]
    # If this is a candidate frame, map back into audit-like frame expected by tpa.
    if set(cols).issubset(df.columns):
        # Provide minimal columns required by build_answer_level_audit_frame.
        # Use coarse approximations for missing count/rate columns to keep behavior deterministic.
        a = pd.DataFrame(
            {
                "neg_lead_true": df["neg_lead_true"].to_numpy(),
                "neg_lead_false": df["neg_lead_false"].to_numpy(),
                "neg_cnt_true": df["neg_cnt_true"].to_numpy(),
                "neg_cnt_false": df["neg_cnt_false"].to_numpy(),
                "hedge_rate_true": df.get("hedge_rate_true", df["hedge_diff"] / 2.0).to_numpy(),
                "hedge_rate_false": df.get("hedge_rate_false", np.zeros(len(df))).to_numpy(),
                "auth_rate_true": df.get("auth_rate_true", df["authority_diff"] / 2.0).to_numpy(),
                "auth_rate_false": df.get("auth_rate_false", np.zeros(len(df))).to_numpy(),
                "len_gap": df["length_gap"].to_numpy(),
                "word_count_true": np.full(len(df), 10),
                "word_count_false": np.full(len(df), 10),
                "sent_count_true": np.ones(len(df)),
                "sent_count_false": np.ones(len(df)),
                "avg_token_len_true": np.full(len(df), 4.0),
                "avg_token_len_false": np.full(len(df), 4.0),
                "type_token_true": np.full(len(df), 1.0),
                "type_token_false": np.full(len(df), 1.0),
                "punc_rate_true": np.zeros(len(df)),
                "punc_rate_false": np.zeros(len(df)),
            }
        )
    else:
        a = df.copy()
    ans = tpa.build_answer_level_audit_frame(a, profile="paper10", copy_audit_meta=False)
    return float(tpa.paper_compatible_audit_oof_auc(ans, profile="paper10", seed=seed, n_splits=CV_SPLITS).auc_oof)

