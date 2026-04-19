#!/usr/bin/env python3
"""
Side experiment (NOT a paper artifact): does extending the negation
lexicon change the grouped-CV AUC of the surface-form audit on full
TruthfulQA?

This script:
  - Reads audits/truthfulqa_style_audit.csv (790 question pairs, the
    canonical source the paper's audit is built from).
  - Builds a 1580-row answer-level dataset by stacking
    (Best Answer, label=1) and (Best Incorrect Answer, label=0) for each
    question, with pair_id = row index in the audit CSV.
  - For each of three lexicon variants A / B / C, RECOMPUTES neg_lead
    and neg_cnt from the raw answer text using word-boundary regex
    matching, case-insensitive. The other 8 features (hedge_rate,
    auth_rate, len_gap, word_count, sent_count, avg_token_len,
    type_token, punc_rate) are taken UNCHANGED from the audit CSV's
    pre-computed *_true / *_false columns. This isolates the lexicon
    swap from any other implementation differences.
  - Fits make_pipeline(StandardScaler(), LogisticRegression(max_iter=
    1000, solver="liblinear", random_state=42)) on all 10 features and
    reports GroupKFold(n_splits=5, groups=pair_id) AUC and accuracy.
  - Reports a "negation removed" ablation per variant (neg_lead = 0,
    neg_cnt = 0) to quantify how much signal the negation family carries
    under each lexicon.
  - Reports per-side coverage counts (how many y=1 vs y=0 rows the
    lexicon flags) so the AUC change can be interpreted.

Self-check: if Lexicon A's AUC is more than 0.02 away from the
published 0.716, the script aborts with a clear error before printing
the B / C numbers, because the baseline reproduction would be off and
the comparison would be untrustworthy.

Outputs (artifacts/, all NEW files - no canonical artifact is touched):
  artifacts/experiment_extended_lexicon.json
  artifacts/features/extended_lexicon_A.parquet
  artifacts/features/extended_lexicon_B.parquet
  artifacts/features/extended_lexicon_C.parquet

Known minor implementation note (does NOT affect this experiment):
  The audit CSV's punc_rate uses denom = max(1, num_simple_tokens)
  whereas the canonical scripts/surface_features_text.py extractor
  used to build the surface10 parquets uses denom = len(text). This
  discrepancy is upstream and has nothing to do with the negation
  lexicon being studied here; we use the audit CSV's columns as-is so
  that "Lexicon A" reproduces the notebook's published configuration.
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_CSV = REPO_ROOT / "audits" / "truthfulqa_style_audit.csv"
OUT_JSON = REPO_ROOT / "artifacts" / "experiment_extended_lexicon.json"
OUT_FEAT_DIR = REPO_ROOT / "artifacts" / "features"
OUT_PARQUET = {
    "A": OUT_FEAT_DIR / "extended_lexicon_A.parquet",
    "B": OUT_FEAT_DIR / "extended_lexicon_B.parquet",
    "C": OUT_FEAT_DIR / "extended_lexicon_C.parquet",
}

RANDOM_STATE = 42
CV_SPLITS = 5

# Match against the paper's published surface_lr full-set AUC. Anything
# more than +/- 0.02 from this value triggers an abort.
PUBLISHED_AUC = 0.716
BASELINE_TOLERANCE = 0.02

FEATURE_COLS = [
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
NEG_FEATURES = ["neg_lead", "neg_cnt"]
NON_NEG_TRUE_COLS = [
    "hedge_rate_true", "auth_rate_true", "len_gap",
    "word_count_true", "sent_count_true", "avg_token_len_true",
    "type_token_true", "punc_rate_true",
]
NON_NEG_FALSE_COLS = [
    "hedge_rate_false", "auth_rate_false", "len_gap",
    "word_count_false", "sent_count_false", "avg_token_len_false",
    "type_token_false", "punc_rate_false",
]
NON_NEG_TARGET_COLS = [
    "hedge_rate", "auth_rate", "len_gap",
    "word_count", "sent_count", "avg_token_len",
    "type_token", "punc_rate",
]

# --- Lexicons --------------------------------------------------------------
# Lexicon A (current, paper baseline) - replicated verbatim from
# scripts/build_audit_notebook.py / scripts/surface_features_text.py so
# this script does not depend on importing them.
NEG_LEADS_A: list[str] = [
    r"no\b", r"not\b",
    r"there is no\b", r"there are no\b", r"there's no\b",
    r"none\b", r"never\b", r"nothing\b", r"nobody\b",
]
NEG_WORDS_A: list[str] = [
    r"\bno\b", r"\bnot\b", r"\bnever\b",
    r"\bnone\b", r"\bnothing\b", r"\bnobody\b",
    r"\bcannot\b", r"\bcan't\b",
    r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b",
    r"\bwon't\b", r"\bwasn't\b", r"\bweren't\b",
]

# Lexicon B (extended): synonym-negation tokens. These express negative
# semantics with surface forms the current lexicon cannot catch. They
# extend BOTH the lead list (for sentence-initial occurrence) and the
# count list (for occurrence anywhere in the answer).
LEX_B_EXTRA = [
    "zero", "incapable", "unaffected",
    "fails", "fail", "lacks",
    "without", "unable", "impossible",
]

# Lexicon C (extended further): adverb-style "barely-negation" markers.
# These weaken rather than negate, so they are a sensitivity check.
LEX_C_EXTRA = [
    "hardly", "scarcely", "barely",
    "rarely", "seldom",
]


def _make_neg_leads(extra_words: list[str]) -> list[str]:
    return list(NEG_LEADS_A) + [rf"{w}\b" for w in extra_words]


def _make_neg_words(extra_words: list[str]) -> list[str]:
    return list(NEG_WORDS_A) + [rf"\b{w}\b" for w in extra_words]


LEXICONS: dict[str, dict] = {
    "A": {
        "name": "A (current paper baseline)",
        "extra_words": [],
        "neg_leads": NEG_LEADS_A,
        "neg_words": NEG_WORDS_A,
    },
    "B": {
        "name": "B (extended: synonym-negation)",
        "extra_words": LEX_B_EXTRA,
        "neg_leads": _make_neg_leads(LEX_B_EXTRA),
        "neg_words": _make_neg_words(LEX_B_EXTRA),
    },
    "C": {
        "name": "C (extended + adverb-negation, sensitivity)",
        "extra_words": LEX_B_EXTRA + LEX_C_EXTRA,
        "neg_leads": _make_neg_leads(LEX_B_EXTRA + LEX_C_EXTRA),
        "neg_words": _make_neg_words(LEX_B_EXTRA + LEX_C_EXTRA),
    },
}


# --- Negation feature extractors ------------------------------------------
def _starts_with_any(patterns: list[str], text: str) -> int:
    t = str(text).strip().lower()
    if not t:
        return 0
    for p in patterns:
        if re.match(p, t):
            return 1
    return 0


def _count_matches(patterns: list[str], text: str) -> int:
    text_l = str(text).lower()
    return sum(len(re.findall(p, text_l)) for p in patterns)


def _compute_neg_features(text_series: pd.Series, lexicon: dict
                          ) -> tuple[np.ndarray, np.ndarray]:
    leads = lexicon["neg_leads"]
    words = lexicon["neg_words"]
    neg_lead = np.array(
        [float(_starts_with_any(leads, t)) for t in text_series],
        dtype=float,
    )
    neg_cnt = np.array(
        [float(_count_matches(words, t)) for t in text_series],
        dtype=float,
    )
    return neg_lead, neg_cnt


# --- Build answer-level dataset for one lexicon ---------------------------
def _build_answer_level(audit: pd.DataFrame, lexicon: dict) -> pd.DataFrame:
    """Return 1580-row df with FEATURE_COLS + label + pair_id columns."""
    n_pairs = len(audit)
    pair_ids = np.arange(n_pairs, dtype=int)

    nl_true, nc_true = _compute_neg_features(audit["Best Answer"], lexicon)
    nl_false, nc_false = _compute_neg_features(
        audit["Best Incorrect Answer"], lexicon)

    rows_true = pd.DataFrame({
        "neg_lead": nl_true,
        "neg_cnt": nc_true,
    })
    for src, tgt in zip(NON_NEG_TRUE_COLS, NON_NEG_TARGET_COLS):
        rows_true[tgt] = audit[src].astype(float).values
    rows_true["label"] = 1
    rows_true["pair_id"] = pair_ids
    rows_true["text"] = audit["Best Answer"].values

    rows_false = pd.DataFrame({
        "neg_lead": nl_false,
        "neg_cnt": nc_false,
    })
    for src, tgt in zip(NON_NEG_FALSE_COLS, NON_NEG_TARGET_COLS):
        rows_false[tgt] = audit[src].astype(float).values
    rows_false["label"] = 0
    rows_false["pair_id"] = pair_ids
    rows_false["text"] = audit["Best Incorrect Answer"].values

    df_ans = pd.concat([rows_true, rows_false], ignore_index=True)
    df_ans = df_ans[FEATURE_COLS + ["label", "pair_id", "text"]]
    if df_ans[FEATURE_COLS].isna().any().any():
        bad = df_ans[FEATURE_COLS].isna().sum()
        raise RuntimeError(f"NaNs in feature columns:\n{bad}")
    return df_ans


# --- Fit + grouped-CV evaluation ------------------------------------------
def _make_pipeline():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
    )


def _cv_metrics(X: np.ndarray, y: np.ndarray, groups: np.ndarray
                ) -> tuple[float, float]:
    cv = GroupKFold(n_splits=CV_SPLITS)
    pipe = _make_pipeline()
    proba = cross_val_predict(
        pipe, X, y, cv=cv, groups=groups, method="predict_proba",
    )[:, 1]
    auc = float(roc_auc_score(y, proba))
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y, pred))
    return auc, acc


def _coverage(df_ans: pd.DataFrame) -> dict:
    pos = df_ans[df_ans["label"] == 1]
    neg = df_ans[df_ans["label"] == 0]
    return {
        "neg_lead_gt0_y1": int((pos["neg_lead"] > 0).sum()),
        "neg_lead_gt0_y0": int((neg["neg_lead"] > 0).sum()),
        "neg_cnt_gt0_y1":  int((pos["neg_cnt"] > 0).sum()),
        "neg_cnt_gt0_y0":  int((neg["neg_cnt"] > 0).sum()),
        "neg_cnt_mean_y1": float(pos["neg_cnt"].mean()),
        "neg_cnt_mean_y0": float(neg["neg_cnt"].mean()),
    }


def main() -> int:
    print("=" * 72)
    print("experiment_extended_lexicon.py")
    print("=" * 72)

    if not AUDIT_CSV.exists():
        print(f"ERROR: missing {AUDIT_CSV}", file=sys.stderr)
        return 1

    audit = pd.read_csv(AUDIT_CSV)
    n_pairs = len(audit)
    print(f"Loaded audit CSV: {AUDIT_CSV.relative_to(REPO_ROOT)}  "
          f"(rows={n_pairs})")
    required = {"Best Answer", "Best Incorrect Answer"} | set(
        NON_NEG_TRUE_COLS + NON_NEG_FALSE_COLS)
    missing = required - set(audit.columns)
    if missing:
        print(f"ERROR: audit CSV is missing required columns: "
              f"{sorted(missing)}", file=sys.stderr)
        return 1

    OUT_FEAT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Build per-lexicon feature frames ---------------------------------
    df_by_lex: dict[str, pd.DataFrame] = {}
    for tag, lex in LEXICONS.items():
        df_by_lex[tag] = _build_answer_level(audit, lex)
        n_rows = len(df_by_lex[tag])
        n_g = df_by_lex[tag]["pair_id"].nunique()
        print(f"  Lexicon {tag}: {n_rows} answer rows, {n_g} pairs, "
              f"{len(lex['neg_words'])} neg_words tokens")

    if any(len(d) != 2 * n_pairs for d in df_by_lex.values()):
        print("ERROR: per-lexicon row count mismatch", file=sys.stderr)
        return 1

    # --- Lexicon A baseline first; abort if outside tolerance -------------
    print("\n--- Lexicon A baseline (must reproduce the paper's surface_lr "
          "AUC within +/-{:.2f}) ---".format(BASELINE_TOLERANCE))
    df_A = df_by_lex["A"]
    X_A = df_A[FEATURE_COLS].to_numpy(dtype=float)
    y_A = df_A["label"].to_numpy(dtype=int)
    g_A = df_A["pair_id"].to_numpy(dtype=int)
    auc_A, acc_A = _cv_metrics(X_A, y_A, g_A)
    delta_A = auc_A - PUBLISHED_AUC
    print(f"  Lexicon A: GroupCV AUC = {auc_A:.4f}  "
          f"(published baseline = {PUBLISHED_AUC:.3f}, "
          f"delta = {delta_A:+.4f})")
    if abs(delta_A) > BASELINE_TOLERANCE:
        print(f"\nABORT: Lexicon A AUC delta {delta_A:+.4f} exceeds "
              f"+/-{BASELINE_TOLERANCE:.2f} tolerance.", file=sys.stderr)
        print("The B / C numbers are NOT reported because the baseline "
              "reproduction is off; the comparison would be untrustworthy.",
              file=sys.stderr)
        return 2
    print("  Within tolerance - proceeding to Lexicon B and C.")

    # --- Now compute everything for all 3 -----------------------------------
    results: dict[str, dict] = {}
    for tag, lex in LEXICONS.items():
        df_ans = df_by_lex[tag]
        X = df_ans[FEATURE_COLS].to_numpy(dtype=float)
        y = df_ans["label"].to_numpy(dtype=int)
        g = df_ans["pair_id"].to_numpy(dtype=int)
        auc, acc = _cv_metrics(X, y, g)

        # Negation-removed ablation: zero out neg_lead and neg_cnt,
        # keep all 10 columns (so scaling and LR weights see the same
        # number of features).
        df_zero = df_ans.copy()
        df_zero["neg_lead"] = 0.0
        df_zero["neg_cnt"] = 0.0
        X_zero = df_zero[FEATURE_COLS].to_numpy(dtype=float)
        auc_zero, acc_zero = _cv_metrics(X_zero, y, g)

        cov = _coverage(df_ans)

        results[tag] = {
            "name": lex["name"],
            "extra_words": lex["extra_words"],
            "neg_words_pattern_count": len(lex["neg_words"]),
            "neg_leads_pattern_count": len(lex["neg_leads"]),
            "auc_grouped5": auc,
            "accuracy_grouped5": acc,
            "auc_grouped5_neg_zeroed": auc_zero,
            "accuracy_grouped5_neg_zeroed": acc_zero,
            "auc_drop_when_neg_zeroed": auc - auc_zero,
            "coverage": cov,
        }

        # Save the per-row feature parquet
        out_pq = OUT_PARQUET[tag]
        df_ans.to_parquet(out_pq, index=False)
        print(f"  wrote {out_pq.relative_to(REPO_ROOT)}  "
              f"(rows={len(df_ans)})")

    auc_A_val = results["A"]["auc_grouped5"]

    # --- Comparison table ----------------------------------------------------
    print("\n" + "=" * 88)
    print(f"LEXICON COMPARISON ON FULL TRUTHFULQA  "
          f"(n=1580 rows, {n_pairs} pairs)")
    print("=" * 88)
    short_label = {
        "A": "A (current)",
        "B": "B (extended)",
        "C": "C (+adverbs)",
    }
    hdr = (f"  {'Lexicon':<14s} {'NegTok':>6s}  {'AUC':>7s}  "
           f"{'Acc':>7s}  {'dAUC vs A':>10s}  {'AUC neg=0':>10s}  "
           f"{'AUC drop':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tag in ("A", "B", "C"):
        r = results[tag]
        d_vs_A = r["auc_grouped5"] - auc_A_val
        print(f"  {short_label[tag]:<14s} "
              f"{r['neg_words_pattern_count']:>6d}  "
              f"{r['auc_grouped5']:>7.4f}  "
              f"{r['accuracy_grouped5']:>7.4f}  "
              f"{d_vs_A:>+10.4f}  "
              f"{r['auc_grouped5_neg_zeroed']:>10.4f}  "
              f"{r['auc_drop_when_neg_zeroed']:>+9.4f}")

    print("\n  Extra tokens added vs A:")
    print(f"    B adds: {results['B']['extra_words']}")
    print(f"    C adds: {results['C']['extra_words']}  (B + adverbs)")

    # --- Coverage table ------------------------------------------------------
    print("\n" + "=" * 72)
    print("NEGATION COVERAGE BY LEXICON")
    print("=" * 72)
    cov_hdr = (f"  {'metric':<18s} | "
               f"{'Lex A y=1':>10s} {'Lex A y=0':>10s} | "
               f"{'Lex B y=1':>10s} {'Lex B y=0':>10s} | "
               f"{'Lex C y=1':>10s} {'Lex C y=0':>10s}")
    print(cov_hdr)
    print("  " + "-" * (len(cov_hdr) - 2))
    for metric, key1, key0, fmt in [
        ("neg_lead > 0",  "neg_lead_gt0_y1", "neg_lead_gt0_y0", "{:>10d}"),
        ("neg_cnt > 0",   "neg_cnt_gt0_y1",  "neg_cnt_gt0_y0",  "{:>10d}"),
        ("neg_cnt mean",  "neg_cnt_mean_y1", "neg_cnt_mean_y0", "{:>10.4f}"),
    ]:
        row = f"  {metric:<18s} |"
        for tag in ("A", "B", "C"):
            cov = results[tag]["coverage"]
            row += f" {fmt.format(cov[key1])} {fmt.format(cov[key0])} |"
        print(row.rstrip(" |"))

    # --- Save JSON ----------------------------------------------------------
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "extended_lexicon",
        "n_pairs": int(n_pairs),
        "n_rows": int(2 * n_pairs),
        "cv_splits": CV_SPLITS,
        "random_state": RANDOM_STATE,
        "feature_cols": FEATURE_COLS,
        "non_negation_features_source": (
            "audits/truthfulqa_style_audit.csv pre-computed *_true / *_false "
            "columns (kept verbatim - only neg_lead and neg_cnt are "
            "recomputed per lexicon variant)"
        ),
        "sklearn_recipe": (
            "make_pipeline(StandardScaler(), "
            f"LogisticRegression(max_iter=1000, solver='liblinear', "
            f"random_state={RANDOM_STATE}))"
        ),
        "cv_recipe": f"GroupKFold(n_splits={CV_SPLITS}, groups=pair_id)",
        "published_baseline_auc": PUBLISHED_AUC,
        "baseline_tolerance": BASELINE_TOLERANCE,
        "results_by_lexicon": results,
        "feature_parquets": {
            tag: str(p.relative_to(REPO_ROOT))
            for tag, p in OUT_PARQUET.items()
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in experiment_extended_lexicon.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
