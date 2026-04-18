#!/usr/bin/env python3
"""
Step 2 — Verify scripts.surface_features_text.extract_surface10 against the
precomputed per-answer feature CSV at audits/truthfulqa_style_audit.csv.

Expected outcome per the approved plan (Correction 3):

  MUST match within 1e-6
    neg_lead   — hard stop on any divergence (real bug)
    neg_cnt    — hard stop on any divergence (real bug)

  Expected to diverge (not a stopping condition)
    hedge_rate, auth_rate        — new extractor uses raw text rather
                                    than pair-level diffs
    word_count, sent_count,
    avg_token_len, type_token,
    punc_rate                    — previously stubbed as constants in
                                    search_truthfulqa_pruned_improved
                                    ._ans_frame

  len_gap is pair-level; checked separately against the CSV's `len_gap`
  column using rel_len_gap(Best Answer, Best Incorrect Answer).

Prints a per-column table of max-abs / mean-abs divergence on both answer
columns, plus a summary of how many rows exceed 1e-6.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.surface_features_text import (
    FEATURE_COLS,
    extract_surface10,
    rel_len_gap,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_CSV = REPO_ROOT / "audits" / "truthfulqa_style_audit.csv"

MUST_MATCH_COLUMNS = ("neg_lead", "neg_cnt")
PER_ANSWER_FEATURES = [c for c in FEATURE_COLS if c != "len_gap"]


def _compute_extractor_frame(texts: pd.Series) -> pd.DataFrame:
    rows = [extract_surface10(t) for t in texts.fillna("").astype(str)]
    return pd.DataFrame(rows)


def _divergence_stats(ours: np.ndarray, theirs: np.ndarray) -> dict:
    diff = np.abs(ours - theirs)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    n_gt_1e6 = int((diff > 1e-6).sum())
    return {"max_abs": max_abs, "mean_abs": mean_abs, "n_exceed_1e6": n_gt_1e6}


def main() -> int:
    print("=" * 72)
    print("STEP 2 — verify_extract_surface10.py")
    print(f"Source CSV: {AUDIT_CSV}")
    print("=" * 72)

    if not AUDIT_CSV.exists():
        print(f"ERROR: {AUDIT_CSV} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(AUDIT_CSV)
    n = len(df)
    print(f"Loaded {n} rows from truthfulqa_style_audit.csv")
    print(f"Columns: {list(df.columns)}\n")

    need = {
        "Best Answer", "Best Incorrect Answer",
        "neg_lead_true", "neg_lead_false",
        "neg_cnt_true", "neg_cnt_false",
        "hedge_rate_true", "hedge_rate_false",
        "auth_rate_true", "auth_rate_false",
        "word_count_true", "word_count_false",
        "sent_count_true", "sent_count_false",
        "avg_token_len_true", "avg_token_len_false",
        "type_token_true", "type_token_false",
        "punc_rate_true", "punc_rate_false",
        "len_gap",
    }
    missing = need - set(df.columns)
    if missing:
        print(f"ERROR: missing expected columns in CSV: {sorted(missing)}",
              file=sys.stderr)
        return 1

    print("Running extract_surface10 on Best Answer ...")
    ext_true = _compute_extractor_frame(df["Best Answer"])
    print("Running extract_surface10 on Best Incorrect Answer ...")
    ext_false = _compute_extractor_frame(df["Best Incorrect Answer"])
    print()

    # --- per-answer features -------------------------------------------------
    header = f"{'feature':<16}{'side':<7}{'max_abs':>14}{'mean_abs':>14}{'n>1e-6':>10}{'verdict':>16}"
    print(header)
    print("-" * len(header))

    must_match_ok = True
    per_col_summary: list[dict] = []

    for feat in PER_ANSWER_FEATURES:
        for side, ext in (("true", ext_true), ("false", ext_false)):
            gold_col = f"{feat}_{side}"
            ours = ext[feat].to_numpy(dtype=float)
            theirs = df[gold_col].to_numpy(dtype=float)
            stats = _divergence_stats(ours, theirs)
            if feat in MUST_MATCH_COLUMNS:
                verdict = "OK" if stats["max_abs"] <= 1e-6 else "BUG"
                if stats["max_abs"] > 1e-6:
                    must_match_ok = False
            else:
                verdict = "ok" if stats["max_abs"] <= 1e-6 else "diverges"
            per_col_summary.append({
                "feature": feat, "side": side, "verdict": verdict, **stats,
            })
            print(
                f"{feat:<16}{side:<7}"
                f"{stats['max_abs']:>14.6g}"
                f"{stats['mean_abs']:>14.6g}"
                f"{stats['n_exceed_1e6']:>10d}"
                f"{verdict:>16}"
            )

    # --- pair-level len_gap --------------------------------------------------
    print()
    print("len_gap (pair-level, rel_len_gap(Best Answer, Best Incorrect Answer)):")
    lg_ours = np.array([
        rel_len_gap(a, b)
        for a, b in zip(df["Best Answer"].fillna("").astype(str),
                        df["Best Incorrect Answer"].fillna("").astype(str))
    ], dtype=float)
    lg_theirs = df["len_gap"].to_numpy(dtype=float)
    lg_stats = _divergence_stats(lg_ours, lg_theirs)
    print(
        f"  max_abs={lg_stats['max_abs']:.6g}  "
        f"mean_abs={lg_stats['mean_abs']:.6g}  "
        f"n>1e-6={lg_stats['n_exceed_1e6']}"
    )
    len_gap_match_within_1e6 = lg_stats["max_abs"] <= 1e-6

    # --- summary -------------------------------------------------------------
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    ok_within_1e6 = [
        (r["feature"], r["side"]) for r in per_col_summary if r["max_abs"] <= 1e-6
    ]
    diverge = [
        (r["feature"], r["side"]) for r in per_col_summary if r["max_abs"] > 1e-6
    ]
    print(f"Columns matching within 1e-6 ({len(ok_within_1e6)}):")
    for feat, side in ok_within_1e6:
        print(f"  {feat}_{side}")
    print(f"\nColumns diverging ({len(diverge)}):")
    for feat, side in diverge:
        print(f"  {feat}_{side}")
    print(
        f"\nlen_gap (pair-level): "
        f"{'OK <= 1e-6' if len_gap_match_within_1e6 else 'diverges'}"
    )

    print()
    if must_match_ok:
        print("neg_lead / neg_cnt match within 1e-6 on both answer columns.  [OK]")
    else:
        print("neg_lead or neg_cnt DIVERGE — this is a real bug per Correction 3.")
        print("STOPPING.")
        return 2

    print("\nOther divergences are expected (per Correction 3) and not blocking.")
    print("Step 2 complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in verify_extract_surface10.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
