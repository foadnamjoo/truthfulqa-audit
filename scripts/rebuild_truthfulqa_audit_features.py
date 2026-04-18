#!/usr/bin/env python3
"""
Step 3 — Regenerate per-answer surface10 feature frames using the new
text-level extractor (scripts.surface_features_text.extract_surface10).

Produces two Parquet files (both overwrite any existing file):

    artifacts/features/truthfulqa_full_surface10.parquet    (1580 rows)
    artifacts/features/truthfulqa_tau052_surface10.parquet  (1056 rows)

Each row is one answer. Columns:

    pair_id         int, same as example_id used across the repo.
                    (For TruthfulQA, example_id is the 0..789 row index
                    into TruthfulQA.csv.)
    y               int, 1 if the row is the Best Answer, 0 if Best
                    Incorrect Answer.
    neg_lead, neg_cnt, hedge_rate, auth_rate,
    len_gap,        pair-level rel_len_gap(Best Answer, Best Incorrect
                    Answer); identical for the two rows of a pair.
    word_count, sent_count, avg_token_len, type_token, punc_rate

The cleaned-subset frame is the rows of the full frame whose pair_id
appears in data/subsets/TruthfulQA-Audited/surface_audited/pair_ids/
pair_ids_tau052.json (the tau=0.52 audit-prune selection).

Invariants checked before writing:
  - no NaNs in any feature column
  - len(full)  == 2 * 790
  - len(tau052) == 2 * 528
  - FEATURE_COLS order matches scripts.surface_features_text.FEATURE_COLS
"""
from __future__ import annotations

import json
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
TRUTHFULQA_CSV = REPO_ROOT / "TruthfulQA.csv"
TAU052_PAIR_IDS = (
    REPO_ROOT
    / "data" / "subsets" / "TruthfulQA-Audited" / "surface_audited"
    / "pair_ids" / "pair_ids_tau052.json"
)
OUT_DIR = REPO_ROOT / "artifacts" / "features"
OUT_FULL = OUT_DIR / "truthfulqa_full_surface10.parquet"
OUT_TAU052 = OUT_DIR / "truthfulqa_tau052_surface10.parquet"


def _build_full_answer_frame() -> pd.DataFrame:
    tq = pd.read_csv(TRUTHFULQA_CSV)
    n_pairs = len(tq)
    best_col = tq["Best Answer"].fillna("").astype(str)
    incorrect_col = tq["Best Incorrect Answer"].fillna("").astype(str)

    rows: list[dict] = []
    for pair_id in range(n_pairs):
        best = best_col.iat[pair_id]
        incorrect = incorrect_col.iat[pair_id]
        lg = rel_len_gap(best, incorrect)

        feat_t = extract_surface10(best)
        feat_t["len_gap"] = lg
        feat_t["pair_id"] = pair_id
        feat_t["y"] = 1
        rows.append(feat_t)

        feat_f = extract_surface10(incorrect)
        feat_f["len_gap"] = lg
        feat_f["pair_id"] = pair_id
        feat_f["y"] = 0
        rows.append(feat_f)

    df = pd.DataFrame(rows)
    assert len(df) == 2 * n_pairs, (
        f"Expected {2 * n_pairs} answer rows, got {len(df)}"
    )
    # Column order: pair_id, y, then FEATURE_COLS.
    ordered = ["pair_id", "y"] + FEATURE_COLS
    df = df[ordered]
    df["pair_id"] = df["pair_id"].astype(int)
    df["y"] = df["y"].astype(int)
    return df


def _load_tau052_ids() -> list[int]:
    with open(TAU052_PAIR_IDS, "r", encoding="utf-8") as f:
        blob = json.load(f)
    ids = blob.get("pair_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Bad pair_ids in {TAU052_PAIR_IDS}")
    return [int(x) for x in ids]


def _assert_no_nans(df: pd.DataFrame, label: str) -> None:
    nans = df.isna().sum().sum()
    if nans:
        raise ValueError(f"{label}: {nans} NaN cell(s) detected")


def main() -> int:
    print("=" * 72)
    print("STEP 3 — rebuild_truthfulqa_audit_features.py")
    print("=" * 72)
    print(f"Source: {TRUTHFULQA_CSV}")
    print(f"Tau=0.52 pair ids: {TAU052_PAIR_IDS}")
    print(f"Output dir: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBuilding full per-answer frame (expecting 1580 rows) ...")
    full = _build_full_answer_frame()
    _assert_no_nans(full, "full")
    print(
        f"  rows={len(full)}  "
        f"n_pairs={full['pair_id'].nunique()}  "
        f"y=1: {(full['y'] == 1).sum()}  y=0: {(full['y'] == 0).sum()}"
    )

    if len(full) != 1580:
        raise RuntimeError(f"Unexpected row count: {len(full)} != 1580")

    print("\nFiltering to tau=0.52 cleaned subset ...")
    tau_ids = _load_tau052_ids()
    print(f"  loaded {len(tau_ids)} pair ids")
    tau = full[full["pair_id"].isin(set(tau_ids))].reset_index(drop=True)
    _assert_no_nans(tau, "tau052")
    print(
        f"  rows={len(tau)}  "
        f"n_pairs={tau['pair_id'].nunique()}  "
        f"y=1: {(tau['y'] == 1).sum()}  y=0: {(tau['y'] == 0).sum()}"
    )
    if tau["pair_id"].nunique() != 528:
        raise RuntimeError(
            f"Expected 528 unique pair_ids in tau052 subset, got "
            f"{tau['pair_id'].nunique()}"
        )
    if len(tau) != 1056:
        raise RuntimeError(f"Unexpected tau052 row count: {len(tau)} != 1056")

    # --- Quick sanity summary: per-feature mean by y, for both frames ------
    print("\nPer-feature mean by y (full):")
    print(full.groupby("y")[FEATURE_COLS].mean().round(4).to_string())
    print("\nPer-feature mean by y (tau052):")
    print(tau.groupby("y")[FEATURE_COLS].mean().round(4).to_string())

    # --- Write parquet ------------------------------------------------------
    print(f"\nWriting {OUT_FULL} ...")
    full.to_parquet(OUT_FULL, index=False)
    print(f"Writing {OUT_TAU052} ...")
    tau.to_parquet(OUT_TAU052, index=False)

    # --- Verify round-trip --------------------------------------------------
    full_r = pd.read_parquet(OUT_FULL)
    tau_r = pd.read_parquet(OUT_TAU052)
    assert full_r.equals(full), "Round-trip mismatch for full parquet"
    assert tau_r.equals(tau), "Round-trip mismatch for tau052 parquet"
    print("\nRound-trip read succeeded. Step 3 complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in rebuild_truthfulqa_audit_features.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
