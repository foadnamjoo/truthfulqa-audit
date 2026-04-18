#!/usr/bin/env python3
"""
Step 4 — Refit the 10-dim surface-form LR on the rebuilt per-answer feature
frames (Step 3 output), report GroupKFold AUCs, and pickle the fitted
pipelines for singleton scoring later.

Model recipe (matches the paper / audit_subset_evaluator.py):

    make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="liblinear",
                           random_state=42),
    )

Saves:

    artifacts/surface_lr_full.pkl      joblib blob with keys
    artifacts/surface_lr_cleaned.pkl     "pipeline", "feature_cols",
                                         "train_len_gap_mean", "n_rows",
                                         "cv_auc_group5"

The train_len_gap_mean value is the mean of the `len_gap` column across
the training rows; score_singleton.py substitutes this in when scoring
a standalone text that has no paired counterpart.

PATH B RESULTS block is printed at the end comparing the refit AUCs to
the paper's published numbers.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.surface_features_text import FEATURE_COLS


REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = REPO_ROOT / "artifacts" / "features"
OUT_DIR = REPO_ROOT / "artifacts"

FULL_PARQUET = FEAT_DIR / "truthfulqa_full_surface10.parquet"
TAU052_PARQUET = FEAT_DIR / "truthfulqa_tau052_surface10.parquet"
OUT_FULL = OUT_DIR / "surface_lr_full.pkl"
OUT_CLEANED = OUT_DIR / "surface_lr_cleaned.pkl"

RANDOM_STATE = 42
CV_SPLITS = 5

PUBLISHED_AUC_FULL = 0.716
PUBLISHED_AUC_CLEANED = 0.513
WARN_DELTA = 0.05


def _make_pipeline():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
    )


def _cv_auc_group5(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    cv = GroupKFold(n_splits=CV_SPLITS)
    pipe = _make_pipeline()
    y_proba = cross_val_predict(
        pipe, X, y,
        cv=cv, groups=groups,
        method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(y, y_proba))


def _fit_and_save(
    parquet_path: Path,
    out_path: Path,
    label: str,
) -> tuple[float, dict]:
    print(f"\n--- {label} ---")
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"  rows={len(df)}, pairs={df['pair_id'].nunique()}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing feature columns: {missing}")
    if df[FEATURE_COLS].isna().any().any():
        raise ValueError(f"{label}: NaNs present in feature columns")

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    groups = df["pair_id"].to_numpy(dtype=int)
    train_len_gap_mean = float(np.mean(X[:, FEATURE_COLS.index("len_gap")]))

    print(f"Running GroupKFold(n_splits={CV_SPLITS}) AUC ...")
    cv_auc = _cv_auc_group5(X, y, groups)
    print(f"  CV AUC (GroupKFold={CV_SPLITS}): {cv_auc:.4f}")

    print("Fitting full-dataset pipeline (StandardScaler + LR, liblinear) ...")
    pipeline = _make_pipeline()
    pipeline.fit(X, y)

    artifact = {
        "pipeline": pipeline,
        "feature_cols": list(FEATURE_COLS),
        "train_len_gap_mean": train_len_gap_mean,
        "n_rows": int(len(df)),
        "n_pairs": int(df["pair_id"].nunique()),
        "cv_auc_group5": cv_auc,
        "random_state": RANDOM_STATE,
        "cv_splits": CV_SPLITS,
        "source_parquet": str(parquet_path.relative_to(REPO_ROOT)),
        "sklearn_recipe": (
            "make_pipeline(StandardScaler(), "
            "LogisticRegression(max_iter=1000, solver='liblinear', "
            f"random_state={RANDOM_STATE}))"
        ),
    }
    joblib.dump(artifact, out_path)
    print(f"  saved {out_path}  (train_len_gap_mean={train_len_gap_mean:.6f})")
    return cv_auc, artifact


def main() -> int:
    print("=" * 72)
    print("STEP 4 — refit_surface_lr.py")
    print("=" * 72)

    for p in (FULL_PARQUET, TAU052_PARQUET):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing feature parquet: {p}. Run Step 3 first."
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    auc_full, art_full = _fit_and_save(FULL_PARQUET, OUT_FULL, "FULL (790 pairs, 1580 rows)")
    auc_cleaned, art_cleaned = _fit_and_save(
        TAU052_PARQUET, OUT_CLEANED, "CLEANED tau=0.52 (528 pairs, 1056 rows)"
    )

    delta_full = auc_full - PUBLISHED_AUC_FULL
    delta_cleaned = auc_cleaned - PUBLISHED_AUC_CLEANED

    print()
    print("=" * 72)
    print("PATH B RESULTS")
    print("=" * 72)
    print(f"  Full-set AUC (published, stubbed features):   {PUBLISHED_AUC_FULL:.3f}")
    print(f"  Full-set AUC (refit, real features):          {auc_full:.4f}")
    print(f"  Cleaned-set AUC (published, stubbed):         {PUBLISHED_AUC_CLEANED:.3f}")
    print(f"  Cleaned-set AUC (refit, real features):       {auc_cleaned:.4f}")
    print(f"  Delta full:    {delta_full:+.4f}")
    print(f"  Delta cleaned: {delta_cleaned:+.4f}")

    warnings: list[str] = []
    if abs(delta_full) > WARN_DELTA:
        warnings.append(
            f"  WARNING: |Delta full| = {abs(delta_full):.4f} exceeds "
            f"{WARN_DELTA}. Real-feature classifier has shifted materially."
        )
    if abs(delta_cleaned) > WARN_DELTA:
        warnings.append(
            f"  WARNING: |Delta cleaned| = {abs(delta_cleaned):.4f} exceeds "
            f"{WARN_DELTA}. Real-feature classifier has shifted materially."
        )
    if warnings:
        print()
        for w in warnings:
            print(w)
        print("  These are heads-ups, not stopping conditions.")
    else:
        print("\n  No |delta| > 0.05 — refit is close to the published numbers.")

    print("\nStep 4 complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in refit_surface_lr.py:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
