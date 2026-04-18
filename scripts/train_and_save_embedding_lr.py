#!/usr/bin/env python3
"""
Step 7 — Fit a StandardScaler + LogisticRegression head on the BGE-large
embeddings produced in Step 6, report GroupKFold AUC, and pickle each
Pipeline for singleton scoring.

Recipe (per plan §5.2):
    make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                           random_state=42),
    )

Inputs (Step 6 output):
    artifacts/embeddings/full_X.npy, full_y.npy, full_pair_id.npy
    artifacts/embeddings/cleaned_X.npy, cleaned_y.npy, cleaned_pair_id.npy

Outputs:
    artifacts/embedding_lr_full.pkl
    artifacts/embedding_lr_cleaned.pkl

Each pickle is a joblib-dumped dict: {"pipeline", "cv_auc_group5",
"n_rows", "n_pairs", "random_state", "cv_splits", "recipe"}.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = REPO_ROOT / "artifacts" / "embeddings"
OUT_DIR = REPO_ROOT / "artifacts"
OUT_FULL = OUT_DIR / "embedding_lr_full.pkl"
OUT_CLEANED = OUT_DIR / "embedding_lr_cleaned.pkl"

RANDOM_STATE = 42
CV_SPLITS = 5


def _make_pipeline():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
    )


def _cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    cv = GroupKFold(n_splits=CV_SPLITS)
    pipe = _make_pipeline()
    proba = cross_val_predict(
        pipe, X, y, cv=cv, groups=groups, method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(y, proba))


def _fit_one(prefix: str, label: str, out_path: Path) -> dict:
    print(f"\n--- {label} ---")
    X_p = EMB_DIR / f"{prefix}_X.npy"
    y_p = EMB_DIR / f"{prefix}_y.npy"
    g_p = EMB_DIR / f"{prefix}_pair_id.npy"
    for p in (X_p, y_p, g_p):
        if not p.exists():
            raise FileNotFoundError(f"Missing embedding blob: {p}")

    X = np.load(X_p)
    y = np.load(y_p).astype(int)
    groups = np.load(g_p).astype(int)
    n, d = X.shape
    print(f"  X: {X.shape}, dtype={X.dtype}   y=1:{(y==1).sum()} y=0:{(y==0).sum()}")

    print(f"  Running GroupKFold(n_splits={CV_SPLITS}) AUC ...")
    cv_auc = _cv_auc(X, y, groups)
    print(f"  CV AUC (GroupKFold={CV_SPLITS}): {cv_auc:.4f}")

    print(f"  Fitting full-data pipeline (StandardScaler + LR, liblinear) ...")
    pipeline = _make_pipeline()
    pipeline.fit(X, y)

    artifact = {
        "pipeline": pipeline,
        "cv_auc_group5": cv_auc,
        "n_rows": int(n),
        "n_pairs": int(len(np.unique(groups))),
        "random_state": RANDOM_STATE,
        "cv_splits": CV_SPLITS,
        "embedding_dim": int(d),
        "recipe": (
            "make_pipeline(StandardScaler(), "
            "LogisticRegression(max_iter=2000, C=1.0, solver='liblinear', "
            f"random_state={RANDOM_STATE}))"
        ),
    }
    joblib.dump(artifact, out_path)
    print(f"  saved {out_path}")
    return artifact


def main() -> int:
    print("=" * 72)
    print("STEP 7 — train_and_save_embedding_lr.py")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    art_full = _fit_one("full",    "FULL (790 pairs, 1580 rows)",    OUT_FULL)
    art_clean = _fit_one("cleaned", "CLEANED tau=0.52 (528 pairs, 1056 rows)", OUT_CLEANED)

    print()
    print("=" * 72)
    print("EMBEDDING-LR RESULTS")
    print("=" * 72)
    print(f"  Surface-LR full    CV AUC (from Step 4): see artifacts/surface_lr_full.pkl")
    print(f"  Embedding-LR full  CV AUC: {art_full['cv_auc_group5']:.4f}")
    print(f"  Embedding-LR clean CV AUC: {art_clean['cv_auc_group5']:.4f}")
    if art_full["cv_auc_group5"] <= 0.716:
        print("  NOTE: embedding-LR full AUC is not above surface-LR published 0.716.")
        print("        This is unexpected; worth flagging but not a stopping condition.")
    print("\nStep 7 complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in train_and_save_embedding_lr.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
