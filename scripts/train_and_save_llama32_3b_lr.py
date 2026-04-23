#!/usr/bin/env python3
"""
Fit StandardScaler + LogisticRegression heads on the Llama-3.2-3B-Instruct
mean-pooled embeddings produced by scripts/build_llama32_3b_embeddings.py.

Same recipe as the other additive families.

Inputs:
    artifacts/embeddings/llama32_3b_full_X.npy,    _y.npy,    _pair_id.npy
    artifacts/embeddings/llama32_3b_cleaned_X.npy, _y.npy,    _pair_id.npy

Outputs:
    artifacts/llama32_3b_lr_full.pkl
    artifacts/llama32_3b_lr_cleaned.pkl

ADDITIVE only.
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
OUT_FULL = OUT_DIR / "llama32_3b_lr_full.pkl"
OUT_CLEANED = OUT_DIR / "llama32_3b_lr_cleaned.pkl"

PRIOR_PKLS = [
    ("surface_lr_full      ", OUT_DIR / "surface_lr_full.pkl"),
    ("surface_lr_cleaned   ", OUT_DIR / "surface_lr_cleaned.pkl"),
    ("embedding_lr_full    ", OUT_DIR / "embedding_lr_full.pkl"),
    ("embedding_lr_cleaned ", OUT_DIR / "embedding_lr_cleaned.pkl"),
    ("modernbert_lr_full   ", OUT_DIR / "modernbert_lr_full.pkl"),
    ("modernbert_lr_cleaned", OUT_DIR / "modernbert_lr_cleaned.pkl"),
    ("qwen_lr_full         ", OUT_DIR / "qwen_lr_full.pkl"),
    ("qwen_lr_cleaned      ", OUT_DIR / "qwen_lr_cleaned.pkl"),
    ("qwen15b_lr_full      ", OUT_DIR / "qwen15b_lr_full.pkl"),
    ("qwen15b_lr_cleaned   ", OUT_DIR / "qwen15b_lr_cleaned.pkl"),
    ("smollm2_lr_full      ", OUT_DIR / "smollm2_lr_full.pkl"),
    ("smollm2_lr_cleaned   ", OUT_DIR / "smollm2_lr_cleaned.pkl"),
    ("qwen3b_lr_full       ", OUT_DIR / "qwen3b_lr_full.pkl"),
    ("qwen3b_lr_cleaned    ", OUT_DIR / "qwen3b_lr_cleaned.pkl"),
    ("phi35_lr_full        ", OUT_DIR / "phi35_lr_full.pkl"),
    ("phi35_lr_cleaned     ", OUT_DIR / "phi35_lr_cleaned.pkl"),
]

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
    print(f"  X: {X.shape}, dtype={X.dtype}   "
          f"y=1:{(y==1).sum()} y=0:{(y==0).sum()}")

    print(f"  Running GroupKFold(n_splits={CV_SPLITS}) AUC ...")
    cv_auc = _cv_auc(X, y, groups)
    print(f"  CV AUC (GroupKFold={CV_SPLITS}): {cv_auc:.4f}")

    print("  Fitting full-data pipeline (StandardScaler + LR, liblinear) ...")
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
        "source_prefix": prefix,
    }
    joblib.dump(artifact, out_path)
    print(f"  saved {out_path}")
    return artifact


def _try_read_cv_auc(pkl: Path) -> float | None:
    if not pkl.exists():
        return None
    try:
        blob = joblib.load(pkl)
        return float(blob.get("cv_auc_group5"))
    except Exception:
        return None


def main() -> int:
    print("=" * 72)
    print("train_and_save_llama32_3b_lr.py  (NEW family: Llama-3.2-3B-Instruct)")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    art_full = _fit_one(
        "llama32_3b_full",    "FULL (790 pairs, 1580 rows)",    OUT_FULL,
    )
    art_clean = _fit_one(
        "llama32_3b_cleaned", "CLEANED tau=0.52 (528 pairs, 1056 rows)",
        OUT_CLEANED,
    )

    print()
    print("=" * 72)
    print("CROSS-FAMILY CV-AUC COMPARISON")
    print("=" * 72)
    for label, p in PRIOR_PKLS:
        v = _try_read_cv_auc(p)
        if v is None:
            print(f"  {label} CV AUC: (not found)")
        else:
            print(f"  {label} CV AUC: {v:.4f}")
    print(f"  llama32_3b_lr_full    CV AUC: {art_full['cv_auc_group5']:.4f}  "
          "<-- NEW")
    print(f"  llama32_3b_lr_cleaned CV AUC: {art_clean['cv_auc_group5']:.4f}  "
          "<-- NEW")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in train_and_save_llama32_3b_lr.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
