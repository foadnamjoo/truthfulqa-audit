#!/usr/bin/env python3
"""
Step 5 — Singleton-scoring sanity tests for the refit surface-LR pipelines.

Replaces the §4-plan single-test with two separate tests per Mod 2:

Test A — exact reproducibility
    Load artifacts/surface_lr_full.pkl and predict on the full training
    parquet. Independently fit the same pipeline factory from scratch on
    the same data with the same random_state and predict again.
    Assert max abs diff < 1e-9 over all 1580 rows. Failure => pickling
    or environment bug; stop and surface it.

Test B — OOF correlation
    Run GroupKFold(5) on the full set with the same pipeline factory to
    get OOF predictions. Compare OOF predictions to the full-refit
    predictions via Spearman rho. Assert rho > 0.9. Absolute differences
    of up to ~0.15 per row are expected (OOF excludes the containing
    fold); correlation is the right check.

If either test fails, exit non-zero and print the traceback. Both tests
must pass before any Stage-0 adversarial scoring runs.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.surface_features_text import FEATURE_COLS


REPO_ROOT = Path(__file__).resolve().parent.parent
FULL_PARQUET = REPO_ROOT / "artifacts" / "features" / "truthfulqa_full_surface10.parquet"
ART_FULL = REPO_ROOT / "artifacts" / "surface_lr_full.pkl"

RANDOM_STATE = 42
CV_SPLITS = 5

TEST_A_TOL = 1e-9
TEST_B_MIN_RHO = 0.9


def _make_pipeline():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
    )


def test_a_exact_reproducibility(X: np.ndarray, y: np.ndarray) -> dict:
    print("--- Test A: exact reproducibility (pickled vs fresh-fit) ---")
    artifact = joblib.load(ART_FULL)
    loaded_pipe = artifact["pipeline"]
    loaded_cols = artifact.get("feature_cols")
    if loaded_cols != FEATURE_COLS:
        raise RuntimeError(
            "Pickled feature_cols differ from scripts.surface_features_text."
            f"FEATURE_COLS.\n  pickled:  {loaded_cols}\n  current:  {FEATURE_COLS}"
        )

    p_loaded = loaded_pipe.predict_proba(X)[:, 1]

    fresh = _make_pipeline()
    fresh.fit(X, y)
    p_fresh = fresh.predict_proba(X)[:, 1]

    diff = np.abs(p_loaded - p_fresh)
    max_diff = float(diff.max())
    print(f"  pickled proba:       min={p_loaded.min():.6f}  max={p_loaded.max():.6f}")
    print(f"  fresh-fit proba:     min={p_fresh.min():.6f}  max={p_fresh.max():.6f}")
    print(f"  max abs diff:        {max_diff:.3e}  (tol {TEST_A_TOL:.0e})")

    ok = max_diff < TEST_A_TOL
    print(f"  Test A: {'PASS' if ok else 'FAIL'}")
    return {"max_abs_diff": max_diff, "pass": ok, "p_refit": p_loaded}


def test_b_oof_correlation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    p_refit: np.ndarray,
) -> dict:
    print("\n--- Test B: OOF-vs-refit Spearman correlation ---")
    cv = GroupKFold(n_splits=CV_SPLITS)
    pipe = _make_pipeline()
    p_oof = cross_val_predict(
        pipe, X, y,
        cv=cv, groups=groups,
        method="predict_proba",
    )[:, 1]

    rho, pval = spearmanr(p_oof, p_refit)
    max_abs = float(np.max(np.abs(p_oof - p_refit)))
    mean_abs = float(np.mean(np.abs(p_oof - p_refit)))
    print(f"  OOF proba:           min={p_oof.min():.6f}  max={p_oof.max():.6f}")
    print(f"  Spearman rho:        {rho:.4f}  (p={pval:.2e})")
    print(f"  |OOF - refit|:       mean={mean_abs:.4f}  max={max_abs:.4f}")

    ok = rho > TEST_B_MIN_RHO
    print(f"  Test B: {'PASS' if ok else 'FAIL'}  (require rho > {TEST_B_MIN_RHO})")
    return {"spearman_rho": float(rho), "pass": ok,
            "mean_abs": mean_abs, "max_abs": max_abs}


def main() -> int:
    print("=" * 72)
    print("STEP 5 — test_singleton_roundtrip.py")
    print("=" * 72)
    for p in (FULL_PARQUET, ART_FULL):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}. Run prior steps first.")

    df = pd.read_parquet(FULL_PARQUET)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    groups = df["pair_id"].to_numpy(dtype=int)
    print(
        f"Loaded full parquet: rows={len(df)}, pairs={df['pair_id'].nunique()}"
    )

    res_a = test_a_exact_reproducibility(X, y)
    res_b = test_b_oof_correlation(X, y, groups, res_a["p_refit"])

    print()
    print("=" * 72)
    print("STEP 5 SUMMARY")
    print("=" * 72)
    print(f"  Test A (exact repro, tol {TEST_A_TOL:.0e}): "
          f"{'PASS' if res_a['pass'] else 'FAIL'}  "
          f"(max abs diff={res_a['max_abs_diff']:.3e})")
    print(f"  Test B (OOF-vs-refit rho > {TEST_B_MIN_RHO}): "
          f"{'PASS' if res_b['pass'] else 'FAIL'}  "
          f"(rho={res_b['spearman_rho']:.4f})")

    if not (res_a["pass"] and res_b["pass"]):
        print("\nOne or both tests FAILED. Do not proceed to Stage 0 scoring.")
        return 1

    print("\nBoth tests passed. Step 5 complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in test_singleton_roundtrip.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
