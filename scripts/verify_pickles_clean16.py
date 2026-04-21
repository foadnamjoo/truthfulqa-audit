#!/usr/bin/env python3
"""
One-off verification: load every Stage 0 classifier pickle and print
- the training set size it was fit on
- the feature dimensionality (length of the LR coef_ vector)
- a stable SHA-256 hash of the first 5 coefficient values

Used as a sanity check that the pickles consumed by
scripts/score_paired_classifiers_clean16.py are the same artifacts that
were committed at the end of Stage 0; nothing was silently swapped.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"

PICKLES: list[tuple[str, str, Path, int, int]] = [
    # (family, train_set, path, expected_n_rows, expected_dim)
    ("surface_lr",      "full",    ARTIFACTS / "surface_lr_full.pkl",       1580,   10),
    ("surface_lr",      "cleaned", ARTIFACTS / "surface_lr_cleaned.pkl",    1056,   10),
    ("BGE-large",       "full",    ARTIFACTS / "embedding_lr_full.pkl",     1580, 1024),
    ("BGE-large",       "cleaned", ARTIFACTS / "embedding_lr_cleaned.pkl",  1056, 1024),
    ("ModernBERT-base", "full",    ARTIFACTS / "modernbert_lr_full.pkl",    1580,  768),
    ("ModernBERT-base", "cleaned", ARTIFACTS / "modernbert_lr_cleaned.pkl", 1056,  768),
]


def _sha256_first5(coef: np.ndarray) -> str:
    head = np.asarray(coef, dtype=np.float64).ravel()[:5]
    canonical = ",".join(f"{x:.17g}" for x in head)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    print("=" * 96)
    print("Stage 0 classifier pickle verification (used by clean-16 paired probe)")
    print("=" * 96)
    hdr = (f"  {'Family':<16s} {'Train':<8s} "
           f"{'n_rows':>7s} {'n_pairs':>8s} {'dim':>5s}  "
           f"{'cv_auc_g5':>10s}  {'first5_coef_sha256':<64s}  exp")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_ok = True
    for fam, split, path, exp_rows, exp_dim in PICKLES:
        if not path.exists():
            print(f"  {fam:<16s} {split:<8s}  MISSING: {path}")
            all_ok = False
            continue
        art = joblib.load(path)
        pipe = art["pipeline"]
        clf = pipe.steps[-1][1]
        coef = np.asarray(clf.coef_).ravel()
        n_rows = int(art.get("n_rows", -1))
        n_pairs = int(art.get("n_pairs", -1))
        cv = float(art.get("cv_auc_group5", float("nan")))
        sha = _sha256_first5(coef)

        rows_ok = (n_rows == exp_rows)
        dim_ok = (coef.size == exp_dim)
        ok = rows_ok and dim_ok
        flag = "OK" if ok else "MISMATCH"
        if not ok:
            all_ok = False
        print(f"  {fam:<16s} {split:<8s} "
              f"{n_rows:>7d} {n_pairs:>8d} {coef.size:>5d}  "
              f"{cv:>10.4f}  {sha}  {flag}")
        if not rows_ok:
            print(f"      ! expected n_rows={exp_rows}, got {n_rows}")
        if not dim_ok:
            print(f"      ! expected dim={exp_dim}, got {coef.size}")

    print()
    print("All six pickles match expected (n_rows, dim)." if all_ok
          else "AT LEAST ONE PICKLE DOES NOT MATCH EXPECTED SHAPE.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
