#!/usr/bin/env python3
"""
Import CHPC prediction CSVs into the repo root as model_predictions_*.csv.

Why:
  - §7c in the notebook scans for model_predictions*.csv in the repo root.
  - CHPC runs are often stored in scratch folders with names like:
      seed_sweep_.../out_<...>.csv
      family_sweep_.../<run>/out/model_predictions.csv
  - This script finds any CSV with columns (model_name, pair_id, correct),
    infers a run tag (and seed when possible), and writes a renamed copy into
    the repo root so the notebook can aggregate them.

Safety:
  - Does NOT overwrite existing files unless --overwrite is passed.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Optional


REQUIRED_COLS = {"model_name", "pair_id", "correct"}


def safe_slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")


def detect_seed(text: str) -> Optional[int]:
    m = re.search(r"(?:seed|SEED)(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{2,4})\b", text)
    # Avoid treating years/ports/etc. as seeds; only accept typical seed sizes.
    if m:
        val = int(m.group(1))
        if 0 <= val <= 10_000:
            return val
    return None


def is_prediction_csv(path: Path) -> bool:
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            return REQUIRED_COLS.issubset(fields)
    except Exception:
        return False


def read_first_model_name(path: Path) -> str:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            return str(r["model_name"])
    raise ValueError(f"{path} appears empty (no rows).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=str,
        required=True,
        help="Directory containing CHPC outputs (copied locally).",
    )
    ap.add_argument(
        "--dest_root",
        type=str,
        default=".",
        help="Repo root to write model_predictions_*.csv into (default: .).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()

    if not src.exists():
        raise SystemExit(f"Source directory not found: {src}")
    if not dest_root.exists():
        raise SystemExit(f"Destination root not found: {dest_root}")

    candidates = sorted([p for p in src.rglob("*.csv") if p.is_file()])
    pred_paths = [p for p in candidates if is_prediction_csv(p)]
    if not pred_paths:
        raise SystemExit(f"No prediction-like CSVs found under: {src}")

    imported = 0
    skipped = 0
    for p in pred_paths:
        model = read_first_model_name(p)
        model_slug = safe_slug(model)
        seed = detect_seed(str(p))

        # Tag includes the leaf directory name so multiple runs don't collide.
        tag = safe_slug(p.parent.name or "run")
        seed_part = f"_seed{seed}" if seed is not None else ""
        out_name = f"model_predictions_{model_slug}{seed_part}_{tag}.csv"
        out_path = dest_root / out_name

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        shutil.copy2(p, out_path)
        imported += 1

    print(f"Found {len(pred_paths)} prediction CSV(s) under {src}")
    print(f"Imported: {imported} | Skipped (already existed): {skipped}")
    print("Notebook §7c will pick these up via model_predictions*.csv glob.")


if __name__ == "__main__":
    main()

