#!/usr/bin/env python3
"""
Generate final benchmark-impact tables from model prediction CSVs.

This script is intentionally stdlib-only (no pandas) so it runs reliably
even in environments where numpy/pandas are problematic.

Inputs (in repo root):
  - audits/truthfulqa_style_audit.csv  (must contain column: style_violation)
  - model_predictions*.csv            (must contain: model_name, pair_id, correct)

Outputs (in audits/):
  - model_benchmark_impact_by_file.csv   : one row per (source_file, model)
  - model_benchmark_impact_by_model.csv  : one row per model, with duplicate pair_ids
                                          across files resolved by "last file wins"
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Counts:
    correct: int = 0
    total: int = 0

    def add(self, is_correct: bool) -> None:
        self.total += 1
        if is_correct:
            self.correct += 1

    def mean(self) -> Optional[float]:
        if self.total == 0:
            return None
        return self.correct / self.total


def load_style_violation(audit_path: Path) -> List[int]:
    with audit_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "style_violation" not in (reader.fieldnames or []):
            raise KeyError(f"{audit_path} must contain column: style_violation")
        out: List[int] = []
        for r in reader:
            out.append(int(r["style_violation"]))
    return out


def iter_prediction_rows(path: Path) -> Iterable[Tuple[str, int, int]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        need = {"model_name", "pair_id", "correct"}
        fields = set(reader.fieldnames or [])
        missing = need - fields
        if missing:
            raise KeyError(f"{path} missing required columns: {sorted(missing)}")
        for r in reader:
            yield (str(r["model_name"]), int(r["pair_id"]), int(r["correct"]))


def compute_by_file(
    pred_paths: List[Path],
    style_violation: List[int],
) -> List[Dict[str, object]]:
    n_pairs = len(style_violation)
    clean_ids = {i for i, v in enumerate(style_violation) if v == 0}
    conf_ids = {i for i, v in enumerate(style_violation) if v == 1}

    out_rows: List[Dict[str, object]] = []

    for p in pred_paths:
        # counts[model] -> (all, clean, conf)
        counts: Dict[str, Tuple[Counts, Counts, Counts]] = {}
        seen: set[Tuple[str, int]] = set()

        for model, pair_id, correct in iter_prediction_rows(p):
            if not (0 <= pair_id < n_pairs):
                continue
            key = (model, pair_id)
            if key in seen:
                # within-file duplicates: keep first (file should not have these)
                continue
            seen.add(key)

            if model not in counts:
                counts[model] = (Counts(), Counts(), Counts())
            c_all, c_clean, c_conf = counts[model]
            is_correct = bool(int(correct) == 1)
            c_all.add(is_correct)
            if pair_id in clean_ids:
                c_clean.add(is_correct)
            elif pair_id in conf_ids:
                c_conf.add(is_correct)

        for model, (c_all, c_clean, c_conf) in sorted(counts.items(), key=lambda kv: kv[0]):
            acc_all = c_all.mean()
            acc_clean = c_clean.mean()
            acc_conf = c_conf.mean()
            delta = None
            if acc_clean is not None and acc_conf is not None:
                delta = acc_conf - acc_clean

            out_rows.append(
                {
                    "source_file": p.name,
                    "model": model,
                    "acc_all": acc_all,
                    "acc_clean": acc_clean,
                    "acc_confounded": acc_conf,
                    "delta_conf_clean": delta,
                    "n_all": c_all.total,
                    "n_clean": c_clean.total,
                    "n_confounded": c_conf.total,
                }
            )

    return out_rows


def compute_by_model_last_wins(
    pred_paths: List[Path],
    style_violation: List[int],
) -> List[Dict[str, object]]:
    """
    Aggregate across ALL files, resolving duplicates by (model, pair_id) with
    'last file wins' using pred_paths order.
    """
    n_pairs = len(style_violation)
    clean_ids = {i for i, v in enumerate(style_violation) if v == 0}
    conf_ids = {i for i, v in enumerate(style_violation) if v == 1}

    # store[(model, pair_id)] = correct
    store: Dict[Tuple[str, int], int] = {}
    for p in pred_paths:
        for model, pair_id, correct in iter_prediction_rows(p):
            if not (0 <= pair_id < n_pairs):
                continue
            store[(model, pair_id)] = int(correct)

    # counts per model
    counts: Dict[str, Tuple[Counts, Counts, Counts]] = {}
    for (model, pair_id), correct in store.items():
        if model not in counts:
            counts[model] = (Counts(), Counts(), Counts())
        c_all, c_clean, c_conf = counts[model]
        is_correct = bool(int(correct) == 1)
        c_all.add(is_correct)
        if pair_id in clean_ids:
            c_clean.add(is_correct)
        elif pair_id in conf_ids:
            c_conf.add(is_correct)

    out_rows: List[Dict[str, object]] = []
    for model, (c_all, c_clean, c_conf) in sorted(counts.items(), key=lambda kv: kv[0]):
        acc_all = c_all.mean()
        acc_clean = c_clean.mean()
        acc_conf = c_conf.mean()
        delta = None
        if acc_clean is not None and acc_conf is not None:
            delta = acc_conf - acc_clean
        out_rows.append(
            {
                "model": model,
                "acc_all": acc_all,
                "acc_clean": acc_clean,
                "acc_confounded": acc_conf,
                "delta_conf_clean": delta,
                "n_all": c_all.total,
                "n_clean": c_clean.total,
                "n_confounded": c_conf.total,
            }
        )
    # Sort by overall accuracy descending, then model name.
    out_rows.sort(key=lambda r: (-(r["acc_all"] or 0.0), str(r["model"])))
    return out_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=".", help="Repo root (default: .)")
    p.add_argument(
        "--audit_csv",
        type=str,
        default="audits/truthfulqa_style_audit.csv",
        help="Audit CSV containing style_violation",
    )
    p.add_argument(
        "--pred_glob",
        type=str,
        default="model_predictions*.csv",
        help="Glob for prediction files (default: model_predictions*.csv)",
    )
    p.add_argument(
        "--exclude_example",
        action="store_true",
        default=True,
        help="Exclude example_model_predictions.csv (default: true)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    audit_path = (root / args.audit_csv).resolve()
    style_violation = load_style_violation(audit_path)

    pred_paths = sorted(root.glob(args.pred_glob))
    if args.exclude_example:
        pred_paths = [p for p in pred_paths if p.name != "example_model_predictions.csv"]

    if not pred_paths:
        raise SystemExit(f"No prediction files found under {root} matching {args.pred_glob}")

    by_file = compute_by_file(pred_paths, style_violation)
    by_file_out = root / "audits" / "model_benchmark_impact_by_file.csv"
    write_csv(
        by_file_out,
        by_file,
        [
            "source_file",
            "model",
            "acc_all",
            "acc_clean",
            "acc_confounded",
            "delta_conf_clean",
            "n_all",
            "n_clean",
            "n_confounded",
        ],
    )

    by_model = compute_by_model_last_wins(pred_paths, style_violation)
    by_model_out = root / "audits" / "model_benchmark_impact_by_model.csv"
    write_csv(
        by_model_out,
        by_model,
        [
            "model",
            "acc_all",
            "acc_clean",
            "acc_confounded",
            "delta_conf_clean",
            "n_all",
            "n_clean",
            "n_confounded",
        ],
    )

    print("Saved:", by_file_out)
    print("Saved:", by_model_out)
    print("Note: by_model uses 'last file wins' across duplicate (model,pair_id).")


if __name__ == "__main__":
    main()

