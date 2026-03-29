#!/usr/bin/env python3
"""
Build user-facing TruthfulQA subset CSVs from canonical JSON pair lists.

Source rows: TruthfulQA.csv (0-based row index == pair_id in JSON).
style_violation: audits/truthfulqa_style_audit.csv (same row order as TruthfulQA.csv).

Does not modify results/final_near_random_truthfulqa_subset/*.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBSETS_DIR = ROOT / "data" / "subsets"
JSON_DIR = ROOT / "results" / "final_near_random_truthfulqa_subset"
TRUTHFULQA_CSV = ROOT / "TruthfulQA.csv"
AUDIT_CSV = ROOT / "audits" / "truthfulqa_style_audit.csv"
METRICS_CSV = JSON_DIR / "best_method_by_target.csv"

# Columns from official CSV + audit + release metadata
BASE_COLS = [
    "Type",
    "Category",
    "Question",
    "Best Answer",
    "Best Incorrect Answer",
]


def load_metrics_by_target() -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with METRICS_CSV.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(row["target_pairs"])
            rows[t] = row
    return rows


def main() -> int:
    if not TRUTHFULQA_CSV.is_file():
        print(f"Missing {TRUTHFULQA_CSV}", file=sys.stderr)
        return 1

    df_t = pd.read_csv(TRUTHFULQA_CSV)
    df_a = pd.read_csv(AUDIT_CSV)
    if len(df_t) != len(df_a):
        print(
            f"Row count mismatch: TruthfulQA.csv={len(df_t)} audit={len(df_a)}",
            file=sys.stderr,
        )
        return 1
    style = df_a["style_violation"].to_numpy()

    metrics = load_metrics_by_target()
    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []

    configs: list[tuple[int, str, str]] = [
        (350, "truthfulqa_subset_350.csv", "main"),
        (375, "truthfulqa_subset_375.csv", "evaluated_intermediate"),
        (400, "truthfulqa_subset_400.csv", "secondary_larger"),
    ]

    for target, csv_name, paper_role in configs:
        json_path = JSON_DIR / f"final_subset_ids_{target}.json"
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        pair_ids: list[int] = meta["pair_ids"]
        method = meta["method"]
        subset_name = f"truthfulqa_subset_{target}"

        if len(pair_ids) != target:
            print(
                f"Expected {target} pair_ids in {json_path}, got {len(pair_ids)}",
                file=sys.stderr,
            )
            return 1

        out_path = SUBSETS_DIR / csv_name
        canonical_rel = f"results/final_near_random_truthfulqa_subset/final_subset_ids_{target}.json"

        m = metrics[target]
        fieldnames = [
            "pair_id",
            *BASE_COLS,
            "style_violation",
            "subset_name",
            "subset_size",
            "selection_method",
            "source_dataset",
            "canonical_json",
        ]

        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for pid in pair_ids:
                if not (0 <= pid < len(df_t)):
                    print(f"Invalid pair_id {pid} for {target}", file=sys.stderr)
                    return 1
                row = df_t.iloc[pid]
                w.writerow(
                    {
                        "pair_id": pid,
                        **{c: row[c] for c in BASE_COLS},
                        "style_violation": int(style[pid]),
                        "subset_name": subset_name,
                        "subset_size": target,
                        "selection_method": method,
                        "source_dataset": "TruthfulQA.csv",
                        "canonical_json": canonical_rel,
                    }
                )

        manifest_rows.append(
            {
                "subset_name": subset_name,
                "subset_size": str(target),
                "csv_path": f"data/subsets/{csv_name}",
                "canonical_json": canonical_rel,
                "selection_method": method,
                "mean_heldout_auc": m["mean_heldout_auc"],
                "std_heldout_auc": m["std_heldout_auc"],
                "mean_dist_to_chance": m["mean_dist_to_chance"],
                "mean_clean_pairs": m["mean_clean_pair_count"],
                "mean_confounded_fraction": m["mean_confounded_fraction"],
                "paper_role": paper_role,
            }
        )

        print(f"Wrote {out_path} ({len(pair_ids)} rows)")

    manifest_path = SUBSETS_DIR / "subset_manifest.csv"
    mf = [
        "subset_name",
        "subset_size",
        "csv_path",
        "canonical_json",
        "selection_method",
        "mean_heldout_auc",
        "std_heldout_auc",
        "mean_dist_to_chance",
        "mean_clean_pairs",
        "mean_confounded_fraction",
        "paper_role",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mf)
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)
    print(f"Wrote {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
