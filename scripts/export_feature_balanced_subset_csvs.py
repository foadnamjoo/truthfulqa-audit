#!/usr/bin/env python3
"""
Export reference TruthfulQA subset CSVs for the fixed-kept feature_balanced
pruning protocol (surface10 audit: ten interpretable surface features; same ordering
as truthfulqa_pruning_final_verification).

Pair lists are deterministic for a chosen GroupShuffleSplit seed. The multi-seed
verification in results/truthfulqa_pruning_final_verification/ reports mean/std
held-out AUC across seeds; these files use split seed = --reference-seed (default
42, i.e. first run of the locked protocol with base_seed 42) as a concrete
downloadable reference for each target kept count.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from search_truthfulqa_pruned_improved import _apply_prefix_keep
from truthfulqa_pruning_utils import load_candidates_with_features, repo_root

BASE_COLS = [
    "Type",
    "Category",
    "Question",
    "Best Answer",
    "Best Incorrect Answer",
]


def sorted_df_feature_balanced(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Match scripts/truthfulqa_pruning_final_verification._sorted_df_feature_balanced."""
    d = df.copy()
    d["len_bin"] = pd.qcut(d["length_gap"], q=4, labels=False, duplicates="drop")
    parts = []
    for b in sorted(d["len_bin"].dropna().unique()):
        sub = d[d["len_bin"] == b]
        parts.append(sub.sample(frac=1.0, random_state=seed + int(b)))
    mixed = pd.concat(parts, ignore_index=True)
    return mixed.sort_values(["negation_flag", "length_gap", "example_id"], ascending=[True, True, True]).reset_index(
        drop=True
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truthfulqa-csv", type=str, default="TruthfulQA.csv")
    p.add_argument("--audit-csv", type=str, default="audits/truthfulqa_style_audit.csv")
    p.add_argument(
        "--summary-csv",
        type=str,
        default="results/truthfulqa_pruning_final_verification/fixed_kept_count_summary.csv",
        help="Used to fill manifest means (feature_balanced rows).",
    )
    p.add_argument("--output-dir", type=str, default="truthfulqaPro")
    p.add_argument("--json-dir", type=str, default="truthfulqaPro/pair_ids")
    p.add_argument("--holdout-fraction", type=float, default=0.25)
    p.add_argument("--reference-seed", type=int, default=42)
    p.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[650, 595, 550, 500, 450, 400, 350, 300],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    tq = (root / args.truthfulqa_csv).resolve()
    audit = (root / args.audit_csv).resolve()
    if not tq.is_file():
        print(f"Missing {tq}", file=sys.stderr)
        return 1
    if not audit.is_file():
        print(f"Missing {audit}", file=sys.stderr)
        return 1

    df_t = pd.read_csv(tq)
    df_a = pd.read_csv(audit)
    if len(df_t) != len(df_a):
        print(f"Row mismatch TruthfulQA={len(df_t)} audit={len(df_a)}", file=sys.stderr)
        return 1
    style = df_a["style_violation"].to_numpy()

    df = load_candidates_with_features(tq, audit)
    groups = df["example_id"].astype(int).to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_fraction, random_state=args.reference_seed)
    tr_idx, ho_idx = next(gss.split(np.arange(len(df)), groups=groups))
    train_ids = set(df.iloc[tr_idx]["example_id"].astype(int).tolist())
    hold_ids = set(df.iloc[ho_idx]["example_id"].astype(int).tolist())

    d_sorted = sorted_df_feature_balanced(df, args.reference_seed)
    summary_path = (root / args.summary_csv).resolve()
    summ = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    summ_fb = summ[summ["method"] == "feature_balanced"].copy() if len(summ) else pd.DataFrame()

    out_dir = (root / args.output_dir).resolve()
    json_dir = (root / args.json_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | float | int]] = []

    for target in args.targets:
        row_fb = summ_fb[summ_fb["target_kept_count"] == target]
        mean_auc = float(row_fb["mean_heldout_auc"].iloc[0]) if len(row_fb) else float("nan")
        std_auc = float(row_fb["std_heldout_auc"].iloc[0]) if len(row_fb) else float("nan")
        mean_conf = float(row_fb["mean_retained_confounded_fraction"].iloc[0]) if len(row_fb) else float("nan")

        kept = _apply_prefix_keep(d_sorted, target)
        pair_ids = [int(x) for x in kept["example_id"].tolist()]
        if len(pair_ids) != target:
            print(f"Expected {target} rows, got {len(pair_ids)}", file=sys.stderr)
            return 1
        n_clean = int((~kept["confounded_flag"].astype(bool)).sum())

        meta = {
            "selection_method": "feature_balanced_length_stratified_prefix",
            "audit_profile": "surface10",
            "target_kept_count": target,
            "reference_split_seed": args.reference_seed,
            "holdout_fraction": args.holdout_fraction,
            "pair_ids": pair_ids,
            "train_pair_count_reference_split": len([p for p in pair_ids if p in train_ids]),
            "hold_pair_count_reference_split": len([p for p in pair_ids if p in hold_ids]),
            "clean_pairs_reference_split": n_clean,
            "confounded_pairs_reference_split": int(kept["confounded_flag"].sum()),
            "verification_note": "Held-out AUC mean/std in manifest are from 10 GroupShuffleSplit seeds (see results/truthfulqa_pruning_final_verification/). This JSON is the pair list for reference_seed only.",
            "mean_heldout_auc_across_seeds": mean_auc,
            "std_heldout_auc_across_seeds": std_auc,
            "mean_retained_confounded_fraction_across_seeds": mean_conf,
            "source_dataset": "TruthfulQA.csv",
            "audit_csv": str(Path("audits/truthfulqa_style_audit.csv")),
        }
        jname = f"pair_ids_{target}_seed{args.reference_seed}.json"
        jpath = json_dir / jname
        jpath.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        csv_name = f"truthfulqaPro_{target}.csv"
        cpath = out_dir / csv_name
        fieldnames = [
            "pair_id",
            *BASE_COLS,
            "style_violation",
            "subset_name",
            "subset_size",
            "selection_method",
            "reference_split_seed",
            "source_dataset",
            "canonical_json",
        ]
        subset_name = f"truthfulqaPro_{target}"
        canonical_rel = f"truthfulqaPro/pair_ids/{jname}"

        with cpath.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for pid in pair_ids:
                row = df_t.iloc[pid]
                w.writerow(
                    {
                        "pair_id": pid,
                        **{c: row[c] for c in BASE_COLS},
                        "style_violation": int(style[pid]),
                        "subset_name": subset_name,
                        "subset_size": target,
                        "selection_method": meta["selection_method"],
                        "reference_split_seed": args.reference_seed,
                        "source_dataset": "TruthfulQA.csv",
                        "canonical_json": canonical_rel,
                    }
                )

        manifest_rows.append(
            {
                "subset_name": subset_name,
                "target_kept_count": target,
                "csv_path": str(Path(args.output_dir) / csv_name),
                "canonical_json": canonical_rel,
                "reference_split_seed": args.reference_seed,
                "mean_heldout_auc": mean_auc,
                "std_heldout_auc": std_auc,
                "mean_retained_confounded_fraction": mean_conf,
                "clean_pairs_reference_seed": n_clean,
                "paper_role": "pruning_verification_reference",
            }
        )

    manifest_path = out_dir / "subset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
        dw = csv.DictWriter(f, fieldnames=fieldnames)
        dw.writeheader()
        for r in manifest_rows:
            dw.writerow(r)

    print(f"Wrote {len(manifest_rows)} CSVs under {out_dir}")
    print(f"Wrote JSON pair lists under {json_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
