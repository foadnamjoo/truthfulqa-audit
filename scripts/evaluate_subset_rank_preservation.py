#!/usr/bin/env python3
"""
Measure whether TruthfulQA subsets preserve model ranking vs full benchmark (TinyBenchmarks-style).

Loads predictions with the same glob as make_final_tables.py (excluding example_model_predictions.csv).
Within-file duplicate (model_name, pair_id): first row wins.
Across files (sorted paths): later file overwrites (last-file-wins), matching make_final_tables.compute_by_model_last_wins.

When models have unequal pair_id coverage, all accuracies use the intersection of pair_ids across models;
documented in results/subset_evaluation_preservation/config.json.

Does not modify subset-selection pipelines.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from scipy.stats import kendalltau, spearmanr

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

def rel_under(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def iter_prediction_rows(path: Path) -> Iterable[Tuple[str, int, int]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"model_name", "pair_id", "correct"}
        fields = set(reader.fieldnames or [])
        missing = need - fields
        if missing:
            raise KeyError(f"{path} missing required columns: {sorted(missing)}")
        for r in reader:
            yield (str(r["model_name"]), int(r["pair_id"]), int(r["correct"]))


def discover_pred_paths(root: Path, pred_glob: str, exclude_example: bool) -> List[Path]:
    pred_paths = sorted(root.glob(pred_glob))
    if not pred_paths and pred_glob == "data/predictions/model_predictions*.csv":
        pred_paths = sorted(root.glob("model_predictions*.csv"))
    if exclude_example:
        pred_paths = [p for p in pred_paths if p.name != "example_model_predictions.csv"]
    return pred_paths


def load_predictions_merged(
    pred_paths: List[Path],
    n_pairs: int,
) -> Dict[Tuple[str, int], int]:
    """
    Per file: first (model, pair_id) wins within the file.
    Across files (sorted order): later files overwrite -> last-file-wins on conflicts.
    """
    merged: Dict[Tuple[str, int], int] = {}
    for p in pred_paths:
        local: Dict[Tuple[str, int], int] = {}
        seen: Set[Tuple[str, int]] = set()
        for model, pair_id, correct in iter_prediction_rows(p):
            if not (0 <= pair_id < n_pairs):
                continue
            key = (model, pair_id)
            if key in seen:
                continue
            seen.add(key)
            local[key] = int(correct)
        for k, v in local.items():
            merged[k] = v
    return merged


def pair_intersection_across_models(merged: Dict[Tuple[str, int], int]) -> Tuple[List[str], Set[int]]:
    by_model: Dict[str, Set[int]] = defaultdict(set)
    for (model, pid), _ in merged.items():
        by_model[model].add(pid)
    if not by_model:
        return [], set()
    common: Optional[Set[int]] = None
    for s in by_model.values():
        common = s if common is None else common & s
    assert common is not None
    models = sorted(by_model.keys())
    return models, common


def mean_accuracy_on_ids(merged: Dict[Tuple[str, int], int], model: str, ids: List[int]) -> Optional[float]:
    if not ids:
        return None
    tot = 0
    ok = 0
    for pid in ids:
        key = (model, pid)
        if key not in merged:
            return None
        tot += 1
        ok += int(merged[key] == 1)
    return ok / tot if tot else None


def load_pair_ids_from_json(root: Path, rel_json: str) -> List[int]:
    path = (root / rel_json).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if "pair_ids" not in data:
        raise KeyError(f"{path} missing 'pair_ids'")
    return [int(x) for x in data["pair_ids"]]


def retained_pairs_from_manifest_row(row: Dict[str, str]) -> Optional[int]:
    for k in ("retained_pairs", "pruned_pair_count", "target_kept_count"):
        if k in row and str(row[k]).strip() != "":
            try:
                return int(float(row[k]))
            except ValueError:
                pass
    return None


def grouped_metrics_from_manifest_row(row: Dict[str, str]) -> Tuple[Optional[float], Optional[float]]:
    auc: Optional[float] = None
    acc: Optional[float] = None
    if "grouped_cv_oof_auc" in row and str(row["grouped_cv_oof_auc"]).strip():
        try:
            auc = float(row["grouped_cv_oof_auc"])
        except ValueError:
            pass
    if "mean_heldout_auc" in row and str(row["mean_heldout_auc"]).strip():
        try:
            auc = float(row["mean_heldout_auc"])
        except ValueError:
            pass
    if "grouped_cv_oof_accuracy" in row and str(row["grouped_cv_oof_accuracy"]).strip():
        try:
            acc = float(row["grouped_cv_oof_accuracy"])
        except ValueError:
            pass
    return auc, acc


def subset_family_from_name(name: str) -> str:
    if name.startswith("truthfulqaPro_"):
        return "fixed_prefix"
    if name.startswith("truthfulqaAuditPruneImproved_"):
        return "improved_search"
    if name.startswith("truthfulqaAuditPrune_"):
        return "audit_prune"
    return "other"


def collect_subset_jobs(root: Path, manifest_paths: List[Path]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for mp in manifest_paths:
        if not mp.exists():
            continue
        with mp.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("subset_name", "").strip()
                cj = row.get("canonical_json", "").strip()
                if not name or not cj:
                    continue
                rk = retained_pairs_from_manifest_row(row)
                auc, acc = grouped_metrics_from_manifest_row(row)
                jobs.append(
                    {
                        "subset_name": name,
                        "subset_family": subset_family_from_name(name),
                        "manifest_path": rel_under(root, mp),
                        "canonical_json": cj,
                        "retained_pairs_manifest": rk,
                        "grouped_cv_auc": auc,
                        "grouped_cv_accuracy": acc,
                    }
                )
    return jobs


def correlations(
    acc_full: List[float],
    acc_sub: List[float],
) -> Tuple[float, float]:
    if len(acc_full) < 2:
        return float("nan"), float("nan")
    r_s, _ = spearmanr(acc_full, acc_sub)
    r_k, _ = kendalltau(acc_full, acc_sub)
    rs = float(r_s) if r_s == r_s else float("nan")
    rk = float(r_k) if r_k == r_k else float("nan")
    return rs, rk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=str, default=".", help="Repository root")
    p.add_argument(
        "--audit-csv",
        type=str,
        default="audits/truthfulqa_style_audit.csv",
        help="Used only for n_pairs = number of rows (790)",
    )
    p.add_argument(
        "--pred-glob",
        type=str,
        default="data/predictions/model_predictions*.csv",
        help="Same default as make_final_tables.py",
    )
    p.add_argument(
        "--manifests",
        type=str,
        nargs="*",
        default=[
            "truthfulqaPro/subset_manifest.csv",
            "truthfulqaAuditPrune/subset_manifest.csv",
            "truthfulqaAuditPruneImproved/subset_manifest.csv",
        ],
        help="Subset manifests to evaluate",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/subset_evaluation_preservation",
        help="Output directory under root",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    audit_path = (root / args.audit_csv).resolve()
    if not audit_path.is_file():
        raise SystemExit(f"Missing audit CSV: {audit_path}")
    with audit_path.open(newline="", encoding="utf-8") as f:
        n_pairs = sum(1 for _ in csv.DictReader(f))

    pred_paths = discover_pred_paths(root, args.pred_glob, exclude_example=True)
    if not pred_paths:
        raise SystemExit(f"No prediction files under {root} matching {args.pred_glob}")

    merged = load_predictions_merged(pred_paths, n_pairs)
    models, intersection_ids = pair_intersection_across_models(merged)
    intersection_list = sorted(intersection_ids)
    if not intersection_list:
        raise SystemExit(
            "Empty intersection of pair_ids across models: cannot align accuracies. Check prediction files."
        )

    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "pred_glob": args.pred_glob,
        "pred_files": [rel_under(root, p) for p in pred_paths],
        "n_prediction_files": len(pred_paths),
        "models_in_correlation": models,
        "within_file_duplicate_policy": "first_row_wins",
        "across_file_duplicate_policy": "last_file_wins_sorted_paths",
        "n_pairs_audit_csv": n_pairs,
        "n_models_in_merged_store": len(models),
        "intersection_pair_ids_count": len(intersection_ids),
        "intersection_note": (
            "acc_full and acc_subset for correlations use only pair_ids in this intersection "
            "across all models with any prediction in merged store."
        ),
        "manifests": args.manifests,
        "scipy": "spearmanr, kendalltau",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    manifest_paths = [(root / m).resolve() for m in args.manifests]
    jobs = collect_subset_jobs(root, manifest_paths)

    summary_rows: List[Dict[str, Any]] = []
    breakdown_rows: List[Dict[str, Any]] = []

    best_kendall: Tuple[float, str] = (-2.0, "")
    best_spearman: Tuple[float, str] = (-2.0, "")

    for job in jobs:
        name = job["subset_name"]
        try:
            pids = load_pair_ids_from_json(root, job["canonical_json"])
        except Exception as e:
            rp = job["retained_pairs_manifest"] or 0
            summary_rows.append(
                {
                    "subset_name": name,
                    "subset_family": job["subset_family"],
                    "retained_pairs": rp,
                    "retained_fraction": (rp / max(1, n_pairs)) if rp else float("nan"),
                    "grouped_cv_auc": job["grouped_cv_auc"],
                    "grouped_cv_accuracy": job["grouped_cv_accuracy"],
                    "spearman_correlation": float("nan"),
                    "kendall_tau": float("nan"),
                    "n_models_correlation": 0,
                    "intersection_n": len(intersection_list),
                    "eval_pair_count": 0,
                    "error": str(e),
                }
            )
            continue

        subset_ids_sorted = sorted(set(pids) & set(intersection_list))
        if not subset_ids_sorted:
            summary_rows.append(
                {
                    "subset_name": name,
                    "subset_family": job["subset_family"],
                    "retained_pairs": len(pids),
                    "retained_fraction": len(pids) / max(1, n_pairs),
                    "grouped_cv_auc": job["grouped_cv_auc"],
                    "grouped_cv_accuracy": job["grouped_cv_accuracy"],
                    "spearman_correlation": float("nan"),
                    "kendall_tau": float("nan"),
                    "n_models": len(models),
                    "intersection_n": len(intersection_list),
                    "eval_pair_count": 0,
                    "error": "empty subset after intersection with model coverage",
                }
            )
            continue

        acc_full_list: List[float] = []
        acc_sub_list: List[float] = []
        model_order: List[str] = []

        for m in models:
            af = mean_accuracy_on_ids(merged, m, intersection_list)
            asub = mean_accuracy_on_ids(merged, m, subset_ids_sorted)
            if af is None or asub is None:
                continue
            model_order.append(m)
            acc_full_list.append(af)
            acc_sub_list.append(asub)
            breakdown_rows.append(
                {
                    "subset_name": name,
                    "subset_family": job["subset_family"],
                    "model_name": m,
                    "prediction_source": f"merged:{args.pred_glob};within_file=first_wins;across_files=last_wins",
                    "acc_full_intersection": af,
                    "acc_subset": asub,
                    "n_pairs_full_eval": len(intersection_list),
                    "n_pairs_subset_eval": len(subset_ids_sorted),
                    "intersection_n_global": len(intersection_list),
                }
            )

        sp, kd = correlations(acc_full_list, acc_sub_list)
        rk = job["retained_pairs_manifest"]
        if rk is None:
            rk = len(pids)

        summary_rows.append(
            {
                "subset_name": name,
                "subset_family": job["subset_family"],
                "retained_pairs": len(pids),
                "retained_fraction": len(pids) / max(1, n_pairs),
                "grouped_cv_auc": job["grouped_cv_auc"],
                "grouped_cv_accuracy": job["grouped_cv_accuracy"],
                "spearman_correlation": sp,
                "kendall_tau": kd,
                "n_models_correlation": len(model_order),
                "intersection_n": len(intersection_list),
                "eval_pair_count": len(subset_ids_sorted),
                "error": "",
            }
        )

        if kd == kd and kd > best_kendall[0]:
            best_kendall = (kd, name)
        if sp == sp and sp > best_spearman[0]:
            best_spearman = (sp, name)

    summ_path = out_dir / "summary_table.csv"
    if summary_rows:
        fields = list(summary_rows[0].keys())
        with summ_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    br_path = out_dir / "per_model_accuracy_breakdown.csv"
    if breakdown_rows:
        bf = list(breakdown_rows[0].keys())
        with br_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=bf)
            w.writeheader()
            for r in breakdown_rows:
                w.writerow(r)

    print(f"Wrote {summ_path}")
    print(f"Wrote {br_path}")
    print(f"Models in merged store: {len(models)}; intersection pair_ids: {len(intersection_list)}")
    print("\nCorrelations by subset (Spearman, Kendall):")
    for r in summary_rows:
        if r.get("error"):
            print(f"  {r['subset_name']}: ERROR {r['error']}")
        else:
            print(
                f"  {r['subset_name']}: spearman={r['spearman_correlation']:.4f} "
                f"kendall={r['kendall_tau']:.4f} (n_models={r.get('n_models_correlation', 0)})"
            )
    if best_kendall[1]:
        print(f"\nBest Kendall tau (rank preservation): {best_kendall[1]} tau={best_kendall[0]:.4f}")
    if best_spearman[1]:
        print(f"Best Spearman: {best_spearman[1]} rho={best_spearman[0]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
