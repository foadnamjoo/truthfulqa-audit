#!/usr/bin/env python3
"""
Aggregate v4 multi-seed robustness results across 5 seeds at n=40.

Reads per-seed score JSONs at:
  stage0_paired_tqa/v4_multi_seed/scores_seed{S}_n{N}.json
  stage0_paired_tqa/v4_multi_seed/judge_seed{S}_n{N}.json   (for pair_passes)
  stage0_paired_tqa/v4_multi_seed/generations_seed{S}_n{N}.json (provenance)

Prints:
  1. Per-seed pair_passes counts.
  2. Cleaning delta (pair_acc_cleaned - pair_acc_full) per classifier per seed,
     plus mean and stdev across seeds. Same on the pair_passes-only subset.
  3. Mean gap (mean P(A) - mean P(B)) per classifier per seed, plus mean and
     stdev. Same on the pair_passes-only subset.

Saves the full aggregate to
  stage0_paired_tqa/v4_multi_seed/summary_5seeds_n{N}.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "stage0_paired_tqa" / "v4_multi_seed"

FAMILIES = ["surface_lr", "BGE-large", "ModernBERT-base"]


def _load(p: Path) -> dict | list:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean_stdev(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[1, 7, 13, 42, 99])
    p.add_argument("--n", type=int, default=40)
    args = p.parse_args()
    seeds: list[int] = sorted(args.seeds)
    n = args.n

    print("=" * 88)
    print(f"V4 MULTI-SEED ROBUSTNESS  ({len(seeds)} seeds x n={n})")
    print("=" * 88)

    per_seed: dict[int, dict] = {}
    for s in seeds:
        sc_path = ROOT / f"scores_seed{s}_n{n}.json"
        ju_path = ROOT / f"judge_seed{s}_n{n}.json"
        ge_path = ROOT / f"generations_seed{s}_n{n}.json"
        sel_path = ROOT / f"selected_pair_ids_v4_seed{s}.json"
        for q in (sc_path, ju_path, ge_path, sel_path):
            if not q.exists():
                raise FileNotFoundError(f"Missing artifact for seed={s}: {q}")
        sc = _load(sc_path)
        ju = _load(ju_path)
        ge = _load(ge_path)
        sel = _load(sel_path)
        per_seed[s] = {
            "scores": sc,
            "judge": ju,
            "generations": ge,
            "selected": sel,
        }

    # -------- Per-seed pair_passes counts --------
    print("\nPer-seed pair_passes:")
    pass_counts: dict[int, int] = {}
    for s in seeds:
        sc = per_seed[s]["scores"]
        n_total = sc["n_pairs"]
        n_pass = sc["n_pair_passes"]
        pass_counts[s] = n_pass
        print(f"  seed {s:>3}: {n_pass}/{n_total}")
    pp_vals = [pass_counts[s] / n for s in seeds]
    pp_mean, pp_std = _mean_stdev(pp_vals)
    print(f"  mean = {pp_mean:.3f}  stdev = {pp_std:.3f}  "
          f"(per-pair fraction)")

    # -------- Cleaning delta tables --------
    def _delta(scores: dict, fam: str, subset: str) -> float:
        agg = scores["aggregates_all"] if subset == "all" \
            else scores["aggregates_pair_passes"]
        full_acc = agg[f"{fam}_full"]["pair_accuracy"]
        clean_acc = agg[f"{fam}_cleaned"]["pair_accuracy"]
        if full_acc != full_acc or clean_acc != clean_acc:
            return float("nan")
        return clean_acc - full_acc

    def _gap(scores: dict, fam: str, split: str, subset: str) -> float:
        agg = scores["aggregates_all"] if subset == "all" \
            else scores["aggregates_pair_passes"]
        return agg[f"{fam}_{split}"]["mean_gap"]

    def _print_delta_table(title: str, subset: str) -> dict:
        print("\n" + "-" * 88)
        print(f"CLEANING DELTA  (pair_acc_cleaned - pair_acc_full)  [{title}]")
        print("-" * 88)
        h = "  Classifier        " + "  ".join(
            f"seed={s:>3}".rjust(8) for s in seeds
        ) + "      mean    stdev"
        print(h)
        out: dict[str, dict] = {}
        for fam in FAMILIES:
            vals = [_delta(per_seed[s]["scores"], fam, subset) for s in seeds]
            m, sd = _mean_stdev(vals)
            cells = "  ".join(f"{v:+.3f}".rjust(8) for v in vals)
            out[fam] = {"per_seed": dict(zip(seeds, vals)),
                        "mean": m, "stdev": sd}
            print(f"  {fam:<17s} {cells}    {m:+.3f}   {sd:.3f}")
        return out

    def _print_gap_table(title: str, split: str, subset: str) -> dict:
        print("\n" + "-" * 88)
        print(f"MEAN GAP  (mean P(A) - mean P(B))  [{title}, {split}-trained]")
        print("-" * 88)
        h = "  Classifier        " + "  ".join(
            f"seed={s:>3}".rjust(8) for s in seeds
        ) + "      mean    stdev"
        print(h)
        out: dict[str, dict] = {}
        for fam in FAMILIES:
            vals = [_gap(per_seed[s]["scores"], fam, split, subset)
                    for s in seeds]
            m, sd = _mean_stdev(vals)
            cells = "  ".join(f"{v:+.3f}".rjust(8) for v in vals)
            out[fam] = {"per_seed": dict(zip(seeds, vals)),
                        "mean": m, "stdev": sd}
            print(f"  {fam:<17s} {cells}    {m:+.3f}   {sd:.3f}")
        return out

    delta_all  = _print_delta_table("ALL n=40 per seed", "all")
    delta_pass = _print_delta_table("pair_passes subset only", "pair_passes")

    print("\n  Pair accuracy per (family, split) per seed (all-40 subset):")
    print("  " + "Classifier        TrainedOn  " + "  ".join(
        f"seed={s:>3}".rjust(8) for s in seeds))
    for fam in FAMILIES:
        for split in ["full", "cleaned"]:
            vals = [per_seed[s]["scores"]["aggregates_all"]
                    [f"{fam}_{split}"]["pair_accuracy"] for s in seeds]
            cells = "  ".join(f"{v:.3f}".rjust(8) for v in vals)
            print(f"  {fam:<17s} {split:<9s} {cells}")

    gap_full_all   = _print_gap_table("all n=40", "full",    "all")
    gap_clean_all  = _print_gap_table("all n=40", "cleaned", "all")
    gap_full_pass  = _print_gap_table("pair_passes subset", "full",    "pair_passes")
    gap_clean_pass = _print_gap_table("pair_passes subset", "cleaned", "pair_passes")

    # -------- Per-seed substitutions / blocklisted --------
    print("\n" + "-" * 88)
    print("SAMPLING PROVENANCE")
    print("-" * 88)
    for s in seeds:
        sel = per_seed[s]["selected"]
        anchor = sel.get("anchor_strategy", "?")
        bl = sel.get("blocklisted_pair_ids", [])
        print(f"  seed {s}: anchor_strategy={anchor!r}  "
              f"blocklisted={bl}")

    # -------- Save aggregate JSON --------
    summary = {
        "seeds":   seeds,
        "n":       n,
        "per_seed_pair_passes": pass_counts,
        "per_seed_pair_passes_fraction": {s: pass_counts[s] / n
                                          for s in seeds},
        "cleaning_delta_all":         delta_all,
        "cleaning_delta_pair_passes": delta_pass,
        "mean_gap_full_all":          gap_full_all,
        "mean_gap_cleaned_all":       gap_clean_all,
        "mean_gap_full_pair_passes":  gap_full_pass,
        "mean_gap_cleaned_pair_passes": gap_clean_pass,
        "per_seed_aggregates_all": {
            s: per_seed[s]["scores"]["aggregates_all"] for s in seeds
        },
        "per_seed_aggregates_pair_passes": {
            s: per_seed[s]["scores"]["aggregates_pair_passes"] for s in seeds
        },
        "per_seed_selected_pair_ids": {
            s: per_seed[s]["selected"].get("selected_pair_ids", [])
            for s in seeds
        },
        "per_seed_provenance": {
            s: {k: v for k, v in per_seed[s]["selected"].items()
                if k != "selected_pair_ids"} for s in seeds
        },
    }
    out_path = ROOT / f"summary_{len(seeds)}seeds_n{n}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
