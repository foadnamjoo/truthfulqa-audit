#!/usr/bin/env python3
"""
v9 same-question forced-choice aggregator (3 light families only).

Reuses stage0_paired_tqa/stage0_paired_classifier_scores_v4.json (already
computed per-pair P_A, P_B for surface_lr / BGE-large / ModernBERT-base)
and reports pair_accuracy_full / pair_accuracy_cleaned / drop, matching
the v8 schema so §5.3 can present the two probes side by side.

Uses judge_pair_passes == True to restrict to pairs where the judge
confirmed A=TRUE AND B=FALSE (16/20).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V4_SCORES = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_classifier_scores_v4.json"
V9_DIR = REPO_ROOT / "stage0_v9_same_question"
OUT_JSON = V9_DIR / "v9_forced_choice_results.json"

FAMILIES = ["surface_lr", "BGE-large", "ModernBERT-base"]


def metrics(pairs, family, variant):
    key = f"{family}_{variant}"
    pA = [r["scores"][key]["P_A"] for r in pairs]
    pB = [r["scores"][key]["P_B"] for r in pairs]
    n = len(pairs)
    correct = sum(1 for a, b in zip(pA, pB) if a > b)
    return {
        "n": n,
        "pair_correct_count": correct,
        "pair_accuracy": correct / n if n else 0.0,
        "mean_P_gap_A_minus_B": sum(a - b for a, b in zip(pA, pB)) / n if n else 0.0,
    }


def main():
    d = json.load(open(V4_SCORES))
    pairs = d["per_pair"]
    pass_pairs = [r for r in pairs if r.get("judge_pair_passes")]

    print("=" * 88)
    print("v9 SAME-QUESTION FORCED-CHOICE — 3 light families")
    print(f"  source: {V4_SCORES.relative_to(REPO_ROOT)}")
    print(f"  all pairs:          n={len(pairs)}")
    print(f"  judge-passes pairs: n={len(pass_pairs)}")
    print("=" * 88)

    results = {"n_all": len(pairs), "n_pass": len(pass_pairs), "families": {}}
    for fam in FAMILIES:
        full_all = metrics(pairs, fam, "full")
        clean_all = metrics(pairs, fam, "cleaned")
        full_pass = metrics(pass_pairs, fam, "full")
        clean_pass = metrics(pass_pairs, fam, "cleaned")
        results["families"][fam] = {
            "all": {"full": full_all, "cleaned": clean_all,
                    "drop": full_all["pair_accuracy"] - clean_all["pair_accuracy"],
                    "gap_drop": full_all["mean_P_gap_A_minus_B"] - clean_all["mean_P_gap_A_minus_B"]},
            "pass": {"full": full_pass, "cleaned": clean_pass,
                     "drop": full_pass["pair_accuracy"] - clean_pass["pair_accuracy"],
                     "gap_drop": full_pass["mean_P_gap_A_minus_B"] - clean_pass["mean_P_gap_A_minus_B"]},
        }

    def row(fam, blk):
        f = blk["full"]; c = blk["cleaned"]
        return (f"{fam:<18} {f['pair_accuracy']:>9.3f} {c['pair_accuracy']:>12.3f} "
                f"{blk['drop']:>+9.3f}   {f['mean_P_gap_A_minus_B']:>+9.3f} "
                f"{c['mean_P_gap_A_minus_B']:>+12.3f} {blk['gap_drop']:>+9.3f}")

    print()
    print(f"--- all pairs (n={len(pairs)}) ---")
    print(f"{'Family':<18} {'full_acc':>9} {'cleaned_acc':>12} {'drop':>9}   "
          f"{'full_gap':>9} {'cleaned_gap':>12} {'gap_drop':>9}")
    print("-" * 88)
    for fam in FAMILIES:
        print(row(fam, results["families"][fam]["all"]))

    print()
    print(f"--- judge-pair_passes (n={len(pass_pairs)}) ---")
    print(f"{'Family':<18} {'full_acc':>9} {'cleaned_acc':>12} {'drop':>9}   "
          f"{'full_gap':>9} {'cleaned_gap':>12} {'gap_drop':>9}")
    print("-" * 88)
    for fam in FAMILIES:
        print(row(fam, results["families"][fam]["pass"]))

    V9_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
