#!/usr/bin/env python3
"""
v9 same-question forced-choice aggregator — 7 families.

Reads:
  - stage0_paired_tqa/stage0_paired_classifier_scores_v4.json
      (surface_lr / BGE-large / ModernBERT-base, full + cleaned, per pair)
  - stage0_v9_same_question/v9_<fam>_scores.json  for heavy families
      (Qwen2.5-0.5B, SmolLM2-1.7B, Qwen2.5-3B, Phi-3.5-mini)
  - stage0_paired_tqa/stage0_paired_judge_v4.json  (pair_passes mask)

For each family and variant (full / cleaned), reports pair_accuracy =
fraction of pairs where P_A > P_B (A = TRUE side). Chance = 0.50.
Reports both all-20 and judge-pair-passes (16/20) subsets, matching the
v8 schema so §5.3 can compare the cross-topic (v8) and same-question (v9)
probes side by side.

Output: stage0_v9_same_question/v9_forced_choice_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V4_SCORES = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_classifier_scores_v4.json"
V4_JUDGE  = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_judge_v4.json"
V9_DIR    = REPO_ROOT / "stage0_v9_same_question"
OUT_JSON  = V9_DIR / "v9_forced_choice_results.json"

LIGHT_FAMILIES = ["surface_lr", "BGE-large", "ModernBERT-base"]

HEAVY_FAMILY_FILES = {
    "Qwen2.5-0.5B": V9_DIR / "v9_qwen_scores.json",
    "SmolLM2-1.7B": V9_DIR / "v9_smollm2_scores.json",
    "Qwen2.5-3B":   V9_DIR / "v9_qwen3b_scores.json",
    "Phi-3.5-mini": V9_DIR / "v9_phi35_scores.json",
}

FAMILY_ORDER = LIGHT_FAMILIES + list(HEAVY_FAMILY_FILES.keys())


def _metrics(pA, pB):
    n = len(pA)
    assert n == len(pB)
    correct = sum(1 for a, b in zip(pA, pB) if a > b)
    ties    = sum(1 for a, b in zip(pA, pB) if a == b)
    return {
        "n": n,
        "pair_correct_count": correct,
        "ties": ties,
        "pair_accuracy": correct / n if n else 0.0,
        "mean_P_gap_A_minus_B": sum(a - b for a, b in zip(pA, pB)) / n if n else 0.0,
    }


def _block(full_pA, full_pB, clean_pA, clean_pB):
    full  = _metrics(full_pA, full_pB)
    clean = _metrics(clean_pA, clean_pB)
    return {
        "full": full, "cleaned": clean,
        "drop": full["pair_accuracy"] - clean["pair_accuracy"],
        "gap_drop": full["mean_P_gap_A_minus_B"] - clean["mean_P_gap_A_minus_B"],
    }


def _light_probs(per_pair, fam: str):
    f_pA  = [r["scores"][f"{fam}_full"]["P_A"]    for r in per_pair]
    f_pB  = [r["scores"][f"{fam}_full"]["P_B"]    for r in per_pair]
    c_pA  = [r["scores"][f"{fam}_cleaned"]["P_A"] for r in per_pair]
    c_pB  = [r["scores"][f"{fam}_cleaned"]["P_B"] for r in per_pair]
    return f_pA, f_pB, c_pA, c_pB


def _heavy_probs(rows_by_pid, pair_ids):
    f_pA = [rows_by_pid[p]["P_A_truthful_full"]    for p in pair_ids]
    f_pB = [rows_by_pid[p]["P_B_truthful_full"]    for p in pair_ids]
    c_pA = [rows_by_pid[p]["P_A_truthful_cleaned"] for p in pair_ids]
    c_pB = [rows_by_pid[p]["P_B_truthful_cleaned"] for p in pair_ids]
    return f_pA, f_pB, c_pA, c_pB


def main() -> int:
    d = json.load(open(V4_SCORES))
    all_pairs   = d["per_pair"]
    pass_pairs  = [r for r in all_pairs if r.get("judge_pair_passes")]
    all_pids    = [r["pair_id"] for r in all_pairs]
    pass_pids   = [r["pair_id"] for r in pass_pairs]

    print("=" * 96)
    print("v9 SAME-QUESTION FORCED-CHOICE — 7 families")
    print(f"  source (light): {V4_SCORES.relative_to(REPO_ROOT)}")
    print(f"  source (heavy): {V9_DIR.relative_to(REPO_ROOT)}/v9_<fam>_scores.json")
    print(f"  all pairs          n = {len(all_pairs)}")
    print(f"  judge-pair_passes  n = {len(pass_pairs)}")
    print("=" * 96)

    results = {
        "n_all":  len(all_pairs),
        "n_pass": len(pass_pairs),
        "families": {},
    }

    for fam in LIGHT_FAMILIES:
        all_fp = _light_probs(all_pairs,  fam)
        pp_fp  = _light_probs(pass_pairs, fam)
        results["families"][fam] = {
            "status": "ok",
            "all":  _block(*all_fp),
            "pass": _block(*pp_fp),
        }

    for fam, path in HEAVY_FAMILY_FILES.items():
        if not path.exists():
            results["families"][fam] = {"status": "missing",
                                        "path": str(path.relative_to(REPO_ROOT))}
            continue
        rows = json.load(open(path))
        by_pid = {int(r["pair_id"]): r for r in rows}
        missing = [p for p in all_pids if p not in by_pid]
        if missing:
            results["families"][fam] = {"status": "incomplete",
                                        "path": str(path.relative_to(REPO_ROOT)),
                                        "missing_pair_ids": missing}
            continue
        all_fp = _heavy_probs(by_pid, all_pids)
        pp_fp  = _heavy_probs(by_pid, pass_pids)
        results["families"][fam] = {
            "status": "ok",
            "path":   str(path.relative_to(REPO_ROOT)),
            "all":  _block(*all_fp),
            "pass": _block(*pp_fp),
        }

    def _row(fam, blk):
        f = blk["full"]; c = blk["cleaned"]
        return (f"{fam:<16} {f['pair_accuracy']:>9.3f} {c['pair_accuracy']:>12.3f} "
                f"{blk['drop']:>+9.3f}   {f['mean_P_gap_A_minus_B']:>+9.3f} "
                f"{c['mean_P_gap_A_minus_B']:>+12.3f} {blk['gap_drop']:>+9.3f}")

    def _hdr():
        return (f"{'Family':<16} {'full_acc':>9} {'cleaned_acc':>12} {'drop':>9}   "
                f"{'full_gap':>9} {'cleaned_gap':>12} {'gap_drop':>9}")

    print(f"\n--- all pairs (n={len(all_pairs)}) ---")
    print(_hdr())
    print("-" * 96)
    for fam in FAMILY_ORDER:
        block = results["families"].get(fam, {})
        if block.get("status") != "ok":
            print(f"{fam:<16}  (missing heavy-family score file: {block.get('path')})")
            continue
        print(_row(fam, block["all"]))

    print(f"\n--- judge-pair_passes (n={len(pass_pairs)}) ---")
    print(_hdr())
    print("-" * 96)
    for fam in FAMILY_ORDER:
        block = results["families"].get(fam, {})
        if block.get("status") != "ok":
            continue
        print(_row(fam, block["pass"]))

    V9_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print("\nInterpretation:")
    print("  pair_accuracy = P(classifier picks TRUE side in same-question forced choice).")
    print("  Chance = 0.500.  drop = full_acc - cleaned_acc.")
    print("  NEGATIVE drop  =>  cleaning HELPS the classifier flip to the right side.")
    print("  NEGATIVE gap   =>  classifier leans toward the FALSE side (fooled by surface).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
