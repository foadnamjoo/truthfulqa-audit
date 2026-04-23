#!/usr/bin/env python3
"""
v8 bilateral cue-inversion probe — Step 4 forced-choice scorer.

For each of the 20 pairs, each classifier (full, cleaned) outputs a pair
prediction: predicted_true = side with higher P(truthful). Pair accuracy =
(predicted_true == A) / 20. Chance = 0.50.

Primary metric: pair_accuracy_full, pair_accuracy_cleaned, drop.
Secondary: mean P(A) - P(B) gap per family.

Sources of P(A), P(B) per classifier:
  3 families reused directly from the v7a-v1 scoring file (same corpus):
    - surface_lr     full/cleaned  <- stage0_singleton_v7a_classifier_scores.json
    - BGE-large      full/cleaned  <- stage0_singleton_v7a_classifier_scores.json
    - ModernBERT     full/cleaned  <- stage0_singleton_v7a_classifier_scores.json

  4 families require running the embedding model + LR on the v7a-v1 A-side
  and §5.1 B-side. If the per-family score JSONs already exist on disk, they
  are loaded; otherwise the family is skipped and flagged.
    - Qwen2.5-0.5B   <- stage0_singleton_v8_qwen_scores.json
    - SmolLM2-1.7B   <- stage0_singleton_v8_smollm2_scores.json
    - Qwen2.5-3B     <- stage0_singleton_v8_qwen3b_scores.json
    - Phi-3.5-mini   <- stage0_singleton_v8_phi35_scores.json

Output: stage0_v8_inverted_bilateral/v8_forced_choice_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V8_DIR = REPO_ROOT / "stage0_v8_inverted_bilateral"
PAIRS_JSONL = V8_DIR / "v8_bilateral_inverted.jsonl"
V7A_V1_SCORES = REPO_ROOT / "stage0_v7a_bilateral" / "stage0_singleton_v7a_classifier_scores.json"
OUT_JSON = V8_DIR / "v8_forced_choice_results.json"

# v8-native score files for the 4 heavy families; produced by a separate
# runner when the torch/transformers environment is available. Expected
# schema: list of {"pair_id": int, "P_A_truthful_full": float,
# "P_B_truthful_full": float, "P_A_truthful_cleaned": float,
# "P_B_truthful_cleaned": float}.
HEAVY_FAMILY_FILES = {
    "Qwen2.5-0.5B":   V8_DIR / "v8_qwen_scores.json",
    "SmolLM2-1.7B":   V8_DIR / "v8_smollm2_scores.json",
    "Qwen2.5-3B":     V8_DIR / "v8_qwen3b_scores.json",
    "Phi-3.5-mini":   V8_DIR / "v8_phi35_scores.json",
}

REUSE_FAMILIES = ["surface_lr", "BGE-large", "ModernBERT-base"]


def load_pairs():
    pairs = []
    with open(PAIRS_JSONL) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def forced_choice_metrics(pA_list, pB_list):
    n = len(pA_list)
    assert n == len(pB_list)
    pair_correct = [1 if a > b else 0 for a, b in zip(pA_list, pB_list)]
    ties = sum(1 for a, b in zip(pA_list, pB_list) if a == b)
    acc = sum(pair_correct) / n if n else 0.0
    mean_gap = sum(a - b for a, b in zip(pA_list, pB_list)) / n if n else 0.0
    return {
        "n": n,
        "pair_accuracy": acc,
        "pair_correct_count": sum(pair_correct),
        "ties": ties,
        "mean_P_gap_A_minus_B": mean_gap,
    }


def load_reuse_scores(pair_ids):
    d = json.load(open(V7A_V1_SCORES))
    per_pair = d["per_pair_same_index"]
    indexed = {r["a_id"]: r for r in per_pair}
    out = {}
    for fam in REUSE_FAMILIES:
        out[fam] = {"full": {"pA": [], "pB": []}, "cleaned": {"pA": [], "pB": []}}
        for pid in pair_ids:
            row = indexed[pid]
            for var, key in [("full", f"{fam}_full"), ("cleaned", f"{fam}_cleaned")]:
                s = row["scores"][key]
                out[fam][var]["pA"].append(s["P_A_truthful"])
                out[fam][var]["pB"].append(s["P_B_truthful"])
    return out


def load_heavy_scores(pair_ids):
    out = {}
    for fam, path in HEAVY_FAMILY_FILES.items():
        if not path.exists():
            out[fam] = {"status": "missing", "path": str(path.relative_to(REPO_ROOT))}
            continue
        rows = json.load(open(path))
        indexed = {int(r["pair_id"]): r for r in rows}
        data = {"full": {"pA": [], "pB": []}, "cleaned": {"pA": [], "pB": []}}
        for pid in pair_ids:
            if pid not in indexed:
                out[fam] = {"status": "incomplete", "path": str(path.relative_to(REPO_ROOT)),
                            "missing_pair_id": pid}
                break
            r = indexed[pid]
            data["full"]["pA"].append(r["P_A_truthful_full"])
            data["full"]["pB"].append(r["P_B_truthful_full"])
            data["cleaned"]["pA"].append(r["P_A_truthful_cleaned"])
            data["cleaned"]["pB"].append(r["P_B_truthful_cleaned"])
        else:
            out[fam] = {"status": "ok", "data": data, "path": str(path.relative_to(REPO_ROOT))}
    return out


def main():
    pairs = load_pairs()
    pair_ids = [p["pair_id"] for p in pairs]
    print(f"Loaded {len(pairs)} pairs from {PAIRS_JSONL.relative_to(REPO_ROOT)}")

    results = {"n_pairs": len(pairs), "pair_ids": pair_ids, "families": {}}

    reuse = load_reuse_scores(pair_ids)
    for fam, data in reuse.items():
        full = forced_choice_metrics(data["full"]["pA"], data["full"]["pB"])
        cleaned = forced_choice_metrics(data["cleaned"]["pA"], data["cleaned"]["pB"])
        results["families"][fam] = {
            "source": "reused from stage0_singleton_v7a_classifier_scores.json",
            "full": full, "cleaned": cleaned,
            "accuracy_drop_full_minus_cleaned": full["pair_accuracy"] - cleaned["pair_accuracy"],
            "mean_gap_drop_full_minus_cleaned": full["mean_P_gap_A_minus_B"] - cleaned["mean_P_gap_A_minus_B"],
        }

    heavy = load_heavy_scores(pair_ids)
    for fam, block in heavy.items():
        if block["status"] != "ok":
            results["families"][fam] = {"source": block["path"], "status": block["status"]}
            continue
        data = block["data"]
        full = forced_choice_metrics(data["full"]["pA"], data["full"]["pB"])
        cleaned = forced_choice_metrics(data["cleaned"]["pA"], data["cleaned"]["pB"])
        results["families"][fam] = {
            "source": block["path"], "status": "ok",
            "full": full, "cleaned": cleaned,
            "accuracy_drop_full_minus_cleaned": full["pair_accuracy"] - cleaned["pair_accuracy"],
            "mean_gap_drop_full_minus_cleaned": full["mean_P_gap_A_minus_B"] - cleaned["mean_P_gap_A_minus_B"],
        }

    V8_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 88)
    print("v8 FORCED-CHOICE PAIR ACCURACY  (n=20, chance=0.50)")
    print("=" * 88)
    print(f"{'Family':<18} {'full_acc':>9} {'cleaned_acc':>12} {'drop':>9}   "
          f"{'full_gap':>9} {'cleaned_gap':>12} {'gap_drop':>9}")
    print("-" * 88)
    family_order = REUSE_FAMILIES + list(HEAVY_FAMILY_FILES.keys())
    for fam in family_order:
        block = results["families"].get(fam, {})
        if block.get("status") in ("missing", "incomplete"):
            print(f"{fam:<18} {'(pending heavy-family run)':>42}   source={block.get('source')}")
            continue
        f_ = block["full"]; c_ = block["cleaned"]
        print(f"{fam:<18} {f_['pair_accuracy']:>9.3f} {c_['pair_accuracy']:>12.3f} "
              f"{block['accuracy_drop_full_minus_cleaned']:>+9.3f}   "
              f"{f_['mean_P_gap_A_minus_B']:>+9.3f} {c_['mean_P_gap_A_minus_B']:>+12.3f} "
              f"{block['mean_gap_drop_full_minus_cleaned']:>+9.3f}")
    print("=" * 88)
    print()
    print("Interpretation:")
    print("  pair_accuracy = P(classifier picks the TRUE-with-FALSE-cues A-side).")
    print("  Chance = 0.500. Negative drop (full - cleaned < 0) = cleaning HELPS.")
    print("  Negative gap = classifier leans toward the FALSE side (fooled).")
    print()
    print(f"Results JSON: {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
