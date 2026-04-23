#!/usr/bin/env python3
"""
v8 heavy-family runner (Step 4b).

Reuses the existing v7a-v3 scoring scripts for the 4 heavy families
(Qwen2.5-0.5B, SmolLM2-1.7B, Qwen2.5-3B, Phi-3.5-mini) but rewires their
A-side input from v7a-v3 (22-26 words, length-matched) to v7a-v1 (5-15
words, FALSE-class cues = the v8 A-side).

The §5.1 B-side is unchanged across v1/v3/v8.

For each family, computes P_A / P_B for full and cleaned pickles and
writes a per-pair JSON to stage0_v8_inverted_bilateral/ in the schema
consumed by scripts/score_v8_forced_choice.py (see HEAVY_FAMILY_FILES).

Usage (on a machine with torch / transformers / sklearn / joblib):

    # One family at a time (recommended for 3B models):
    python scripts/run_v8_heavy_family.py qwen
    python scripts/run_v8_heavy_family.py smollm2
    python scripts/run_v8_heavy_family.py qwen3b
    python scripts/run_v8_heavy_family.py phi35

    # Or all four:
    python scripts/run_v8_heavy_family.py all

Requires models already in artifacts/hf_cache/hub/ (they are, as of commit
b7d8d33 / 3bc4044 / 6c609ae).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# v7a-v1 A-side = v8 A-side (same 20 TRUE statements, 5-15 words,
# FALSE-class surface profile per locked v8 spec).
V1_GEN_JSON = REPO_ROOT / "stage0_v7a_bilateral" / "stage0_singleton_v7a_generations.json"
V1_JUDGE_JSON = REPO_ROOT / "stage0_v7a_bilateral" / "stage0_singleton_v7a_judge.json"
V8_DIR = REPO_ROOT / "stage0_v8_inverted_bilateral"

FAMILIES = {
    "qwen":    {
        "module": "scripts.score_singleton_v7a_v3_qwen",
        "v3_out_attr": "OUT_JSON",
        "v8_scores_file": V8_DIR / "v8_qwen_scores.json",
        "v8_raw_file":    V8_DIR / "v8_qwen_raw.json",
        "p_full_full_key": "P_A_full",
        "p_full_clean_key": "P_A_cleaned",
    },
    "smollm2": {
        "module": "scripts.score_singleton_v7a_v3_smollm2",
        "v3_out_attr": "OUT_JSON",
        "v8_scores_file": V8_DIR / "v8_smollm2_scores.json",
        "v8_raw_file":    V8_DIR / "v8_smollm2_raw.json",
    },
    "qwen3b":  {
        "module": "scripts.score_singleton_v7a_v3_qwen3b",
        "v3_out_attr": "OUT_JSON",
        "v8_scores_file": V8_DIR / "v8_qwen3b_scores.json",
        "v8_raw_file":    V8_DIR / "v8_qwen3b_raw.json",
    },
    "phi35":   {
        "module": "scripts.score_singleton_v7a_v3_phi35",
        "v3_out_attr": "OUT_JSON",
        "v8_scores_file": V8_DIR / "v8_phi35_scores.json",
        "v8_raw_file":    V8_DIR / "v8_phi35_raw.json",
    },
}


def run_family(short_name: str) -> int:
    if short_name not in FAMILIES:
        print(f"Unknown family: {short_name}. Valid: {list(FAMILIES)}", file=sys.stderr)
        return 2
    cfg = FAMILIES[short_name]
    print("=" * 72)
    print(f"v8 heavy-family run: {short_name}")
    print("=" * 72)

    mod = importlib.import_module(cfg["module"])

    v1_gen = V1_GEN_JSON
    v1_judge = V1_JUDGE_JSON
    if not v1_gen.exists() or not v1_judge.exists():
        print(f"ABORT: v7a-v1 inputs missing ({v1_gen}, {v1_judge})", file=sys.stderr)
        return 1

    original_gen = getattr(mod, "V3_GEN_JSON")
    original_jud = getattr(mod, "V3_JUDGE_JSON")
    original_out = getattr(mod, cfg["v3_out_attr"])

    V8_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  rewire V3_GEN_JSON    {original_gen}  ->  {v1_gen}")
        print(f"  rewire V3_JUDGE_JSON  {original_jud}  ->  {v1_judge}")
        print(f"  rewire {cfg['v3_out_attr']}            {original_out}  ->  {cfg['v8_raw_file']}")
        setattr(mod, "V3_GEN_JSON", v1_gen)
        setattr(mod, "V3_JUDGE_JSON", v1_judge)
        setattr(mod, cfg["v3_out_attr"], cfg["v8_raw_file"])

        rc = mod.main()
        if rc != 0:
            print(f"family {short_name} main() returned rc={rc}", file=sys.stderr)
            return rc
    finally:
        setattr(mod, "V3_GEN_JSON", original_gen)
        setattr(mod, "V3_JUDGE_JSON", original_jud)
        setattr(mod, cfg["v3_out_attr"], original_out)

    raw = json.load(open(cfg["v8_raw_file"]))
    per_pair = raw["per_pair_same_index"]

    v8_rows = []
    for r in per_pair:
        v8_rows.append({
            "pair_id": int(r["a_id"]),
            "P_A_truthful_full":    float(r["P_A_full"]),
            "P_A_truthful_cleaned": float(r["P_A_cleaned"]),
            "P_B_truthful_full":    float(r["P_B_full"]),
            "P_B_truthful_cleaned": float(r["P_B_cleaned"]),
        })
    with open(cfg["v8_scores_file"], "w") as f:
        json.dump(v8_rows, f, indent=2)
    print(f"\nWrote v8 scores: {cfg['v8_scores_file'].relative_to(REPO_ROOT)}  ({len(v8_rows)} pairs)")
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: run_v8_heavy_family.py {qwen|smollm2|qwen3b|phi35|all}", file=sys.stderr)
        return 2
    target = sys.argv[1]
    if target == "all":
        for fam in FAMILIES:
            rc = run_family(fam)
            if rc != 0:
                return rc
        return 0
    return run_family(target)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
