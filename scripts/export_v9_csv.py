#!/usr/bin/env python3
"""
Export v9 same-question forced-choice data to a single CSV.

Columns:
  pair_id, judge_pair_passes, question, a_side (TRUE), b_side (FALSE),
  then for each (family, variant): P_A, P_B, picked ('A' or 'B'),
  correct (1 if picked == 'A' else 0).

Families: surface_lr, BGE-large, ModernBERT-base, Qwen2.5-0.5B,
          SmolLM2-1.7B, Qwen2.5-3B, Phi-3.5-mini
Variants: full, cleaned
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_JSON  = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations_v4.json"
SCORES_V4 = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_classifier_scores_v4.json"
V9_DIR    = REPO_ROOT / "stage0_v9_same_question"
OUT_CSV   = V9_DIR / "v9_data.csv"

LIGHT = ["surface_lr", "BGE-large", "ModernBERT-base"]
HEAVY = {
    "Qwen2.5-0.5B": V9_DIR / "v9_qwen_scores.json",
    "SmolLM2-1.7B": V9_DIR / "v9_smollm2_scores.json",
    "Qwen2.5-3B":   V9_DIR / "v9_qwen3b_scores.json",
    "Phi-3.5-mini": V9_DIR / "v9_phi35_scores.json",
}
VARIANTS = ["full", "cleaned"]
FAMILIES = LIGHT + list(HEAVY.keys())


def main() -> int:
    gens = {int(r["pair_id"]): r for r in json.load(open(GEN_JSON))}
    scores_v4 = json.load(open(SCORES_V4))
    light_by_pid = {r["pair_id"]: r for r in scores_v4["per_pair"]}

    heavy_by_pid = {fam: {} for fam in HEAVY}
    for fam, path in HEAVY.items():
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        rows = json.load(open(path))
        heavy_by_pid[fam] = {int(r["pair_id"]): r for r in rows}

    pair_ids = sorted(light_by_pid.keys())

    # Build columns
    header = ["pair_id", "judge_pair_passes", "question", "a_side_TRUE", "b_side_FALSE"]
    for fam in FAMILIES:
        for var in VARIANTS:
            prefix = f"{fam}_{var}"
            header += [f"{prefix}_P_A", f"{prefix}_P_B",
                       f"{prefix}_picked", f"{prefix}_correct"]

    V9_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for pid in pair_ids:
            g = gens[pid]
            l = light_by_pid[pid]
            row = [pid, int(bool(l["judge_pair_passes"])),
                   g.get("question", ""), g.get("a_side", ""), g.get("b_side", "")]
            for fam in FAMILIES:
                for var in VARIANTS:
                    if fam in LIGHT:
                        s = l["scores"][f"{fam}_{var}"]
                        pa, pb = s["P_A"], s["P_B"]
                    else:
                        r = heavy_by_pid[fam].get(pid, {})
                        pa = r.get(f"P_A_truthful_{var}", "")
                        pb = r.get(f"P_B_truthful_{var}", "")
                    if pa == "" or pb == "":
                        row += ["", "", "", ""]
                        continue
                    picked = "A" if pa > pb else "B"
                    correct = 1 if picked == "A" else 0
                    row += [f"{pa:.6f}", f"{pb:.6f}", picked, correct]
            w.writerow(row)

    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)}  ({len(pair_ids)} rows, "
          f"{len(header)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
