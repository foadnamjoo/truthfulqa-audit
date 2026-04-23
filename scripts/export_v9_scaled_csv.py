#!/usr/bin/env python3
"""Export v9 scaled (n=79) forced-choice data to CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V9_DIR    = REPO_ROOT / "stage0_v9_same_question"
GEN_JSON  = V9_DIR / "v9_scaled_generations.json"
LIGHT     = V9_DIR / "v9_scaled_scores_light.json"
MANIFEST  = V9_DIR / "v9_scaled_manifest.json"
OUT_CSV   = V9_DIR / "v9_scaled_data.csv"

LIGHT_FAMS = ["surface_lr", "BGE-large", "ModernBERT-base"]
HEAVY = {
    "Qwen2.5-0.5B": V9_DIR / "v9_scaled_qwen_scores.json",
    "Qwen2.5-1.5B": V9_DIR / "v9_scaled_qwen15b_scores.json",
    "SmolLM2-1.7B": V9_DIR / "v9_scaled_smollm2_scores.json",
    "Qwen2.5-3B":   V9_DIR / "v9_scaled_qwen3b_scores.json",
    "Llama-3.2-3B": V9_DIR / "v9_scaled_llama32_3b_scores.json",
    "Phi-3.5-mini": V9_DIR / "v9_scaled_phi35_scores.json",
}
VARIANTS = ["full", "cleaned"]
FAMS = LIGHT_FAMS + list(HEAVY.keys())


def main() -> int:
    gens = {int(r["pair_id"]): r for r in json.load(open(GEN_JSON))}
    light_per = {int(r["pair_id"]): r
                 for r in json.load(open(LIGHT))["per_pair"]}
    src_by_pid = {int(r["pair_id"]): r["source"]
                  for r in json.load(open(MANIFEST))["manifest"]}

    heavy_by_pid: dict[str, dict[int, dict]] = {}
    for fam, path in HEAVY.items():
        if not path.exists():
            heavy_by_pid[fam] = {}
            continue
        heavy_by_pid[fam] = {int(r["pair_id"]): r
                             for r in json.load(open(path))}

    pids = sorted(gens.keys())
    header = ["pair_id", "source_seed", "judge_pair_passes",
              "question", "a_side_TRUE", "b_side_FALSE"]
    for fam in FAMS:
        for var in VARIANTS:
            h = f"{fam}_{var}"
            header += [f"{h}_P_A", f"{h}_P_B", f"{h}_picked", f"{h}_correct"]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for pid in pids:
            g = gens[pid]; lp = light_per[pid]
            row = [pid, src_by_pid.get(pid, "?"),
                   int(bool(lp.get("judge_pair_passes"))),
                   g.get("question", ""), g.get("a_side", ""),
                   g.get("b_side", "")]
            for fam in FAMS:
                for var in VARIANTS:
                    if fam in LIGHT_FAMS:
                        s = lp["scores"].get(f"{fam}_{var}", {})
                        pa, pb = s.get("P_A"), s.get("P_B")
                    else:
                        r = heavy_by_pid.get(fam, {}).get(pid, {})
                        pa = r.get(f"P_A_truthful_{var}")
                        pb = r.get(f"P_B_truthful_{var}")
                    if pa is None or pb is None:
                        row += ["", "", "", ""]; continue
                    picked = "A" if pa > pb else "B"
                    row += [f"{pa:.6f}", f"{pb:.6f}", picked,
                            1 if picked == "A" else 0]
            w.writerow(row)

    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)}  "
          f"({len(pids)} rows, {len(header)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
