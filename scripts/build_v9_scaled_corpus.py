#!/usr/bin/env python3
"""
Build a scaled v9 same-question corpus by merging:
  - stage0_paired_tqa/stage0_paired_generations_v4.json  (n=20, pass=16)
  - stage0_paired_tqa/v4_multi_seed/generations_seed{1,7,42}_n40.json

For each distinct pair_id, keep ONE generation (preferring judge-passing
ones). Priority: v4 (if pass) > seed1 > seed42 > seed7.  Only pair_ids
with at least one passing generation are kept.  Non-passing v4 pair_ids
are included only if no seed has a passing generation (they will carry
judge_pair_passes=False and can be filtered at aggregation time).

For the same pair_id chosen generation, also carry forward the matching
scores row from the seed's score file — covering the 3 light families
(surface_lr, BGE-large, ModernBERT-base) in full+cleaned splits.

Outputs (stage0_v9_same_question/):
  v9_scaled_generations.json        (merged per-pair gens)
  v9_scaled_judge.json              (merged per-pair judge rows)
  v9_scaled_scores_light.json       (per-pair scores for 3 light families)
  v9_scaled_manifest.json           (provenance: which seed each pair came from)
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRED = REPO_ROOT / "stage0_paired_tqa"
V9_DIR = REPO_ROOT / "stage0_v9_same_question"

SOURCES = [
    # (name,              gens,                                       judge,                                scores)
    ("v4",     PAIRED / "stage0_paired_generations_v4.json",
               PAIRED / "stage0_paired_judge_v4.json",
               PAIRED / "stage0_paired_classifier_scores_v4.json"),
    ("seed1",  PAIRED / "v4_multi_seed" / "generations_seed1_n40.json",
               PAIRED / "v4_multi_seed" / "judge_seed1_n40.json",
               PAIRED / "v4_multi_seed" / "scores_seed1_n40.json"),
    ("seed42", PAIRED / "v4_multi_seed" / "generations_seed42_n40.json",
               PAIRED / "v4_multi_seed" / "judge_seed42_n40.json",
               PAIRED / "v4_multi_seed" / "scores_seed42_n40.json"),
    ("seed7",  PAIRED / "v4_multi_seed" / "generations_seed7_n40.json",
               PAIRED / "v4_multi_seed" / "judge_seed7_n40.json",
               PAIRED / "v4_multi_seed" / "scores_seed7_n40.json"),
    ("seed100", PAIRED / "v4_multi_seed" / "generations_seed100_n50.json",
                PAIRED / "v4_multi_seed" / "judge_seed100_n50.json",
                PAIRED / "v4_multi_seed" / "scores_seed100_n50.json"),
    ("seed101", PAIRED / "v4_multi_seed" / "generations_seed101_n50.json",
                PAIRED / "v4_multi_seed" / "judge_seed101_n50.json",
                PAIRED / "v4_multi_seed" / "scores_seed101_n50.json"),
    ("seed102", PAIRED / "v4_multi_seed" / "generations_seed102_n50.json",
                PAIRED / "v4_multi_seed" / "judge_seed102_n50.json",
                PAIRED / "v4_multi_seed" / "scores_seed102_n50.json"),
    ("seed103", PAIRED / "v4_multi_seed" / "generations_seed103_n50.json",
                PAIRED / "v4_multi_seed" / "judge_seed103_n50.json",
                PAIRED / "v4_multi_seed" / "scores_seed103_n50.json"),
    ("seed104", PAIRED / "v4_multi_seed" / "generations_seed104_n50.json",
                PAIRED / "v4_multi_seed" / "judge_seed104_n50.json",
                PAIRED / "v4_multi_seed" / "scores_seed104_n50.json"),
    ("seed105", PAIRED / "v4_multi_seed" / "generations_seed105_n50.json",
                PAIRED / "v4_multi_seed" / "judge_seed105_n50.json",
                PAIRED / "v4_multi_seed" / "scores_seed105_n50.json"),
]


def _load_src(name, gens_p, judge_p, scores_p):
    gens  = {int(r["pair_id"]): r for r in json.load(open(gens_p))}
    judge = {int(r["pair_id"]): r for r in json.load(open(judge_p))}
    sc    = json.load(open(scores_p))
    per_pair = {int(r["pair_id"]): r for r in sc.get("per_pair", [])}
    return gens, judge, per_pair


def main() -> int:
    srcs = [(name, *_load_src(name, g, j, s)) for name, g, j, s in SOURCES]

    chosen: dict[int, dict] = {}
    for name, gens, judge, scores in srcs:
        for pid, gen in gens.items():
            jrow = judge.get(pid, {})
            passes = bool(jrow.get("pair_passes", False))
            prev = chosen.get(pid)
            if prev is None:
                chosen[pid] = {
                    "src": name, "gen": gen, "judge": jrow,
                    "scores": scores.get(pid), "passes": passes,
                }
            else:
                if passes and not prev["passes"]:
                    chosen[pid] = {
                        "src": name, "gen": gen, "judge": jrow,
                        "scores": scores.get(pid), "passes": passes,
                    }

    chosen_items = sorted(chosen.items())
    merged_pass = [c for _, c in chosen_items if c["passes"]]
    print(f"Distinct pair_ids total:    {len(chosen_items)}")
    print(f"Distinct passing pair_ids:  {len(merged_pass)}")
    src_counts: dict[str, int] = {}
    for _, c in chosen_items:
        if c["passes"]:
            src_counts[c["src"]] = src_counts.get(c["src"], 0) + 1
    print(f"Source distribution (passing only): {src_counts}")

    keep_pids = sorted(pid for pid, c in chosen.items() if c["passes"])
    print(f"\nKeeping {len(keep_pids)} passing pair_ids for scaled corpus.")

    gens_out:   list[dict] = []
    judge_out:  list[dict] = []
    scores_out: list[dict] = []
    manifest: list[dict] = []
    for pid in keep_pids:
        c = chosen[pid]
        gens_out.append(c["gen"])
        judge_out.append(c["judge"])
        if c["scores"] is None:
            raise RuntimeError(f"pid {pid} has no score row in source {c['src']}")
        scores_out.append(c["scores"])
        manifest.append({
            "pair_id": pid,
            "source":  c["src"],
            "pair_passes": c["passes"],
        })

    V9_DIR.mkdir(parents=True, exist_ok=True)
    (V9_DIR / "v9_scaled_generations.json").write_text(
        json.dumps(gens_out,   ensure_ascii=False, indent=2))
    (V9_DIR / "v9_scaled_judge.json").write_text(
        json.dumps(judge_out,  ensure_ascii=False, indent=2))
    (V9_DIR / "v9_scaled_scores_light.json").write_text(
        json.dumps({"n_pairs": len(scores_out), "per_pair": scores_out},
                   ensure_ascii=False, indent=2))
    (V9_DIR / "v9_scaled_manifest.json").write_text(
        json.dumps({"n_pairs": len(keep_pids),
                    "pair_ids": keep_pids,
                    "source_counts": src_counts,
                    "manifest": manifest},
                   ensure_ascii=False, indent=2))
    print(f"\nWrote:")
    for p in ("v9_scaled_generations.json", "v9_scaled_judge.json",
              "v9_scaled_scores_light.json", "v9_scaled_manifest.json"):
        print(f"  stage0_v9_same_question/{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
