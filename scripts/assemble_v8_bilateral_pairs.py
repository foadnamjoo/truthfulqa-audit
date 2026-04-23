#!/usr/bin/env python3
"""
v8 bilateral cue-inversion probe — Step 3 pair assembler.

Freezes the 20 (A=TRUE-with-FALSE-cues, B=FALSE-with-TRUE-cues) pairs
as stage0_v8_inverted_bilateral/v8_bilateral_inverted.jsonl.

Inputs
------
- stage0_v7a_bilateral/stage0_singleton_v7a_generations.json  (A-side, n=20)
- stage0_v7a_bilateral/stage0_singleton_v7a_judge.json         (A-side GPT-5.4 truth judge)
- stage0/stage0_generations.json                                (B-side, n=20)
- stage0/stage0_judge.json                                      (B-side GPT-5.4 truth judge)
- stage0_v8_inverted_bilateral/v8_verification_report.json      (locked features from Step 2)

Output
------
- stage0_v8_inverted_bilateral/v8_bilateral_inverted.jsonl   (one pair per line)
- stage0_v8_inverted_bilateral/v8_bilateral_inverted_manifest.json  (corpus-level audit)

Pre-conditions enforced
-----------------------
- A-side judge_agrees_true == True for all 20
- B-side judged_false == True for all 20 and judge_disagreement_flag == False
- Verification report shows zero A-side failures, zero pair delta failures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
A_GEN = REPO_ROOT / "stage0_v7a_bilateral" / "stage0_singleton_v7a_generations.json"
A_JUD = REPO_ROOT / "stage0_v7a_bilateral" / "stage0_singleton_v7a_judge.json"
B_GEN = REPO_ROOT / "stage0" / "stage0_generations.json"
B_JUD = REPO_ROOT / "stage0" / "stage0_judge.json"
V8_DIR = REPO_ROOT / "stage0_v8_inverted_bilateral"
VERIFY = V8_DIR / "v8_verification_report.json"
OUT_JSONL = V8_DIR / "v8_bilateral_inverted.jsonl"
OUT_MANIFEST = V8_DIR / "v8_bilateral_inverted_manifest.json"


def load(p: Path):
    if not p.exists():
        print(f"MISSING: {p}", file=sys.stderr)
        sys.exit(1)
    return json.load(open(p, "r", encoding="utf-8"))


def index_by_id(records):
    return {int(r["id"]): r for r in records}


def main():
    a_gen = index_by_id(load(A_GEN))
    a_jud = index_by_id(load(A_JUD))
    b_gen = index_by_id(load(B_GEN))
    b_jud = index_by_id(load(B_JUD))
    verify = load(VERIFY)

    verify_a = {r["id"]: r for r in verify["a_side"]}
    verify_b = {r["id"]: r for r in verify["b_side"]}
    verify_pair = {r["id"]: r for r in verify["pair_rows"]}

    summary = verify["summary"]
    if summary["a_side_fail_ids"]:
        print(f"ABORT: A-side verification failures: {summary['a_side_fail_ids']}", file=sys.stderr)
        return 1
    if summary["pair_delta_fail_ids"]:
        print(f"ABORT: Pair length-delta failures: {summary['pair_delta_fail_ids']}", file=sys.stderr)
        return 1

    pair_ids = sorted(set(a_gen) & set(b_gen))
    pairs = []
    judge_issues = []

    for pid in pair_ids:
        a = a_gen[pid]
        b = b_gen[pid]
        aj = a_jud[pid]
        bj = b_jud[pid]

        if not aj.get("judge_agrees_true", False):
            judge_issues.append(("A-side not judge-true", pid))
            continue
        if not bj.get("judged_false", False) or bj.get("judge_disagreement_flag", False):
            judge_issues.append(("B-side not judge-false or flagged", pid))
            continue

        pair = {
            "pair_id": pid,
            "A_side": {
                "statement": a["statement"],
                "truth_label": "TRUE",
                "surface_profile": "FALSE_class",
                "topic": a.get("topic"),
                "source_file": "stage0_v7a_bilateral/stage0_singleton_v7a_generations.json",
                "source_id": pid,
                "generator_model": a.get("model"),
                "generator_confidence": a.get("generator_confidence"),
                "generator_truth_rationale": a.get("generator_truth_rationale"),
                "judge_agrees_true": aj.get("judge_agrees_true"),
                "judge_confidence": aj.get("judge_confidence"),
                "judge_model": aj.get("judge_model"),
                "judge_criterion": aj.get("judge_criterion"),
                "surface_features": verify_a[pid]["features"],
            },
            "B_side": {
                "statement": b["statement"],
                "truth_label": "FALSE",
                "surface_profile": "TRUE_class",
                "topic": b.get("topic"),
                "intended_cues": b.get("intended_cues"),
                "source_file": "stage0/stage0_generations.json",
                "source_id": pid,
                "generator_model": b.get("model"),
                "generator_confidence_in_falsity": b.get("generator_confidence_in_falsity"),
                "judged_false": bj.get("judged_false"),
                "judge_confidence": bj.get("judge_confidence"),
                "judge_model": bj.get("judge_model"),
                "judge_disagreement_flag": bj.get("judge_disagreement_flag"),
                "surface_features": verify_b[pid]["features"],
            },
            "pair_constraints": {
                "word_count_delta_B_minus_A": verify_pair[pid]["delta"],
                "delta_ok_ge_8": verify_pair[pid]["ok"],
                "same_topic": (a.get("topic") == b.get("topic")),
            },
        }
        pairs.append(pair)

    if judge_issues:
        print("ABORT: judge preconditions failed:", judge_issues, file=sys.stderr)
        return 1

    V8_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    deltas = [p["pair_constraints"]["word_count_delta_B_minus_A"] for p in pairs]
    same_topic_n = sum(1 for p in pairs if p["pair_constraints"]["same_topic"])
    manifest = {
        "spec_version": "v8.0",
        "generated_from_spec": "stage0_v8_inverted_bilateral/v8_cue_profiles.json",
        "description": (
            "Bilateral cue-inversion pairs. Each pair: A = TRUE statement "
            "dressed in FALSE-class surface cues (short, no-negation, "
            "no-hedge, no-auth, no-contrast, single sentence), B = FALSE "
            "statement dressed in TRUE-class surface cues (longer, "
            "negation/hedging, from §5.1). Evaluation is forced-choice per "
            "pair: classifier picks higher-P(truthful); chance = 0.50."
        ),
        "n_pairs": len(pairs),
        "pair_ids": [p["pair_id"] for p in pairs],
        "A_side_summary": {
            "truth_label": "TRUE (judge-confirmed, criterion=factual_correctness_v7a, model=gpt-5.4-2026-03-05)",
            "word_count_range": [min(p["A_side"]["surface_features"]["word_count"] for p in pairs),
                                  max(p["A_side"]["surface_features"]["word_count"] for p in pairs)],
            "word_count_mean": sum(p["A_side"]["surface_features"]["word_count"] for p in pairs) / len(pairs),
            "all_neg_cnt_zero": all(p["A_side"]["surface_features"]["neg_cnt"] == 0 for p in pairs),
            "all_hedge_cnt_zero": all(p["A_side"]["surface_features"]["hedge_cnt"] == 0 for p in pairs),
            "all_auth_cnt_zero": all(p["A_side"]["surface_features"]["auth_cnt"] == 0 for p in pairs),
            "all_contrast_cnt_zero": all(p["A_side"]["surface_features"]["contrast_cnt"] == 0 for p in pairs),
        },
        "B_side_summary": {
            "truth_label": "FALSE (judge-confirmed, model=gpt-5.4-2026-03-05, no disagreement flags)",
            "word_count_range": [min(p["B_side"]["surface_features"]["word_count"] for p in pairs),
                                  max(p["B_side"]["surface_features"]["word_count"] for p in pairs)],
            "word_count_mean": sum(p["B_side"]["surface_features"]["word_count"] for p in pairs) / len(pairs),
        },
        "pair_delta_summary": {
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "mean_delta": sum(deltas) / len(deltas),
            "all_delta_ge_8": all(d >= 8 for d in deltas),
        },
        "same_topic_pair_count": same_topic_n,
        "topic_independence_note": (
            "A-side topic and B-side topic do NOT always match. Forced-choice "
            "evaluation is over surface style given per-side truth label, not "
            "topic-conditional truth evaluation. Topic mismatch is by design."
        ),
        "output_file": str(OUT_JSONL.relative_to(REPO_ROOT)),
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print(f"Wrote {len(pairs)} pairs to {OUT_JSONL.relative_to(REPO_ROOT)}")
    print(f"Wrote manifest to     {OUT_MANIFEST.relative_to(REPO_ROOT)}")
    print("=" * 70)
    print(f"A-side wc range: {manifest['A_side_summary']['word_count_range']} "
          f"mean={manifest['A_side_summary']['word_count_mean']:.1f}")
    print(f"B-side wc range: {manifest['B_side_summary']['word_count_range']} "
          f"mean={manifest['B_side_summary']['word_count_mean']:.1f}")
    print(f"Pair delta: min={manifest['pair_delta_summary']['min_delta']}, "
          f"max={manifest['pair_delta_summary']['max_delta']}, "
          f"mean={manifest['pair_delta_summary']['mean_delta']:.1f}")
    print(f"Same-topic pairs: {same_topic_n}/{len(pairs)} "
          f"(topic-independence is by design — forced-choice is over style not topic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
