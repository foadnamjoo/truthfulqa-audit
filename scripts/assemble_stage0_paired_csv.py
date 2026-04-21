#!/usr/bin/env python3
"""
Step 8a (paired-pair audit probe) - merge generations + judge + classifier
scores into one human-reviewable CSV.

Output: stage0_paired_tqa/stage0_paired_human_validation.csv

Columns (in order):
    pair_id
    question
    original_correct
    original_incorrect
    a_side_rewritten
    b_side_rewritten
    a_side_faithful                       (generator self-check)
    b_side_faithful                       (generator self-check)
    generator_confidence_in_faithfulness
    a_judge_same_proposition              (GPT-5.4 verdict, A-side)
    a_judge_confidence
    a_judge_rationale
    b_judge_same_proposition              (GPT-5.4 verdict, B-side)
    b_judge_confidence
    b_judge_rationale
    pair_faithful                         (a AND b)
    surface_lr_full_P_a, surface_lr_full_P_b, surface_lr_full_correct
    surface_lr_cleaned_P_a, surface_lr_cleaned_P_b, surface_lr_cleaned_correct
    embedding_lr_full_P_a, embedding_lr_full_P_b, embedding_lr_full_correct
    embedding_lr_cleaned_P_a, embedding_lr_cleaned_P_b, embedding_lr_cleaned_correct
    modernbert_lr_full_P_a, modernbert_lr_full_P_b, modernbert_lr_full_correct
    modernbert_lr_cleaned_P_a, modernbert_lr_cleaned_P_b, modernbert_lr_cleaned_correct
    my_a_faithful                         (blank, human fills in)
    my_b_faithful                         (blank, human fills in)
    my_pair_notes                         (blank, human fills in)

Safety rail: if the file exists AND any of the three human-review columns
have non-empty values, refuse to overwrite (mirrors the safety check in
scripts/assemble_stage0_csv.py).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DIR = REPO_ROOT / "stage0_paired_tqa"
GEN = DIR / "stage0_paired_generations.json"
JDG = DIR / "stage0_paired_judge.json"
SCR = DIR / "stage0_paired_classifier_scores.json"
OUT = DIR / "stage0_paired_human_validation.csv"

PROBA_DECIMALS = 4
HUMAN_COLS = ["my_a_faithful", "my_b_faithful", "my_pair_notes"]
FAMILIES = ["surface_lr", "embedding_lr", "modernbert_lr"]
SPLITS = ["full", "cleaned"]


def _fmt(x: float) -> str:
    return f"{float(x):.{PROBA_DECIMALS}f}"


def _safety_check_existing() -> tuple[bool, list[tuple[int, str, str]]]:
    if not OUT.exists():
        return False, []
    with open(OUT, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    filled: list[tuple[int, str, str]] = []
    for r in rows:
        for col in HUMAN_COLS:
            v = (r.get(col) or "").strip()
            if v:
                try:
                    pid = int(r.get("pair_id", "-1"))
                except ValueError:
                    pid = -1
                filled.append((pid, col, v))
    return (len(filled) > 0), filled


def main() -> int:
    print("=" * 72)
    print("Step 8a - assemble_stage0_paired_csv.py")
    print("=" * 72)

    any_filled, filled = _safety_check_existing()
    if any_filled:
        print(f"STOP: {OUT} has non-empty human-review columns. "
              "Refusing to overwrite. Entries that would be destroyed:",
              file=sys.stderr)
        for pid, col, v in filled:
            print(f"  pid={pid} / {col} = {v!r}", file=sys.stderr)
        print("\nIf you really want to regenerate, back up the existing CSV "
              "and clear the human-review columns first.", file=sys.stderr)
        return 1
    if OUT.exists():
        print(f"Note: {OUT} exists but human-review columns are empty - "
              f"safe to regenerate.")

    for p in (GEN, JDG, SCR):
        if not p.exists():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            return 1

    with open(GEN, "r", encoding="utf-8") as f:
        gens = {int(r["pair_id"]): r for r in json.load(f)}
    with open(JDG, "r", encoding="utf-8") as f:
        judges = {int(r["pair_id"]): r for r in json.load(f)}
    with open(SCR, "r", encoding="utf-8") as f:
        scores = {int(r["pair_id"]): r
                  for r in json.load(f)["pair_records"]}

    pair_ids = sorted(gens.keys())
    if set(pair_ids) != set(judges.keys()):
        print(f"ERROR: gen vs judge id mismatch: gen={sorted(gens)} "
              f"judge={sorted(judges)}", file=sys.stderr)
        return 1
    if set(pair_ids) != set(scores.keys()):
        print(f"ERROR: gen vs score id mismatch", file=sys.stderr)
        return 1

    headers = [
        "pair_id", "question", "original_correct", "original_incorrect",
        "a_side_rewritten", "b_side_rewritten",
        "a_side_faithful", "b_side_faithful",
        "generator_confidence_in_faithfulness",
        "a_judge_same_proposition", "a_judge_confidence", "a_judge_rationale",
        "b_judge_same_proposition", "b_judge_confidence", "b_judge_rationale",
        "pair_faithful",
    ]
    for fam in FAMILIES:
        for split in SPLITS:
            headers += [
                f"{fam}_{split}_P_a",
                f"{fam}_{split}_P_b",
                f"{fam}_{split}_correct",
            ]
    headers += HUMAN_COLS

    rows = []
    for pid in pair_ids:
        g = gens[pid]; j = judges[pid]; s = scores[pid]
        row = {
            "pair_id": pid,
            "question": g["question"],
            "original_correct": g["original_correct"],
            "original_incorrect": g["original_incorrect"],
            "a_side_rewritten": g["a_side_rewritten"],
            "b_side_rewritten": g["b_side_rewritten"],
            "a_side_faithful": g["a_side_faithful"],
            "b_side_faithful": g["b_side_faithful"],
            "generator_confidence_in_faithfulness": _fmt(
                g["generator_confidence_in_faithfulness"]),
            "a_judge_same_proposition": bool(j["a_judge_same_proposition"]),
            "a_judge_confidence": _fmt(j["a_judge_confidence"]),
            "a_judge_rationale": j["a_judge_rationale"],
            "b_judge_same_proposition": bool(j["b_judge_same_proposition"]),
            "b_judge_confidence": _fmt(j["b_judge_confidence"]),
            "b_judge_rationale": j["b_judge_rationale"],
            "pair_faithful": bool(j["pair_faithful"]),
        }
        for fam in FAMILIES:
            for split in SPLITS:
                row[f"{fam}_{split}_P_a"] = _fmt(s[f"{fam}_{split}_P_a"])
                row[f"{fam}_{split}_P_b"] = _fmt(s[f"{fam}_{split}_P_b"])
                row[f"{fam}_{split}_correct"] = bool(
                    s[f"{fam}_{split}_correct"])
        for c in HUMAN_COLS:
            row[c] = ""
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} rows, {len(headers)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
