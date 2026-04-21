#!/usr/bin/env python3
"""
Build a review-friendly CSV of the paired probe rewrites for human
inspection.

Inputs:
    stage0_paired_tqa/stage0_paired_generations.json
        Authoritative source for original_correct, original_incorrect,
        a_side_rewritten, b_side_rewritten, and the question text.
    audits/truthfulqa_style_audit.csv
        The original TruthfulQA audit table; the row whose 0-based index
        equals the pair_id is the source row. We use this only to
        cross-check the Question field against the question stored in
        the generation JSON; if they disagree we abort.
    stage0_paired_tqa/stage0_paired_judge_truthclass.json
        Read-only reference column showing what GPT-5.4 decided under
        the truth-class criterion (a_truthclass_same, b_truthclass_same).

Output:
    stage0_paired_tqa/stage0_paired_human_review_truthclass.csv
        One row per pair, 13 columns:

          pair_id,
          question,
          original_correct, a_side_rewritten,
          a_my_truthclass_faithful  (blank: y / n / unsure),
          a_my_notes                (blank),
          a_gpt5_truthclass_faithful (read-only reference),
          original_incorrect, b_side_rewritten,
          b_my_truthclass_faithful  (blank: y / n / unsure),
          b_my_notes                (blank),
          b_gpt5_truthclass_faithful (read-only reference),
          pair_truthclass_faithful  (blank - compute after review)

Criterion reminder for the human reviewer:
    "Truth-class faithfulness" means: if the original is FALSE the
    rewrite must also be FALSE; if the original is TRUE the rewrite must
    also be TRUE. Stronger or weaker rewrites that stay on the SAME side
    of the true/false divide count as "y". Only mark "n" if the rewrite
    has flipped to the OPPOSITE truth class. Mark "unsure" only when
    even after looking up the fact you cannot decide which side the
    rewrite lands on.
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations.json"
JUDGE_JSON = (REPO_ROOT / "stage0_paired_tqa"
              / "stage0_paired_judge_truthclass.json")
AUDIT_CSV = REPO_ROOT / "audits" / "truthfulqa_style_audit.csv"
OUT_CSV = (REPO_ROOT / "stage0_paired_tqa"
           / "stage0_paired_human_review_truthclass.csv")

COLUMNS = [
    "pair_id",
    "question",
    "original_correct",
    "a_side_rewritten",
    "a_my_truthclass_faithful",
    "a_my_notes",
    "a_gpt5_truthclass_faithful",
    "original_incorrect",
    "b_side_rewritten",
    "b_my_truthclass_faithful",
    "b_my_notes",
    "b_gpt5_truthclass_faithful",
    "pair_truthclass_faithful",
]


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"Missing input: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"{path} is not a JSON list.")
    return data


def _load_audit_questions() -> dict[int, str]:
    if not AUDIT_CSV.exists():
        raise RuntimeError(f"Missing input: {AUDIT_CSV}")
    out: dict[int, str] = {}
    with open(AUDIT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            out[i] = (row.get("Question") or "").strip()
    return out


def main() -> int:
    print("=" * 72)
    print("export_paired_for_human_review.py")
    print("=" * 72)

    gens = sorted(_load_json(GEN_JSON), key=lambda r: int(r["pair_id"]))
    judge = {int(r["pair_id"]): r for r in _load_json(JUDGE_JSON)}
    audit_q = _load_audit_questions()

    print(f"Loaded {len(gens)} generations from {GEN_JSON.name}")
    print(f"Loaded {len(judge)} truth-class judgments from {JUDGE_JSON.name}")
    print(f"Loaded {len(audit_q)} questions from {AUDIT_CSV.name}")

    rows: list[dict] = []
    mismatches: list[tuple[int, str, str]] = []
    missing_judge: list[int] = []

    for g in gens:
        pid = int(g["pair_id"])
        gen_q = (g.get("question") or "").strip()
        audit_row_q = audit_q.get(pid, "")
        if audit_row_q and gen_q and audit_row_q != gen_q:
            mismatches.append((pid, gen_q, audit_row_q))

        j = judge.get(pid)
        if j is None:
            missing_judge.append(pid)
            a_gpt = ""
            b_gpt = ""
        else:
            a_gpt = "y" if j["a_truthclass_same"] else "n"
            b_gpt = "y" if j["b_truthclass_same"] else "n"

        rows.append({
            "pair_id": pid,
            "question": gen_q or audit_row_q,
            "original_correct": g["original_correct"],
            "a_side_rewritten": g["a_side_rewritten"],
            "a_my_truthclass_faithful": "",
            "a_my_notes": "",
            "a_gpt5_truthclass_faithful": a_gpt,
            "original_incorrect": g["original_incorrect"],
            "b_side_rewritten": g["b_side_rewritten"],
            "b_my_truthclass_faithful": "",
            "b_my_notes": "",
            "b_gpt5_truthclass_faithful": b_gpt,
            "pair_truthclass_faithful": "",
        })

    if mismatches:
        # Hard fail: the question text is supposed to be canonical, and
        # if the JSON disagrees with the audit CSV the reviewer should
        # not be shown ambiguous source material. Print a clear summary
        # and abort before writing.
        print("ERROR: question-text mismatch between generation JSON and "
              "audit CSV for the following pair_ids:", file=sys.stderr)
        for pid, gen_q, audit_row_q in mismatches:
            print(f"  pid={pid}", file=sys.stderr)
            print(f"    gen JSON : {gen_q!r}", file=sys.stderr)
            print(f"    audit CSV: {audit_row_q!r}", file=sys.stderr)
        return 1
    print("Cross-check OK: every generation-JSON question matches the "
          "audit-CSV row at index == pair_id.")

    if missing_judge:
        print(f"WARNING: {len(missing_judge)} pair_ids have no truth-class "
              f"judgment; their reference columns will be blank: "
              f"{missing_judge}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")
    print(f"Columns ({len(COLUMNS)}): {COLUMNS}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in export_paired_for_human_review.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
