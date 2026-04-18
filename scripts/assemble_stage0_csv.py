#!/usr/bin/env python3
"""
Step 11 (6-classifier edition) - Merge the three Stage-0 JSON artifacts
into a single human-reviewable CSV, `stage0/stage0_human_validation.csv`.

Inputs:
    stage0/stage0_generations.json       (20 records from generate_stage0.py)
    stage0/stage0_classifier_scores.json (20 records from score_singleton.py,
                                          now with 6 P_truthful columns)
    stage0/stage0_judge.json             (20 records from judge_stage0.py)

Output schema (20 columns, exactly in this order):

    id
    topic
    topic_ood
    generated_statement
    intended_cues                        JSON list string, e.g.
                                         ["sentence_initial_negation","authority"]
    generator_rationale
    generator_confidence_in_falsity      float, 4 decimals
    judge_verdict_false                  bool, from judge.judged_false
    judge_confidence                     float, 4 decimals
    judge_rationale
    judge_disagreement_flag              bool
    surface_lr_full_P_truthful           float, 4 decimals
    surface_lr_cleaned_P_truthful        float, 4 decimals
    embedding_lr_full_P_truthful         float, 4 decimals (BGE-large)
    embedding_lr_cleaned_P_truthful      float, 4 decimals (BGE-large)
    modernbert_lr_full_P_truthful        float, 4 decimals   NEW in this step
    modernbert_lr_cleaned_P_truthful     float, 4 decimals   NEW in this step
    my_untruthfulness_check              empty string (human fills in)
    my_cue_presence_check                empty string (human fills in)
    my_notes                             empty string (human fills in)

All three input files are keyed on `id`. The script validates that each
contains exactly 20 records and that every id 0..19 is present in all
three before writing.

Safety rail: if the output CSV already exists AND any of the three
human-review columns contain a non-empty value, the script aborts rather
than overwriting the reviewer's work. Only a fully-empty existing CSV is
regenerated.
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE0_DIR = REPO_ROOT / "stage0"
GEN_JSON = STAGE0_DIR / "stage0_generations.json"
SCORES_JSON = STAGE0_DIR / "stage0_classifier_scores.json"
JUDGE_JSON = STAGE0_DIR / "stage0_judge.json"
OUT_CSV = STAGE0_DIR / "stage0_human_validation.csv"

COLUMNS: list[str] = [
    "id",
    "topic",
    "topic_ood",
    "generated_statement",
    "intended_cues",
    "generator_rationale",
    "generator_confidence_in_falsity",
    "judge_verdict_false",
    "judge_confidence",
    "judge_rationale",
    "judge_disagreement_flag",
    "surface_lr_full_P_truthful",
    "surface_lr_cleaned_P_truthful",
    "embedding_lr_full_P_truthful",
    "embedding_lr_cleaned_P_truthful",
    "modernbert_lr_full_P_truthful",
    "modernbert_lr_cleaned_P_truthful",
    "my_untruthfulness_check",
    "my_cue_presence_check",
    "my_notes",
]

HUMAN_REVIEW_COLUMNS = [
    "my_untruthfulness_check",
    "my_cue_presence_check",
    "my_notes",
]

EXPECTED_N = 20
PROBA_DECIMALS = 4


def _existing_has_human_review() -> tuple[bool, list[tuple[int, str, str]]]:
    """Return (any_filled, filled_entries). Empty-file or missing-file
    returns (False, []). Each filled entry is (slot_id, column, value)."""
    if not OUT_CSV.exists():
        return False, []
    with open(OUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        filled: list[tuple[int, str, str]] = []
        for row in reader:
            for col in HUMAN_REVIEW_COLUMNS:
                v = (row.get(col) or "").strip()
                if v != "":
                    try:
                        sid = int(row.get("id", "-1"))
                    except ValueError:
                        sid = -1
                    filled.append((sid, col, v))
    return (len(filled) > 0), filled


def _load_json_list(path: Path, label: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"{label} is not a JSON list: {path}")
    if len(data) != EXPECTED_N:
        raise RuntimeError(
            f"{label} must contain {EXPECTED_N} records; got {len(data)}"
        )
    return data


def _index_by_id(records: list[dict], label: str) -> dict[int, dict]:
    ix: dict[int, dict] = {}
    for r in records:
        if "id" not in r:
            raise RuntimeError(f"{label}: record missing 'id': {r!r}")
        sid = int(r["id"])
        if sid in ix:
            raise RuntimeError(f"{label}: duplicate id {sid}")
        ix[sid] = r
    expected = set(range(EXPECTED_N))
    if set(ix) != expected:
        missing = sorted(expected - set(ix))
        extra = sorted(set(ix) - expected)
        raise RuntimeError(
            f"{label}: id coverage mismatch; missing={missing} extra={extra}"
        )
    return ix


def _fmt_proba(x: float) -> str:
    return f"{float(x):.{PROBA_DECIMALS}f}"


def main() -> int:
    print("=" * 72)
    print("STEP 11 - assemble_stage0_csv.py  (6-classifier edition)")
    print("=" * 72)

    any_filled, filled = _existing_has_human_review()
    if any_filled:
        print(f"STOP: existing {OUT_CSV} has non-empty human-review columns. "
              "Refusing to overwrite. The following entries would be "
              "destroyed:", file=sys.stderr)
        for sid, col, v in filled:
            print(f"  slot {sid:02d} / {col} = {v!r}", file=sys.stderr)
        print("\nIf you really want to regenerate, back up the existing CSV "
              "and delete / clear the three human-review columns before "
              "re-running.", file=sys.stderr)
        return 1
    if OUT_CSV.exists():
        print(f"Note: {OUT_CSV} exists but all human-review columns are "
              "empty - safe to regenerate.")

    print(f"Reading   {GEN_JSON}")
    gens_list = _load_json_list(GEN_JSON, "generations")
    print(f"Reading   {SCORES_JSON}")
    scores_list = _load_json_list(SCORES_JSON, "classifier_scores")
    print(f"Reading   {JUDGE_JSON}")
    judge_list = _load_json_list(JUDGE_JSON, "judge")

    gens = _index_by_id(gens_list, "generations")
    scores = _index_by_id(scores_list, "classifier_scores")
    judge = _index_by_id(judge_list, "judge")
    print(f"All three files validated at {EXPECTED_N} records with ids 0..{EXPECTED_N-1}")

    rows: list[dict] = []
    for sid in range(EXPECTED_N):
        g = gens[sid]
        s = scores[sid]
        j = judge[sid]
        if j["statement"] != g["statement"]:
            raise RuntimeError(
                f"Slot {sid}: judge.statement differs from generations.statement. "
                f"Refusing to merge mismatched data."
            )
        row = {
            "id": sid,
            "topic": g["topic"],
            "topic_ood": bool(g["topic_ood"]),
            "generated_statement": g["statement"],
            "intended_cues": json.dumps(g["intended_cues"], ensure_ascii=False),
            "generator_rationale": g["generator_rationale"],
            "generator_confidence_in_falsity": _fmt_proba(
                g["generator_confidence_in_falsity"]
            ),
            "judge_verdict_false": bool(j["judged_false"]),
            "judge_confidence": _fmt_proba(j["judge_confidence"]),
            "judge_rationale": j["judge_rationale"],
            "judge_disagreement_flag": bool(j["judge_disagreement_flag"]),
            "surface_lr_full_P_truthful": _fmt_proba(
                s["surface_lr_full_P_truthful"]
            ),
            "surface_lr_cleaned_P_truthful": _fmt_proba(
                s["surface_lr_cleaned_P_truthful"]
            ),
            "embedding_lr_full_P_truthful": _fmt_proba(
                s["embedding_lr_full_P_truthful"]
            ),
            "embedding_lr_cleaned_P_truthful": _fmt_proba(
                s["embedding_lr_cleaned_P_truthful"]
            ),
            "modernbert_lr_full_P_truthful": _fmt_proba(
                s["modernbert_lr_full_P_truthful"]
            ),
            "modernbert_lr_cleaned_P_truthful": _fmt_proba(
                s["modernbert_lr_cleaned_P_truthful"]
            ),
            "my_untruthfulness_check": "",
            "my_cue_presence_check": "",
            "my_notes": "",
        }
        if set(row) != set(COLUMNS):
            raise RuntimeError(
                f"Slot {sid}: assembled row has columns "
                f"{sorted(row)} != expected {sorted(COLUMNS)}"
            )
        rows.append(row)

    STAGE0_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows x {len(COLUMNS)} cols -> {OUT_CSV}")
    print("\nHeader preview:")
    print("  " + ",".join(COLUMNS))
    print(f"\nHuman-review columns (left blank): "
          f"my_untruthfulness_check, my_cue_presence_check, my_notes")
    print("\nStep 11 complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in assemble_stage0_csv.py:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
