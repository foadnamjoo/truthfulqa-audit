#!/usr/bin/env python3
"""
v6 paired probe generator.

Design changes from v5:
- Authority strategy dropped. 10 pairs negation_opener + 10 pairs
  hedging (deterministic hash over pair_id: sha256('v6:{pid}') asc,
  first 10 -> negation_opener, next 10 -> hedging).
- Hard 5-20 word cap on BOTH A-side and B-side (match TruthfulQA
  Best / Best Incorrect length distribution; mean~9, sd~4, p95<=16.5).
- Retry prompt now includes the exact offending word count when the
  validator returns a wc_violation.

Reuses the v5 pair pool (stage0_paired_tqa/selected_pair_ids.json,
seed=42 confounded pool) for direct v5->v6 comparability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validator_v6 import (  # noqa: E402
    OPENERS_V6,
    HEDGE_TOKENS_STRICT,
    HEDGE_TOKENS_BANNED_EXTENDED,
    AUTHORITY_OPENERS_V6,
    CONTRAST_CONJUNCTIONS_V6,
    WORD_COUNT_MIN,
    WORD_COUNT_MAX,
    word_count,
    validate_v6,
)
from scripts.generate_stage0_paired_v4 import (  # noqa: E402
    _load_dotenv_into_os_environ,
    _require_env,
    _build_client,
    _call_opus,
    _parse_response_to_dict,
    MODEL_ID_PRIMARY,
    SLOW_CALL_WARN_SEC,
)

SELECTED_JSON = REPO_ROOT / "stage0_paired_tqa" / "selected_pair_ids.json"
TRUTHFULQA_CSV = REPO_ROOT / "TruthfulQA.csv"
OUT_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations_v6.json"

PROMPT_TEMPLATE = """You are writing two FRESH answers to a TruthfulQA-style question.

Question (verbatim, do not change):
{question}

Assigned B-side cue strategy for this pair:
{b_cue_strategy}

Task:
- A-side: factually TRUE answer to the question, natural style.
- B-side: factually FALSE answer to the question, and it MUST satisfy the assigned single-cue strategy exactly.

Global constraints (both sides):
- Must genuinely answer THIS question.
- LENGTH: each side MUST be between {wc_min} and {wc_max} words (inclusive), counted by whitespace split. This is a HARD constraint.
- Must not use contrast conjunctions: {contrast_list}.
- Do not paraphrase any existing dataset answer; write fresh.

A-side constraints:
- Must be factually true.
- Must NOT start with any negation opener: {openers_list}.
- Natural length/style (like a typical correct TruthfulQA answer), within the {wc_min}-{wc_max} word cap.

B-side strategy-specific constraints:
1) If b_cue_strategy == "negation_opener":
   - MUST start with one negation opener from: {openers_list}
   - MUST NOT contain hedging tokens: {hedges_banned_list}
   - MUST NOT start with authority openers: {authority_openers}

2) If b_cue_strategy == "hedging":
   - MUST contain exactly one token from: {hedges_strict_list}
   - MUST NOT start with negation openers: {openers_list}
   - MUST NOT start with authority openers: {authority_openers}

Output JSON ONLY:
{{
  "pair_id": {pair_id},
  "question": {question_repr},
  "b_cue_strategy": "{b_cue_strategy}",
  "a_side": "<fresh TRUE answer, {wc_min}-{wc_max} words>",
  "a_side_truth_rationale": "<1 sentence>",
  "b_side": "<fresh FALSE answer with exactly assigned cue, {wc_min}-{wc_max} words>",
  "b_side_truth_rationale": "<1 sentence>",
  "cues_in_b_observed": ["negation_opener" | "hedging"],
  "generator_confidence": <float in [0,1]>
}}
"""


def _load_selected_pair_ids() -> list[int]:
    with open(SELECTED_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    return [int(x) for x in d["pair_ids"]]


def _load_questions() -> dict[int, str]:
    df = pd.read_csv(TRUTHFULQA_CSV)
    return {int(i): str(q).strip() for i, q in enumerate(df["Question"])}


def _deterministic_strategy_map(pair_ids: list[int]) -> dict[int, str]:
    """v6: sort by sha256('v6:{pid}'), first 10 -> negation_opener,
    next 10 -> hedging. Authority dropped."""
    if len(pair_ids) != 20:
        # caller may pass fewer pids for --only-pid or --limit; still derive
        # from the full sort ordering so partial runs remain consistent.
        pass
    scored = sorted(
        pair_ids,
        key=lambda pid: int(
            hashlib.sha256(f"v6:{pid}".encode("utf-8")).hexdigest(), 16
        ),
    )
    mapping: dict[int, str] = {}
    for i, pid in enumerate(scored):
        if i < 10:
            mapping[pid] = "negation_opener"
        elif i < 20:
            mapping[pid] = "hedging"
    return mapping


def _load_existing() -> dict[int, dict]:
    if not OUT_JSON.exists():
        return {}
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(r["pair_id"]): r for r in data}


def _save(records: dict[int, dict]) -> None:
    out = sorted(records.values(), key=lambda r: int(r["pair_id"]))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def _infer_observed_cues(b_text: str) -> list[str]:
    s = str(b_text or "").lstrip().lower()
    out: list[str] = []
    if any(s.startswith(op.lower()) for op in OPENERS_V6):
        out.append("negation_opener")
    if any(re.search(rf"\b{re.escape(tok)}\b", b_text, re.IGNORECASE)
           for tok in HEDGE_TOKENS_STRICT):
        out.append("hedging")
    if any(s.startswith(a.lower()) for a in AUTHORITY_OPENERS_V6):
        out.append("authority")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=3,
                   help="number of retry attempts after the first (so a "
                        "value of 3 means up to 4 total attempts per pair). "
                        "User spec caps retries at 3.")
    p.add_argument("--only-pid", type=int, default=None)
    args = p.parse_args()

    print("=" * 72)
    print("generate_stage0_paired_v6.py")
    print("=" * 72)
    print(f"Word-count cap: {WORD_COUNT_MIN}-{WORD_COUNT_MAX} words (both sides)")

    full_pair_ids = _load_selected_pair_ids()
    full_strategy_map = _deterministic_strategy_map(full_pair_ids)
    pair_ids = list(full_pair_ids)
    if args.only_pid is not None:
        pair_ids = [args.only_pid]
    elif args.limit is not None:
        pair_ids = pair_ids[:args.limit]

    strategy_map = {pid: full_strategy_map[pid] for pid in pair_ids}
    counts = {"negation_opener": 0, "hedging": 0}
    for pid in pair_ids:
        counts[strategy_map[pid]] += 1
    print(f"Selected pair_ids ({len(pair_ids)}): {pair_ids}")
    print(f"Strategy distribution: {counts}")
    print(f"Full 20-pair strategy map:")
    for pid in full_pair_ids:
        print(f"  pid={pid:>3d} -> {full_strategy_map[pid]}")

    _load_dotenv_into_os_environ()
    _require_env("ANTHROPIC_API_KEY")
    questions = _load_questions()
    existing = _load_existing()
    todo = [pid for pid in pair_ids if pid not in existing]
    print(f"Existing v6 records: {len(existing)}")
    print(f"To generate this run: {len(todo)} -> {todo}")
    if not todo:
        print("Nothing to do.")
        return 0

    client = _build_client()
    effort_state: dict = {}
    n_ok = 0
    n_fail = 0
    failed: list[tuple[int, str]] = []

    for pid in todo:
        question = questions[pid]
        b_strategy = strategy_map[pid]
        base_prompt = PROMPT_TEMPLATE.format(
            question=question,
            pair_id=pid,
            question_repr=json.dumps(question),
            b_cue_strategy=b_strategy,
            wc_min=WORD_COUNT_MIN,
            wc_max=WORD_COUNT_MAX,
            openers_list=", ".join(repr(x) for x in OPENERS_V6),
            hedges_strict_list=", ".join(repr(x) for x in HEDGE_TOKENS_STRICT),
            hedges_banned_list=", ".join(
                repr(x) for x in HEDGE_TOKENS_BANNED_EXTENDED
            ),
            authority_openers=", ".join(
                repr(x) for x in AUTHORITY_OPENERS_V6
            ),
            contrast_list=", ".join(
                repr(x) for x in CONTRAST_CONJUNCTIONS_V6
            ),
        )
        print(f"\n--- pid={pid} strategy={b_strategy} ---")
        record = None
        last_err = None
        extra_retry_note = ""

        # Spec: up to 3 retries per pair = up to 4 total attempts.
        total_attempts = args.max_retries + 1
        for attempt in range(total_attempts):
            prompt = base_prompt + extra_retry_note
            try:
                raw, elapsed = _call_opus(client, prompt, effort_state)
            except Exception as e:
                last_err = (f"API error attempt {attempt}: "
                            f"{type(e).__name__}: {str(e)[:180]}")
                print(f"   {last_err}")
                continue
            tag = " WARN slow" if elapsed > SLOW_CALL_WARN_SEC else ""
            print(f"   attempt {attempt}: {elapsed:.1f}s{tag}")

            try:
                cand = _parse_response_to_dict(raw)
            except Exception as e:
                last_err = f"JSON parse fail: {e}"
                print(f"   {last_err}")
                continue

            cand["b_cue_strategy"] = b_strategy
            ok, why, warns = validate_v6(cand)
            if not ok:
                last_err = f"validator: {why}"
                print(f"   {last_err}")

                # Build a specific retry hint based on the violation.
                why_str = str(why or "")
                if why_str.startswith("wc_violation:"):
                    # Format: wc_violation:<side>:<count>:<message>
                    parts = why_str.split(":", 3)
                    side = parts[1] if len(parts) > 1 else "?"
                    count = parts[2] if len(parts) > 2 else "?"
                    extra_retry_note = (
                        f"\n\nCRITICAL RETRY NOTE:\n"
                        f"Your previous {side}-side was {count} words, "
                        f"which is outside the hard cap of "
                        f"{WORD_COUNT_MIN}-{WORD_COUNT_MAX} words. On this "
                        f"retry, BOTH A-side and B-side MUST be between "
                        f"{WORD_COUNT_MIN} and {WORD_COUNT_MAX} words "
                        f"(inclusive). Count words by whitespace split. "
                        f"Prioritize hitting the length target; prune "
                        f"modifiers and prepositional phrases if needed."
                    )
                elif "A-side starts with forbidden negation opener" in why_str:
                    extra_retry_note = (
                        "\n\nCRITICAL RETRY NOTE:\n"
                        "Your previous A-side started with a forbidden "
                        "negation opener. A-side MUST begin with a "
                        "non-negating affirmative subject phrase. Also "
                        f"keep both sides in the {WORD_COUNT_MIN}-"
                        f"{WORD_COUNT_MAX} word cap."
                    )
                else:
                    extra_retry_note = (
                        "\n\nCRITICAL RETRY NOTE:\n"
                        f"Your previous output failed validation: {why_str}. "
                        f"Fix it on this retry. Also keep both sides in the "
                        f"{WORD_COUNT_MIN}-{WORD_COUNT_MAX} word cap."
                    )
                continue

            b_text = str(cand["b_side"]).strip()
            a_text = str(cand["a_side"]).strip()
            record = {
                "pair_id": int(pid),
                "question": question,
                "b_cue_strategy": b_strategy,
                "a_side": a_text,
                "a_side_word_count": word_count(a_text),
                "a_side_truth_rationale":
                    str(cand["a_side_truth_rationale"]).strip(),
                "b_side": b_text,
                "b_side_word_count": word_count(b_text),
                "b_side_truth_rationale":
                    str(cand["b_side_truth_rationale"]).strip(),
                "cues_in_b_observed": _infer_observed_cues(b_text),
                "generator_confidence": float(cand["generator_confidence"]),
                "model": MODEL_ID_PRIMARY,
                "effort_kwarg":
                    effort_state.get("kwargs", {}).get("effort", "(none)"),
                "retries": attempt,
                "elapsed_sec_last_attempt": float(elapsed),
                "soft_warnings": warns,
            }
            break

        if record is None:
            n_fail += 1
            failed.append((pid, last_err or "unknown"))
            print(f"   FAILED after {total_attempts} attempts")
            continue

        existing[pid] = record
        _save(existing)
        n_ok += 1
        print(
            f"   OK retries={record['retries']} "
            f"A_wc={record['a_side_word_count']} "
            f"B_wc={record['b_side_word_count']} "
            f"observed={record['cues_in_b_observed']} "
            f"conf={record['generator_confidence']:.2f}"
        )

    print("\n" + "=" * 72)
    print(f"Run summary: ok={n_ok} fail={n_fail} of {len(todo)} attempted")
    if failed:
        print("  Failed pids:")
        for pid, reason in failed:
            print(f"    pid={pid}: {reason}")
    print(f"  Output: {OUT_JSON} ({len(existing)} total records)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in generate_stage0_paired_v6.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
