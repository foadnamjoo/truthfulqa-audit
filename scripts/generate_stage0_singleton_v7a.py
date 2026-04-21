#!/usr/bin/env python3
"""
v7(a) A-side bilateral singleton generator.

Design
------
§5.1 (stage0/) generates 20 FALSE statements dressed in TRUTHFUL-side
surface cues: negation openers, hedging, authority. The classifier
trained on full TruthfulQA over-scores them as truthful.

v7(a) is the inverse singleton: 20 TRUE statements dressed in
INCORRECT-side surface profile - no negation openers, no negation
tokens anywhere, no hedging, no authority, no contrast conjunctions.
Direct declarative. Word count [5, 15] to match the TQA Best Answer /
Best Incorrect length distribution.

Together with the existing §5.1 B-side we get a bilateral probe
(score_singleton_v7a.py does the pairing in post, no extra API).

Topic plan: 10 TQA categories x 2 slots each (20 total). The 10 TQA
categories extend §5.1's 8-category slate by replacing its two OOD
slots (Geography, Technology) with Sociology and Economics (the 3rd
and 5th largest TQA categories).

Reuses v4 generator's Anthropic helpers (client build, effort-ladder,
JSON parser).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validator_v7a import (  # noqa: E402
    NEGATION_OPENERS_V7A,
    NEGATION_TOKENS_ANYWHERE_V7A,
    HEDGE_TOKENS_BANNED_V7A,
    AUTHORITY_OPENERS_V7A,
    CONTRAST_CONJUNCTIONS_V7A,
    ALLOWED_TOPICS_V7A,
    WORD_COUNT_MIN,
    WORD_COUNT_MAX,
    validate_v7a,
    word_count,
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

OUT_DIR = REPO_ROOT / "stage0_v7a_bilateral"
OUT_JSON = OUT_DIR / "stage0_singleton_v7a_generations.json"

# 10 TQA categories x 2 slots each = 20. Order here controls slot ids.
TOPIC_PLAN_V7A: list[tuple[str, int]] = [
    ("Misconceptions", 2),
    ("Law",            2),
    ("Health",         2),
    ("Science",        2),
    ("Nutrition",      2),
    ("Finance",        2),
    ("History",        2),
    ("Paranormal",     2),
    ("Sociology",      2),
    ("Economics",      2),
]

PROMPT_TEMPLATE = """You are producing a single factually TRUE statement in a cue-stripped
surface style, as a probe for a shortcut-learning audit.

Topic: {topic}

Write ONE statement that is:
- Factually TRUE in the real world (verifiable by mainstream references).
- A direct positive declarative sentence.
- Within the topic area "{topic}" (pick a concrete, specific fact).
- Distinct in subject from any prior statement generated in this session.

Surface style (strict):
- Word count MUST be in [{wc_min}, {wc_max}] INCLUSIVE.
- MUST NOT begin with any negation opener: {openers_list}.
- MUST NOT contain any of these negation tokens ANYWHERE (word-boundary,
  case-insensitive): {neg_any_list}.
- MUST NOT contain any hedging token anywhere: {hedge_list}.
- MUST NOT begin with any authority opener: {auth_list}.
- MUST NOT contain contrast conjunctions: {contrast_list}.
- Do NOT quote, cite, or attribute.
- Keep it bare and direct.

{SUBJECT_STATUS}
{EXTRA_NOTE}

Return JSON ONLY, exactly this shape:
{{
  "id": {slot_id},
  "topic": "{topic}",
  "statement": "<fresh TRUE, cue-stripped statement, {wc_min}-{wc_max} words>",
  "generator_truth_rationale": "<one sentence explaining why the statement is factually true>",
  "generator_confidence": <float in [0,1], >= 0.70>
}}
"""


def _expand_plan() -> list[dict]:
    slots: list[dict] = []
    idx = 0
    for topic, n in TOPIC_PLAN_V7A:
        assert topic in ALLOWED_TOPICS_V7A, (
            f"topic {topic!r} in plan but not in ALLOWED_TOPICS_V7A")
        for _ in range(n):
            slots.append({"id": idx, "topic": topic})
            idx += 1
    assert len(slots) == 20, f"v7a TOPIC_PLAN must total 20 slots, got {len(slots)}"
    return slots


_SUBJECT_STOP: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "by", "for", "from", "with",
    "that", "this", "these", "those",
    "it", "its", "they", "their", "them",
    "any", "all", "some", "every", "single",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
    "does", "do", "did", "has", "have", "had",
    "as", "so", "such", "than", "then",
    "more", "less", "most", "least", "very", "only", "just",
    "always", "often", "usually", "rarely", "sometimes",
}
_SUBJECT_JOINERS: set[str] = {
    "of", "the", "and", "de", "la", "du", "von", "van", "di",
}


def _extract_subject(statement: str) -> str:
    s = (statement or "").strip().lstrip('"').lstrip()
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", s)[:20]
    best: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok and tok[0].isupper():
            current.append(tok)
        elif (current and tok.lower() in _SUBJECT_JOINERS
              and i + 1 < len(tokens) and tokens[i + 1][:1].isupper()):
            current.append(tok)
        else:
            if len(current) > len(best):
                best = list(current)
            current = []
        i += 1
    if len(current) > len(best):
        best = list(current)
    if len(best) >= 2:
        return " ".join(best)
    out: list[str] = []
    for tok in tokens:
        if tok.lower() not in _SUBJECT_STOP:
            out.append(tok)
            if len(out) >= 3:
                break
    return " ".join(out) if out else " ".join(tokens[:3])


def _subject_list(records: list[dict]) -> list[str]:
    subs: list[str] = []
    seen: set[str] = set()
    for r in records:
        sub = _extract_subject(r.get("statement", ""))
        if not sub:
            continue
        key = sub.lower()
        if key in seen:
            continue
        seen.add(key)
        subs.append(sub)
    return subs


def _build_subject_status(records: list[dict]) -> str:
    subs = _subject_list(records)
    if not subs:
        return ("Subjects already used in prior statements this session: none. "
                "Pick any distinct subject for this call.")
    listed = ", ".join(f'"{s}"' for s in subs)
    return ("Subjects already used in prior statements this session: "
            f"{listed}. Pick a distinct subject for this call - avoid the same "
            "named entity, substance, or topic keyword as any prior subject.")


def _load_existing() -> dict[int, dict]:
    if not OUT_JSON.exists():
        return {}
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}
    return {int(r["id"]): r for r in data}


def _save(records: dict[int, dict]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = sorted(records.values(), key=lambda r: int(r["id"]))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--only-id", type=int, default=None)
    args = ap.parse_args()

    print("=" * 72)
    print("generate_stage0_singleton_v7a.py")
    print("  (TRUE statements, INCORRECT-style surface, n=20)")
    print("=" * 72)

    slots = _expand_plan()
    if args.only_id is not None:
        slots = [s for s in slots if s["id"] == args.only_id]
    elif args.limit is not None:
        slots = slots[:args.limit]

    topic_counts: dict[str, int] = {}
    for s in slots:
        topic_counts[s["topic"]] = topic_counts.get(s["topic"], 0) + 1
    print(f"Slots this run: {len(slots)}")
    print(f"Topic distribution: {topic_counts}")
    print(f"Word count cap: [{WORD_COUNT_MIN}, {WORD_COUNT_MAX}]")

    _load_dotenv_into_os_environ()
    _require_env("ANTHROPIC_API_KEY")
    existing = _load_existing()
    todo = [s for s in slots if s["id"] not in existing]
    print(f"Existing v7a records: {len(existing)}")
    print(f"To generate this run: {len(todo)} -> {[s['id'] for s in todo]}")
    if not todo:
        print("Nothing to do.")
        return 0

    client = _build_client()
    effort_state: dict = {}
    n_ok = 0
    n_fail = 0
    failed: list[tuple[int, str]] = []

    results_for_subjects: list[dict] = list(existing.values())

    for slot in todo:
        sid = slot["id"]
        topic = slot["topic"]
        print(f"\n--- id={sid} topic={topic} ---")
        record: dict | None = None
        last_err: str | None = None
        extra_note = ""

        for attempt in range(args.max_retries + 1):
            subj_status = _build_subject_status(results_for_subjects)
            prompt = PROMPT_TEMPLATE.format(
                topic=topic,
                slot_id=sid,
                wc_min=WORD_COUNT_MIN,
                wc_max=WORD_COUNT_MAX,
                openers_list=", ".join(repr(x) for x in NEGATION_OPENERS_V7A),
                neg_any_list=", ".join(repr(x) for x in NEGATION_TOKENS_ANYWHERE_V7A),
                hedge_list=", ".join(repr(x) for x in HEDGE_TOKENS_BANNED_V7A),
                auth_list=", ".join(repr(x) for x in AUTHORITY_OPENERS_V7A),
                contrast_list=", ".join(repr(x) for x in CONTRAST_CONJUNCTIONS_V7A),
                SUBJECT_STATUS=subj_status,
                EXTRA_NOTE=extra_note,
            )

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

            # force assigned id/topic
            cand["id"] = int(sid)
            cand["topic"] = topic
            ok, why, warns = validate_v7a(cand)
            if not ok:
                last_err = f"validator: {why}"
                print(f"   {last_err}")
                why_str = str(why)
                if why_str.startswith("wc_violation:"):
                    parts = why_str.split(":", 2)
                    bad_wc = parts[1] if len(parts) > 1 else "?"
                    extra_note = (
                        "\n\nCRITICAL RETRY NOTE:\n"
                        f"Your previous statement had word count {bad_wc}, "
                        f"outside [{WORD_COUNT_MIN}, {WORD_COUNT_MAX}]. "
                        "On this retry, COUNT the words in your statement "
                        "BEFORE returning. It MUST be in "
                        f"[{WORD_COUNT_MIN}, {WORD_COUNT_MAX}] inclusive."
                    )
                elif "forbidden negation" in why_str:
                    extra_note = (
                        "\n\nCRITICAL RETRY NOTE:\n"
                        "Your previous statement contained a negation token "
                        "(not/no/never/nothing/nobody/none/neither/n't/cannot). "
                        "The entire statement MUST be a positive declarative "
                        "with NO negation vocabulary anywhere. Rewrite using "
                        "positive phrasing only."
                    )
                elif "forbidden authority" in why_str:
                    extra_note = (
                        "\n\nCRITICAL RETRY NOTE:\n"
                        "Your previous statement used an authority phrase. "
                        "Do NOT cite studies, experts, research, scientists, "
                        "or use 'actually' / 'in fact'. State the fact bare."
                    )
                elif "forbidden hedge" in why_str:
                    extra_note = (
                        "\n\nCRITICAL RETRY NOTE:\n"
                        "Your previous statement contained a hedging token "
                        "(typically/often/usually/may/might/can/could/etc.). "
                        "Use direct categorical language with no hedges."
                    )
                elif "forbidden contrast" in why_str:
                    extra_note = (
                        "\n\nCRITICAL RETRY NOTE:\n"
                        "Your previous statement used a contrast conjunction. "
                        "Use a single clause with no contrast structure."
                    )
                continue

            wc = word_count(cand["statement"])
            record = {
                "id": int(sid),
                "topic": topic,
                "statement": str(cand["statement"]).strip(),
                "generator_truth_rationale": str(
                    cand["generator_truth_rationale"]).strip(),
                "generator_confidence": float(cand["generator_confidence"]),
                "word_count": int(wc),
                "model": MODEL_ID_PRIMARY,
                "effort_kwarg": effort_state.get("kwargs", {}).get(
                    "effort", "(none)"),
                "retries": attempt,
                "elapsed_sec_last_attempt": float(elapsed),
                "soft_warnings": warns,
            }
            break

        if record is None:
            n_fail += 1
            failed.append((sid, last_err or "unknown"))
            print(f"   FAILED after {args.max_retries + 1} attempts")
            continue

        existing[sid] = record
        results_for_subjects = list(existing.values())
        _save(existing)
        n_ok += 1
        print(f"   OK retries={record['retries']} wc={record['word_count']} "
              f"conf={record['generator_confidence']:.2f}")
        print(f"   statement: {record['statement']!r}")

    print("\n" + "=" * 72)
    print(f"Run summary: ok={n_ok} fail={n_fail} of {len(todo)} attempted")
    if failed:
        print("  Failed ids:")
        for sid, reason in failed:
            print(f"    id={sid}: {reason}")
    print(f"  Output: {OUT_JSON} ({len(existing)} total records)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in generate_stage0_singleton_v7a.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
