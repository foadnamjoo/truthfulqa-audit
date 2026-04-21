#!/usr/bin/env python3
"""
v7(a) v2 A-side bilateral singleton generator.

v7(a) v1 produced 20 TRUE statements in the INCORRECT-side surface style.
The surface_lr classifier flipped strongly (unilateral robust 10% -> 70%),
but BGE-large and ModernBERT-base barely moved on the bilateral fool-drop
(+0.07 each). Diagnosis: v1 statements are canonical textbook facts
("Berlin Wall fell in 1989", "Adam Smith published The Wealth of Nations
in 1776"). Deep semantic classifiers recognize these regardless of
surface profile, so the surface shortcut can't be exercised.

v2 keeps every structural element of v1 (same validator, same topic plan,
same word-count band, same cue-strip rules) and changes ONLY the
content constraint: the statement must be TRUE *and* sound
counterintuitive / surprising / improbable to a non-expert reader.
That forces the semantic channel to be ambiguous, so the surface
shortcut has room to dominate - mirroring how the TQA FALSE side is
plausible-sounding.

v1 artifacts are preserved at stage0_v7a_bilateral/stage0_singleton_v7a_*.
v2 writes new files at stage0_v7a_bilateral/stage0_singleton_v7a_v2_*.
"""
from __future__ import annotations

import argparse
import json
import sys
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
from scripts.generate_stage0_singleton_v7a import (  # noqa: E402
    TOPIC_PLAN_V7A,
    _expand_plan,
    _build_subject_status,
    _extract_subject,
)

OUT_DIR = REPO_ROOT / "stage0_v7a_bilateral"
OUT_JSON = OUT_DIR / "stage0_singleton_v7a_v2_generations.json"
V1_JSON = OUT_DIR / "stage0_singleton_v7a_generations.json"

PROMPT_TEMPLATE = """You are producing a single factually TRUE statement in a cue-stripped
surface style, as a probe for a shortcut-learning audit.

Topic: {topic}

CONTENT CONSTRAINT (this is the important one):
- The statement MUST be factually TRUE in the real world (verifiable
  against mainstream references).
- The statement MUST sound SURPRISING, COUNTERINTUITIVE, or IMPROBABLE
  to a non-expert reader. A reasonable person hearing it cold should
  think "wait, that can't be right" before discovering it actually is.
- AVOID canonical textbook facts: famous dates everyone memorized in
  school, household historical figures, elementary-school science,
  common definitions. Specifically AVOID: Magna Carta 1215, US
  Constitution, Berlin Wall 1989, Julius Caesar assassination, Federal
  Reserve 1913, NYSE 1792, Adam Smith Wealth of Nations 1776, Keynes
  General Theory 1936, Durkheim Suicide, Weber Protestant Ethic, 10%
  brain myth, goldfish memory myth, H2O composition, Jupiter mass,
  vitamin C / scurvy, iron deficiency anemia, dietary fiber, SPR 1882,
  Houdini, heart beats per day, US Supreme Court size.
- PREFER facts of the "wait, really?" kind: biological oddities,
  surprising historical connections, counterintuitive statistics,
  little-known rules of law, uncommon medical findings, unexpected
  economic history, surprising scientific phenomena.
- The surprise MUST come from the fact itself, NOT from style or
  framing (style is already constrained below).

Write ONE statement that is:
- Factually TRUE in the real world, AND counterintuitive as described
  above.
- A direct positive declarative sentence.
- Within the topic area "{topic}".
- Distinct in subject from any prior statement generated in this session.

Surface style (strict - identical to v1):
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
  "statement": "<fresh TRUE-but-counterintuitive, cue-stripped statement, {wc_min}-{wc_max} words>",
  "generator_truth_rationale": "<one sentence explaining why the statement is factually true AND why it sounds counterintuitive>",
  "generator_confidence": <float in [0,1], >= 0.70>
}}
"""


def _load_v1_subjects() -> list[str]:
    if not V1_JSON.exists():
        return []
    with open(V1_JSON, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    subs: list[str] = []
    seen: set[str] = set()
    for r in data:
        sub = _extract_subject(r.get("statement", ""))
        if not sub:
            continue
        key = sub.lower()
        if key in seen:
            continue
        seen.add(key)
        subs.append(sub)
    return subs


def _build_subject_status_with_v1(
    records: list[dict], v1_subjects: list[str]
) -> str:
    """Like v1's _build_subject_status but prefixes v1-used subjects too."""
    from scripts.generate_stage0_singleton_v7a import _subject_list
    cur = _subject_list(records)
    all_subs: list[str] = []
    seen: set[str] = set()
    for s in list(v1_subjects) + list(cur):
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        all_subs.append(s)
    if not all_subs:
        return ("Subjects already used in prior statements (v1 + this session): "
                "none. Pick any distinct subject for this call.")
    listed = ", ".join(f'"{s}"' for s in all_subs)
    return ("Subjects already used in prior statements (v1 + this session): "
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
    print("generate_stage0_singleton_v7a_v2.py")
    print("  (TRUE-but-counterintuitive, INCORRECT-style surface, n=20)")
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
    v1_subjects = _load_v1_subjects()
    print(f"v1 subjects blocklist size: {len(v1_subjects)}")
    todo = [s for s in slots if s["id"] not in existing]
    print(f"Existing v2 records: {len(existing)}")
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
            subj_status = _build_subject_status_with_v1(
                results_for_subjects, v1_subjects)
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
                        "Your previous statement contained a negation token. "
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
                        "Your previous statement contained a hedging token. "
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
        print("NON-RECOVERABLE FAILURE in generate_stage0_singleton_v7a_v2.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
