#!/usr/bin/env python3
"""
Re-judge the existing v3 paraphrase rewrites under a RELAXED criterion:
"truth-class preservation" rather than "strict propositional equivalence".

Background. The first judge (scripts/judge_stage0_paired_tqa.py) asked
"do these two sentences assert the same factual proposition?". That is a
strict truth-conditional equivalence test, and it produced 5/20 faithful
pairs on v3 because a hedge insertion ("X is Y" -> "X is typically Y")
shifts truth conditions even when both sentences land on the same side
of the truth/false divide. Per advisor feedback, the experiment only
needs truth-class preservation: did the rewrite stay on the same side
of the truth/false divide as the original?

This script does NOT modify the existing strict judge, the existing
generation file, or the existing strict judge's output. It writes a
parallel artifact at stage0_paired_tqa/stage0_paired_judge_truthclass.json
with the relaxed verdicts.

Per slot we make TWO independent calls (same shape as the strict judge):

    Call 1 (A-side):  original_correct   + a_side_rewritten
    Call 2 (B-side):  original_incorrect + b_side_rewritten

A pair is `pair_truthclass_faithful` iff BOTH sides preserve their
truth class.

Per-slot record (one per pair_id):

    pair_id                            int
    a_truthclass_same                  bool
    a_truthclass_confidence            float 0..1
    a_truthclass_rationale             str
    a_truthclass_elapsed_sec           float
    b_truthclass_same                  bool
    b_truthclass_confidence            float 0..1
    b_truthclass_rationale             str
    b_truthclass_elapsed_sec           float
    pair_truthclass_faithful           bool
    judge_model                        str
    judge_criterion                    str   (constant: "truth_class_v1")

Model: gpt-5.4-2026-03-05 (advisor pin); same Responses-API kwarg ladder
and resolver as scripts/judge_stage0_paired_tqa.py.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations.json"
OUT_JSON = (REPO_ROOT / "stage0_paired_tqa"
            / "stage0_paired_judge_truthclass.json")

JUDGE_CRITERION_TAG = "truth_class_v1"


# --- workspace-local .env loader (mirrors judge_stage0_paired_tqa.py) ------
_ENV_PATH = REPO_ROOT / ".env"
_ENV_PLACEHOLDER = "REPLACE_WITH_REAL_KEY"


def _load_dotenv_into_os_environ(env_path: Path = _ENV_PATH) -> None:
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _require_env(name: str) -> None:
    val = os.environ.get(name, "")
    if not val:
        print(f"ERROR: {name} is not set. Populate {_ENV_PATH}.",
              file=sys.stderr)
        raise SystemExit(1)
    if val == _ENV_PLACEHOLDER:
        print(f"ERROR: {name} is still the placeholder.", file=sys.stderr)
        raise SystemExit(1)


_load_dotenv_into_os_environ()

# --- constants -------------------------------------------------------------
MODEL_ID_PRIMARY = "gpt-5.4"
MODEL_FALLBACK_PREFIX = "gpt-5.4"
MODEL_PINNED_TARGET = "gpt-5.4-2026-03-05"

CLIENT_TIMEOUT_SEC = 30.0
CLIENT_SDK_RETRIES = 0
SLOW_CALL_WARN_SEC = 10.0
MAX_RETRIES_DEFAULT = 3

# Prompt copied verbatim from advisor feedback (2026-04-17). The judge is
# NOT told whether the original is true or false; it must determine that
# itself from world knowledge, then check whether the paraphrase lands on
# the same side of the truth/false divide.
JUDGE_PROMPT_TEMPLATE = """You will see an original statement and a paraphrased statement. The original is known to be either a true statement or a false statement. Your question: is the paraphrased statement on the SAME TRUTH CLASS as the original? That is, if the original is true, is the paraphrase also true? If the original is false, is the paraphrase also false? The paraphrase may be STRONGER or WEAKER than the original; that's fine as long as the truth class is preserved. Reply in JSON: {{same_truth_class: bool, confidence: float, rationale: string}}.

Original: {ORIGINAL}
Paraphrased: {REWRITTEN}

Return JSON ONLY, matching this schema exactly:
{{
  "same_truth_class": <true or false>,
  "confidence": <float between 0 and 1>,
  "rationale": "<one concise sentence>"
}}"""


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_to_json(text: str) -> str | None:
    m = _JSON_OBJECT_RE.search(text or "")
    return m.group(0) if m else None


def _extract_text(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt:
        return txt
    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            t = getattr(content, "text", None)
            if isinstance(t, str):
                parts.append(t)
            elif hasattr(t, "value"):
                parts.append(t.value)
    return "".join(parts)


# --- model resolution (same logic as judge_stage0_paired_tqa.py) -----------
def _resolve_judge_model(client, primary: str) -> str:
    import openai
    try:
        client.models.retrieve(primary)
        return primary
    except openai.NotFoundError:
        print(f"NOTE: requested model {primary!r} not found; searching for "
              f"most recent {MODEL_FALLBACK_PREFIX}* snapshot.")
    except openai.APIStatusError as e:
        print(f"WARNING: models.retrieve({primary!r}) returned APIStatusError "
              f"(status={getattr(e, 'status_code', '?')}); attempting "
              f"optimistic call.")
        return primary
    except Exception as e:
        print(f"WARNING: models.retrieve({primary!r}) raised "
              f"{type(e).__name__}: {e}. Falling through to list().")

    try:
        listing = client.models.list()
    except Exception as e:
        raise RuntimeError(
            f"Could not list models after {primary!r} not found: "
            f"{type(e).__name__}: {e}"
        )
    candidates = [m.id for m in listing.data
                  if m.id.startswith(MODEL_FALLBACK_PREFIX)]
    if not candidates:
        raise RuntimeError(
            f"Requested model {primary!r} not found and no "
            f"{MODEL_FALLBACK_PREFIX}* snapshots are available for this key."
        )
    candidates.sort(reverse=True)
    return candidates[0]


# --- Responses-API kwargs ladder (cached across calls) ---------------------
_CALL_KWARGS_RESOLVED: dict | None = None
_CALL_KWARGS_LABEL: str | None = None


def _kwarg_candidates() -> list[tuple[dict, str]]:
    # temperature=0.0 first per advisor instruction.
    return [
        ({"text": {"format": {"type": "json_object"}}, "temperature": 0.0},
         "text.format + temperature=0"),
        ({"text": {"format": {"type": "json_object"}}}, "text.format only"),
        ({"response_format": {"type": "json_object"}, "temperature": 0.0},
         "response_format + temperature=0"),
        ({"response_format": {"type": "json_object"}}, "response_format only"),
        ({"temperature": 0.0}, "temperature=0 only"),
        ({}, "plain"),
    ]


def _is_kwarg_rejection(err: Exception) -> bool:
    from openai import APIStatusError, BadRequestError
    msg = str(err).lower()
    if isinstance(err, TypeError):
        return ("unexpected keyword argument" in msg
                or "response_format" in msg
                or "text" in msg
                or "temperature" in msg)
    if isinstance(err, (BadRequestError, APIStatusError)):
        if getattr(err, "status_code", None) not in (400, None):
            return False
        return ("response_format" in msg
                or "text.format" in msg
                or "temperature" in msg
                or "unsupported" in msg
                or "not supported" in msg)
    return False


def _call_responses(client, model_id: str, prompt: str
                    ) -> tuple[str, str, float]:
    global _CALL_KWARGS_RESOLVED, _CALL_KWARGS_LABEL
    from openai import (APITimeoutError, RateLimitError,
                        APIStatusError, BadRequestError)

    if _CALL_KWARGS_RESOLVED is not None:
        attempts = [(_CALL_KWARGS_RESOLVED, f"cached {_CALL_KWARGS_LABEL}")]
    else:
        attempts = _kwarg_candidates()

    last_err: Exception | None = None
    resp = None
    used_kwargs, used_label = {}, ""
    t0 = time.perf_counter()

    for kwargs, label in attempts:
        try:
            resp = client.responses.create(
                model=model_id, input=prompt,
                timeout=CLIENT_TIMEOUT_SEC, **kwargs,
            )
            used_kwargs, used_label = kwargs, label
            break
        except (APITimeoutError, RateLimitError):
            raise
        except (TypeError, BadRequestError, APIStatusError) as e:
            if _is_kwarg_rejection(e) and _CALL_KWARGS_RESOLVED is None:
                last_err = e
                print(f"  [{_now_stamp()}] probe rejected {label}: "
                      f"{type(e).__name__}: {str(e)[:140]}")
                continue
            raise
        except Exception:
            raise

    if resp is None:
        raise RuntimeError(
            f"Every Responses-API kwarg candidate rejected. "
            f"Last error: {type(last_err).__name__ if last_err else 'None'}: "
            f"{last_err}"
        )

    elapsed = time.perf_counter() - t0
    if _CALL_KWARGS_RESOLVED is None:
        _CALL_KWARGS_RESOLVED = used_kwargs
        _CALL_KWARGS_LABEL = used_label
        print(f"  [{_now_stamp()}] Responses kwargs resolved to: {used_label}")
    if elapsed > SLOW_CALL_WARN_SEC:
        print(f"  [{_now_stamp()}] WARNING: slow call - {elapsed:.2f}s "
              f"exceeds {SLOW_CALL_WARN_SEC:.0f}s.")

    resolved = getattr(resp, "model", model_id) or model_id
    return _extract_text(resp), resolved, elapsed


def _judge_pair(
    client, model_id: str, original: str, rewritten: str,
    side_label: str, pair_id: int, max_retries: int,
) -> tuple[dict, str]:
    """Returns (parsed_record_subset, resolved_model_id_used)."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(ORIGINAL=original,
                                          REWRITTEN=rewritten)
    last_err: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw, resolved, elapsed = _call_responses(client, model_id, prompt)
        except Exception as e:
            last_err = (f"API error {side_label} attempt {attempt}: "
                        f"{type(e).__name__}: {e}")
            print(f"  pid={pair_id} {side_label}: {last_err}")
            if max_retries == 1:
                raise
            time.sleep(1.5 * attempt)
            continue

        blob = _strip_to_json(raw)
        if blob is None:
            last_err = f"no JSON object in response (attempt {attempt})"
            print(f"  pid={pair_id} {side_label}: {last_err}; "
                  f"head={raw[:120]!r}")
            time.sleep(0.5)
            continue
        try:
            data = json.loads(blob)
        except Exception as e:
            last_err = f"JSON parse failed (attempt {attempt}): {e}"
            print(f"  pid={pair_id} {side_label}: {last_err}")
            time.sleep(0.5)
            continue

        required = {"same_truth_class", "confidence", "rationale"}
        missing = required - data.keys()
        if missing:
            last_err = f"missing keys {sorted(missing)} (attempt {attempt})"
            print(f"  pid={pair_id} {side_label}: {last_err}")
            continue

        try:
            same = bool(data["same_truth_class"])
            conf = float(data["confidence"])
        except Exception as e:
            last_err = f"type coercion failed (attempt {attempt}): {e}"
            print(f"  pid={pair_id} {side_label}: {last_err}")
            continue

        return ({
            "same_truth_class": same,
            "confidence": conf,
            "rationale": str(data["rationale"]),
            "elapsed_sec": round(float(elapsed), 3),
        }, resolved)

    raise RuntimeError(
        f"pid={pair_id} side={side_label}: judge failed after "
        f"{max_retries} attempts. Last error: {last_err}"
    )


def _load_generations() -> list[dict]:
    if not GEN_JSON.exists():
        raise RuntimeError(f"{GEN_JSON} does not exist; run Step 4 first.")
    with open(GEN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"{GEN_JSON} is not a JSON list.")
    return sorted(data, key=lambda r: int(r["pair_id"]))


def _load_existing() -> list[dict]:
    if not OUT_JSON.exists():
        return []
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    args = p.parse_args()

    print("=" * 72)
    print("rejudge_paired_tqa_truthclass.py "
          f"(criterion={JUDGE_CRITERION_TAG})")
    print("=" * 72)
    _require_env("OPENAI_API_KEY")

    from openai import OpenAI
    client = OpenAI(timeout=CLIENT_TIMEOUT_SEC, max_retries=CLIENT_SDK_RETRIES)
    print(f"OpenAI client: timeout={CLIENT_TIMEOUT_SEC}s, "
          f"max_retries={CLIENT_SDK_RETRIES}")

    model_id = _resolve_judge_model(client, MODEL_ID_PRIMARY)
    if model_id != MODEL_PINNED_TARGET:
        print(f"NOTE: resolver returned {model_id!r}; advisor's pin target "
              f"was {MODEL_PINNED_TARGET!r}. Recording {model_id!r} in every "
              f"record so the actual judge model is unambiguous.")
    else:
        print(f"Resolved judge model: {model_id!r} (matches advisor pin)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    gens = _load_generations()
    existing = _load_existing()
    done_ids = {int(r["pair_id"]) for r in existing}
    results: list[dict] = list(existing)
    if existing:
        print(f"Resume: {len(existing)} existing pair_ids in {OUT_JSON}")

    target_total = args.limit if args.limit is not None else len(gens)
    print(f"Will judge up to {target_total} of {len(gens)} pairs")

    n_done_now = 0
    actual_models: set[str] = set()
    try:
        for gen_rec in gens:
            if len(results) >= target_total:
                break
            pid = int(gen_rec["pair_id"])
            if pid in done_ids:
                continue
            print(f"\n[pid={pid}] truth-class judging A-side and B-side ...")

            a_rec, m1 = _judge_pair(
                client, model_id,
                gen_rec["original_correct"],
                gen_rec["a_side_rewritten"],
                "A", pid, args.max_retries,
            )
            actual_models.add(m1)
            tag_a = "SAME" if a_rec["same_truth_class"] else "diff"
            print(f"  A: {tag_a}  conf={a_rec['confidence']:.2f}  "
                  f"elapsed={a_rec['elapsed_sec']:.2f}s")

            b_rec, m2 = _judge_pair(
                client, model_id,
                gen_rec["original_incorrect"],
                gen_rec["b_side_rewritten"],
                "B", pid, args.max_retries,
            )
            actual_models.add(m2)
            tag_b = "SAME" if b_rec["same_truth_class"] else "diff"
            print(f"  B: {tag_b}  conf={b_rec['confidence']:.2f}  "
                  f"elapsed={b_rec['elapsed_sec']:.2f}s")

            pair_faithful = (a_rec["same_truth_class"]
                             and b_rec["same_truth_class"])
            tag_pair = "FAITHFUL" if pair_faithful else "UNFAITHFUL"
            print(f"  pair_truthclass_faithful = {tag_pair}")

            results.append({
                "pair_id": pid,
                "a_truthclass_same":          a_rec["same_truth_class"],
                "a_truthclass_confidence":    a_rec["confidence"],
                "a_truthclass_rationale":     a_rec["rationale"],
                "a_truthclass_elapsed_sec":   a_rec["elapsed_sec"],
                "b_truthclass_same":          b_rec["same_truth_class"],
                "b_truthclass_confidence":    b_rec["confidence"],
                "b_truthclass_rationale":     b_rec["rationale"],
                "b_truthclass_elapsed_sec":   b_rec["elapsed_sec"],
                "pair_truthclass_faithful":   pair_faithful,
                "judge_model":                m1 if m1 == m2 else f"{m1}|{m2}",
                "judge_criterion":            JUDGE_CRITERION_TAG,
            })
            done_ids.add(pid)
            n_done_now += 1
    finally:
        if results:
            results.sort(key=lambda r: int(r["pair_id"]))
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nWrote {len(results)} truth-class judgments -> "
                  f"{OUT_JSON}")

    n = len(results)
    n_pair_faithful = sum(1 for r in results if r["pair_truthclass_faithful"])
    n_a_faithful = sum(1 for r in results if r["a_truthclass_same"])
    n_b_faithful = sum(1 for r in results if r["b_truthclass_same"])
    print("\nSummary (truth-class criterion):")
    print(f"  pairs judged (this run):              {n_done_now}")
    print(f"  pairs in output file:                 {n}")
    print(f"  a_truthclass_faithful:                {n_a_faithful}/{n}")
    print(f"  b_truthclass_faithful:                {n_b_faithful}/{n}")
    print(f"  pair_truthclass_faithful (A and B):   {n_pair_faithful}/{n}")
    print(f"  resolved judge model(s):              {sorted(actual_models)}")
    print("\n  Slots flagged pair_truthclass_faithful=False:")
    flagged_any = False
    for r in results:
        if not r["pair_truthclass_faithful"]:
            flagged_any = True
            why = []
            if not r["a_truthclass_same"]:
                why.append(f"A-not-same (conf="
                           f"{r['a_truthclass_confidence']:.2f})")
            if not r["b_truthclass_same"]:
                why.append(f"B-not-same (conf="
                           f"{r['b_truthclass_confidence']:.2f})")
            print(f"    pid={r['pair_id']:>3d}: {'; '.join(why)}")
    if not flagged_any:
        print("    (none)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in rejudge_paired_tqa_truthclass.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
