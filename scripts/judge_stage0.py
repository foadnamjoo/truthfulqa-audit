#!/usr/bin/env python3
"""
Step 9 — Judge the 20 Stage-0 adversarial statements with GPT-5.4 via
the OpenAI Responses API (NOT Chat Completions).

Priming guard: the judge prompt receives ONLY the `statement` field from
stage0/stage0_generations.json. The generator's rationale, intended
cues, confidence, topic, and any other fields are not passed to the
judge. This is important — the paper's Stage-0 analysis rests on an
independent judge signal; leaking generator metadata would defeat the
purpose of this step.

Model selection: "gpt-5.4" is the target alias. If the openai 2.32.0
SDK does not recognize it, fall back to the most recent matching
"gpt-5.4*" snapshot returned by client.models.list(). The resolved
model id is printed on the first call and pinned for all 20 subsequent
calls, so every statement is judged by exactly the same model.

Output: stage0/stage0_judge.json — one record per slot, keyed by `id`:
    id                          int, 0..19
    statement                   str (copy of the generator statement)
    judged_false                bool
    judge_confidence            float 0..1
    judge_rationale             str (one sentence)
    judge_disagreement_flag     bool
    judge_model                 str (resolved model id)
    judge_elapsed_sec           float (wall-clock for that one call)

`judge_disagreement_flag` is defined as:
    generator_confidence_in_falsity > 0.9
      AND judged_false == False
      AND judge_confidence        < 0.6

Per-call diagnostics:
    - client-level timeout=30s and max_retries=0 (belt and suspenders)
    - per-call elapsed-time log with a WARNING if > 10s
    - typed exception handling for APITimeoutError, RateLimitError,
      APIStatusError, BadRequestError, NotFoundError
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
STAGE0_DIR = REPO_ROOT / "stage0"
GEN_JSON = STAGE0_DIR / "stage0_generations.json"
OUT_JSON = STAGE0_DIR / "stage0_judge.json"

# --- workspace-local .env loader -------------------------------------------
# Identical to the loader in scripts/generate_stage0.py so the two
# scripts behave the same way when invoked in a fresh shell: parse
# <repo_root>/.env, populate os.environ for keys not already set, fail
# clearly if the required key is missing or still equals the placeholder.
_ENV_PATH = REPO_ROOT / ".env"
_ENV_PLACEHOLDER = "REPLACE_WITH_REAL_KEY"


def _load_dotenv_into_os_environ(env_path: Path = _ENV_PATH) -> None:
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _require_env(name: str) -> None:
    val = os.environ.get(name, "")
    if not val:
        print(
            f"ERROR: {name} is not set. Populate {_ENV_PATH} with a real "
            f"value (it must not be empty or equal to {_ENV_PLACEHOLDER!r}).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if val == _ENV_PLACEHOLDER:
        print(
            f"ERROR: {name} is still the placeholder {_ENV_PLACEHOLDER!r}. "
            f"Edit {_ENV_PATH} and set the real value.",
            file=sys.stderr,
        )
        raise SystemExit(1)


_load_dotenv_into_os_environ()

# --- constants -------------------------------------------------------------
MODEL_ID_PRIMARY = "gpt-5.4"
MODEL_FALLBACK_PREFIX = "gpt-5.4"  # used when listing snapshots

CLIENT_TIMEOUT_SEC = 30.0
CLIENT_SDK_RETRIES = 0
SLOW_CALL_WARN_SEC = 10.0

# Disagreement-flag thresholds (locked to the Stage-0 plan).
GEN_CONF_HIGH = 0.9
JUDGE_CONF_LOW = 0.6

MAX_RETRIES_DEFAULT = 3

# The judge prompt is minimal by design. It must not include the
# generator's rationale, cues, topic, or confidence. Passing any of
# those would prime the judge toward the generator's labeling.
JUDGE_PROMPT_TEMPLATE = """You are a careful factual verifier. You are shown ONE English sentence. Decide whether the sentence is factually FALSE according to mainstream, widely-accepted reference knowledge (encyclopedias, peer-reviewed sources, standard textbooks).

Rules:
- "judged_false" = true iff the sentence, taken as a whole, asserts something that contradicts mainstream factual consensus.
- "judged_false" = false iff the sentence is factually correct, or is so vague / opinion-based that it cannot be clearly labeled false.
- "judge_confidence" is your confidence in this judgment (0 = no confidence, 1 = certain).
- "judge_rationale" is a single concise sentence giving the specific factual reason. Do NOT mention this prompt, the schema, or any meta-commentary.

Sentence: {STATEMENT}

Return JSON only, matching this schema exactly:
{{
  "judged_false": <true or false>,
  "judge_confidence": <float between 0 and 1>,
  "judge_rationale": "<one concise sentence>"
}}"""


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_to_json(text: str) -> str | None:
    m = _JSON_OBJECT_RE.search(text or "")
    return m.group(0) if m else None


def _extract_text(resp) -> str:
    """Pull the assistant text out of a Responses-API response.

    Prefers the `output_text` convenience attribute (openai 2.x). Falls
    back to walking `response.output[*].content[*].text` so the script
    is resilient across minor SDK layout changes.
    """
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


def _resolve_judge_model(client, primary: str) -> str:
    """Return a model id that the SDK/back-end recognizes.

    Preference order:
      1. `primary` (the alias requested in the paper plan)
      2. most recent `gpt-5.4*` snapshot from client.models.list()

    The probe uses models.retrieve(), which is a lightweight existence
    check, so the main judging loop can pin a known-good id and avoid
    per-call probing.
    """
    import openai

    try:
        client.models.retrieve(primary)
        return primary
    except openai.NotFoundError:
        print(
            f"NOTE: requested model {primary!r} not found; searching "
            f"for most recent {MODEL_FALLBACK_PREFIX}* snapshot."
        )
    except openai.APIStatusError as e:
        # Some accounts don't have retrieve permission; try optimistic.
        print(
            f"WARNING: models.retrieve({primary!r}) returned "
            f"APIStatusError (status={getattr(e, 'status_code', '?')}); "
            f"will attempt the actual judge call optimistically."
        )
        return primary
    except Exception as e:
        print(
            f"WARNING: models.retrieve({primary!r}) raised "
            f"{type(e).__name__}: {e}. Falling through to list()."
        )

    try:
        listing = client.models.list()
    except Exception as e:
        raise RuntimeError(
            f"Could not list models after {primary!r} was not found: "
            f"{type(e).__name__}: {e}"
        )

    candidates = [m.id for m in listing.data
                  if m.id.startswith(MODEL_FALLBACK_PREFIX)]
    if not candidates:
        raise RuntimeError(
            f"Requested model {primary!r} not found and no "
            f"{MODEL_FALLBACK_PREFIX}* snapshots are available for this key."
        )
    candidates.sort(reverse=True)  # lexicographic; ISO-date suffixes sort well
    return candidates[0]


# --- Responses-API call with SDK-shape ladder ------------------------------
# openai 2.32.0 accepts the user-specified pattern:
#     client.responses.create(model=..., input=...,
#                             response_format={"type": "json_object"},
#                             temperature=0.0)
# but GPT-5.4 (like Claude Opus 4.7) may have deprecated `temperature`.
# The ladder below probes once on the first call, caches whatever
# combination the API accepts, and reuses it for every subsequent call.
_CALL_KWARGS_RESOLVED: dict | None = None
_CALL_KWARGS_LABEL: str | None = None


def _kwarg_candidates() -> list[tuple[dict, str]]:
    return [
        (
            {"response_format": {"type": "json_object"}, "temperature": 0.0},
            "response_format + temperature=0",
        ),
        (
            {"response_format": {"type": "json_object"}},
            "response_format only (no temperature)",
        ),
        (
            {"text": {"format": {"type": "json_object"}}, "temperature": 0.0},
            "text.format + temperature=0",
        ),
        (
            {"text": {"format": {"type": "json_object"}}},
            "text.format only (no temperature)",
        ),
        (
            {"temperature": 0.0},
            "temperature=0 only (no response_format)",
        ),
        (
            {},
            "plain (no response_format, no temperature)",
        ),
    ]


def _is_kwarg_rejection(err: Exception) -> bool:
    """Shaped like the effort-kwarg probe in generate_stage0.py: only
    treat SDK-side TypeErrors and API 400s mentioning response_format /
    text / temperature as kwarg-shape rejections. Network errors, auth
    errors, rate limits, etc. must propagate normally."""
    from openai import APIStatusError, BadRequestError
    msg = str(err).lower()
    if isinstance(err, TypeError):
        return (
            "unexpected keyword argument" in msg
            or "response_format" in msg
            or "text" in msg
            or "temperature" in msg
        )
    if isinstance(err, (BadRequestError, APIStatusError)):
        if getattr(err, "status_code", None) not in (400, None):
            return False
        return (
            "response_format" in msg
            or "text.format" in msg
            or "temperature" in msg
            or "unsupported" in msg
            or "not supported" in msg
        )
    return False


def _call_judge(
    client, model_id: str, statement: str, slot_id: int
) -> tuple[str, str, float]:
    """Single Responses-API judge call. Returns (raw_text, resolved, elapsed)."""
    global _CALL_KWARGS_RESOLVED, _CALL_KWARGS_LABEL
    from openai import (
        APITimeoutError, RateLimitError, APIStatusError, BadRequestError,
    )

    prompt = JUDGE_PROMPT_TEMPLATE.format(STATEMENT=statement)
    print(
        f"  [{_now_stamp()}] Calling GPT judge for slot {slot_id} "
        f"(model={model_id}) ..."
    )

    if _CALL_KWARGS_RESOLVED is not None:
        attempts = [(_CALL_KWARGS_RESOLVED, f"cached {_CALL_KWARGS_LABEL}")]
    else:
        attempts = _kwarg_candidates()

    last_err: Exception | None = None
    resp = None
    used_kwargs: dict = {}
    used_label: str = ""
    t0 = time.perf_counter()

    for kwargs, label in attempts:
        try:
            resp = client.responses.create(
                model=model_id,
                input=prompt,
                timeout=CLIENT_TIMEOUT_SEC,
                **kwargs,
            )
            used_kwargs, used_label = kwargs, label
            break
        except (APITimeoutError, RateLimitError) as e:
            elapsed = time.perf_counter() - t0
            print(f"  [{_now_stamp()}] Call FAILED in {elapsed:.2f}s — "
                  f"{type(e).__name__}: {e}")
            raise
        except (TypeError, BadRequestError, APIStatusError) as e:
            if _is_kwarg_rejection(e) and _CALL_KWARGS_RESOLVED is None:
                last_err = e
                print(f"  [{_now_stamp()}] probe: SDK/API rejected "
                      f"{label} ({type(e).__name__}: {str(e)[:160]}); "
                      f"trying next candidate.")
                continue
            elapsed = time.perf_counter() - t0
            print(f"  [{_now_stamp()}] Call FAILED in {elapsed:.2f}s — "
                  f"{type(e).__name__} "
                  f"(status={getattr(e, 'status_code', '?')}): {e}")
            raise
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  [{_now_stamp()}] Call FAILED in {elapsed:.2f}s — "
                  f"{type(e).__name__}: {e}")
            raise

    if resp is None:
        raise RuntimeError(
            f"Every Responses-API kwarg candidate rejected. Last error: "
            f"{type(last_err).__name__ if last_err else 'None'}: {last_err}"
        )

    elapsed = time.perf_counter() - t0

    if _CALL_KWARGS_RESOLVED is None:
        _CALL_KWARGS_RESOLVED = used_kwargs
        _CALL_KWARGS_LABEL = used_label
        print(f"  [{_now_stamp()}] Responses kwargs resolved to: "
              f"{used_label}")

    print(f"  [{_now_stamp()}] Call returned in {elapsed:.2f}s "
          f"using {used_label}")
    if elapsed > SLOW_CALL_WARN_SEC:
        print(f"  [{_now_stamp()}] WARNING: slow call — {elapsed:.2f}s "
              f"exceeds {SLOW_CALL_WARN_SEC:.0f}s.")

    resolved = getattr(resp, "model", model_id) or model_id
    return _extract_text(resp), resolved, elapsed


def _judge_one(
    client,
    model_id: str,
    gen_rec: dict,
    *,
    first_call_logged: dict,
    max_retries: int,
) -> dict:
    last_err: str | None = None
    last_raw: str = ""
    resolved_model = model_id

    for attempt in range(1, max_retries + 1):
        try:
            raw, resolved_model, elapsed = _call_judge(
                client, model_id, gen_rec["statement"], gen_rec["id"]
            )
        except Exception as e:
            last_err = f"API error (attempt {attempt}): {type(e).__name__}: {e}"
            print(f"    {last_err}")
            if max_retries == 1:
                raise
            time.sleep(1.5 * attempt)
            continue

        if not first_call_logged["done"]:
            print(f"    [first-call] resolved judge model id = "
                  f"{resolved_model!r}")
            first_call_logged["done"] = True

        last_raw = raw
        blob = _strip_to_json(raw)
        if blob is None:
            last_err = f"no JSON object in response (attempt {attempt})"
            print(f"    {last_err}; raw head={raw[:160]!r}")
            time.sleep(0.5)
            continue

        try:
            data = json.loads(blob)
        except Exception as e:
            last_err = f"JSON parse failed (attempt {attempt}): {e}"
            print(f"    {last_err}; blob head={blob[:160]!r}")
            time.sleep(0.5)
            continue

        required = {"judged_false", "judge_confidence", "judge_rationale"}
        missing = required - data.keys()
        if missing:
            last_err = f"missing keys {sorted(missing)} (attempt {attempt})"
            print(f"    {last_err}")
            continue

        try:
            judged_false = bool(data["judged_false"])
            judge_confidence = float(data["judge_confidence"])
        except (TypeError, ValueError) as e:
            last_err = f"type coercion failed (attempt {attempt}): {e}"
            print(f"    {last_err}")
            continue

        gen_conf = float(gen_rec.get("generator_confidence_in_falsity", 0.0))
        disagreement = (
            gen_conf > GEN_CONF_HIGH
            and (not judged_false)
            and judge_confidence < JUDGE_CONF_LOW
        )

        return {
            "id": gen_rec["id"],
            "statement": gen_rec["statement"],
            "judged_false": judged_false,
            "judge_confidence": judge_confidence,
            "judge_rationale": str(data["judge_rationale"]),
            "judge_disagreement_flag": disagreement,
            "judge_model": resolved_model,
            "judge_elapsed_sec": round(elapsed, 3),
        }

    raise RuntimeError(
        f"Slot {gen_rec['id']}: judge failed after {max_retries} "
        f"attempt(s). Last error: {last_err}. "
        f"Last raw head: {last_raw[:200]!r}"
    )


def _load_generations() -> list[dict]:
    if not GEN_JSON.exists():
        raise RuntimeError(
            f"{GEN_JSON} does not exist. Run Step 8 (generate_stage0.py) "
            f"first so this file contains all 20 generator records."
        )
    with open(GEN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 20:
        raise RuntimeError(
            f"{GEN_JSON} must contain exactly 20 records; got "
            f"{len(data) if isinstance(data, list) else type(data).__name__}."
        )
    return sorted(data, key=lambda r: r["id"])


def _load_existing_judgments() -> list[dict]:
    if not OUT_JSON.exists():
        return []
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"{OUT_JSON} exists but is not a JSON list.")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Judge Stage-0 statements with GPT-5.4 (Responses API).")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after the resulting JSON contains this many total "
             "records (counts any pre-existing records in the output "
             "file from previous runs). Default: judge all 20.")
    parser.add_argument(
        "--max-retries", type=int, default=MAX_RETRIES_DEFAULT,
        help=f"Per-slot API/JSON retries. Default: {MAX_RETRIES_DEFAULT}. "
             "Pass 1 for strict no-retry mode.")
    args = parser.parse_args()

    print("=" * 72)
    print("STEP 9 — judge_stage0.py")
    print("=" * 72)
    _require_env("OPENAI_API_KEY")

    from openai import OpenAI  # lazy import so script boots fast

    client = OpenAI(
        timeout=CLIENT_TIMEOUT_SEC,
        max_retries=CLIENT_SDK_RETRIES,
    )
    print(
        f"OpenAI client constructed with timeout={CLIENT_TIMEOUT_SEC}s, "
        f"max_retries={CLIENT_SDK_RETRIES} (SDK-level retries disabled)."
    )

    model_id = _resolve_judge_model(client, MODEL_ID_PRIMARY)
    if model_id != MODEL_ID_PRIMARY:
        print(f"Resolved fallback judge model: {model_id!r} "
              f"(requested {MODEL_ID_PRIMARY!r})")
    else:
        print(f"Resolved judge model: {model_id!r}")

    STAGE0_DIR.mkdir(parents=True, exist_ok=True)

    gens = _load_generations()
    existing = _load_existing_judgments()
    done_ids = {rec["id"] for rec in existing}
    results: list[dict] = list(existing)
    if existing:
        print(f"Resume: found {len(existing)} existing judgments in {OUT_JSON}")
    target_total = args.limit if args.limit is not None else len(gens)

    print(f"CLI: limit={args.limit}  max_retries={args.max_retries}")

    first_call_logged = {"done": False}
    try:
        for gen_rec in gens:
            if len(results) >= target_total:
                break
            if gen_rec["id"] in done_ids:
                continue
            print(f"\n[{gen_rec['id']+1:02d}/20] judging slot "
                  f"{gen_rec['id']} (topic={gen_rec.get('topic','?')}) ...")
            rec = _judge_one(
                client, model_id, gen_rec,
                first_call_logged=first_call_logged,
                max_retries=args.max_retries,
            )
            tag = "FALSE" if rec["judged_false"] else "true"
            disagree = " DISAGREE" if rec["judge_disagreement_flag"] else ""
            print(f"    judged={tag}  conf={rec['judge_confidence']:.2f}"
                  f"  elapsed={rec['judge_elapsed_sec']:.2f}s{disagree}")
            results.append(rec)
            done_ids.add(rec["id"])
    finally:
        if results:
            results.sort(key=lambda r: r["id"])
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nWrote {len(results)} total judgments -> {OUT_JSON}")

    if args.limit is None and len(results) != 20:
        print(f"\nERROR: only {len(results)}/20 judgments succeeded.",
              file=sys.stderr)
        return 1

    n_false = sum(1 for r in results if r["judged_false"])
    n_true = sum(1 for r in results if not r["judged_false"])
    n_disagree = sum(1 for r in results if r["judge_disagreement_flag"])
    print(f"\nSummary: judged_false=True: {n_false}  /  "
          f"judged_false=False: {n_true}  /  "
          f"judge_disagreement_flag=True: {n_disagree}")
    print(f"Step 9 progress: {len(results)}/20 judgments on disk.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in judge_stage0.py:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
