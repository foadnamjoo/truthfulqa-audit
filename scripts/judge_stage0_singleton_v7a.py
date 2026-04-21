#!/usr/bin/env python3
"""
v7(a) A-side singleton judge.

Mirrors scripts/judge_stage0.py (the §5.1 judge) prompt verbatim so the
factual-correctness signal is comparable across §5.1 and v7(a). Each
statement is judged in isolation - the generator's rationale,
confidence, topic, etc. are NOT sent to the judge. A v7(a) record is
considered to "pass" the judge when judged_false == False (i.e. the
judge agrees the statement is factually true).

Model: gpt-5.4, pinned to gpt-5.4-2026-03-05 via the same resolver used
in the paired judges.
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

DEFAULT_GEN_JSON = REPO_ROOT / "stage0_v7a_bilateral" / "stage0_singleton_v7a_generations.json"
DEFAULT_OUT_JSON = REPO_ROOT / "stage0_v7a_bilateral" / "stage0_singleton_v7a_judge.json"
GEN_JSON: Path = DEFAULT_GEN_JSON
OUT_JSON: Path = DEFAULT_OUT_JSON
JUDGE_CRITERION_TAG = "factual_correctness_v7a"

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
        print(f"ERROR: {name} is still placeholder.", file=sys.stderr)
        raise SystemExit(1)


_load_dotenv_into_os_environ()

MODEL_ID_PRIMARY = "gpt-5.4"
MODEL_FALLBACK_PREFIX = "gpt-5.4"
MODEL_PINNED_TARGET = "gpt-5.4-2026-03-05"
CLIENT_TIMEOUT_SEC = 30.0
CLIENT_SDK_RETRIES = 0
SLOW_CALL_WARN_SEC = 10.0
MAX_RETRIES_DEFAULT = 3

# Mirrors scripts/judge_stage0.py verbatim so results are comparable.
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
    import openai
    try:
        client.models.retrieve(primary)
        return primary
    except openai.NotFoundError:
        pass
    except Exception:
        return primary
    listing = client.models.list()
    cands = [m.id for m in listing.data
             if m.id.startswith(MODEL_FALLBACK_PREFIX)]
    if not cands:
        raise RuntimeError(f"No {MODEL_FALLBACK_PREFIX}* models available")
    cands.sort(reverse=True)
    return cands[0]


_CALL_KWARGS_RESOLVED: dict | None = None
_CALL_KWARGS_LABEL: str | None = None


def _kwarg_candidates() -> list[tuple[dict, str]]:
    return [
        ({"text": {"format": {"type": "json_object"}}, "temperature": 0.0},
         "text.format + temperature=0"),
        ({"text": {"format": {"type": "json_object"}}},
         "text.format only"),
        ({"response_format": {"type": "json_object"}, "temperature": 0.0},
         "response_format + temperature=0"),
        ({"response_format": {"type": "json_object"}},
         "response_format only"),
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

    attempts = ([(_CALL_KWARGS_RESOLVED, f"cached {_CALL_KWARGS_LABEL}")]
                if _CALL_KWARGS_RESOLVED is not None
                else _kwarg_candidates())
    resp = None
    used_kwargs, used_label = {}, ""
    last_err: Exception | None = None
    t0 = time.perf_counter()
    for kwargs, label in attempts:
        kwargs = kwargs or {}
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
                continue
            raise
    if resp is None:
        raise RuntimeError(f"All Responses kwargs rejected: {last_err}")
    elapsed = time.perf_counter() - t0
    if _CALL_KWARGS_RESOLVED is None:
        _CALL_KWARGS_RESOLVED = used_kwargs
        _CALL_KWARGS_LABEL = used_label
        print(f"  [{_now_stamp()}] Responses kwargs resolved: {used_label}")
    if elapsed > SLOW_CALL_WARN_SEC:
        print(f"  [{_now_stamp()}] WARNING slow call: {elapsed:.2f}s")
    resolved = getattr(resp, "model", model_id) or model_id
    return _extract_text(resp), resolved, elapsed


def _judge_one(client, model_id: str, statement: str, slot_id: int,
               max_retries: int) -> tuple[dict, str]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(STATEMENT=json.dumps(statement))
    last_err: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw, resolved, elapsed = _call_responses(client, model_id, prompt)
        except Exception as e:
            last_err = (f"API error id={slot_id} attempt {attempt}: "
                        f"{type(e).__name__}: {e}")
            time.sleep(1.2 * attempt)
            continue
        blob = _strip_to_json(raw)
        if blob is None:
            last_err = f"no JSON object attempt {attempt}"
            continue
        try:
            d = json.loads(blob)
            judged_false = bool(d["judged_false"])
            conf = float(d["judge_confidence"])
            rationale = str(d["judge_rationale"])
        except Exception as e:
            last_err = f"parse/type failure attempt {attempt}: {e}"
            continue
        return ({
            "judged_false": judged_false,
            "judge_confidence": conf,
            "judge_rationale": rationale,
            "elapsed_sec": round(float(elapsed), 3),
        }, resolved)
    raise RuntimeError(
        f"id={slot_id} judge failed after {max_retries} attempts: {last_err}"
    )


def _load_generations() -> list[dict]:
    if not GEN_JSON.exists():
        raise RuntimeError(f"Missing {GEN_JSON}")
    with open(GEN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data, key=lambda r: int(r["id"]))


def _load_existing() -> list[dict]:
    if not OUT_JSON.exists():
        return []
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else []


def main() -> int:
    global GEN_JSON, OUT_JSON
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    p.add_argument("--gen-json", type=str, default=None)
    p.add_argument("--out-json", type=str, default=None)
    args = p.parse_args()
    if args.gen_json:
        GEN_JSON = Path(args.gen_json).resolve()
    if args.out_json:
        OUT_JSON = Path(args.out_json).resolve()

    print("=" * 72)
    print("judge_stage0_singleton_v7a.py")
    print("=" * 72)
    _require_env("OPENAI_API_KEY")

    from openai import OpenAI
    client = OpenAI(timeout=CLIENT_TIMEOUT_SEC,
                    max_retries=CLIENT_SDK_RETRIES)
    model_id = _resolve_judge_model(client, MODEL_ID_PRIMARY)
    print(f"Resolved model: {model_id!r} "
          f"(pin target={MODEL_PINNED_TARGET!r})")

    gens = _load_generations()
    existing = _load_existing()
    done_ids = {int(r["id"]) for r in existing}
    results = list(existing)
    target_total = args.limit if args.limit is not None else len(gens)
    actual_models: set[str] = set()
    n_done_now = 0

    for g in gens:
        if len(results) >= target_total:
            break
        sid = int(g["id"])
        if sid in done_ids:
            continue
        statement = str(g["statement"])
        topic = str(g.get("topic", "?"))
        print(f"\n[id={sid} topic={topic}] judging ...")
        rec, m = _judge_one(client, model_id, statement, sid, args.max_retries)
        actual_models.add(m)
        judge_agrees_true = (rec["judged_false"] is False)
        print(f"  judged_false={rec['judged_false']} "
              f"conf={rec['judge_confidence']:.2f} "
              f"agrees_true={judge_agrees_true}")
        results.append({
            "id": sid,
            "topic": topic,
            "statement": statement,
            "judged_false": rec["judged_false"],
            "judge_agrees_true": judge_agrees_true,
            "judge_confidence": rec["judge_confidence"],
            "judge_rationale": rec["judge_rationale"],
            "judge_elapsed_sec": rec["elapsed_sec"],
            "judge_model": m,
            "judge_criterion": JUDGE_CRITERION_TAG,
        })
        done_ids.add(sid)
        n_done_now += 1
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(sorted(results, key=lambda r: int(r["id"])),
                      f, ensure_ascii=False, indent=2)

    results = sorted(results, key=lambda r: int(r["id"]))
    n = len(results)
    n_agree = sum(1 for r in results if r["judge_agrees_true"])
    print("\nSummary:")
    print(f"  judged this run: {n_done_now}")
    print(f"  total in output: {n}")
    print(f"  judge agrees (true): {n_agree}/{n}")
    print(f"  resolved models: {sorted(actual_models)}")

    topic_counts: dict[str, list[int]] = {}
    for r in results:
        t = str(r.get("topic", ""))
        if t not in topic_counts:
            topic_counts[t] = [0, 0]
        topic_counts[t][1] += 1
        if r["judge_agrees_true"]:
            topic_counts[t][0] += 1
    print("\nPer-topic judge_agrees_true:")
    for t in sorted(topic_counts.keys()):
        ok, total = topic_counts[t]
        print(f"  {t}: {ok}/{total}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in judge_stage0_singleton_v7a.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
