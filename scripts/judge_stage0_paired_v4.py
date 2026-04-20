#!/usr/bin/env python3
"""
v4 paired-probe judge: factual-correctness judge over fresh A/B answers.

For each v4 pair we make two independent GPT-5.4 calls:

    Call 1 (A-side, expected TRUE):  question + a_side
    Call 2 (B-side, expected FALSE): question + b_side

Judge prompt: "Is the following statement, as an answer to the question
'<Q>', factually correct? Reply JSON: {correct: bool, confidence: float,
rationale: str}".

A pair "passes" iff A-side.correct=True AND B-side.correct=False (i.e.
the generator hit both intended truth values). All judge bookkeeping
mirrors scripts/rejudge_paired_tqa_truthclass.py: same Responses-API
kwarg-ladder probe, same model resolver, same .env loader, model pinned
to gpt-5.4-2026-03-05.
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
DEFAULT_GEN_JSON = (REPO_ROOT / "stage0_paired_tqa"
                    / "stage0_paired_generations_v4.json")
DEFAULT_OUT_JSON = (REPO_ROOT / "stage0_paired_tqa"
                    / "stage0_paired_judge_v4.json")
# Mutable module-globals: the multi-seed orchestrator overrides these via
# CLI flags so a single judge implementation serves the original v4 file
# AND the per-seed generations. Default behaviour is unchanged.
GEN_JSON: Path = DEFAULT_GEN_JSON
OUT_JSON: Path = DEFAULT_OUT_JSON

JUDGE_CRITERION_TAG = "factual_correctness_v4"

# --- workspace-local .env loader ------------------------------------------
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

JUDGE_PROMPT_TEMPLATE = """Is the following statement, as an answer to the question {QUESTION}, factually correct? Use widely-accepted real-world knowledge. Be strict: if the statement asserts something that is false in the actual world, mark correct=false even if the wording is hedged.

Statement: {STATEMENT}

Reply JSON ONLY, matching this schema exactly:
{{
  "correct": <true or false>,
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


# --- model resolution -----------------------------------------------------
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


# --- Responses-API kwargs ladder (cached across calls) -------------------
_CALL_KWARGS_RESOLVED: dict | None = None
_CALL_KWARGS_LABEL: str | None = None


def _kwarg_candidates() -> list[tuple[dict, str]]:
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


def _judge_one(
    client, model_id: str, question: str, statement: str,
    side_label: str, pair_id: int, max_retries: int,
) -> tuple[dict, str]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        QUESTION=json.dumps(question), STATEMENT=json.dumps(statement),
    )
    last_err: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw, resolved, elapsed = _call_responses(client, model_id, prompt)
        except Exception as e:
            last_err = (f"API error {side_label} attempt {attempt}: "
                        f"{type(e).__name__}: {e}")
            print(f"  pid={pair_id} {side_label}: {last_err}")
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

        required = {"correct", "confidence", "rationale"}
        missing = required - data.keys()
        if missing:
            last_err = f"missing keys {sorted(missing)} (attempt {attempt})"
            print(f"  pid={pair_id} {side_label}: {last_err}")
            continue
        try:
            correct = bool(data["correct"])
            conf = float(data["confidence"])
        except Exception as e:
            last_err = f"type coercion failed (attempt {attempt}): {e}"
            print(f"  pid={pair_id} {side_label}: {last_err}")
            continue

        return ({
            "correct": correct,
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
        raise RuntimeError(f"{GEN_JSON} does not exist; run generator first.")
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
    global GEN_JSON, OUT_JSON
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    p.add_argument("--gen-json", type=str, default=None,
                   help="Override input generations JSON path "
                        "(default: stage0_paired_generations_v4.json).")
    p.add_argument("--out-json", type=str, default=None,
                   help="Override output judge JSON path "
                        "(default: stage0_paired_judge_v4.json).")
    args = p.parse_args()
    if args.gen_json:
        GEN_JSON = Path(args.gen_json).resolve()
    if args.out_json:
        OUT_JSON = Path(args.out_json).resolve()

    print("=" * 72)
    print("judge_stage0_paired_v4.py "
          f"(criterion={JUDGE_CRITERION_TAG})")
    print("=" * 72)
    _require_env("OPENAI_API_KEY")

    from openai import OpenAI
    client = OpenAI(timeout=CLIENT_TIMEOUT_SEC, max_retries=CLIENT_SDK_RETRIES)
    print(f"OpenAI client: timeout={CLIENT_TIMEOUT_SEC}s, "
          f"max_retries={CLIENT_SDK_RETRIES}")

    model_id = _resolve_judge_model(client, MODEL_ID_PRIMARY)
    if model_id != MODEL_PINNED_TARGET:
        print(f"NOTE: resolver returned {model_id!r}; pin target was "
              f"{MODEL_PINNED_TARGET!r}.")
    else:
        print(f"Resolved judge model: {model_id!r} (matches pin)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    gens = _load_generations()
    existing = _load_existing()
    done_ids = {int(r["pair_id"]) for r in existing}
    results: list[dict] = list(existing)
    if existing:
        print(f"Resume: {len(existing)} existing pair_ids in {OUT_JSON}")

    target_total = args.limit if args.limit is not None else len(gens)
    print(f"Will judge up to {target_total} of {len(gens)} pairs")

    actual_models: set[str] = set()
    n_done_now = 0
    try:
        for gen_rec in gens:
            if len(results) >= target_total:
                break
            pid = int(gen_rec["pair_id"])
            if pid in done_ids:
                continue
            print(f"\n[pid={pid}] judging A-side and B-side ...")

            q = str(gen_rec["question"])
            a_text = str(gen_rec["a_side"])
            b_text = str(gen_rec["b_side"])

            a_rec, m1 = _judge_one(client, model_id, q, a_text,
                                   "A", pid, args.max_retries)
            actual_models.add(m1)
            print(f"  A: correct={a_rec['correct']}  "
                  f"conf={a_rec['confidence']:.2f}  "
                  f"elapsed={a_rec['elapsed_sec']:.2f}s")

            b_rec, m2 = _judge_one(client, model_id, q, b_text,
                                   "B", pid, args.max_retries)
            actual_models.add(m2)
            print(f"  B: correct={b_rec['correct']}  "
                  f"conf={b_rec['confidence']:.2f}  "
                  f"elapsed={b_rec['elapsed_sec']:.2f}s")

            pair_passes = (a_rec["correct"] is True
                           and b_rec["correct"] is False)
            print(f"  pair_passes (A=True AND B=False) = {pair_passes}")

            results.append({
                "pair_id": pid,
                "question": q,
                "a_side": a_text,
                "a_correct":         a_rec["correct"],
                "a_confidence":      a_rec["confidence"],
                "a_rationale":       a_rec["rationale"],
                "a_elapsed_sec":     a_rec["elapsed_sec"],
                "b_side": b_text,
                "b_correct":         b_rec["correct"],
                "b_confidence":      b_rec["confidence"],
                "b_rationale":       b_rec["rationale"],
                "b_elapsed_sec":     b_rec["elapsed_sec"],
                "pair_passes":       pair_passes,
                "judge_model":       m1 if m1 == m2 else f"{m1}|{m2}",
                "judge_criterion":   JUDGE_CRITERION_TAG,
            })
            done_ids.add(pid)
            n_done_now += 1
    finally:
        if results:
            results.sort(key=lambda r: int(r["pair_id"]))
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nWrote {len(results)} v4 judgments -> {OUT_JSON}")

    n = len(results)
    n_a = sum(1 for r in results if r["a_correct"])
    n_b_false = sum(1 for r in results if r["b_correct"] is False)
    n_pair = sum(1 for r in results if r["pair_passes"])
    print("\nSummary (v4 factual-correctness criterion):")
    print(f"  pairs judged this run:                {n_done_now}")
    print(f"  pairs in output file:                 {n}")
    print(f"  A-side correct=True:                  {n_a}/{n}")
    print(f"  B-side correct=False:                 {n_b_false}/{n}")
    print(f"  pair_passes (A=True AND B=False):     {n_pair}/{n}")
    print(f"  resolved judge model(s):              {sorted(actual_models)}")
    print("\n  Pairs that failed pair_passes:")
    fail = [r for r in results if not r["pair_passes"]]
    if not fail:
        print("    (none)")
    for r in fail:
        why = []
        if not r["a_correct"]:
            why.append(f"A-not-correct (conf={r['a_confidence']:.2f})")
        if r["b_correct"]:
            why.append(f"B-not-false (conf={r['b_confidence']:.2f})")
        print(f"    pid={r['pair_id']:>3d}: {'; '.join(why)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in judge_stage0_paired_v4.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
