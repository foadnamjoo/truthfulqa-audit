#!/usr/bin/env python3
"""
v4 multi-seed generator (robustness check).

Per seed S we sample N pair_ids from the same density-filtered confounded
pool that the original v4 used (seed-42 single-shot run), then run the
v4 fresh-answer Opus generator with the same prompt + validator.

Pool reproduction (verified to match v4's selected_pair_ids.json):

  confounded = (neg_lead_true>0) | (neg_lead_false>0)
               | (|len_gap| > 0.20)
               | (|hedge_rate_true - hedge_rate_false| > 0.02)
               | (|auth_rate_true - auth_rate_false| > 0.01)
  density    = (word_count_true >= 5)
               & (word_count_false >= 5)
               & not (Best Answer            startswith "i have no comment")
               & not (Best Incorrect Answer  startswith "i have no comment")
  pool       = audit_df[confounded & density].pair_id    -> 433 entries

Sampler:

  rng = np.random.default_rng(seed)
  pair_ids = sorted(rng.choice(pool, size=N, replace=False).tolist())

Special case S=42: the original v4 sampler used a complex 2-stage
process (initial 20 -> density rejection -> resample -> manual safety
substitution 597 -> 474). It is not identical to the clean rng.choice
sampler used here. To honour the spec's "Reuse those 20 generations"
requirement, the seed=42 sampler ANCHORS its first 20 pair_ids to the
existing v4 list verbatim, then samples (N - 20) additional pair_ids
from (pool minus existing 20) using rng = default_rng(42). This is
documented in the per-seed selected_pair_ids JSON as
`anchor_strategy: "v4_existing_20_then_rng_choice"`.

Outputs (per seed):
  stage0_paired_tqa/v4_multi_seed/selected_pair_ids_v4_seed{S}.json
  stage0_paired_tqa/v4_multi_seed/generations_seed{S}_n{N}.json

The generator reuses scripts/generate_stage0_paired_v4.PROMPT_TEMPLATE,
its Anthropic call ladder, and scripts/validator_v4.validate_v4 to
keep behaviour identical to the seed-42 single-shot v4 run. Resume:
re-running the script skips pair_ids already present in the per-seed
generations JSON.
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

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validator_v4 import validate_v4  # noqa: E402
from scripts.generate_stage0_paired_v4 import (  # noqa: E402
    PROMPT_TEMPLATE, OPENERS_V4, HEDGE_TOKENS_V4, AUTH_CUE_PHRASES_V4,
    CONTRAST_CONJUNCTIONS_V4, MODEL_ID_PRIMARY, SLOW_CALL_WARN_SEC,
    _build_client, _call_opus, _observe_b_strategy,
    _parse_response_to_dict, _load_dotenv_into_os_environ, _require_env,
)

AUDIT_CSV = REPO_ROOT / "audits" / "truthfulqa_style_audit.csv"
TRUTHFULQA_CSV = REPO_ROOT / "TruthfulQA.csv"
V4_EXISTING_GEN = (REPO_ROOT / "stage0_paired_tqa"
                   / "stage0_paired_generations_v4.json")

OUT_ROOT = REPO_ROOT / "stage0_paired_tqa" / "v4_multi_seed"

# Known Anthropic safety refusal from the original v4 sampling pass. This
# pair_id (597, "I hit my partner every day") was refused 6/6 times in
# stage0_paired_tqa/selected_pair_ids.json under
# safety_refusal_substitutions; we exclude it from every multi-seed
# sample and document the exclusion in the per-seed provenance JSON.
KNOWN_REFUSAL_BLOCKLIST: set[int] = {597}


# --- pool reproduction ----------------------------------------------------
def build_pool() -> list[int]:
    df = pd.read_csv(AUDIT_CSV).reset_index().rename(
        columns={"index": "pair_id"})
    neg = (df["neg_lead_true"] > 0) | (df["neg_lead_false"] > 0)
    lg  = df["len_gap"].abs() > 0.20
    hd  = (df["hedge_rate_true"] - df["hedge_rate_false"]).abs() > 0.02
    au  = (df["auth_rate_true"] - df["auth_rate_false"]).abs() > 0.01
    confounded = neg | lg | hd | au
    no_comment_a = df["Best Answer"].astype(str).str.lower() \
        .str.startswith("i have no comment")
    no_comment_b = df["Best Incorrect Answer"].astype(str).str.lower() \
        .str.startswith("i have no comment")
    density = ((df["word_count_true"] >= 5)
               & (df["word_count_false"] >= 5)
               & ~no_comment_a & ~no_comment_b)
    pool = sorted(df.loc[confounded & density, "pair_id"].tolist())
    return [int(x) for x in pool]


# --- sampler --------------------------------------------------------------
def _load_v4_existing_pair_ids() -> list[int]:
    if not V4_EXISTING_GEN.exists():
        raise FileNotFoundError(f"Missing {V4_EXISTING_GEN}")
    with open(V4_EXISTING_GEN, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(int(r["pair_id"]) for r in data)


def sample_seed(seed: int, n: int, pool: list[int],
                anchor_v4_existing: bool = True) -> tuple[list[int], dict]:
    """Return (sorted_pair_ids, sampling_provenance_dict).

    Pre-emptively excludes KNOWN_REFUSAL_BLOCKLIST from the sampling
    pool. Records both the original size and the refusal-filtered size
    in provenance.
    """
    pool_clean = [pid for pid in pool if pid not in KNOWN_REFUSAL_BLOCKLIST]
    excluded = sorted(set(pool) - set(pool_clean))

    if seed == 42 and anchor_v4_existing:
        anchor = _load_v4_existing_pair_ids()
        for pid in anchor:
            if pid in KNOWN_REFUSAL_BLOCKLIST:
                raise RuntimeError(
                    f"v4 anchor pair_id {pid} is in KNOWN_REFUSAL_BLOCKLIST"
                )
        if not all(pid in pool_clean for pid in anchor):
            missing = [pid for pid in anchor if pid not in pool_clean]
            raise RuntimeError(
                f"v4 existing pair_ids not in current pool: {missing}")
        if n < len(anchor):
            raise ValueError(f"n={n} smaller than v4 anchor len={len(anchor)}")
        remaining_pool = [pid for pid in pool_clean if pid not in anchor]
        rng = np.random.default_rng(42)
        extra_n = n - len(anchor)
        if extra_n > 0:
            extra = sorted(rng.choice(remaining_pool, size=extra_n,
                                      replace=False).tolist())
        else:
            extra = []
        merged = sorted(set(anchor) | set(int(x) for x in extra))
        prov = {
            "seed": 42,
            "n": n,
            "pool_size_raw": len(pool),
            "pool_size_after_blocklist": len(pool_clean),
            "blocklisted_pair_ids": excluded,
            "anchor_strategy": "v4_existing_20_then_rng_choice",
            "v4_existing_pair_ids": anchor,
            "extra_pair_ids": [int(x) for x in extra],
            "selected_pair_ids": merged,
        }
        return merged, prov

    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(pool_clean, size=n, replace=False).tolist())
    chosen = [int(x) for x in chosen]
    prov = {
        "seed": seed,
        "n": n,
        "pool_size_raw": len(pool),
        "pool_size_after_blocklist": len(pool_clean),
        "blocklisted_pair_ids": excluded,
        "anchor_strategy": "rng_choice_size_n_replace_false",
        "selected_pair_ids": chosen,
    }
    return chosen, prov


# --- per-seed paths --------------------------------------------------------
def _paths(seed: int, n: int) -> tuple[Path, Path]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sel = OUT_ROOT / f"selected_pair_ids_v4_seed{seed}.json"
    gen = OUT_ROOT / f"generations_seed{seed}_n{n}.json"
    return sel, gen


# --- generation helpers (mirrors generate_stage0_paired_v4.main loop) -----
def _load_question(pair_id: int) -> str:
    df = pd.read_csv(TRUTHFULQA_CSV)
    return str(df.iloc[pair_id]["Question"]).strip()


def _load_existing_generations(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}
    return {int(r["pair_id"]): r for r in data}


def _save_generations(path: Path, records: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sorted(records.values(), key=lambda r: int(r["pair_id"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def _save_provenance(path: Path, prov: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)


def _seed_42_reuse_v4(target_records: dict[int, dict]) -> int:
    """Copy existing v4 generations into the seed=42 file. Returns count."""
    if not V4_EXISTING_GEN.exists():
        return 0
    with open(V4_EXISTING_GEN, "r", encoding="utf-8") as f:
        v4 = json.load(f)
    n = 0
    for r in v4:
        pid = int(r["pair_id"])
        if pid in target_records:
            continue
        # Carry the original record verbatim, but tag its origin so the
        # provenance is unambiguous when these generations are mixed
        # with newly-sampled extras.
        rec = dict(r)
        rec["source"] = "reused_from_stage0_paired_generations_v4"
        target_records[pid] = rec
        n += 1
    return n


def _generate_one(client, effort_state: dict, pid: int, max_retries: int
                  ) -> tuple[dict | None, str | None]:
    question = _load_question(pid)
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        question_repr=json.dumps(question),
        pair_id=pid,
        openers_list=", ".join(repr(op) for op in OPENERS_V4),
        hedge_list=", ".join(repr(h) for h in HEDGE_TOKENS_V4),
        auth_phrases=", ".join(repr(a) for a in AUTH_CUE_PHRASES_V4[:5]),
        contrast_conj=", ".join(repr(c) for c in CONTRAST_CONJUNCTIONS_V4),
    )
    last_err: str | None = None
    for attempt in range(max_retries + 1):
        try:
            raw, elapsed = _call_opus(client, prompt, effort_state)
        except Exception as e:
            last_err = (f"API error attempt {attempt}: "
                        f"{type(e).__name__}: {str(e)[:160]}")
            print(f"   {last_err}")
            continue
        tag = "  WARN slow" if elapsed > SLOW_CALL_WARN_SEC else ""
        print(f"   attempt {attempt}: {elapsed:.1f}s{tag}")

        try:
            cand = _parse_response_to_dict(raw)
        except Exception as e:
            last_err = f"JSON parse fail: {e}"
            print(f"   {last_err}")
            continue

        ok, why, soft = validate_v4(cand)
        if not ok:
            last_err = f"validator: {why}"
            print(f"   {last_err}")
            continue
        for w in soft:
            print(f"   WARN soft: {w}")

        a_text = str(cand["a_side"]).strip()
        b_text = str(cand["b_side"]).strip()
        b_strategy = _observe_b_strategy(b_text)

        rec = {
            "pair_id": int(pid),
            "question": question,
            "a_side": a_text,
            "a_side_truth_rationale":
                str(cand["a_side_truth_rationale"]).strip(),
            "b_side": b_text,
            "b_side_truth_rationale":
                str(cand["b_side_truth_rationale"]).strip(),
            "b_side_strategy_observed": b_strategy,
            "cues_in_b": list(cand.get("cues_in_b", [])),
            "generator_confidence": float(cand["generator_confidence"]),
            "model": MODEL_ID_PRIMARY,
            "effort_kwarg":
                effort_state.get("kwargs", {}).get("effort", "(none)"),
            "retries": attempt,
            "elapsed_sec_last_attempt": float(elapsed),
            "soft_warnings": soft,
            "source": "newly_generated",
        }
        return rec, None
    return None, last_err


# --- main ------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true",
                   help="Print sampled pair_ids and provenance, but do not "
                        "call the Anthropic API or write generations JSON.")
    p.add_argument("--plan-only", action="store_true",
                   help="Like --dry-run but also writes the per-seed "
                        "selected_pair_ids JSON for inspection.")
    args = p.parse_args()

    print("=" * 72)
    print(f"generate_stage0_paired_v4_multi.py  seed={args.seed}  "
          f"n={args.n}")
    print("=" * 72)

    pool = build_pool()
    print(f"Confounded+density pool size: {len(pool)}")

    pids, prov = sample_seed(args.seed, args.n, pool)
    print(f"Sampled {len(pids)} pair_ids: {pids}")

    sel_path, gen_path = _paths(args.seed, args.n)

    if args.dry_run or args.plan_only:
        print("\n(dry-run) provenance:")
        for k, v in prov.items():
            if isinstance(v, list) and len(v) > 8:
                print(f"  {k}: [{len(v)} items] head={v[:8]}")
            else:
                print(f"  {k}: {v}")
        if args.plan_only:
            _save_provenance(sel_path, prov)
            print(f"\nWrote provenance: {sel_path.relative_to(REPO_ROOT)}")
        return 0

    _save_provenance(sel_path, prov)
    print(f"Wrote provenance: {sel_path.relative_to(REPO_ROOT)}")

    existing = _load_existing_generations(gen_path)
    n_reused_v4 = 0
    if args.seed == 42:
        n_reused_v4 = _seed_42_reuse_v4(existing)
        if n_reused_v4:
            _save_generations(gen_path, existing)
            print(f"Seed=42: reused {n_reused_v4} v4 generations into "
                  f"{gen_path.name}")

    print(f"Existing records on disk: {len(existing)}")
    todo = [pid for pid in pids if pid not in existing]
    print(f"To generate this run: {len(todo)} -> {todo}")
    if not todo:
        print("Nothing to do.")
        return 0

    _load_dotenv_into_os_environ()
    _require_env("ANTHROPIC_API_KEY")
    client = _build_client()
    effort_state: dict = {}

    n_ok, n_fail = 0, 0
    failed: list[tuple[int, str]] = []
    for pid in todo:
        print(f"\n--- pid={pid} ---")
        rec, err = _generate_one(client, effort_state, pid, args.max_retries)
        if rec is None:
            n_fail += 1
            failed.append((pid, err or "unknown"))
            print(f"   FAILED; last_err={err}")
            continue
        existing[pid] = rec
        _save_generations(gen_path, existing)
        n_ok += 1
        print(f"   OK retries={rec['retries']}  "
              f"b_strategy={rec['b_side_strategy_observed']}  "
              f"conf={rec['generator_confidence']:.2f}")

    print("\n" + "=" * 72)
    print(f"Run summary seed={args.seed}: ok={n_ok} fail={n_fail} of "
          f"{len(todo)} attempted; total in file = {len(existing)}/{args.n}")
    if failed:
        print("  Failed pids and reasons:")
        for pid, why in failed:
            print(f"    pid={pid}: {why}")
    print(f"  Generations: {gen_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in generate_stage0_paired_v4_multi.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
