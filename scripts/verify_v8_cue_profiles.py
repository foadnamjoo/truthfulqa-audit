#!/usr/bin/env python3
"""
v8 bilateral cue-inversion probe — Step 2 verifier.

Checks whether the already-generated v7a-v1 A-side (20 TRUE, 5-15 words,
cue-stripped) satisfies the locked FALSE-class surface profile from
stage0_v8_inverted_bilateral/v8_cue_profiles.json, and that the existing
§5.1 B-side (20 FALSE, ~23.5 words, TRUE-class cues) satisfies its side.

If A-side items fail, lists them for regeneration. Also computes
per-pair word_count delta against the same-id B-side to flag pair
constraints (delta >= 8).

No numpy dependency — pure stdlib.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

A_SIDE_JSON = REPO_ROOT / "stage0_v7a_bilateral" / "stage0_singleton_v7a_generations.json"
B_SIDE_JSON = REPO_ROOT / "stage0" / "stage0_generations.json"
SPEC_JSON = REPO_ROOT / "stage0_v8_inverted_bilateral" / "v8_cue_profiles.json"
OUT_JSON = REPO_ROOT / "stage0_v8_inverted_bilateral" / "v8_verification_report.json"

# Mirrors scripts/validator_v7a.py lexicons (and the surface_features_text.py
# NEG_WORDS / HEDGES / AUTH_CUES lists); duplicated here so the verifier has
# no runtime dependency beyond stdlib.
NEG_LEADS = [
    r"^no\b", r"^not\b",
    r"^there is no\b", r"^there are no\b", r"^there's no\b",
    r"^none\b", r"^never\b", r"^nothing\b", r"^nobody\b",
    r"^neither\b", r"^contrary to\b", r"^it is not the case that\b",
]
NEG_WORDS = [
    r"\bno\b", r"\bnot\b", r"\bnever\b",
    r"\bnone\b", r"\bnothing\b", r"\bnobody\b", r"\bneither\b",
    r"\bcannot\b", r"\bcan't\b",
    r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b",
    r"\bwon't\b", r"\bwasn't\b", r"\bweren't\b",
]
HEDGES = [
    r"\bmight\b", r"\bcould\b", r"\bmay\b",
    r"\bgenerally\b", r"\btypically\b", r"\boften\b",
    r"\busually\b", r"\bsometimes\b", r"\blikely\b",
    r"\bprobably\b", r"\bpossibly\b",
    r"\bappears to\b", r"\bseems\b", r"\bsuggests\b",
    r"\broughly\b", r"\bapproximately\b",
    r"\bperhaps\b", r"\bmaybe\b", r"\breportedly\b", r"\ballegedly\b",
]
AUTH_ANYWHERE = [
    r"according to", r"experts say", r"scientists say",
    r"scientists have confirmed", r"researchers say",
    r"peer[- ]?reviewed", r"study shows", r"studies show",
    r"research indicates", r"historians note",
    r"records show", r"it is well known", r"as reported by",
]
CONTRAST_CONJUNCTIONS = [
    r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bthough\b",
    r"\byet\b", r"\bwhereas\b", r"\bwhile\b",
    r"\bnevertheless\b", r"\bnonetheless\b",
]

SENT_SPLIT = re.compile(r"[.!?]+")


def count_matches(patterns, text_lower):
    return sum(len(re.findall(p, text_lower)) for p in patterns)


def starts_with_neg_lead(text_lower_stripped):
    return any(re.match(p, text_lower_stripped) for p in NEG_LEADS)


def sentence_count(text):
    s = text.strip()
    if not s:
        return 1
    segs = [seg for seg in SENT_SPLIT.split(s) if seg.strip()]
    return max(1, len(segs))


def check_a_side(statement):
    """Return (ok, failures_list) for a single A-side candidate."""
    failures = []
    t = str(statement)
    t_lower = t.lower()
    t_lower_stripped = t_lower.strip()

    wc = len(t.split())
    if not (5 <= wc <= 15):
        failures.append(f"word_count={wc} out of [5,15]")

    if starts_with_neg_lead(t_lower_stripped):
        failures.append("neg_lead matched")

    neg_cnt = count_matches(NEG_WORDS, t_lower)
    if neg_cnt > 0:
        failures.append(f"neg_cnt={neg_cnt} > 0")

    hedge_cnt = count_matches(HEDGES, t_lower)
    if hedge_cnt > 0:
        failures.append(f"hedge_cnt={hedge_cnt} > 0")

    auth_cnt = count_matches(AUTH_ANYWHERE, t_lower)
    if auth_cnt > 0:
        failures.append(f"auth_cnt={auth_cnt} > 0")

    contrast_cnt = count_matches(CONTRAST_CONJUNCTIONS, t_lower)
    if contrast_cnt > 0:
        failures.append(f"contrast_cnt={contrast_cnt} > 0")

    sc = sentence_count(t)
    if not (1 <= sc <= 3):
        failures.append(f"sent_count={sc} out of [1,3]")

    return (len(failures) == 0, failures, {
        "word_count": wc, "neg_cnt": neg_cnt, "hedge_cnt": hedge_cnt,
        "auth_cnt": auth_cnt, "contrast_cnt": contrast_cnt, "sent_count": sc,
    })


def check_b_side(statement):
    """Verify B-side carries TRUE-class cues (word_count in [18,29] AND at
    least one of neg_cnt/neg_lead/hedge nonzero)."""
    failures = []
    t = str(statement)
    t_lower = t.lower()
    t_lower_stripped = t_lower.strip()

    wc = len(t.split())
    if not (18 <= wc <= 29):
        failures.append(f"word_count={wc} out of [18,29]")

    neg_cnt = count_matches(NEG_WORDS, t_lower)
    hedge_cnt = count_matches(HEDGES, t_lower)
    neg_lead = 1 if starts_with_neg_lead(t_lower_stripped) else 0
    if neg_cnt == 0 and hedge_cnt == 0 and neg_lead == 0:
        failures.append("no TRUE-class cue present (neg_cnt=neg_lead=hedge=0)")

    return (len(failures) == 0, failures, {
        "word_count": wc, "neg_cnt": neg_cnt, "hedge_cnt": hedge_cnt,
        "neg_lead": neg_lead,
    })


def main():
    if not A_SIDE_JSON.exists():
        print(f"MISSING: {A_SIDE_JSON}", file=sys.stderr)
        return 1
    if not B_SIDE_JSON.exists():
        print(f"MISSING: {B_SIDE_JSON}", file=sys.stderr)
        return 1
    if not SPEC_JSON.exists():
        print(f"MISSING: {SPEC_JSON}", file=sys.stderr)
        return 1

    a_side = json.load(open(A_SIDE_JSON))
    b_side = json.load(open(B_SIDE_JSON))

    print("=" * 70)
    print("v8 VERIFICATION — A-side (v7a-v1): TRUE with FALSE-class cues")
    print("=" * 70)
    a_results = []
    a_fail_ids = []
    for r in a_side:
        sid = r.get("id")
        stmt = r.get("statement", "")
        ok, failures, feats = check_a_side(stmt)
        a_results.append({
            "id": sid, "statement": stmt, "ok": ok,
            "failures": failures, "features": feats,
        })
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] id={sid:>2}  wc={feats['word_count']:>2}  "
              f"neg={feats['neg_cnt']}  hedge={feats['hedge_cnt']}  "
              f"auth={feats['auth_cnt']}  contrast={feats['contrast_cnt']}  "
              f"sent={feats['sent_count']}  :: {stmt[:70]}")
        if not ok:
            a_fail_ids.append(sid)
            for f in failures:
                print(f"         -> {f}")

    print()
    print("=" * 70)
    print("v8 VERIFICATION — B-side (§5.1): FALSE with TRUE-class cues")
    print("=" * 70)
    b_results = []
    b_fail_ids = []
    for r in b_side:
        sid = r.get("id")
        stmt = r.get("statement", "")
        ok, failures, feats = check_b_side(stmt)
        b_results.append({
            "id": sid, "statement": stmt, "ok": ok,
            "failures": failures, "features": feats,
        })
        tag = "PASS" if ok else "WARN"
        print(f"  [{tag}] id={sid:>2}  wc={feats['word_count']:>2}  "
              f"neg={feats['neg_cnt']}  hedge={feats['hedge_cnt']}  "
              f"neg_lead={feats['neg_lead']}  :: {stmt[:70]}")
        if not ok:
            b_fail_ids.append(sid)
            for f in failures:
                print(f"         -> {f}")

    print()
    print("=" * 70)
    print("PAIR CONSTRAINT — |wc_A - wc_B| >= 8 (same id)")
    print("=" * 70)
    a_by_id = {r["id"]: r for r in a_results}
    b_by_id = {r["id"]: r for r in b_results}
    common = sorted(set(a_by_id) & set(b_by_id))
    pair_rows = []
    pair_fail = []
    for pid in common:
        wa = a_by_id[pid]["features"]["word_count"]
        wb = b_by_id[pid]["features"]["word_count"]
        delta = wb - wa
        ok = delta >= 8
        pair_rows.append({
            "id": pid, "wc_A": wa, "wc_B": wb, "delta": delta, "ok": ok,
        })
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] id={pid:>2}  wc_A={wa:>2}  wc_B={wb:>2}  delta={delta:+d}")
        if not ok:
            pair_fail.append(pid)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"A-side: {len(a_side) - len(a_fail_ids)}/{len(a_side)} pass, fails={a_fail_ids}")
    print(f"B-side: {len(b_side) - len(b_fail_ids)}/{len(b_side)} pass, warns={b_fail_ids}")
    print(f"Pair length-delta: {len(pair_rows) - len(pair_fail)}/{len(pair_rows)} pass, fails={pair_fail}")
    union_drop = sorted(set(a_fail_ids) | set(pair_fail))
    print(f"Pairs to drop or regenerate: {union_drop}  ({len(union_drop)} of {len(common)})")
    print(f"Usable n = {len(common) - len(union_drop)}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "spec": str(SPEC_JSON.relative_to(REPO_ROOT)),
            "a_side": a_results,
            "b_side": b_results,
            "pair_rows": pair_rows,
            "summary": {
                "a_side_pass": len(a_side) - len(a_fail_ids),
                "a_side_total": len(a_side),
                "a_side_fail_ids": a_fail_ids,
                "b_side_pass": len(b_side) - len(b_fail_ids),
                "b_side_total": len(b_side),
                "b_side_warn_ids": b_fail_ids,
                "pair_delta_pass": len(pair_rows) - len(pair_fail),
                "pair_delta_total": len(pair_rows),
                "pair_delta_fail_ids": pair_fail,
                "drop_or_regenerate": union_drop,
                "usable_n": len(common) - len(union_drop),
            },
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
