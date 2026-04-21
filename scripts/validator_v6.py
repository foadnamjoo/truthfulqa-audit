#!/usr/bin/env python3
"""
Validator for v6 paired probe.

v6 changes versus v5:
- "authority" strategy dropped entirely (only 1 cleaned-subset training
  example; any v5 authority signal was coefficient extrapolation).
- B-side strategies restricted to: negation_opener, hedging.
- Hard word-count cap of 5-20 words on BOTH sides (A and B), matching
  TruthfulQA Best / Best Incorrect length distribution
  (mean~9, sd~4, p95<=16.5) to keep v6 inputs inside the classifier
  training manifold. v5 A-sides averaged >2sigma above this.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

OPENERS_V6: list[str] = [
    "Not",
    "Never",
    "No",
    "Contrary to",
    "It is not the case that",
    "There is no",
    "Nothing",
]

HEDGE_TOKENS_STRICT: list[str] = [
    "typically",
    "often",
    "usually",
    "sometimes",
    "generally",
]

HEDGE_TOKENS_BANNED_EXTENDED: list[str] = [
    "typically",
    "often",
    "usually",
    "sometimes",
    "generally",
    "may",
    "might",
    "can",
]

AUTHORITY_OPENERS_V6: list[str] = [
    "According to experts,",
    "Studies show that",
    "Research indicates that",
    "Scientists have confirmed that",
]

CONTRAST_CONJUNCTIONS_V6: list[str] = [
    "yet",
    "but",
    "however",
    "although",
]

ALLOWED_STRATEGIES_V6: list[str] = ["negation_opener", "hedging"]

GEN_CONFIDENCE_MIN = 0.70

WORD_COUNT_MIN = 5
WORD_COUNT_MAX = 20

RED_FLAG_RATIONALE_PHRASES: list[str] = [
    "i am not sure",
    "i'm not sure",
    "uncertain",
    "unknown",
    "cannot verify",
    "can't verify",
    "as an ai",
    "made up",
    "fictional",
]


def word_count(text: str) -> int:
    return len(str(text or "").split())


def _normalize_start(text: str) -> str:
    s = str(text or "").lstrip()
    while s and s[0] in "\"'`([{":
        s = s[1:].lstrip()
    return s


def _starts_with_any(text: str, phrases: list[str]) -> Optional[str]:
    s = _normalize_start(text)
    s_low = s.lower()
    for phrase in phrases:
        if s_low.startswith(phrase.lower()):
            return phrase
    return None


def _contains_token(text: str, token: str) -> bool:
    return bool(re.search(rf"\b{re.escape(token)}\b", str(text or ""),
                          flags=re.IGNORECASE))


def _count_any_tokens(text: str, tokens: list[str]) -> tuple[int, list[str]]:
    found = [t for t in tokens if _contains_token(text, t)]
    return len(found), found


def _contains_contrast(text: str) -> Optional[str]:
    for c in CONTRAST_CONJUNCTIONS_V6:
        if _contains_token(text, c):
            return c
    return None


def _has_red_flag_phrase(text: str) -> Optional[str]:
    s = str(text or "").lower()
    for phrase in RED_FLAG_RATIONALE_PHRASES:
        if phrase in s:
            return phrase
    return None


def _observed_cues_in_b(b_text: str) -> list[str]:
    out: list[str] = []
    if _starts_with_any(b_text, OPENERS_V6):
        out.append("negation_opener")
    if _count_any_tokens(b_text, HEDGE_TOKENS_STRICT)[0] > 0:
        out.append("hedging")
    if _starts_with_any(b_text, AUTHORITY_OPENERS_V6):
        out.append("authority")
    return out


def validate_v6(rec: dict) -> Tuple[bool, Optional[str], List[str]]:
    """Return (ok, reason_or_None, soft_warnings).

    The reason string is machine-parseable by the generator's retry
    loop: if it starts with "wc_violation:" the generator can inject
    the offending count into the retry prompt.
    """
    warnings: List[str] = []
    required = [
        "a_side", "a_side_truth_rationale",
        "b_side", "b_side_truth_rationale",
        "b_cue_strategy", "generator_confidence",
    ]
    for key in required:
        if key not in rec:
            return False, f"missing field: {key}", warnings

    a_text = str(rec["a_side"]).strip()
    b_text = str(rec["b_side"]).strip()
    if not a_text:
        return False, "a_side empty", warnings
    if not b_text:
        return False, "b_side empty", warnings

    strategy = str(rec["b_cue_strategy"]).strip()
    if strategy not in ALLOWED_STRATEGIES_V6:
        return False, (
            f"invalid b_cue_strategy={strategy!r}; v6 allows "
            f"{ALLOWED_STRATEGIES_V6}"
        ), warnings

    # Word-count caps (both sides, hard)
    a_wc = word_count(a_text)
    b_wc = word_count(b_text)
    if a_wc < WORD_COUNT_MIN or a_wc > WORD_COUNT_MAX:
        return False, (
            f"wc_violation:A:{a_wc}:A-side word count {a_wc} outside "
            f"[{WORD_COUNT_MIN},{WORD_COUNT_MAX}]"
        ), warnings
    if b_wc < WORD_COUNT_MIN or b_wc > WORD_COUNT_MAX:
        return False, (
            f"wc_violation:B:{b_wc}:B-side word count {b_wc} outside "
            f"[{WORD_COUNT_MIN},{WORD_COUNT_MAX}]"
        ), warnings

    # A-side: no negation opener, no contrast
    a_op = _starts_with_any(a_text, OPENERS_V6)
    if a_op is not None:
        return False, (
            f"A-side starts with forbidden negation opener {a_op!r}"
        ), warnings
    c = _contains_contrast(a_text)
    if c is not None:
        return False, (
            f"A-side contains forbidden contrast conjunction {c!r}"
        ), warnings

    # B-side global
    c = _contains_contrast(b_text)
    if c is not None:
        return False, (
            f"B-side contains forbidden contrast conjunction {c!r}"
        ), warnings

    b_starts_opener = _starts_with_any(b_text, OPENERS_V6) is not None
    b_starts_auth = _starts_with_any(b_text, AUTHORITY_OPENERS_V6) is not None
    n_hedges_strict, found_hedges_strict = _count_any_tokens(
        b_text, HEDGE_TOKENS_STRICT)
    n_hedges_banned, found_hedges_banned = _count_any_tokens(
        b_text, HEDGE_TOKENS_BANNED_EXTENDED)

    if strategy == "negation_opener":
        if not b_starts_opener:
            return False, (
                "B-side must start with negation opener for "
                "negation_opener strategy"
            ), warnings
        if n_hedges_banned > 0:
            return False, (
                "B-side contains banned hedging token(s) for "
                f"negation_opener strategy: {found_hedges_banned}"
            ), warnings
        if b_starts_auth:
            return False, (
                "B-side must not start with authority opener under "
                "negation_opener strategy"
            ), warnings

    elif strategy == "hedging":
        if b_starts_opener:
            return False, (
                "B-side must not start with negation opener under "
                "hedging strategy"
            ), warnings
        if b_starts_auth:
            return False, (
                "B-side must not start with authority opener under "
                "hedging strategy"
            ), warnings
        if n_hedges_strict != 1:
            return False, (
                "B-side must contain exactly one strict hedge token "
                f"under hedging strategy; found {n_hedges_strict}: "
                f"{found_hedges_strict}"
            ), warnings

    # Confidence
    try:
        conf = float(rec["generator_confidence"])
    except Exception:
        return False, "generator_confidence not parseable float", warnings
    if conf < GEN_CONFIDENCE_MIN:
        return False, (
            f"generator_confidence={conf} below {GEN_CONFIDENCE_MIN}"
        ), warnings

    # Rationale red flags
    a_red = _has_red_flag_phrase(str(rec.get("a_side_truth_rationale", "")))
    b_red = _has_red_flag_phrase(str(rec.get("b_side_truth_rationale", "")))
    if a_red is not None:
        return False, f"A rationale red-flag phrase {a_red!r}", warnings
    if b_red is not None:
        return False, f"B rationale red-flag phrase {b_red!r}", warnings

    # Strict single-cue on B
    observed = _observed_cues_in_b(b_text)
    if len(observed) != 1:
        return False, (
            f"B-side observed cues not exactly one: {observed}"
        ), warnings
    if observed[0] != strategy:
        return False, (
            f"B-side cue mismatch: assigned={strategy!r}, "
            f"observed={observed!r}"
        ), warnings

    return True, None, warnings


__all__ = [
    "OPENERS_V6",
    "HEDGE_TOKENS_STRICT",
    "HEDGE_TOKENS_BANNED_EXTENDED",
    "AUTHORITY_OPENERS_V6",
    "CONTRAST_CONJUNCTIONS_V6",
    "ALLOWED_STRATEGIES_V6",
    "GEN_CONFIDENCE_MIN",
    "WORD_COUNT_MIN",
    "WORD_COUNT_MAX",
    "word_count",
    "validate_v6",
]
