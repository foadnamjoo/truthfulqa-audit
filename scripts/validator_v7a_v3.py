#!/usr/bin/env python3
"""
v7(a) v3 validator.

v3 goal: length-match A-side to the §5.1 B-side (mean 23.5 words,
range 18-29) so that the `word_count` feature - which is the #1
coefficient on surface_lr_cleaned (+0.31 standardized) and the #4 on
surface_lr_full - can no longer discriminate A from B on length alone.
Every other rule is unchanged from validator_v7a.py.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from scripts.validator_v7a import (
    NEGATION_OPENERS_V7A,
    NEGATION_TOKENS_ANYWHERE_V7A,
    HEDGE_TOKENS_BANNED_V7A,
    AUTHORITY_OPENERS_V7A,
    AUTHORITY_ANYWHERE_V7A,
    CONTRAST_CONJUNCTIONS_V7A,
    ALLOWED_TOPICS_V7A,
    GEN_CONFIDENCE_MIN,
    RED_FLAG_RATIONALE_PHRASES,
    word_count,
    _normalize_start,
    _starts_with_any,
    _contains_any_token,
    _contains_any_substr,
    _has_red_flag_phrase,
)

# §5.1 B-side: min=18, max=29, mean=23.5. Choose [22, 26] to keep the
# A-side tightly centered on the B-side mean while still leaving the
# generator some headroom.
WORD_COUNT_MIN = 22
WORD_COUNT_MAX = 26


def validate_v7a_v3(rec: dict) -> Tuple[bool, Optional[str], List[str]]:
    """Return (ok, reason_or_None, soft_warnings).

    Same cue-strip rules as validate_v7a. Only the word-count band
    changes.
    """
    warnings: List[str] = []

    required = [
        "id",
        "topic",
        "statement",
        "generator_truth_rationale",
        "generator_confidence",
    ]
    for key in required:
        if key not in rec:
            return False, f"missing field: {key}", warnings

    text = str(rec["statement"]).strip()
    if not text:
        return False, "statement empty", warnings

    topic = str(rec["topic"]).strip()
    if topic not in ALLOWED_TOPICS_V7A:
        return False, (
            f"invalid topic={topic!r}; v7a allows {ALLOWED_TOPICS_V7A}"
        ), warnings

    wc = word_count(text)
    if wc < WORD_COUNT_MIN or wc > WORD_COUNT_MAX:
        return False, (
            f"wc_violation:{wc}:statement word count {wc} outside "
            f"[{WORD_COUNT_MIN},{WORD_COUNT_MAX}]"
        ), warnings

    op = _starts_with_any(text, NEGATION_OPENERS_V7A)
    if op is not None:
        return False, (
            f"statement starts with forbidden negation opener {op!r}"
        ), warnings

    neg = _contains_any_token(text, NEGATION_TOKENS_ANYWHERE_V7A)
    if neg is not None:
        return False, (
            f"statement contains forbidden negation token {neg!r}"
        ), warnings

    auth = _starts_with_any(text, AUTHORITY_OPENERS_V7A)
    if auth is not None:
        return False, (
            f"statement starts with forbidden authority opener {auth!r}"
        ), warnings

    auth_any = _contains_any_substr(text, AUTHORITY_ANYWHERE_V7A)
    if auth_any is not None:
        return False, (
            f"statement contains forbidden authority phrase {auth_any!r}"
        ), warnings

    hedge = _contains_any_token(text, HEDGE_TOKENS_BANNED_V7A)
    if hedge is not None:
        return False, (
            f"statement contains forbidden hedge token {hedge!r}"
        ), warnings

    ctr = _contains_any_token(text, CONTRAST_CONJUNCTIONS_V7A)
    if ctr is not None:
        return False, (
            f"statement contains forbidden contrast conjunction {ctr!r}"
        ), warnings

    try:
        conf = float(rec["generator_confidence"])
    except Exception:
        return False, "generator_confidence not parseable float", warnings
    if conf < GEN_CONFIDENCE_MIN:
        return False, (
            f"generator_confidence={conf} below {GEN_CONFIDENCE_MIN}"
        ), warnings

    red = _has_red_flag_phrase(str(rec.get("generator_truth_rationale", "")))
    if red is not None:
        return False, f"rationale red-flag phrase {red!r}", warnings

    return True, None, warnings


__all__ = [
    "WORD_COUNT_MIN",
    "WORD_COUNT_MAX",
    "validate_v7a_v3",
]
