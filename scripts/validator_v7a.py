#!/usr/bin/env python3
"""
Validator for v7(a) A-side bilateral singleton probe.

Design:
- v7(a) produces 20 factually TRUE statements that are dressed in the
  INCORRECT-side surface profile that the classifiers trained on full
  TruthfulQA over-associate with the false label.
- Each A-side statement is therefore stripped of every surface cue the
  §5.1 singleton probe exploited on the false side (negation openers,
  hedging tokens, authority phrases, contrast conjunctions).
- Direct declarative. Word count [5, 15] (mirrors the TQA Best Answer /
  Best Incorrect length distribution; mean~9.4, sd~4.1).

A paired v7 B-side is NOT generated in this round (deferred pending
v7(a) headline result). The B-side used for bilateral scoring in
score_singleton_v7a.py is the existing §5.1 singleton.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Mirrors the cue lists used across §5.1 / v5 / v6 so the A-side here is
# scrubbed against the same surface vocabulary the classifier was
# shortcut-trained on.
NEGATION_OPENERS_V7A: list[str] = [
    "Not",
    "Never",
    "No",
    "Nothing",
    "Nobody",
    "None",
    "Neither",
    "Contrary to",
    "It is not the case that",
    "There is no",
    "There are no",
]

NEGATION_TOKENS_ANYWHERE_V7A: list[str] = [
    "not",
    "no",
    "never",
    "nothing",
    "nobody",
    "none",
    "neither",
    "n't",
    "cannot",
]

HEDGE_TOKENS_BANNED_V7A: list[str] = [
    "typically",
    "often",
    "usually",
    "sometimes",
    "generally",
    "may",
    "might",
    "can",
    "could",
    "possibly",
    "reportedly",
    "allegedly",
    "perhaps",
    "likely",
    "maybe",
]

AUTHORITY_OPENERS_V7A: list[str] = [
    "According to experts",
    "Studies show",
    "Research indicates",
    "Scientists have confirmed",
    "As reported by",
    "It is well known",
    "Historians note",
    "Experts say",
    "Scientists say",
    "Researchers say",
    "Records show",
    "In fact",
    "Actually",
]

AUTHORITY_ANYWHERE_V7A: list[str] = [
    "according to experts",
    "studies show",
    "research indicates",
    "scientists have confirmed",
    "as reported by",
    "it is well known",
    "historians note",
    "experts say",
    "scientists say",
    "researchers say",
    "records show",
]

CONTRAST_CONJUNCTIONS_V7A: list[str] = [
    "yet",
    "but",
    "however",
    "although",
    "though",
    "while",
    "whereas",
    "nevertheless",
    "still",
    "rather",
]

GEN_CONFIDENCE_MIN = 0.70

WORD_COUNT_MIN = 5
WORD_COUNT_MAX = 15

ALLOWED_TOPICS_V7A: list[str] = [
    "Misconceptions",
    "Law",
    "Health",
    "Science",
    "Nutrition",
    "Finance",
    "History",
    "Paranormal",
    "Sociology",
    "Economics",
]

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
    "actually false",
    "is false",
    "statement is false",
    "this is false",
    "i cannot confirm",
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
    pat = rf"(?<![\w']){re.escape(token)}(?![\w])"
    return bool(re.search(pat, str(text or ""), flags=re.IGNORECASE))


def _contains_any_token(text: str, tokens: list[str]) -> Optional[str]:
    for tok in tokens:
        if _contains_token(text, tok):
            return tok
    return None


def _contains_any_substr(text: str, phrases: list[str]) -> Optional[str]:
    s = str(text or "").lower()
    for p in phrases:
        if p.lower() in s:
            return p
    return None


def _has_red_flag_phrase(text: str) -> Optional[str]:
    s = str(text or "").lower()
    for phrase in RED_FLAG_RATIONALE_PHRASES:
        if phrase in s:
            return phrase
    return None


def validate_v7a(rec: dict) -> Tuple[bool, Optional[str], List[str]]:
    """Return (ok, reason_or_None, soft_warnings).

    `reason` for word-count violations uses the machine-parseable
    prefix `wc_violation:<count>:...` so the generator's retry loop can
    feed the exact offending count back into the prompt (matches the v6
    convention).
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

    # Cue-strip: negation opener
    op = _starts_with_any(text, NEGATION_OPENERS_V7A)
    if op is not None:
        return False, (
            f"statement starts with forbidden negation opener {op!r}"
        ), warnings

    # Cue-strip: negation token anywhere
    neg = _contains_any_token(text, NEGATION_TOKENS_ANYWHERE_V7A)
    if neg is not None:
        return False, (
            f"statement contains forbidden negation token {neg!r}"
        ), warnings

    # Cue-strip: authority opener
    auth = _starts_with_any(text, AUTHORITY_OPENERS_V7A)
    if auth is not None:
        return False, (
            f"statement starts with forbidden authority opener {auth!r}"
        ), warnings

    # Cue-strip: authority phrase anywhere
    auth_any = _contains_any_substr(text, AUTHORITY_ANYWHERE_V7A)
    if auth_any is not None:
        return False, (
            f"statement contains forbidden authority phrase {auth_any!r}"
        ), warnings

    # Cue-strip: hedge anywhere
    hedge = _contains_any_token(text, HEDGE_TOKENS_BANNED_V7A)
    if hedge is not None:
        return False, (
            f"statement contains forbidden hedge token {hedge!r}"
        ), warnings

    # Cue-strip: contrast conjunctions
    ctr = _contains_any_token(text, CONTRAST_CONJUNCTIONS_V7A)
    if ctr is not None:
        return False, (
            f"statement contains forbidden contrast conjunction {ctr!r}"
        ), warnings

    # Confidence in truthfulness
    try:
        conf = float(rec["generator_confidence"])
    except Exception:
        return False, "generator_confidence not parseable float", warnings
    if conf < GEN_CONFIDENCE_MIN:
        return False, (
            f"generator_confidence={conf} below {GEN_CONFIDENCE_MIN}"
        ), warnings

    # Rationale red flags
    red = _has_red_flag_phrase(str(rec.get("generator_truth_rationale", "")))
    if red is not None:
        return False, f"rationale red-flag phrase {red!r}", warnings

    return True, None, warnings


__all__ = [
    "NEGATION_OPENERS_V7A",
    "NEGATION_TOKENS_ANYWHERE_V7A",
    "HEDGE_TOKENS_BANNED_V7A",
    "AUTHORITY_OPENERS_V7A",
    "AUTHORITY_ANYWHERE_V7A",
    "CONTRAST_CONJUNCTIONS_V7A",
    "ALLOWED_TOPICS_V7A",
    "GEN_CONFIDENCE_MIN",
    "WORD_COUNT_MIN",
    "WORD_COUNT_MAX",
    "word_count",
    "validate_v7a",
]
