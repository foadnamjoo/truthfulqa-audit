#!/usr/bin/env python3
"""
Validator for the v4 paired probe.

v4 differs from v3 (paraphrase) in that the generator writes FRESH
TruthfulQA-style answers rather than paraphrasing the existing ones.
We therefore enforce the surface-cue rules tightly:

A-side (factually TRUE answer with INCORRECT-style surface profile):
  - MUST NOT start with any of the 7 OPENERS_V4.
  - MUST NOT contain any token in NEG_LEXICON_V4 (word-boundary,
    case-insensitive). NEG_LEXICON_V4 is a strict superset of the
    audit's NEG_WORDS lexicon, plus the synonym-negation tokens that
    a thoughtful reviewer would mark as semantic negation
    (zero / incapable / unaffected / fails / lacks / without /
    unable / impossible) and all common denial contractions.
  - MUST NOT contain any hedging token (HEDGE_TOKENS_V4).
  - MUST NOT start with any AUTH_CUE_PHRASES_V4 phrase.

B-side (factually FALSE answer with CORRECT-style surface profile):
  - MUST start with one of OPENERS_V4 (case-insensitive, word-bounded).
  - SHOULD contain >= 1 hedging token (HEDGE_TOKENS_V4). Absence is
    logged as a soft warning, not a hard reject.

Both sides:
  - MUST NOT contain the myth-vs-truth contrast conjunctions in
    CONTRAST_CONJUNCTIONS_V4 (yet / but / however / although).
  - generator_confidence >= GEN_CONFIDENCE_MIN.

Returned by validate_v4(...): (ok: bool, hard_fail: str|None,
soft_warnings: list[str]).
"""
from __future__ import annotations

import re
from typing import Tuple, List, Optional


OPENERS_V4: list[str] = [
    "Not", "Never", "No", "Contrary to",
    "It is not the case that", "There is no", "Nothing",
]

# Strict superset of the audit's NEG_WORDS lexicon plus the synonym-
# negation tokens used in scripts/experiment_extended_lexicon.py
# (Lexicon B). Contractions are listed both with apostrophe and curly
# quote variants so the regex matches whatever quote-style Opus emits.
NEG_LEXICON_V4: list[str] = [
    # Audit's NEG_WORDS (verbatim, sans regex anchors)
    "no", "not", "never", "none", "nothing", "nobody",
    "cannot", "can't", "don't", "doesn't", "didn't",
    "won't", "wasn't", "weren't",
    # Additional contractions for full coverage
    "isn't", "aren't", "hasn't", "haven't", "hadn't",
    "wouldn't", "couldn't", "shouldn't",
    "can\u2019t", "don\u2019t", "doesn\u2019t", "didn\u2019t",
    "won\u2019t", "wasn\u2019t", "weren\u2019t",
    "isn\u2019t", "aren\u2019t", "hasn\u2019t", "haven\u2019t",
    "hadn\u2019t", "wouldn\u2019t", "couldn\u2019t", "shouldn\u2019t",
    # Synonym-negation extension (Lexicon B from
    # scripts/experiment_extended_lexicon.py)
    "zero", "incapable", "unaffected",
    "fails", "fail", "lacks", "lack",
    "without", "unable", "impossible",
]

# Per spec: {typically, often, usually, sometimes, can, may, might,
# generally}. Note "can" overlaps with the audit's HEDGES list.
HEDGE_TOKENS_V4: list[str] = [
    "typically", "often", "usually", "sometimes",
    "can", "may", "might", "generally",
]

# Authority-cue starters; matches scripts/generate_stage0_paired_tqa.py's
# AUTH_CUE_PHRASES family. Forbidden as A-side openers.
AUTH_CUE_PHRASES_V4: list[str] = [
    "according to", "studies show", "experts say",
    "scientists say", "research shows", "researchers say",
    "records show", "in fact", "actually", "historians note",
    "it is well known", "as reported by", "study shows",
]

# Spec list: yet / but / however / although. We add common siblings
# (though / while / whereas / nevertheless) used elsewhere in the v3
# validator to keep the v4 prompt internally consistent.
CONTRAST_CONJUNCTIONS_V4: list[str] = [
    "yet", "but", "however", "although",
    "though", "while", "whereas", "nevertheless",
]

GEN_CONFIDENCE_MIN: float = 0.70


# --- Regex helpers ---------------------------------------------------------
def _starts_with_opener(text: str) -> Optional[str]:
    s = (text or "").lstrip()
    if not s:
        return None
    s_low = s.lower()
    for op in OPENERS_V4:
        op_low = op.lower()
        if not s_low.startswith(op_low):
            continue
        tail = s[len(op):]
        # Word boundary: next char must be space, comma, end-of-string,
        # or punctuation that isn't a letter/apostrophe.
        if not tail or not (tail[0].isalpha() or tail[0] == "'"):
            return op
    return None


def _starts_with_auth_phrase(text: str) -> Optional[str]:
    s = (text or "").lstrip().lower()
    for phrase in AUTH_CUE_PHRASES_V4:
        if s.startswith(phrase):
            tail = s[len(phrase):]
            if not tail or not tail[0].isalpha():
                return phrase
    return None


_NEG_TOKEN_RES = [
    (tok, re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE))
    for tok in NEG_LEXICON_V4
]
_HEDGE_TOKEN_RES = [
    (tok, re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE))
    for tok in HEDGE_TOKENS_V4
]
_CONTRAST_RES = [
    (c, re.compile(rf"\b{re.escape(c)}\b", re.IGNORECASE))
    for c in CONTRAST_CONJUNCTIONS_V4
]


def _find_token(rx_list: list[tuple[str, re.Pattern]], text: str
                ) -> Optional[str]:
    for tok, rx in rx_list:
        if rx.search(text or ""):
            return tok
    return None


# --- Validator -------------------------------------------------------------
def validate_v4(rec: dict) -> Tuple[bool, Optional[str], List[str]]:
    """Return (ok, hard_fail_reason, soft_warnings)."""
    warnings: List[str] = []

    required = [
        "a_side", "a_side_truth_rationale",
        "b_side", "b_side_truth_rationale",
        "cues_in_b", "generator_confidence",
    ]
    for k in required:
        if k not in rec:
            return False, f"missing field: {k}", warnings

    a_text = str(rec["a_side"]).strip()
    b_text = str(rec["b_side"]).strip()
    if not a_text:
        return False, "a_side empty", warnings
    if not b_text:
        return False, "b_side empty", warnings

    # --- A-side hard checks ---
    a_op = _starts_with_opener(a_text)
    if a_op is not None:
        return False, f"A-side starts with negation opener {a_op!r}", warnings

    a_neg = _find_token(_NEG_TOKEN_RES, a_text)
    if a_neg is not None:
        return False, (f"A-side contains forbidden neg-lexicon token "
                       f"{a_neg!r}"), warnings

    a_hedge = _find_token(_HEDGE_TOKEN_RES, a_text)
    if a_hedge is not None:
        return False, (f"A-side contains forbidden hedging token "
                       f"{a_hedge!r}"), warnings

    a_auth = _starts_with_auth_phrase(a_text)
    if a_auth is not None:
        return False, (f"A-side starts with forbidden authority phrase "
                       f"{a_auth!r}"), warnings

    # --- B-side hard checks ---
    b_op = _starts_with_opener(b_text)
    if b_op is None:
        return False, (f"B-side does not start with any of the required "
                       f"openers {OPENERS_V4}"), warnings

    b_hedge = _find_token(_HEDGE_TOKEN_RES, b_text)
    if b_hedge is None:
        warnings.append("B-side has no hedging token (soft warning)")

    # --- Both sides: contrast conjunction ---
    for label, txt in [("A", a_text), ("B", b_text)]:
        c = _find_token(_CONTRAST_RES, txt)
        if c is not None:
            return False, (f"{label}-side contains forbidden contrast "
                           f"conjunction {c!r}"), warnings

    # --- generator_confidence ---
    try:
        conf = float(rec["generator_confidence"])
    except Exception:
        return False, "generator_confidence is not a parseable float", warnings
    if conf < GEN_CONFIDENCE_MIN:
        return False, (f"generator_confidence={conf} below "
                       f"{GEN_CONFIDENCE_MIN}"), warnings

    # --- cues_in_b sanity (informational; do not hard-fail) ---
    cues = rec.get("cues_in_b", [])
    if not isinstance(cues, list):
        warnings.append(f"cues_in_b is not a list (got {type(cues).__name__})")

    return True, None, warnings


__all__ = [
    "OPENERS_V4", "NEG_LEXICON_V4", "HEDGE_TOKENS_V4",
    "AUTH_CUE_PHRASES_V4", "CONTRAST_CONJUNCTIONS_V4",
    "GEN_CONFIDENCE_MIN", "validate_v4",
]


if __name__ == "__main__":
    samples = [
        # Good A-side, good B-side
        {
            "a_side": "Earth orbits the sun once every year.",
            "a_side_truth_rationale": "...",
            "b_side": "Not all planets typically orbit the sun annually.",
            "b_side_truth_rationale": "...",
            "cues_in_b": ["negation_opener", "hedging_token"],
            "generator_confidence": 0.95,
        },
        # Bad A-side: contains "without"
        {
            "a_side": "Earth orbits the sun without exception.",
            "a_side_truth_rationale": "...",
            "b_side": "Not all planets typically orbit the sun.",
            "b_side_truth_rationale": "...",
            "cues_in_b": ["negation_opener"],
            "generator_confidence": 0.95,
        },
        # Bad B-side: no opener
        {
            "a_side": "Earth orbits the sun every year.",
            "a_side_truth_rationale": "...",
            "b_side": "All planets typically orbit the moon.",
            "b_side_truth_rationale": "...",
            "cues_in_b": ["hedging_token"],
            "generator_confidence": 0.95,
        },
    ]
    for i, s in enumerate(samples):
        ok, why, warn = validate_v4(s)
        print(f"sample {i}: ok={ok}  why={why!r}  warn={warn}")
