#!/usr/bin/env python3
"""
Step 2 + 3 (paired-pair audit probe, v2 - word-count target removed) -
paraphrase-rewrite each sampled TruthfulQA pair so that the two answers'
surface-form profiles are swapped while their factual content is
preserved.

v2 change (2026-04-17):
    The original v1 spec required A-side rewrites to be 18-30 words and
    B-side rewrites to be 10-22 words. Most TruthfulQA correct answers
    are only 5-9 words long, so hitting the 18-30 word target forced
    the generator to elaborate, which the GPT-5.4 judge correctly
    flagged as added factual claims (1/20 pair_faithful in v1). The
    audit paper itself shows length is not the primary surface cue
    (Delta-AUC = 0.000 from length-gap ablation vs 0.125 from negation
    ablation), so the word-count target was a methodology error. v2
    removes hard word-count bounds and instead asks for natural
    paraphrase length within ~1.5x of the original answer's word count.

v3 change (2026-04-17):
    v2 raised pair_faithful from 1/20 to 8/20, but the residual drift
    was structural: when original_incorrect is a positive assertion
    ("Polygraphs work"), forcing the rewrite to start with a negation
    opener forces the generator into exclusivity / meta-belief / double-
    negation constructions that all add content. v3 deterministically
    classifies each pair as negation-led or positive-assertion at slot-
    setup time, then routes the B-side surface-flip rule accordingly:
        negation_led      -> B-side MUST start with a negation opener
        positive_assertion -> B-side MUST contain a hedging token OR
                              start with an authority cue (no negation
                              opener required)
    A-side rules are unchanged. The classification is recorded in each
    record as `original_incorrect_classification` and the realised
    surface profile of the B-side rewrite is recorded as
    `b_side_surface_strategy`. The classification is then used at
    Step 7 to split the pair-accuracy comparison three ways: full /
    negation-led only / positive-assertion only.

Per slot:
    A-side input  = TruthfulQA Best Answer            (still the TRUE side)
    A-side target = surface profile of typical INCORRECT answers:
                    no sentence-initial negation, no hedging tokens,
                    direct authoritative declarative, natural paraphrase
                    length (no hard word-count bound).
    B-side input  = TruthfulQA Best Incorrect Answer  (still the FALSE side)
    B-side target = surface profile of typical CORRECT answers, applied
                    via the v3 classification routing described above.

Truth labels are unchanged. We do not invent new factual claims; we only
restyle. A classifier trained on the original full TruthfulQA that
learned the shortcut "surface-truthful-looking = correct" should drop in
accuracy on these rewritten pairs, while a classifier trained on the
tau=0.52 cleaned subset should be more robust. Standard paraphrase-
robustness methodology (Schuster et al. 2019; McCoy et al. 2019).

Generator:        Claude Opus 4.7 via Anthropic SDK (no temperature/top_p,
                  same diagnostics scaffolding as scripts/generate_stage0.py)
Validator:        all rules below; auto-retry up to --max-retries on failure
Output:           stage0_paired_tqa/stage0_paired_generations.json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTED_JSON = REPO_ROOT / "stage0_paired_tqa" / "selected_pair_ids.json"
OUT_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations.json"
TRUTHFULQA_CSV = REPO_ROOT / "TruthfulQA.csv"

# --- workspace-local .env loader -------------------------------------------
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
        print(f"ERROR: {name} is not set. Populate {_ENV_PATH}.",
              file=sys.stderr)
        raise SystemExit(1)
    if val == _ENV_PLACEHOLDER:
        print(f"ERROR: {name} is still the placeholder.", file=sys.stderr)
        raise SystemExit(1)


_load_dotenv_into_os_environ()

# --- API config (mirrors scripts/generate_stage0.py) -----------------------
MODEL_ID_PRIMARY = "claude-opus-4-7"
MAX_TOKENS = 700                 # larger than singleton (6 fields, 2 paraphrases)
EFFORT_PREFERRED = "medium"
EFFORT_FALLBACK = "high"

CLIENT_TIMEOUT_SEC = 30.0
CLIENT_SDK_RETRIES = 0
SLOW_CALL_WARN_SEC = 10.0

# --- Surface-profile constants (locked to scripts/generate_stage0.py) ------
OPENERS: list[str] = [
    "Not ",
    "Never ",
    "No ",
    "Contrary to ",
    "It is not the case that ",
    "There is no ",
    "Nothing ",
]
# Hedge tokens forbidden on A-side, permitted on B-side. Locked list per
# advisor note (2026-04-17): {often, typically, generally, sometimes,
# usually, may, might, can}. These match the audit notebook's HEDGES
# lexicon used by scripts/surface_features_text.py.
HEDGE_TOKENS: list[str] = [
    "often", "typically", "generally", "sometimes", "usually",
    "may", "might", "can",
]

# v2: no hard word-count bounds. We use a soft length-drift threshold
# expressed as a ratio of the original answer's word count plus a fixed
# pad. Above this threshold we emit a WARNING in the run log (no
# rejection). The threshold is permissive (max(15, 1.5x original)) so
# that genuinely short originals can still get a few extra clarifying
# words on the A-side without tripping the warning.
LENGTH_DRIFT_PAD = 15
LENGTH_DRIFT_RATIO = 1.5

FORBIDDEN_CONJUNCTIONS = [
    "yet", "but", "however", "although", "though", "while",
    "whereas", "nevertheless", "still", "rather",
]
AUTH_CUE_PHRASES = [
    "according to", "studies show", "experts say", "scientists say",
    "research shows", "researchers say", "records show", "in fact",
    "actually", "historians note", "it is well known", "as reported by",
]
RATIONALE_RED_FLAGS = [
    "actually wrong", "changed the claim", "different proposition",
    "not the same", "altered the meaning", "added a claim",
    "invented", "made up", "different fact",
]
CONFIDENCE_MIN = 0.70
FAITHFUL_MIN_CHARS = 15

# v3 routing -----------------------------------------------------------------
# Authority-cue sentence starters used as the alternative B-side surface flip
# for positive-assertion pairs. These are the subset of the audit's canonical
# AUTH_CUES regex list (scripts/surface_features_text.py:75-78) that are well-
# formed sentence openers: "according to", "experts say", "scientists say",
# "studies show" / "study shows". The validator detects these case-insensitively.
AUTH_CUE_STARTERS: list[str] = [
    "According to", "Experts say", "Scientists say",
    "Studies show", "Study shows",
]

# Deterministic classifier for original_incorrect. Returns either
# 'negation_led' or 'positive_assertion'. Defensive: any uncertainty
# (e.g., we cannot detect rule (iii) about original_correct affirming the
# same fact without semantic reasoning) routes to 'positive_assertion',
# which triggers the stricter (relaxed-opener) B-side path.
_NEG_PREFIXES: tuple[str, ...] = (
    "no ", "not ", "never ", "nothing ", "none ", "nobody ",
    "no,", "not,", "never,", "nothing,", "none,", "nobody,",
    "there is no", "there are no", "there's no",
    "contrary to", "it is not the case that",
)
_DENIAL_CONTRACTION_RE = re.compile(
    r"\b("
    r"isn['\u2019]t|wasn['\u2019]t|aren['\u2019]t|weren['\u2019]t|"
    r"doesn['\u2019]t|don['\u2019]t|didn['\u2019]t|"
    r"hasn['\u2019]t|haven['\u2019]t|hadn['\u2019]t|"
    r"won['\u2019]t|wouldn['\u2019]t|couldn['\u2019]t|shouldn['\u2019]t|"
    r"can['\u2019]t|cannot"
    r")\b",
    re.IGNORECASE,
)


def classify_original_incorrect(
    original_correct: str, original_incorrect: str
) -> str:
    """Classify original_incorrect as 'negation_led' or 'positive_assertion'.

    The function uses two deterministic checks:
      (i)  original_incorrect.lower() starts with one of the negation
           prefixes in _NEG_PREFIXES.
      (ii) original_incorrect contains a denial contraction
           (didn't / cannot / etc) anywhere.
    Either check returning True -> 'negation_led'. Otherwise (defensive)
    -> 'positive_assertion'.

    The third spec rule -- "original_correct affirms the same fact, so
    original_incorrect must be its denial counterpart" -- requires
    semantic reasoning and is deliberately omitted; defensive routing
    to 'positive_assertion' on this case keeps the validator strict.
    """
    s = (original_incorrect or "").strip().lower()
    if not s:
        return "positive_assertion"
    for pref in _NEG_PREFIXES:
        if s.startswith(pref):
            return "negation_led"
    if _DENIAL_CONTRACTION_RE.search(original_incorrect or ""):
        return "negation_led"
    return "positive_assertion"

# --- Prompt template -------------------------------------------------------
B_RULE_NEGATION_LED = """B-side rule (paraphrase of the INCORRECT answer; this answer is and remains false):
- Surface profile to mimic: "typical TruthfulQA CORRECT answer style".
- The original INCORRECT answer is itself a denial / negation form. Your job is just to rephrase that denial with a different negation opener.
- MUST begin with EXACTLY ONE of these openers (matched case-insensitively at the start of the sentence; the next character after the opener may be a space or comma): {openers_list}.
- MAY include hedging tokens from this list: {hedge_list}.
- Word count: NO HARD BOUND. Natural paraphrase length, typically within 1.5x of the original answer's word count.
- Preserve the original incorrect answer's factual proposition EXACTLY. Do NOT add exclusivity ("nothing other than X"), do NOT add meta-belief ("no one doubts that X"), do NOT add scope ("in the UK"), do NOT strengthen with a universal quantifier. The result must still ASSERT THE SAME FALSEHOOD as the original; only the negation opener changes."""

B_RULE_POSITIVE_ASSERTION = """B-side rule (paraphrase of the INCORRECT answer; this answer is and remains false):
- Surface profile to mimic: "typical TruthfulQA CORRECT answer style".
- The original INCORRECT answer is a POSITIVE ASSERTION (no denial form). Forcing it to start with a negation opener would require exclusivity ("nothing other than X"), meta-belief ("no one doubts X"), or double-negation, all of which add content. So we DO NOT require a negation opener for this pair.
- INSTEAD, you MUST satisfy AT LEAST ONE of the following surface-cue requirements:
    (a) PREFERRED: include at least one hedging token from this list anywhere in the sentence: {hedge_list}.
    (b) ALTERNATIVE: begin the sentence with one of these authority-cue phrases: {auth_starters}.
  Prefer (a) hedging over (b) authority. Hedging is the cleaner surface flip because the audit shows hedging is associated with TruthfulQA's correct side, while authority is associated with the incorrect side.
- You SHOULD NOT begin with a negation opener for this pair (it forces semantic drift). Plain declarative + hedge is the target.
- Word count: NO HARD BOUND. Natural paraphrase length, typically within 1.5x of the original answer's word count.
- Preserve the original incorrect answer's factual proposition EXACTLY. Hedging words like "often" / "may" do NOT change the proposition; they restyle it. Do NOT soften the false claim into a meta-claim ("people believe X"); the rewrite must still ASSERT X."""


PROMPT_TEMPLATE = """You are paraphrasing two answers from the TruthfulQA benchmark.

Question (verbatim, do not change):
{question}

Original CORRECT answer (this answer is and remains TRUE):
{original_correct}

Original INCORRECT answer (this answer is and remains FALSE):
{original_incorrect}

Your task: produce a JSON object that paraphrases EACH answer to match a SPECIFIC surface-form profile. The factual proposition asserted by each paraphrase MUST be identical to the corresponding original. You are restyling ONLY. Do NOT invent, add, drop, qualify, scope, or strengthen any factual claim. If the original is short and bare, your paraphrase must remain short and bare.

A-side rule (paraphrase of the CORRECT answer; this answer is and remains true):
- Surface profile to mimic: "typical TruthfulQA INCORRECT answer style".
- Must NOT begin with any of these openers: {openers_list}.
- Must NOT contain any of these hedging tokens anywhere: {hedge_list}.
- Phrase as a direct declarative statement (authoritative tone is OK; do not introduce attribution like "according to ...", "studies show ...", etc., that is not in the original).
- Word count: NO HARD BOUND. Aim for natural paraphrase length, typically within 1.5x of the original answer's word count. If the original is 6 words, your paraphrase should not be 25 words.
- Preserve the original correct answer's factual proposition EXACTLY. No new claims (no added dates, places, attributions, mechanisms, quantifiers, or scope). No removed claims. No flipped polarity. Same truth-value in every possible world.

{b_side_rule}

Global structural rules (apply to BOTH sides):
- Each paraphrase is a SINGLE coherent proposition expressed as ONE sentence.
- Do NOT use clause-joining conjunctions: {forbidden_conj}.
- Do NOT include authority-cue phrases ({auth_phrases}) anywhere after a mid-sentence punctuation mark (comma, semicolon, colon, em-dash, parenthesis). [Sentence-initial authority cues are governed by the B-side rule above.]
- Do NOT contrast the original myth with a correction or vice versa.

Self-check fields (mandatory):
- "a_side_faithful": one or two sentences confirming WHY a_side_rewritten asserts the same factual proposition as original_correct. >= 15 chars. Do NOT include phrases like "actually wrong", "changed the claim", "different proposition", "added a claim".
- "b_side_faithful": one or two sentences confirming WHY b_side_rewritten asserts the same factual proposition as original_incorrect. Same constraints.
- "generator_confidence_in_faithfulness": float in [0,1] indicating your confidence that BOTH paraphrases preserve their respective original propositions exactly. Must be >= 0.70.

Output FORMAT - return ONLY this JSON object, nothing else:
{{
  "a_side_rewritten": "<paraphrase of original_correct, A-side profile>",
  "a_side_faithful":  "<self-check sentence(s)>",
  "b_side_rewritten": "<paraphrase of original_incorrect, B-side profile>",
  "b_side_faithful":  "<self-check sentence(s)>",
  "generator_confidence_in_faithfulness": <float in [0,1]>
}}
"""


# --- Validator -------------------------------------------------------------
_word_re = re.compile(r"\b\w[\w\-']*\b")


def _wc(s: str) -> int:
    return len(_word_re.findall(s or ""))


_OPENER_STEMS: list[str] = [op.rstrip() for op in OPENERS]


def _starts_with_opener(s: str) -> str | None:
    """Word-boundary match against the negation-opener stems. We accept
    "No, humans..." as starting with "No" (the comma is a word boundary,
    same convention as the audit's NEG_LEADS regex in
    scripts/surface_features_text.py). Reject "Note", "Northern", etc."""
    s = (s or "").lstrip()
    for stem in _OPENER_STEMS:
        if not s.lower().startswith(stem.lower()):
            continue
        tail = s[len(stem):]
        if not tail:
            return stem
        nxt = tail[0]
        if not (nxt.isalpha() or nxt == "'"):
            return stem
    return None


def _has_forbidden_conj(s: str) -> str | None:
    low = " " + (s or "").lower() + " "
    for c in FORBIDDEN_CONJUNCTIONS:
        if f" {c} " in low or f", {c} " in low or f"; {c} " in low:
            return c
    return None


_PUNCT_RE = re.compile(r"[,;:\u2014\u2013()]")


def _has_midclause_authority(s: str) -> str | None:
    if not s:
        return None
    text = s.strip()
    parts = _PUNCT_RE.split(text)
    if len(parts) <= 1:
        return None
    for tail in parts[1:]:
        tail_low = tail.strip().lower()
        for cue in AUTH_CUE_PHRASES:
            if tail_low.startswith(cue):
                return cue
    return None


_HEDGE_TOKEN_RES = [
    re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE)
    for tok in HEDGE_TOKENS
]


def _has_hedge_token(s: str) -> str | None:
    for tok, rex in zip(HEDGE_TOKENS, _HEDGE_TOKEN_RES):
        if rex.search(s or ""):
            return tok
    return None


def _starts_with_auth_cue(s: str) -> str | None:
    """Case-insensitive check that the text starts with one of the
    AUTH_CUE_STARTERS phrases. Returns the matched phrase or None."""
    s_low = (s or "").lstrip().lower()
    for cue in AUTH_CUE_STARTERS:
        cue_low = cue.lower()
        if s_low.startswith(cue_low):
            tail = s_low[len(cue_low):]
            if not tail or not tail[0].isalpha():
                return cue
    return None


def detect_b_side_strategy(b_text: str) -> str:
    """Inspect the actual rewritten B-side text and return one of:
        'negation_opener'        - starts with a negation opener, no hedge
        'hedging'                - contains a hedge token, no negation opener
        'authority'              - starts with auth cue, no negation, no hedge
        'negation_and_hedging'   - both
        'none'                   - neither (validator should have rejected)
    """
    has_neg = _starts_with_opener(b_text) is not None
    has_hedge = _has_hedge_token(b_text) is not None
    has_auth = _starts_with_auth_cue(b_text) is not None
    if has_neg and has_hedge:
        return "negation_and_hedging"
    if has_neg:
        return "negation_opener"
    if has_hedge:
        return "hedging"
    if has_auth:
        return "authority"
    return "none"


def _length_drift_warning(label: str, rewritten: str, original: str
                          ) -> str | None:
    rw = _wc(rewritten)
    ow = _wc(original)
    threshold = max(LENGTH_DRIFT_PAD, int(LENGTH_DRIFT_RATIO * ow))
    if abs(rw - ow) > threshold:
        return (f"{label}-side length drift: rewritten={rw} words, "
                f"original={ow} words, |delta|={abs(rw-ow)} > "
                f"threshold={threshold}")
    return None


def _validate_generation(
    rec: dict,
    classification: str = "negation_led",
    original_correct: str | None = None,
    original_incorrect: str | None = None,
) -> tuple[bool, str | None, list[str]]:
    """v3: hard rejects on structure only. The B-side rule branches by
    `classification`:
        'negation_led'        -> B-side MUST start with a negation opener
        'positive_assertion'  -> B-side MUST contain a hedging token OR
                                 start with an authority-cue phrase
    Returns (ok, hard_fail_reason, soft_warnings). Length drift is reported
    as a soft warning, not a hard fail."""
    warnings: list[str] = []

    required = [
        "a_side_rewritten", "a_side_faithful",
        "b_side_rewritten", "b_side_faithful",
        "generator_confidence_in_faithfulness",
    ]
    for k in required:
        if k not in rec:
            return False, f"missing field: {k}", warnings

    a_text = str(rec["a_side_rewritten"]).strip()
    b_text = str(rec["b_side_rewritten"]).strip()
    if not a_text or not b_text:
        return False, "a_side_rewritten or b_side_rewritten empty", warnings

    a_op = _starts_with_opener(a_text)
    if a_op is not None:
        return False, f"A-side starts with negation opener: {a_op!r}", warnings

    if classification == "negation_led":
        b_op = _starts_with_opener(b_text)
        if b_op is None:
            return False, (f"B-side (negation_led route) does not start "
                           f"with any required opener (must be one of "
                           f"{OPENERS})"), warnings
    elif classification == "positive_assertion":
        has_hedge = _has_hedge_token(b_text) is not None
        has_auth = _starts_with_auth_cue(b_text) is not None
        if not (has_hedge or has_auth):
            return False, ("B-side (positive_assertion route) has neither "
                           "a hedging token nor an authority-cue starter; "
                           f"need >=1 of hedges={HEDGE_TOKENS} or auth-"
                           f"starters={AUTH_CUE_STARTERS}"), warnings
    else:
        return False, (f"unknown classification: {classification!r}; "
                       f"expected 'negation_led' or 'positive_assertion'"
                       ), warnings

    a_hedge = _has_hedge_token(a_text)
    if a_hedge is not None:
        return False, (f"A-side contains forbidden hedging token "
                       f"{a_hedge!r}"), warnings

    for label, txt in [("A", a_text), ("B", b_text)]:
        c = _has_forbidden_conj(txt)
        if c:
            return False, (f"{label}-side contains forbidden conjunction: "
                           f"{c!r}"), warnings
        a = _has_midclause_authority(txt)
        if a:
            return False, (f"{label}-side has authority cue {a!r} after "
                           f"mid-sentence punctuation"), warnings

    a_faith = str(rec["a_side_faithful"]).strip()
    b_faith = str(rec["b_side_faithful"]).strip()
    if len(a_faith) < FAITHFUL_MIN_CHARS:
        return False, (f"a_side_faithful too short ({len(a_faith)} chars; "
                       f"need >= {FAITHFUL_MIN_CHARS})"), warnings
    if len(b_faith) < FAITHFUL_MIN_CHARS:
        return False, (f"b_side_faithful too short ({len(b_faith)} chars; "
                       f"need >= {FAITHFUL_MIN_CHARS})"), warnings
    for label, faith in [("a", a_faith), ("b", b_faith)]:
        low = faith.lower()
        for flag in RATIONALE_RED_FLAGS:
            if flag in low:
                return False, (f"{label}_side_faithful contains red-flag "
                               f"phrase: {flag!r}"), warnings

    try:
        conf = float(rec["generator_confidence_in_faithfulness"])
    except Exception:
        return False, ("generator_confidence_in_faithfulness is not a "
                       "parseable float"), warnings
    if conf < CONFIDENCE_MIN:
        return False, (f"generator_confidence_in_faithfulness={conf} "
                       f"below {CONFIDENCE_MIN}"), warnings

    # --- Soft length-drift warnings (no rejection) -------------------------
    if original_correct is not None:
        w = _length_drift_warning("A", a_text, original_correct)
        if w:
            warnings.append(w)
    if original_incorrect is not None:
        w = _length_drift_warning("B", b_text, original_incorrect)
        if w:
            warnings.append(w)

    return True, None, warnings


# --- Anthropic call ladder -------------------------------------------------
def _build_client():
    from anthropic import Anthropic
    return Anthropic(timeout=CLIENT_TIMEOUT_SEC, max_retries=CLIENT_SDK_RETRIES)


def _call_opus(client, prompt: str, effort_kwarg_state: dict) -> tuple[str, float]:
    """Call client.messages.create with the working effort kwarg. Returns
    (raw_text, elapsed_sec). effort_kwarg_state is mutated to remember
    which kwarg shape works after the first call."""
    from anthropic import APIStatusError, APITimeoutError, RateLimitError

    base_kwargs = dict(
        model=MODEL_ID_PRIMARY,
        max_tokens=MAX_TOKENS,
        timeout=CLIENT_TIMEOUT_SEC,
        messages=[{"role": "user", "content": prompt}],
    )

    if not effort_kwarg_state.get("resolved"):
        # Probe once: try effort=medium, then effort=high, then no effort.
        for ladder in (
            {"effort": EFFORT_PREFERRED},
            {"effort": EFFORT_FALLBACK},
            {},
        ):
            t0 = time.time()
            try:
                msg = client.messages.create(**base_kwargs, **ladder)
                effort_kwarg_state["resolved"] = True
                effort_kwarg_state["kwargs"] = ladder
                elapsed = time.time() - t0
                return msg.content[0].text, elapsed
            except (APITimeoutError, RateLimitError):
                raise
            except APIStatusError as e:
                msg_lower = str(e).lower()
                if ("unexpected keyword" in msg_lower
                        or "invalid" in msg_lower
                        or "extra fields" in msg_lower):
                    continue
                raise
            except TypeError:
                continue
        raise RuntimeError("All effort-kwarg shapes rejected by SDK.")
    else:
        ladder = effort_kwarg_state["kwargs"]
        t0 = time.time()
        msg = client.messages.create(**base_kwargs, **ladder)
        return msg.content[0].text, time.time() - t0


# --- IO helpers ------------------------------------------------------------
def _load_selected_pair_ids() -> list[int]:
    if not SELECTED_JSON.exists():
        raise FileNotFoundError(f"Missing {SELECTED_JSON}; run Step 1 first.")
    with open(SELECTED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [int(x) for x in data["pair_ids"]]


def _load_pair_text(pair_id: int) -> tuple[str, str, str]:
    import pandas as pd
    df = pd.read_csv(TRUTHFULQA_CSV)
    row = df.iloc[pair_id]
    return (str(row["Question"]).strip(),
            str(row["Best Answer"]).strip(),
            str(row["Best Incorrect Answer"]).strip())


def _load_existing() -> dict[int, dict]:
    if not OUT_JSON.exists():
        return {}
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}
    return {int(r["pair_id"]): r for r in data}


def _save(records: dict[int, dict]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = sorted(records.values(), key=lambda r: int(r["pair_id"]))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response_to_dict(raw: str) -> dict:
    """Extract the first {...} block and json.loads it."""
    if not raw:
        raise ValueError("empty response")
    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        raise ValueError(f"no JSON object found in response: {raw[:200]!r}")
    return json.loads(m.group(0))


# --- Main loop -------------------------------------------------------------
def _print_classification_breakdown(pair_ids: list[int]) -> None:
    """Diagnostic: print the deterministic v3 classification (negation_led
    vs positive_assertion) for each pair_id, plus the totals."""
    print("\nClassification breakdown (v3 router):")
    print(f"  {'pid':>4}  {'class':<19}  original_incorrect")
    print(f"  {'-'*4}  {'-'*19}  {'-'*60}")
    counts = {"negation_led": 0, "positive_assertion": 0}
    for pid in pair_ids:
        _, best, incorrect = _load_pair_text(pid)
        cls = classify_original_incorrect(best, incorrect)
        counts[cls] += 1
        snippet = incorrect if len(incorrect) <= 60 else incorrect[:57] + "..."
        print(f"  {pid:>4}  {cls:<19}  {snippet}")
    n = len(pair_ids)
    print(f"\n  Totals: negation_led={counts['negation_led']}/{n}  "
          f"positive_assertion={counts['positive_assertion']}/{n}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only generate this many slots from the "
                        "head of the selected_pair_ids list.")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Per-slot validator-failure retry budget.")
    p.add_argument("--print-classification", action="store_true",
                   help="Print the negation_led/positive_assertion split "
                        "for the selected pair_ids and exit (no API calls).")
    args = p.parse_args()

    print("=" * 72)
    print("Step 2/3 - generate_stage0_paired_tqa.py")
    print("=" * 72)

    pair_ids = _load_selected_pair_ids()
    if args.limit is not None:
        pair_ids = pair_ids[:args.limit]
    print(f"Selected pair_ids ({len(pair_ids)}): {pair_ids}")

    if args.print_classification:
        _print_classification_breakdown(pair_ids)
        return 0

    _require_env("ANTHROPIC_API_KEY")

    existing = _load_existing()
    print(f"Existing records on disk: {len(existing)} "
          f"(pair_ids={sorted(existing.keys())})")

    todo = [pid for pid in pair_ids if pid not in existing]
    print(f"To generate this run: {len(todo)} -> {todo}")

    if not todo:
        print("Nothing to do. Exiting.")
        return 0

    client = _build_client()
    effort_state: dict = {}

    n_ok = 0
    n_fail = 0
    elapsed_log: list[tuple[int, float, int, str]] = []

    for pid in todo:
        question, best, incorrect = _load_pair_text(pid)
        classification = classify_original_incorrect(best, incorrect)
        if classification == "negation_led":
            b_rule = B_RULE_NEGATION_LED.format(
                openers_list=", ".join(repr(op.strip()) for op in OPENERS),
                hedge_list=", ".join(repr(h) for h in HEDGE_TOKENS),
            )
        else:
            b_rule = B_RULE_POSITIVE_ASSERTION.format(
                hedge_list=", ".join(repr(h) for h in HEDGE_TOKENS),
                auth_starters=", ".join(repr(a) for a in AUTH_CUE_STARTERS),
            )
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            original_correct=best,
            original_incorrect=incorrect,
            openers_list=", ".join(repr(op.strip()) for op in OPENERS),
            hedge_list=", ".join(repr(h) for h in HEDGE_TOKENS),
            forbidden_conj=", ".join(repr(c) for c in FORBIDDEN_CONJUNCTIONS),
            auth_phrases=", ".join(repr(a) for a in AUTH_CUE_PHRASES),
            b_side_rule=b_rule,
        )

        print(f"\n--- pid={pid} | Q: {question[:80]} ---")
        print(f"   +A: {best}")
        print(f"   -B: {incorrect}")
        print(f"   classification: {classification}")

        last_err: str | None = None
        record: dict | None = None
        for attempt in range(args.max_retries + 1):
            try:
                raw, elapsed = _call_opus(client, prompt, effort_state)
            except Exception as e:
                last_err = f"API error attempt {attempt}: " \
                           f"{type(e).__name__}: {str(e)[:160]}"
                print(f"   {last_err}")
                continue
            tag = ""
            if elapsed > SLOW_CALL_WARN_SEC:
                tag = "  WARN slow"
            print(f"   attempt {attempt}: {elapsed:.1f}s{tag}")
            elapsed_log.append((pid, elapsed, attempt,
                                effort_state.get("kwargs", {}).get("effort", "")))
            try:
                cand = _parse_response_to_dict(raw)
            except Exception as e:
                last_err = f"JSON parse fail: {e}"
                print(f"   {last_err}")
                continue
            ok, why, soft_warnings = _validate_generation(
                cand,
                classification=classification,
                original_correct=best,
                original_incorrect=incorrect,
            )
            if not ok:
                last_err = f"validator: {why}"
                print(f"   {last_err}")
                a_dbg = str(cand.get("a_side_rewritten", ""))[:140]
                b_dbg = str(cand.get("b_side_rewritten", ""))[:140]
                print(f"      A: {a_dbg!r}")
                print(f"      B: {b_dbg!r}")
                continue
            for w in soft_warnings:
                print(f"   WARN soft: {w}")

            b_text = str(cand["b_side_rewritten"]).strip()
            b_strategy = detect_b_side_strategy(b_text)

            record = {
                "pair_id": int(pid),
                "question": question,
                "original_correct": best,
                "original_incorrect": incorrect,
                "original_incorrect_classification": classification,
                "a_side_rewritten": str(cand["a_side_rewritten"]).strip(),
                "a_side_faithful":  str(cand["a_side_faithful"]).strip(),
                "b_side_rewritten": b_text,
                "b_side_faithful":  str(cand["b_side_faithful"]).strip(),
                "b_side_surface_strategy": b_strategy,
                "generator_confidence_in_faithfulness": float(
                    cand["generator_confidence_in_faithfulness"]),
                "model": MODEL_ID_PRIMARY,
                "effort_kwarg": effort_state.get("kwargs", {}).get(
                    "effort", "(none)"),
                "retries": attempt,
                "elapsed_sec_last_attempt": float(elapsed),
                "soft_warnings": soft_warnings,
            }
            break

        if record is None:
            n_fail += 1
            print(f"   FAILED after {args.max_retries+1} attempts; "
                  f"last_err={last_err}")
            continue

        existing[pid] = record
        _save(existing)
        n_ok += 1
        print(f"   OK retries={record['retries']}  "
              f"a_wc={_wc(record['a_side_rewritten'])}  "
              f"b_wc={_wc(record['b_side_rewritten'])}  "
              f"b_strategy={record['b_side_surface_strategy']}  "
              f"conf={record['generator_confidence_in_faithfulness']:.2f}")

    print("\n" + "=" * 72)
    print(f"Run summary: ok={n_ok} fail={n_fail} of {len(todo)} attempted")
    if elapsed_log:
        slow = [(pid, e) for (pid, e, _, _) in elapsed_log
                if e > SLOW_CALL_WARN_SEC]
        print(f"  Slow calls (>{SLOW_CALL_WARN_SEC}s): {len(slow)}")
        for pid, e in slow:
            print(f"    pid={pid}: {e:.1f}s")
        total = sum(e for (_, e, _, _) in elapsed_log)
        n = len(elapsed_log)
        print(f"  Total API time: {total:.1f}s across {n} calls "
              f"(avg {total/n:.1f}s)")
    print(f"  effort_kwarg in use: "
          f"{effort_state.get('kwargs', {}).get('effort', '(none)')}")
    print(f"  Output: {OUT_JSON} ({len(existing)} total records)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in generate_stage0_paired_tqa.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
