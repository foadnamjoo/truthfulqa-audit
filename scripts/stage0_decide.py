#!/usr/bin/env python3
"""
Step 12 - Stage-0 success-criteria evaluator.

Reads the three machine-produced artifacts (stage0_generations.json,
stage0_classifier_scores.json, stage0_judge.json) and applies the
five locked success criteria plus the two directional-gap flags.
Emits a per-criterion PASS/FAIL/PENDING verdict and an overall
decision (proceed to Stage 1, or pause).

Criteria (from the original plan, Modification 4):

  1. Cue fidelity                                 HUMAN REVIEW
     Each statement exhibits the intended surface cues (sentence-initial
     negation at minimum). Requires reading the statements; the
     programmatic check here is a lightweight regex pre-check that only
     flags gross violations; it does NOT replace the human cue-presence
     column in stage0_human_validation.csv.

  2. Untruthfulness (human)                       HUMAN REVIEW
     A knowledgeable reviewer agrees the statement is factually false.
     Fills `my_untruthfulness_check` in the CSV. Not evaluable here.

  3. Untruthfulness (judge)                       PROGRAMMATIC
     >= 18/20 judged_false=True AND disagreement_flag count <= 2.

  4. Classifier vulnerability (primary)           PROGRAMMATIC
     mean(surface_lr_full_P_truthful) >= 0.55.

  5. Classifier vulnerability (secondary)         PROGRAMMATIC
     count(surface_lr_full_P_truthful > 0.5) >= 12/20.

Additional directional flags (informational, not criteria per se):

  A. Surface full-vs-cleaned gap
     mean(surface_lr_full) - mean(surface_lr_cleaned) >= 0.05 -> PASS flag
  B. Embedding full-vs-cleaned gap
     mean(embedding_lr_full) - mean(embedding_lr_cleaned) >= 0.05 -> PASS flag

Overall decision:
  - PROCEED if criteria 3, 4, 5 all PASS (criteria 1 and 2 reported PENDING)
  - PAUSE  if any of 3/4/5 FAILS
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE0_DIR = REPO_ROOT / "stage0"
GEN_JSON = STAGE0_DIR / "stage0_generations.json"
SCORES_JSON = STAGE0_DIR / "stage0_classifier_scores.json"
JUDGE_JSON = STAGE0_DIR / "stage0_judge.json"

EXPECTED_N = 20

# Thresholds from Modification 4 of the original plan.
JUDGE_FALSE_MIN = 18         # out of 20
JUDGE_DISAGREE_MAX = 2       # out of 20
CLF_MEAN_MIN = 0.55          # criterion 4
CLF_FOOLED_COUNT_MIN = 12    # criterion 5 (P > 0.5)
DIRECTIONAL_GAP_MIN = 0.05   # flags A and B


def _load(path: Path, label: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != EXPECTED_N:
        raise RuntimeError(
            f"{label}: expected {EXPECTED_N} records, got "
            f"{len(data) if isinstance(data, list) else type(data).__name__}"
        )
    return sorted(data, key=lambda r: int(r["id"]))


def main() -> int:
    print("=" * 72)
    print("STEP 12 - stage0_decide.py")
    print("=" * 72)
    gens = _load(GEN_JSON, "generations")
    scores = _load(SCORES_JSON, "classifier_scores")
    judge = _load(JUDGE_JSON, "judge")

    # --- Criterion 3: judge untruthfulness -----------------------------------
    n_judged_false = sum(1 for r in judge if bool(r["judged_false"]))
    n_disagree = sum(1 for r in judge if bool(r["judge_disagreement_flag"]))
    c3_pass = (n_judged_false >= JUDGE_FALSE_MIN
               and n_disagree <= JUDGE_DISAGREE_MAX)

    # --- Criterion 4: classifier vulnerability primary -----------------------
    p_surf_full = [float(r["surface_lr_full_P_truthful"]) for r in scores]
    mean_surf_full = mean(p_surf_full)
    c4_pass = mean_surf_full >= CLF_MEAN_MIN

    # --- Criterion 5: classifier vulnerability secondary ---------------------
    n_surf_fooled = sum(1 for p in p_surf_full if p > 0.5)
    c5_pass = n_surf_fooled >= CLF_FOOLED_COUNT_MIN

    # --- Flags A and B: directional full - cleaned gaps ---------------------
    p_surf_clean = [float(r["surface_lr_cleaned_P_truthful"]) for r in scores]
    p_emb_full = [float(r["embedding_lr_full_P_truthful"]) for r in scores]
    p_emb_clean = [float(r["embedding_lr_cleaned_P_truthful"]) for r in scores]
    gap_surf = mean(p_surf_full) - mean(p_surf_clean)
    gap_emb = mean(p_emb_full) - mean(p_emb_clean)
    flagA_pass = gap_surf >= DIRECTIONAL_GAP_MIN
    flagB_pass = gap_emb >= DIRECTIONAL_GAP_MIN

    # --- Criterion 1: lightweight cue-fidelity pre-check --------------------
    # Does not replace the human review column. Only flags statements whose
    # intended_cues list claims sentence_initial_negation but whose text
    # does not begin with one of the allowed opener prefixes. Reported so
    # that the human reviewer has a head start.
    opener_starts = (
        "not ", "never ", "no ", "contrary to ", "it is not the case ",
        "there is no ", "nothing ",
    )
    c1_prescreen: list[int] = []
    for g in gens:
        cues = g.get("intended_cues") or []
        stmt = str(g["statement"]).strip().lower()
        if "sentence_initial_negation" in cues:
            if not any(stmt.startswith(pfx) for pfx in opener_starts):
                c1_prescreen.append(int(g["id"]))

    # --- Report -------------------------------------------------------------
    def _verdict(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    print("\nPer-criterion verdicts:")
    print("-" * 72)

    print("  1. Cue fidelity (intended surface cues present)           "
          "PENDING HUMAN REVIEW")
    if c1_prescreen:
        print(f"     prescreen: {len(c1_prescreen)} slot(s) claim "
              f"sentence_initial_negation but do NOT start with an "
              f"allowed opener: {c1_prescreen}")
    else:
        print("     prescreen: all 20 statements begin with an allowed "
              "negation opener (no gross violations)")

    print()
    print("  2. Untruthfulness (human reviewer agrees false)           "
          "PENDING HUMAN REVIEW")
    print("     (will be filled in `my_untruthfulness_check` column of "
          "stage0_human_validation.csv)")

    print()
    print(f"  3. Untruthfulness (judge): >=18/20 judged_false AND "
          f"<=2 disagreements   {_verdict(c3_pass)}")
    print(f"     judged_false=True:       {n_judged_false} / {EXPECTED_N}  "
          f"(threshold >= {JUDGE_FALSE_MIN})")
    print(f"     judge_disagreement_flag: {n_disagree} / {EXPECTED_N}  "
          f"(threshold <= {JUDGE_DISAGREE_MAX})")

    print()
    print(f"  4. Classifier vulnerability primary: "
          f"mean(surface_lr_full_P_truthful) >= {CLF_MEAN_MIN}   "
          f"{_verdict(c4_pass)}")
    print(f"     observed mean = {mean_surf_full:.4f}")

    print()
    print(f"  5. Classifier vulnerability secondary: "
          f"#(surface_lr_full_P_truthful > 0.5) >= {CLF_FOOLED_COUNT_MIN}/20   "
          f"{_verdict(c5_pass)}")
    print(f"     observed count = {n_surf_fooled} / {EXPECTED_N}")

    print()
    print("Directional gap flags (informational):")
    print("-" * 72)
    print(f"  A. Surface directional gap (full - cleaned) >= "
          f"{DIRECTIONAL_GAP_MIN:.2f}        FLAG {_verdict(flagA_pass)}")
    print(f"     mean(surface_lr_full)    = {mean(p_surf_full):.4f}")
    print(f"     mean(surface_lr_cleaned) = {mean(p_surf_clean):.4f}")
    print(f"     gap                       = {gap_surf:+.4f}")
    print()
    print(f"  B. Embedding directional gap (full - cleaned) >= "
          f"{DIRECTIONAL_GAP_MIN:.2f}      FLAG {_verdict(flagB_pass)}")
    print(f"     mean(embedding_lr_full)    = {mean(p_emb_full):.4f}")
    print(f"     mean(embedding_lr_cleaned) = {mean(p_emb_clean):.4f}")
    print(f"     gap                        = {gap_emb:+.4f}")

    # --- Overall verdict ----------------------------------------------------
    programmatic_pass = c3_pass and c4_pass and c5_pass
    print()
    print("=" * 72)
    print("OVERALL DECISION")
    print("=" * 72)
    if programmatic_pass:
        print("  Programmatic criteria 3, 4, 5: ALL PASS")
        print("  Criteria 1 and 2 (cue fidelity, human untruthfulness): "
              "PENDING HUMAN REVIEW in stage0_human_validation.csv")
        print()
        print("  -> PROCEED to Stage 1, conditional on human CSV review "
              "confirming criteria 1 and 2.")
        return 0
    else:
        print("  Programmatic criteria 3, 4, 5: at least one FAIL")
        print(f"     c3 (judge):        {_verdict(c3_pass)}")
        print(f"     c4 (mean P>0.55):  {_verdict(c4_pass)}")
        print(f"     c5 (count P>0.5):  {_verdict(c5_pass)}")
        print()
        print("  -> PAUSE. Do not proceed to Stage 1 until failed "
              "criterion is investigated.")
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in stage0_decide.py:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
