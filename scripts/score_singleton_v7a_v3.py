#!/usr/bin/env python3
"""
v7(a) v3 bilateral singleton scorer.

Identical pipeline to score_singleton_v7a.py; only the paths change.
The §5.1 B-side and the six pickles are unchanged.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.score_singleton_v7a as v1  # noqa: E402

OUT_DIR = REPO_ROOT / "stage0_v7a_bilateral"
v1.GEN_JSON = OUT_DIR / "stage0_singleton_v7a_v3_generations.json"
v1.JUDGE_JSON = OUT_DIR / "stage0_singleton_v7a_v3_judge.json"
v1.OUT_JSON = OUT_DIR / "stage0_singleton_v7a_v3_classifier_scores.json"


if __name__ == "__main__":
    try:
        print("(score_singleton_v7a_v3.py: v3 paths)")
        print(f"  GEN   : {v1.GEN_JSON}")
        print(f"  JUDGE : {v1.JUDGE_JSON}")
        print(f"  OUT   : {v1.OUT_JSON}")
        raise SystemExit(v1.main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in score_singleton_v7a_v3.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
