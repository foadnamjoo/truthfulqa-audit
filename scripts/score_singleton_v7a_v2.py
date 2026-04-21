#!/usr/bin/env python3
"""
v7(a) v2 bilateral singleton scorer.

Identical pipeline to score_singleton_v7a.py, but points at the v2
generation / judge files and writes to a v2 scores file. The §5.1
B-side (stage0/stage0_classifier_scores.json) and the six pickles are
unchanged.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the v1 scorer module and override its three path constants
# before calling main(). Everything else - pickles, encoders, metrics -
# stays identical.
import scripts.score_singleton_v7a as v1  # noqa: E402

OUT_DIR = REPO_ROOT / "stage0_v7a_bilateral"
v1.GEN_JSON = OUT_DIR / "stage0_singleton_v7a_v2_generations.json"
v1.JUDGE_JSON = OUT_DIR / "stage0_singleton_v7a_v2_judge.json"
v1.OUT_JSON = OUT_DIR / "stage0_singleton_v7a_v2_classifier_scores.json"


if __name__ == "__main__":
    try:
        print("(score_singleton_v7a_v2.py: v2 paths)")
        print(f"  GEN   : {v1.GEN_JSON}")
        print(f"  JUDGE : {v1.JUDGE_JSON}")
        print(f"  OUT   : {v1.OUT_JSON}")
        raise SystemExit(v1.main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in score_singleton_v7a_v2.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
