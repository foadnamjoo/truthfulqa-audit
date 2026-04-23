#!/usr/bin/env python3
"""
Pretty-print the 20 v8 bilateral cue-inversion pairs for human review.

Also writes a CSV template ready for human-eval ratings.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRS = REPO_ROOT / "stage0_v8_inverted_bilateral" / "v8_bilateral_inverted.jsonl"
OUT_CSV = REPO_ROOT / "stage0_v8_inverted_bilateral" / "v8_pairs_human_eval_template.csv"


def load_pairs():
    pairs = []
    with open(PAIRS) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def main():
    pairs = load_pairs()
    print("=" * 96)
    print(f"v8 bilateral cue-inversion pairs, n={len(pairs)}")
    print("A-side  = TRUE statement dressed in FALSE-class surface cues (short, no negation/hedge)")
    print("B-side  = FALSE statement dressed in TRUE-class surface cues (long, negation/hedge)")
    print("=" * 96)
    for p in pairs:
        a = p["A_side"]
        b = p["B_side"]
        topic_a = a.get("topic", "?")
        topic_b = b.get("topic", "?")
        same = "same" if topic_a == topic_b else "diff"
        print()
        print(f"[{p['pair_id']:>2}]  topic A={topic_a!r:<22}  B={topic_b!r:<22}  ({same})  "
              f"wc: A={a['surface_features']['word_count']}, B={b['surface_features']['word_count']}")
        print(f"   A (TRUE):  {a['statement']}")
        print(f"   B (FALSE): {b['statement']}")

    # CSV template for human ratings
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pair_id", "A_topic", "B_topic", "same_topic",
            "A_truth_label", "A_statement",
            "B_truth_label", "B_statement",
            # --- to fill in by human ---
            "A_factually_true_yesno",
            "A_surface_looks_false_1to5",   # 1 = looks TRUE, 5 = looks FALSE
            "B_factually_false_yesno",
            "B_surface_looks_true_1to5",    # 1 = looks FALSE, 5 = looks TRUE
            "pair_is_clean_inversion_1to5", # overall quality of the cue-inversion, 1..5
            "notes",
        ])
        for p in pairs:
            a = p["A_side"]; b = p["B_side"]
            w.writerow([
                p["pair_id"], a.get("topic"), b.get("topic"),
                a.get("topic") == b.get("topic"),
                "TRUE", a["statement"],
                "FALSE", b["statement"],
                "", "", "", "", "", "",
            ])
    print()
    print("=" * 96)
    print(f"Wrote human-eval template: {OUT_CSV.relative_to(REPO_ROOT)}")
    print("Rating scale suggestions:")
    print("  A_surface_looks_false_1to5  : 1 = reads as TRUE (caveats, hedged), 5 = reads as FALSE (bald, too confident to be true)")
    print("  B_surface_looks_true_1to5   : 1 = reads as FALSE, 5 = reads as TRUE (qualified, authoritative)")
    print("  pair_is_clean_inversion_1to5: 1 = broken, 5 = perfect inversion suitable for adversarial probe")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
