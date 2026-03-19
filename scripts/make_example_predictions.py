#!/usr/bin/env python3
"""Generate synthetic demo predictions for §7c (NOT real model outputs)."""
import csv
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
random.seed(42)

# Read style_violation (last column) from audit
with open(ROOT / "audits" / "truthfulqa_style_audit.csv") as f:
    r = csv.reader(f)
    next(r)  # header
    violation = [int(row[-1]) for row in r]
n = len(violation)

pred_dir = ROOT / "data" / "predictions"
pred_dir.mkdir(parents=True, exist_ok=True)
out = pred_dir / "example_model_predictions.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model_name", "pair_id", "correct"])
    for i in range(n):
        p = 0.62 if violation[i] == 1 else 0.52
        w.writerow(["model_A", i, 1 if random.random() < p else 0])
    for i in range(n):
        p = 0.68 if violation[i] == 1 else 0.55
        w.writerow(["model_B", i, 1 if random.random() < p else 0])

print("Wrote", out, "| rows:", 2 * n, "| models: 2")
print("WARNING: This file contains synthetic example predictions for demonstrating the §7c plumbing only.")
print("It is NOT real model output and MUST NOT be interpreted as empirical evidence.")
