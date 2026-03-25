#!/usr/bin/env bash
# Smoke test: run BoolQ with 2 examples on smallest model.
# Verifies the pipeline works before full CHPC run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs data/predictions/boolq

echo "=== Smoke test BoolQ (2 examples, distilgpt2) ==="
python3 scripts/run_binary_choice_eval.py \
    --dataset boolq \
    --model_name distilgpt2 \
    --max_examples 2 \
    --seed 42 \
    --device cpu \
    --dtype none

echo ""
echo "Smoke test OK. Check: data/predictions/boolq/distilgpt2.csv"
