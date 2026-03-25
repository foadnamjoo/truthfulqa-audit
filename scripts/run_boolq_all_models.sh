#!/usr/bin/env bash
#
# Run BoolQ binary-choice evaluation for all 9 models.
# Use with CHPC SLURM or run directly: ./scripts/run_boolq_all_models.sh
#
# On clusters where python3 has no PyTorch, set the venv interpreter, e.g.:
#   export PYTHON="$HOME/venvs/boolq310/bin/python"
#
# Outputs: data/predictions/boolq/<model_safe>.csv
# Logs: logs/boolq_<model_safe>_*.log (if LOG_DIR set)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
mkdir -p "$LOG_DIR"
mkdir -p "$REPO_ROOT/data/predictions/boolq"

MODELS=(
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "microsoft/Phi-3.5-mini-instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    "EleutherAI/pythia-2.8b-deduped"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "distilgpt2"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

MAX_EXAMPLES="${MAX_EXAMPLES:-}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
PYTHON="${PYTHON:-python3}"
# Many HPC login/interactive nodes expose python3 without torch; prefer a known venv.
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    for candidate in "$HOME/venvs/boolq310/bin/python" "$HOME/venvs/boolq/bin/python"; do
        if [[ -x "$candidate" ]] && "$candidate" -c "import torch" 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
fi

EXTRA_ARGS=()
[[ -n "$MAX_EXAMPLES" ]] && EXTRA_ARGS+=(--max_examples "$MAX_EXAMPLES")

echo "=== BoolQ binary-choice eval: $(date) ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "PYTHON=$PYTHON"
echo "Models: ${#MODELS[@]}"
echo ""

for model in "${MODELS[@]}"; do
    safe="${model//\//__}"
    log="$LOG_DIR/boolq_${safe}.log"
    echo "[$(date)] Running $model ..."
    if "$PYTHON" scripts/run_binary_choice_eval.py \
        --dataset boolq \
        --model_name "$model" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$log"; then
        echo "[$(date)] OK $model"
    else
        echo "[$(date)] FAILED $model" >&2
        exit 1
    fi
done

echo ""
echo "=== Done $(date) ==="
