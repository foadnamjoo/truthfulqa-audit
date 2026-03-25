# BoolQ Binary-Choice Evaluation on CHPC (H100)

Run 9 models on BoolQ validation using the same A/B prompt format as TruthfulQA.

## Prerequisites

- Repo cloned on CHPC (or synced)
- Conda/venv with: `transformers`, `torch`, `datasets`
- SLURM account and H100 partition access

## One-Time Setup

```bash
cd /path/to/truthfulqa_audit
mkdir -p logs data/predictions/boolq
```

Edit `chpc_runs/run_boolq_batch.slurm`:
- Set `#SBATCH --account=YOUR_ACCOUNT`
- Adjust `#SBATCH --partition=` if needed (e.g. `kingspeak`)

## Smoke Test (Before Overnight Run)

```bash
cd /path/to/truthfulqa_audit
bash scripts/smoke_test_boolq.sh
```

Runs 2 examples on `distilgpt2` on CPU. Should complete in ~1 min.

## Launch Overnight Run

From repo root:

```bash
cd /path/to/truthfulqa_audit
sbatch chpc_runs/run_boolq_batch.slurm
```

Or with explicit paths:

```bash
cd /path/to/truthfulqa_audit
REPO_ROOT=$(pwd) sbatch chpc_runs/run_boolq_batch.slurm
```

## Output Paths

| Path | Contents |
|------|----------|
| `data/predictions/boolq/Qwen__Qwen2.5-14B-Instruct.csv` | Predictions for that model |
| `logs/boolq_Qwen__Qwen2.5-14B-Instruct.log` | Per-model log |
| `logs/boolq_batch_<jobid>.out` | SLURM stdout |
| `logs/boolq_batch_<jobid>.err` | SLURM stderr |

## Models Run (in order)

1. Qwen/Qwen2.5-14B-Instruct
2. Qwen/Qwen2.5-1.5B-Instruct
3. microsoft/Phi-3.5-mini-instruct
4. mistralai/Mistral-7B-Instruct-v0.2
5. HuggingFaceTB/SmolLM2-1.7B-Instruct
6. EleutherAI/pythia-2.8b-deduped
7. Qwen/Qwen2.5-0.5B-Instruct
8. distilgpt2
9. TinyLlama/TinyLlama-1.1B-Chat-v1.0

## Single-Model Command (Manual Run)

```bash
python3 scripts/run_binary_choice_eval.py \
  --dataset boolq \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --device cuda \
  --dtype float16 \
  --seed 42
```

Output: `data/predictions/boolq/mistralai__Mistral-7B-Instruct-v0.2.csv`
