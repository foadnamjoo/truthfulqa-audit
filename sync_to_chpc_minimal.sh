#!/bin/bash
# Minimal sync for BoolQ run on CHPC. Excludes .git, model caches, old sweeps.
# Run from your Mac: bash sync_to_chpc_minimal.sh
set -e
cd "$(dirname "$0")/.."
REPO="truthfulqa_audit"
REMOTE="${1:-u1419668@notchpeak.chpc.utah.edu}"

rsync -avz --progress \
  --exclude='.git' \
  --exclude='.DS_Store' \
  --exclude='chpc_runs/*/' \
  --exclude='*.ipynb' \
  --exclude='audits/' \
  --exclude='data/' \
  --exclude='logs/' \
  --exclude='paper_assets/' \
  --exclude='docs/' \
  --exclude='**/hf_*/' \
  --exclude='**/transformers/' \
  --exclude='**/*.safetensors' \
  "$REPO/" "$REMOTE:~/$REPO/"

echo ""
echo "Done. On CHPC: cd ~/$REPO && sbatch chpc_runs/run_boolq_batch.slurm"
