#!/bin/bash
#SBATCH --job-name=hpc-inference
#SBATCH -A bsc100
#SBATCH -q acc_debug
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/.env"

echo "[$(date '+%H:%M:%S')] === Feature 5: Batch inference + save outputs ==="
echo "Model: $MODELS_ROOT/meta-llama/Llama-3.1-8B-Instruct"

conda run --no-capture-output -n hpc-inference \
    python src/run_inference.py \
        --model "$MODELS_ROOT/meta-llama/Llama-3.1-8B-Instruct" \
        --tensor-parallel-size 2

echo "[$(date '+%H:%M:%S')] === DONE ==="
