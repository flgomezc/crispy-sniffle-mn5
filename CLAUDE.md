# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HPC inference pipeline on MareNostrum 5 for evaluating LLMs on sentiment analysis (Rotten Tomatoes dataset, binary: Positive/Negative). See [PRD.md](PRD.md) for full requirements.

## Setup

Copy `.env.example` to `.env` and fill in `MODELS_ROOT`, `SLURM_ACCOUNT`, and `SLURM_QUEUE`. `.env` is git-ignored.

## Commands

```bash
make env   # Create conda environment from environment.yaml
make run   # Allocate SLURM resources and launch inference (requires .env)
```

Manual run inside an active allocation:
```bash
conda run --no-capture-output -n hpc-inference \
    python src/run_inference.py --model $MODELS_ROOT/meta-llama/Llama-3.1-8B-Instruct
```

## Architecture

```
src/
└── run_inference.py          # Entry point: loads dataset → builds prompts → vLLM → writes results
results/                      # Git-ignored
├── <model>_predictions.csv   # Per-sample: id, text, true_label, predicted_label, raw_output
└── <model>_summary.json      # Accuracy, F1, model, split, timestamp
environments/hpc-inference.yaml  # Conda env (no system modules)
Makefile                      # env + run recipes (reads .env)
.env.example                  # Template for local config — copy to .env
```

## HPC Config

- **Cluster:** MareNostrum 5 · **GPU:** NVIDIA H100 × 4, single node
- **Max walltime:** 30 min
- **No system modules** — use only the `hpc-inference` conda env

## Inference Design

- vLLM with tensor parallelism across 4 H100s (`tensor_parallel_size=4`)
- Llama chat template; system prompt constrains output strictly to `"Positive"` or `"Negative"`
- Evaluate on **test split** (1,066 samples) by default
- `--model` argument makes the pipeline model-agnostic for future multi-model runs

## Development workflow
- Implement one feature at a time (see feature list below)
- After each feature: submit via SLURM, capture job ID, monitor, check logs
- Log path: repo/logs/$SLURM_JOB_ID
- Only proceed to next feature when current one is verified working
- Commit after each passing feature

## Feature list
- [ ] Feature 1: create the environment
- [ ] Feature 2: load model
- [ ] Feature 3: run single inference
- [ ] Feature 4: batch inference
- [ ] Feature 5: save outputs