# Product Requirements Document: HPC Inference Pipeline

## Overview

An HPC inference pipeline running on MareNostrum 5 to evaluate LLM performance on sentiment analysis. The pipeline runs batch inference over the Rotten Tomatoes dataset and produces per-sample predictions and classification metrics.

## Dataset

- **Source:** `cornell-movie-review-data/rotten_tomatoes` (HuggingFace)
- **Splits:** train (8,530) · validation (1,066) · test (1,066)
- **Labels:** 1 = Positive, 0 = Negative — balanced (5,331 each across full corpus)
- **Default evaluation split:** test (1,066 samples)

## Task

Binary sentiment classification: given a movie review text, predict **"Positive"** or **"Negative"**. No neutral class.

## Model & Framework

- **Initial model:** `meta-llama/Llama-3.1-8B-Instruct` (loaded from local models directory via `$MODELS_ROOT`)
- **Framework:** vLLM
- **Design:** model-agnostic — `--model` CLI argument selects the model path, enabling multi-model evaluation runs.

## Prompt Design

Llama chat template with a system prompt that strictly constrains output:

```
System: You are a sentiment classifier. Given a movie review, classify it as exactly
        one of: "Positive" or "Negative". Respond with only that single word.
        No explanations, no neutral answers.
User:   <review text>
```

## HPC Configuration

- **Cluster:** MareNostrum 5
- **GPU:** NVIDIA H100 × 4 (single node)
- **SLURM allocation:** `salloc -A $SLURM_ACCOUNT -q $SLURM_QUEUE --time=02:15:00 --gres=gpu:4 --cpus-per-task=80`
- **Environment:** conda (see `environment.yaml`) — no system modules loaded
- **Local config:** copy `.env.example` to `.env` and set `MODELS_ROOT`, `SLURM_ACCOUNT`, `SLURM_QUEUE`

## Outputs

| Output | Format | Path |
|--------|--------|------|
| Per-sample predictions | CSV | `results/<model_name>_predictions.csv` |
| Summary report | JSON | `results/<model_name>_summary.json` |

**CSV columns:** `id`, `text`, `true_label`, `predicted_label`, `raw_output`

**Summary fields:** `model`, `dataset_split`, `n_samples`, `accuracy`, `f1`, `timestamp`

## Success Criteria

- Metrics: **Accuracy** and **F1** (binary, positive class)
- Exploratory phase — no target threshold yet
- Pipeline must run end-to-end on the test split within the 2h15m SLURM allocation

## Future Scope

- Multi-model evaluation loop (iterate over models in `$MODELS_ROOT`)
- Prompt ablation studies
- Support for validation split
