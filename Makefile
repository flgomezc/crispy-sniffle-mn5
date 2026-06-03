-include .env

ENV_NAME    := hpc-inference
ENV_FILE    := environments/hpc-inference.yaml
MODEL_NAME  := meta-llama/Llama-3.1-8B-Instruct
MODEL_PATH  := $(MODELS_ROOT)/$(MODEL_NAME)
SLURM_FLAGS := -A $(SLURM_ACCOUNT) -q $(SLURM_QUEUE) --time=00:30:00 --gres=gpu:4 --cpus-per-task=80

.PHONY: env run

env:
	conda env create -f $(ENV_FILE)

run:
ifndef MODELS_ROOT
	$(error MODELS_ROOT is not set. Copy .env.example to .env and fill in the values.)
endif
	salloc $(SLURM_FLAGS) \
		srun conda run --no-capture-output -n $(ENV_NAME) \
		python src/run_inference.py --model $(MODEL_PATH)
