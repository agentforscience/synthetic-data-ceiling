#!/bin/bash
# Start one student fine-tune/eval worker (it picks a GPU with free memory per job).
# Usage: run_worker.sh NAME [FILTER_REGEX]
cd "$(dirname "$0")/../src" && source ../.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python train_eval.py --name $1 --filter "${2:-.*}" 2>&1 \
  | grep --line-buffered -E "^\[" >> ../logs/worker_$1.log
