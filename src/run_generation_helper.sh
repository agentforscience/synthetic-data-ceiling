#!/bin/bash
# Optional third teacher instance: works backwards through seed 2's jobs on a GPU with spare memory.
# Usage: run_generation_helper.sh GPU [GPU_FRAC]
cd "$(dirname "$0")/.." && source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=$1 python src/generate_synth.py --jobs mbpp:2,roc:2,gsm8k:2 --reverse \
  --n_requests mbpp=7000,gsm8k=5200,roc=5200 --gpu_frac ${2:-0.25} >> logs/gen_C.log 2>&1
