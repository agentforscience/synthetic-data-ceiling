#!/bin/bash
# Generate all synthetic pools with two teacher instances.
# Usage: run_generation.sh GPU_A GPU_B [GPU_FRAC]
# The GPUs are shared with other jobs, so each instance reserves only GPU_FRAC of the card and is
# restarted (generation is resumable) if it dies, e.g. from an out-of-memory error.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1   # system nvcc is too old for FlashInfer's JIT
FRAC=${3:-0.30}
NREQ=mbpp=7000,gsm8k=5200,roc=5200
launch () {  # gpu, jobs, logfile
  for try in 1 2 3 4 5 6; do
    CUDA_VISIBLE_DEVICES=$1 python src/generate_synth.py --jobs $2 --n_requests $NREQ --gpu_frac $FRAC >> $3 2>&1 && return 0
    echo "[launcher] attempt $try failed on GPU $1; retrying in 60s" >> $3; sleep 60
  done; return 1
}
launch $1 gsm8k:0,roc:0,mbpp:0,gsm8k:2 logs/gen_A.log &
launch $2 gsm8k:1,roc:1,mbpp:1,roc:2,mbpp:2 logs/gen_B.log &
wait
