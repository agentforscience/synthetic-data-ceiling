# The Synthetic Data Ceiling

How much model-generated data is worth adding to a small real training set? A stronger model
(Qwen2.5-3B) writes synthetic examples from 100 real ones; a small student (Qwen2.5-0.5B, LoRA) is trained
on 100 real + r×100 synthetic examples for r from 0 to 50 and tested on real held-out data, in three
domains: code (MBPP), math (GSM8K) and creative writing (ROCStories). Full write-up: [REPORT.md](REPORT.md).

## Key findings

- **"80% of the benefit by 3:1" did not hold.** At 3:1 the student had 63% of its maximum gain on code
  (CI 13–100%) and 40% on math (CI 6–64%).
- **No single ceiling.** Code pass@1 saturates near 5:1 and stays flat to 30:1; math accuracy keeps rising to
  50:1 (32.6% → 41.7%); creative writing gets worse with *any* synthetic data.
- **Fidelity to real data degrades past 10:1 in all three domains** (loss on real held-out references),
  slightly for code and math, strongly for stories.
- **Generator mode collapse is severe.** 44% of 3,000 synthetic code tasks are exact duplicates; 24% of
  synthetic math problems start with "Sarah". The student inherits these modes.
- **Real data is worth far more for creative writing:** 100 extra real stories helped; 5,000 synthetic ones
  hurt. For math accuracy the opposite was true (synthetic beat real at equal size by 6.5 points at 30:1).

Caveats: one teacher, one student, 3 seeds, 500 test items; accuracy differences under ~4 points are noise.

## Reproduce

```bash
uv venv && source .venv/bin/activate && uv sync      # dependencies are pinned in pyproject.toml / uv.lock
./src/run_generation.sh GPU_A GPU_B [GPU_FRACTION]    # teacher writes the synthetic pools (~35 min)
./src/run_worker.sh w0 &                              # start one or more fine-tune/eval workers;
./src/run_worker.sh w1 &                              #   each needs ~9 GB of GPU memory
cd src && python diversity.py && python analyze.py    # statistics, tables, figures (CPU)
python teacher_eval.py                                # optional: teacher's own test scores
```

Workers share a job queue (`results/locks/`), pick a GPU with free memory per job, and resume after
interruption. Seeds 0–2; all hyperparameters are constants at the top of `src/common.py` and
`src/train_eval.py`. Re-running individual jobs reproduced the logged loss and accuracy exactly.
MBPP is read from a machine-specific offline cache; see `datasets/README.md` to change that.

## Files

| Path | Contents |
|---|---|
| `REPORT.md` | Full report |
| `planning.md` | Pre-registered plan and the logged deviations |
| `resources.md`, `datasets/README.md` | Models, datasets, prior work |
| `src/common.py` | Configuration, data splits, formats |
| `src/generate_synth.py` | Teacher generation, parsing, execution filter |
| `src/train_eval.py` | LoRA fine-tuning, early stopping, evaluation, job queue |
| `src/exec_eval.py` | Subprocess harness for running generated code |
| `src/diversity.py`, `src/analyze.py`, `src/teacher_eval.py` | Analyses and figures |
| `results/runs.jsonl` | One line per run (117 runs) |
| `results/model_outputs/` | Per-item outputs and losses for every run |
| `results/synthetic_pools/` | The generated data (≈52k teacher outputs) |
| `results/tables.md`, `results/analysis.json`, `results/diversity.json`, `results/teacher_baseline.json` | Tables and statistics |
| `figures/` | `dose_response`, `normalised_benefit`, `curve_fits`, `pool_diversity`, `creative_generation` |
