# Datasets

Nothing here is committed; `hf_cache/` is a Hugging Face `datasets` cache (≈45 MB) recreated on first use
by `src/common.py` (it sets `HF_DATASETS_CACHE` to this directory).

| Task | Hub id | Split use |
|---|---|---|
| Code | `google-research-datasets/mbpp` (config `full`) | train 374 → real seed sets / control; validation 90 → early stopping; test 500 → evaluation |
| Math | `openai/gsm8k` (config `main`) | train 7,473 → 200 validation (fixed shuffle, seed 1234) + 7,273 pool; test 1,319 (first 500 for generation metrics) |
| Creative | `igormorgado/ROCStories2018` | only `train` (52,665 titled five-sentence stories) is used: 200 validation, 1,000 test, rest = pool |

```python
from datasets import load_dataset
load_dataset("google-research-datasets/mbpp", "full")
load_dataset("openai/gsm8k", "main")
load_dataset("igormorgado/ROCStories2018")
```

On the machine used for the study MBPP was read from a shared offline cache (`/data/transformers/datasets`,
hard-coded in `load_real`); elsewhere, remove the `cache_dir=` argument and the offline flag there.

Model-generated data (the synthetic pools) is an experiment output and lives in `results/synthetic_pools/`.
