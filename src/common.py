"""Shared configuration, data loading, splits and formatting for all experiments.

Every example, real or synthetic, is normalised to a dict with at least
    {"prompt": <user turn>, "answer": <assistant turn>}
plus task-specific fields used for evaluation (tests, gold answer, title...).
"""
import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "datasets" / "hf_cache"))

TEACHER = "Qwen/Qwen2.5-3B-Instruct"   # 7B left <1 GB KV cache on the shared GPUs (see planning.md deviations)
STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"

TASKS = ["mbpp", "gsm8k", "roc"]
TASK_LABEL = {"mbpp": "Code (MBPP)", "gsm8k": "Math (GSM8K)", "roc": "Creative (ROCStories)"}
SEEDS = [0, 1, 2]
N_REAL = 100                                  # size of the real seed set
RATIOS = [0, 1, 2, 3, 5, 10, 20, 30]          # synthetic : real (core grid, all tasks)
EXTRA_RATIOS = {"gsm8k": [50], "roc": [50], "mbpp": []}   # run last; MBPP pools are too small for 50:1
SPLIT_SEED = 1234                             # fixed seed for val/test carving
N_VAL = {"mbpp": 90, "gsm8k": 200, "roc": 200}
N_TEST = {"mbpp": 500, "gsm8k": 1319, "roc": 1000}
N_GEN_TEST = {"mbpp": 500, "gsm8k": 500, "roc": 500}    # items used for generation-based metrics
POOL_REQUESTS = {"mbpp": 7000, "gsm8k": 5200, "roc": 5200}   # teacher requests per (task, seed) pool
MAX_NEW_TOKENS = {"mbpp": 256, "gsm8k": 256, "roc": 96}
POOL_DIR = ROOT / "results" / "synthetic_pools"
RUNS_FILE = ROOT / "results" / "runs.jsonl"
OUT_DIR = ROOT / "results" / "model_outputs"


# --------------------------------------------------------------------------- formatting
def clean_code(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").strip()


def mbpp_prompt(text: str, tests: list) -> str:
    return (f"{text.strip()}\nYour code should pass these tests:\n\n" + "\n".join(tests)
            + "\n\nRespond with a single Python code block.")


def mbpp_answer(code: str) -> str:
    return f"```python\n{clean_code(code)}\n```"


def gsm_clean_solution(ans: str) -> str:
    """Strip GSM8K calculator annotations <<a=b>> so real and synthetic share one format."""
    return re.sub(r"<<.*?>>", "", ans).strip()


def gsm_gold(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


def gsm_extract(pred: str):
    """Number after the last '####'; None if the format is not followed."""
    if "####" not in pred:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", pred.split("####")[-1])
    return m.group(0).replace(",", "").rstrip(".") if m else None


def gsm_equal(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (TypeError, ValueError):
        return False


def roc_prompt(title: str) -> str:
    return f"Write a five-sentence short story with the title: {title.strip()}"


def make_example(task: str, raw: dict) -> dict:
    """Convert a raw dataset row into the unified example format."""
    if task == "mbpp":
        return {"prompt": mbpp_prompt(raw["text"], raw["test_list"]), "answer": mbpp_answer(raw["code"]),
                "text": raw["text"].strip(), "code": clean_code(raw["code"]), "tests": list(raw["test_list"]),
                "setup": raw.get("test_setup_code", "") or "", "id": raw.get("task_id")}
    if task == "gsm8k":
        sol = gsm_clean_solution(raw["answer"])
        return {"prompt": raw["question"].strip(), "answer": sol, "question": raw["question"].strip(),
                "gold": gsm_gold(raw["answer"])}
    if task == "roc":
        story = " ".join(raw[f"sentence{i}"].strip() for i in range(1, 6))
        return {"prompt": roc_prompt(raw["storytitle"]), "answer": story, "title": raw["storytitle"].strip(),
                "story": story}
    raise ValueError(task)


# --------------------------------------------------------------------------- real data + splits
_CACHE = {}


def load_real(task: str) -> dict:
    """Return {"train": [...], "val": [...], "test": [...]} of unified examples (fixed, seed-independent)."""
    if task in _CACHE:
        return _CACHE[task]
    from datasets import load_dataset
    rng = random.Random(SPLIT_SEED)
    if task == "mbpp":
        os.environ["HF_DATASETS_OFFLINE"] = "1"   # cached copy lives in the shared HF cache
        ds = load_dataset("google-research-datasets/mbpp", "full", cache_dir="/data/transformers/datasets")
        out = {"train": list(ds["train"]), "val": list(ds["validation"]), "test": list(ds["test"])}
    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        tr = list(ds["train"])
        rng.shuffle(tr)
        out = {"val": tr[:N_VAL[task]], "train": tr[N_VAL[task]:], "test": list(ds["test"])}
    elif task == "roc":
        ds = load_dataset("igormorgado/ROCStories2018")
        # Only the train split carries titles (val/test are cloze-format), so carve val/test from it.
        ok = lambda r: r["storytitle"] and all(r[f"sentence{i}"] for i in range(1, 6))
        tr = [r for r in ds["train"] if ok(r)]
        rng.shuffle(tr)
        nv, nt = N_VAL[task], N_TEST[task]
        held = {" ".join(r[f"sentence{i}"].strip() for i in range(1, 6)) for r in tr[:nv + nt]}
        out = {"val": tr[:nv], "test": tr[nv:nv + nt],   # drop the few exact duplicates of held-out stories
               "train": [r for r in tr[nv + nt:]
                         if " ".join(r[f"sentence{i}"].strip() for i in range(1, 6)) not in held]}
    else:
        raise ValueError(task)
    out = {k: [make_example(task, r) for r in v] for k, v in out.items()}
    _CACHE[task] = out
    return out


def real_order(task: str, seed: int) -> list:
    """Seed-specific permutation of the real training pool. The first N_REAL items are the real seed
    set; longer prefixes are the nested all-real control sets."""
    tr = load_real(task)["train"]
    idx = list(range(len(tr)))
    random.Random(10_000 + seed).shuffle(idx)
    return [tr[i] for i in idx]


def real_seed_set(task: str, seed: int) -> list:
    return real_order(task, seed)[:N_REAL]


def pool_path(task: str, seed: int) -> Path:
    return POOL_DIR / f"{task}_seed{seed}.jsonl"


def load_pool(task: str, seed: int, only_valid: bool = True) -> list:
    """Synthetic pool in generation order (order matters: nested subsets are prefixes)."""
    by_req = {}
    for l in open(pool_path(task, seed)):   # keep the first copy if two generator instances overlapped
        r = json.loads(l)
        by_req.setdefault(r["req"], r)
    # A pool is usable only once every request is present: nested subsets are prefixes in request order.
    if len(by_req) < POOL_REQUESTS[task]:
        raise RuntimeError(f"pool {task} seed {seed} incomplete: {len(by_req)}/{POOL_REQUESTS[task]}")
    rows = [by_req[k] for k in sorted(by_req)]
    if only_valid:
        rows = [r for r in rows if r.get("valid", True)]
    return rows


def train_set(task: str, seed: int, cond: str, ratio: float, filtered: bool = True) -> list:
    """Training set for a condition.
    cond == "synth": N_REAL real + round(ratio*N_REAL) synthetic (prefix of the pool).
    cond == "real":  N_REAL*(1+ratio) real examples (all-real control at matched size).
    """
    n_extra = int(round(ratio * N_REAL))
    order = real_order(task, seed)
    if cond == "real":
        assert N_REAL + n_extra <= len(order), "not enough real data"
        return order[:N_REAL + n_extra]
    if cond in ("synth", "synth_unfiltered"):
        if n_extra == 0:
            return order[:N_REAL]
        if cond == "synth":
            pool = load_pool(task, seed)
        else:   # ablation: keep code examples whether or not they pass their own tests
            pool = [r for r in load_pool(task, seed, only_valid=False) if r.get("parsed", True)]
        assert len(pool) >= n_extra, f"pool too small: {len(pool)} < {n_extra}"
        return order[:N_REAL] + pool[:n_extra]
    raise ValueError(cond)
