"""Generate synthetic training pools with the teacher model (vLLM offline inference).

Usage: CUDA_VISIBLE_DEVICES=g python src/generate_synth.py --jobs mbpp:0,gsm8k:0 --n_requests 6000

Request i of (task, seed) is fully determined by (task, seed, i): its 3 few-shot examples are drawn from
that seed's 100 real examples with an RNG keyed on (task, seed, i) and the sampling seed is derived from
the same key, so pools are resumable and extendable (re-running with a larger --n_requests appends).
"""
import argparse
import json
import random
import re
import time
import zlib

import common as C
from exec_eval import check_many, extract_code

K_SHOT = 3
TEMPERATURE, TOP_P, MAX_TOKENS = 1.0, 0.95, 512

SYSTEM = "You are a careful dataset author who writes new training examples for an existing dataset."

INSTR = {
    "mbpp": ("Below are {k} examples from a dataset of short Python programming tasks.\n\n{shots}\n"
             "Write ONE new example for this dataset: a different programming task of similar style and "
             "difficulty, a correct solution, and exactly three assert-based tests that the solution passes. "
             "Use exactly this format and output nothing else:\n\n"
             "Task: <one-sentence task description>\nCode:\n```python\n<solution>\n```\nTests:\n```python\n"
             "<assert 1>\n<assert 2>\n<assert 3>\n```"),
    "gsm8k": ("Below are {k} examples from a dataset of grade-school math word problems.\n\n{shots}\n"
              "Write ONE new example for this dataset: a different word problem of similar style and "
              "difficulty with a correct step-by-step solution. The last line of the solution must be "
              "'#### <final numeric answer>'. Use exactly this format and output nothing else:\n\n"
              "Question: <problem>\nSolution: <step-by-step solution>\n#### <answer>"),
    "roc": ("Below are {k} examples from a dataset of five-sentence everyday short stories written by "
            "people.\n\n{shots}\n"
            "Write ONE new example for this dataset: a different story with a new title, in the same style, "
            "exactly five sentences long. Use exactly this format and output nothing else:\n\n"
            "Title: <title>\nStory: <five-sentence story>"),
}


def render_shot(task: str, ex: dict, i: int) -> str:
    if task == "mbpp":
        return (f"Example {i}:\nTask: {ex['text']}\nCode:\n```python\n{ex['code']}\n```\nTests:\n```python\n"
                + "\n".join(ex["tests"]) + "\n```\n")
    if task == "gsm8k":
        return f"Example {i}:\nQuestion: {ex['question']}\nSolution: {ex['answer']}\n"
    return f"Example {i}:\nTitle: {ex['title']}\nStory: {ex['story']}\n"


def build_prompt(task: str, seed_set: list, seed: int, req: int) -> str:
    rng = random.Random(f"{task}-{seed}-{req}")
    shots = rng.sample(seed_set, K_SHOT)
    return INSTR[task].format(k=K_SHOT, shots="\n".join(render_shot(task, s, i + 1) for i, s in enumerate(shots)))


def parse(task: str, text: str):
    """Parse a teacher completion into a unified example, or None if the format is violated."""
    text = text.strip()
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text):   # teacher code-switched (CJK): format violation
        return None
    if task == "mbpp":
        m = re.search(r"Task:\s*(.+?)\s*Code:\s*```(?:python)?\s*\n(.*?)```\s*Tests:\s*```(?:python)?\s*\n(.*?)```",
                      text, flags=re.S)
        if not m:
            return None
        desc, code = m.group(1).strip(), C.clean_code(m.group(2))
        tests = [l.strip() for l in m.group(3).splitlines() if l.strip().startswith("assert")]
        if not desc or not code or len(tests) < 1:
            return None
        tests = tests[:3]
        return {"prompt": C.mbpp_prompt(desc, tests), "answer": C.mbpp_answer(code), "text": desc,
                "code": code, "tests": tests}
    if task == "gsm8k":
        m = re.search(r"Question:\s*(.+?)\s*Solution:\s*(.+)", text, flags=re.S)
        if not m or "####" not in m.group(2):
            return None
        q, sol = m.group(1).strip(), m.group(2).strip()
        sol = sol[: sol.rfind("####")] + "#### " + sol.split("####")[-1].strip().splitlines()[0].strip()
        gold = C.gsm_extract(sol)
        if not q or gold is None:
            return None
        return {"prompt": q, "answer": sol, "question": q, "gold": gold}
    m = re.search(r"Title:\s*(.+?)\s*\n\s*Story:\s*(.+)", text, flags=re.S)
    if not m:
        return None
    title, story = m.group(1).strip(), " ".join(m.group(2).split())
    if not title or len(story.split()) < 10:
        return None
    return {"prompt": C.roc_prompt(title), "answer": story, "title": title, "story": story}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="comma list of task:seed")
    ap.add_argument("--n_requests", type=str, required=True, help="int, or task=int,task=int")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--reverse", action="store_true", help="work from the end of the request list "
                    "(lets a helper instance share a job with a forward-running one)")
    ap.add_argument("--gpu_frac", type=float, default=0.80, help="fraction of GPU memory vLLM may reserve")
    args = ap.parse_args()
    n_req = ({k: int(v) for k, v in (kv.split("=") for kv in args.n_requests.split(","))}
             if "=" in args.n_requests else None)

    from vllm import LLM, SamplingParams
    llm = LLM(C.TEACHER, dtype="bfloat16", max_model_len=3072, gpu_memory_utilization=args.gpu_frac, seed=0)
    tok = llm.get_tokenizer()
    C.POOL_DIR.mkdir(parents=True, exist_ok=True)

    for job in args.jobs.split(","):
        task, seed = job.split(":")
        seed = int(seed)
        total = n_req[task] if n_req else int(args.n_requests)
        seed_set = C.real_seed_set(task, seed)
        path = C.pool_path(task, seed)
        done = {json.loads(l)["req"] for l in open(path)} if path.exists() else set()
        todo = [i for i in range(total) if i not in done]
        print(f"[{job}] {len(done)} done, {len(todo)} to generate", flush=True)
        if args.reverse:
            todo = todo[::-1]
        for s in range(0, len(todo), args.chunk):
            # re-read the pool: another instance may have produced part of this chunk meanwhile
            done = {json.loads(l)["req"] for l in open(path)} if path.exists() else set()
            reqs = sorted(i for i in todo[s:s + args.chunk] if i not in done)
            if not reqs:
                continue
            t0 = time.time()
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": build_prompt(task, seed_set, seed, i)}],
                tokenize=False, add_generation_prompt=True) for i in reqs]
            sps = [SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS,
                                  seed=(zlib.crc32(f'{task}-{seed}'.encode()) % 1000) * 1_000_000 + i) for i in reqs]
            outs = llm.generate(prompts, sps, use_tqdm=False)
            rows, n_tok = [], 0
            for i, o in zip(reqs, outs):
                txt = o.outputs[0].text
                n_tok += len(o.outputs[0].token_ids)
                ex = parse(task, txt)
                rows.append({"req": i, "parsed": ex is not None, "valid": ex is not None,
                             **(ex or {"raw": txt[:600]})})
            if task == "mbpp":   # execution filter: solution must pass the teacher's own tests
                idx = [k for k, r in enumerate(rows) if r["parsed"]]
                status = check_many([rows[k]["code"] for k in idx], [rows[k]["tests"] for k in idx])
                for k, st in zip(idx, status):
                    rows[k]["exec"] = st
                    rows[k]["valid"] = st == "pass"
            with open(path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            dt = time.time() - t0
            print(f"[{job}] reqs {reqs[0]}-{reqs[-1]}: parsed {sum(r['parsed'] for r in rows)}/{len(rows)}, "
                  f"valid {sum(r['valid'] for r in rows)}, {n_tok / dt:.0f} out tok/s, {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
