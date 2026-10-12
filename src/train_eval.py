"""Fine-tune the student (LoRA) on one training-set condition and evaluate it on real held-out data.

Any number of workers can share the job queue: a job is claimed with a lock file, finished jobs are
appended to results/runs.jsonl, and failed jobs (e.g. out-of-memory on a shared GPU) are released and
retried, so the sweep is resumable.

    python src/train_eval.py --name w1 [--filter REGEX]

No checkpoints are written to disk (the best-validation adapter weights are held in host RAM).
"""
import argparse
import gzip
import json
import math
import os
import random
import re
import subprocess
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common as C
from exec_eval import check_many, extract_code

LR, BATCH, EPOCHS, MIN_STEPS, N_VAL_EVALS = 1e-4, 8, 2, 100, 6
LORA_R, LORA_ALPHA = 32, 64
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
WARMUP_FRAC, WEIGHT_DECAY, MAX_LEN, GRAD_CLIP = 0.1, 0.01, 448, 1.0
# generate() computes vocabulary logits for every prompt position on its first step (batch x prompt_len x
# 151k values), so the batch size is set per task from the typical prompt length.
GEN_BATCH = {"mbpp": 48, "gsm8k": 64, "roc": 125}
REAL_CONTROL = {"gsm8k": [1, 3, 10, 30], "roc": [1, 3, 10, 30], "mbpp": [1, 2.74]}   # 2.74 -> all 374 MBPP train
UNFILTERED = [3, 30]                 # ablation E5 (MBPP only): no execution filter on synthetic code
LOCK_DIR = C.ROOT / "results" / "locks"


def job_key(j):
    return f"{j['task']}|{j['cond']}|{j['ratio']}|{j['seed']}"


def all_jobs():
    """Jobs in priority order: zero-shot, then seed by seed (ratios 0/1/3/10/30, real control, remaining
    ratios, ablation), then the optional 50:1 runs. If compute runs out, what is finished is a usable block."""
    jobs = [{"task": t, "cond": "zeroshot", "ratio": 0, "seed": 0} for t in C.TASKS]
    first = [r for r in C.RATIOS if r in (0, 1, 3, 10, 30)]
    for s in C.SEEDS:
        for t in C.TASKS:
            jobs += [{"task": t, "cond": "synth", "ratio": r, "seed": s} for r in first]
        for t in C.TASKS:
            jobs += [{"task": t, "cond": "real", "ratio": r, "seed": s} for r in REAL_CONTROL[t]]
        for t in C.TASKS:
            jobs += [{"task": t, "cond": "synth", "ratio": r, "seed": s} for r in C.RATIOS if r not in first]
        jobs += [{"task": "mbpp", "cond": "synth_unfiltered", "ratio": r, "seed": s} for r in UNFILTERED]
    for s in C.SEEDS:
        for t in C.TASKS:
            jobs += [{"task": t, "cond": "synth", "ratio": r, "seed": s} for r in C.EXTRA_RATIOS[t]]
    return jobs


def claim(key):
    """Atomically claim a job (lock file) so several workers can share the queue."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(LOCK_DIR / key.replace("|", "_"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def release(key):
    (LOCK_DIR / key.replace("|", "_")).unlink(missing_ok=True)


# --------------------------------------------------------------------------- tokenisation
def encode(tok, ex):
    """Return (input_ids, labels) with the loss restricted to the assistant turn."""
    prefix = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False,
                                     add_generation_prompt=True)
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    a = tok(ex["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(p) + len(a) > MAX_LEN:
        a = a[:max(MAX_LEN - len(p), 8)]
    return p + a, [-100] * len(p) + a


def collate(items, pad_id):
    n = max(len(i[0]) for i in items)
    ids = torch.full((len(items), n), pad_id, dtype=torch.long)
    lab = torch.full((len(items), n), -100, dtype=torch.long)
    att = torch.zeros((len(items), n), dtype=torch.long)
    for k, (x, y) in enumerate(items):
        ids[k, :len(x)] = torch.tensor(x)
        lab[k, :len(y)] = torch.tensor(y)
        att[k, :len(x)] = 1
    return ids, att, lab


def batches(enc, rng):
    """Shuffled batches with light length-bucketing (less padding, same expected gradient)."""
    idx = list(range(len(enc)))
    rng.shuffle(idx)
    out = []
    for s in range(0, len(idx), BATCH * 32):
        chunk = sorted(idx[s:s + BATCH * 32], key=lambda i: len(enc[i][0]))
        out += [chunk[b:b + BATCH] for b in range(0, len(chunk), BATCH)]
    rng.shuffle(out)
    return out


LOSS_CHUNK = 384   # answer tokens projected onto the vocabulary at a time


def _answer_hidden(model, ids, att, lab):
    hid = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    mask = tgt != -100
    return hid[:, :-1][mask], tgt[mask], mask


def train_step_backward(model, ids, att, lab):
    """Backward pass of the mean answer-token cross-entropy, with exact gradients but bounded memory.

    Logits over the 151k-token vocabulary are the dominant memory cost, so they are computed only at
    answer positions and in chunks: each chunk back-propagates into the LM head and into a detached copy
    of the hidden states; the accumulated hidden-state gradient is then pushed through the transformer
    body once. Returns the loss value."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, tgt, _ = _answer_hidden(model, ids, att, lab)
    h_det = h.detach().requires_grad_(True)
    total, n = 0.0, tgt.numel()
    for s in range(0, n, LOSS_CHUNK):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.lm_head(h_det[s:s + LOSS_CHUNK])
        l = torch.nn.functional.cross_entropy(logits.float(), tgt[s:s + LOSS_CHUNK], reduction="sum") / n
        l.backward()
        total += l.item()
    h.backward(h_det.grad)
    return total


@torch.no_grad()
def nll_per_item(model, enc, pad_id, bs=16):
    """Summed NLL and token count of the reference answer for every item."""
    model.eval()
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
    sums, cnts = np.zeros(len(enc)), np.zeros(len(enc))
    for s in range(0, len(order), bs):
        ii = order[s:s + bs]
        ids, att, lab = (t.cuda() for t in collate([enc[i] for i in ii], pad_id))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, tgt, mask = _answer_hidden(model, ids, att, lab)
            loss = torch.cat([torch.nn.functional.cross_entropy(
                model.lm_head(h[c:c + LOSS_CHUNK]).float(), tgt[c:c + LOSS_CHUNK], reduction="none")
                for c in range(0, tgt.numel(), LOSS_CHUNK)])
        row = torch.nonzero(mask)[:, 0]
        sums[ii] = torch.zeros(len(ii), device="cuda", dtype=torch.float64).index_add_(0, row, loss.double()).cpu().numpy()
        cnts[ii] = mask.sum(1).cpu().numpy()
    return sums, cnts


@torch.no_grad()
def generate(model, tok, prompts, max_new, sample, seed, bs):
    model.eval()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    outs = [None] * len(texts)
    torch.manual_seed(seed)
    kw = (dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=0) if sample
          else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
    for s in range(0, len(order), bs):
        ii = order[s:s + bs]
        b = tok([texts[i] for i in ii], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(**b, max_new_tokens=max_new, repetition_penalty=1.0,
                               pad_token_id=tok.pad_token_id, **kw)
        for i, row in zip(ii, g[:, b["input_ids"].shape[1]:]):
            outs[i] = tok.decode(row, skip_special_tokens=True).strip()
    return outs


def last_number(text):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def task_eval(task, model, tok, test, seed):
    """Generation-based evaluation. Returns (metrics dict, per-item records)."""
    items = test[:C.N_GEN_TEST[task]]
    gens = generate(model, tok, [e["prompt"] for e in items], C.MAX_NEW_TOKENS[task],
                    sample=(task == "roc"), seed=777 + seed, bs=GEN_BATCH[task])
    if task == "mbpp":
        st = check_many([extract_code(g) for g in gens], [e["tests"] for e in items], [e["setup"] for e in items])
        return {"score": float(np.mean([s == "pass" for s in st]))}, [{"gen": g, "status": s} for g, s in zip(gens, st)]
    if task == "gsm8k":
        strict = [C.gsm_equal(C.gsm_extract(g), e["gold"]) for g, e in zip(gens, items)]
        lenient = [C.gsm_equal(C.gsm_extract(g) or last_number(g), e["gold"]) for g, e in zip(gens, items)]
        return ({"score": float(np.mean(strict)), "score_lenient": float(np.mean(lenient))},
                [{"gen": g, "correct": bool(c), "correct_lenient": bool(l)} for g, c, l in zip(gens, strict, lenient)])
    return {}, [{"gen": g} for g in gens]   # roc: generation metrics are computed in analysis


# --------------------------------------------------------------------------- one run
def run(job, tok):
    task, cond, ratio, seed = job["task"], job["cond"], job["ratio"], job["seed"]
    t0 = time.time()
    torch.manual_seed(1000 * seed + 17)
    rng = random.Random(1000 * seed + 17)
    data = C.load_real(task)
    val_enc = [encode(tok, e) for e in data["val"]]
    test_enc = [encode(tok, e) for e in data["test"]]
    model = AutoModelForCausalLM.from_pretrained(C.STUDENT, dtype=torch.float32).cuda()
    torch.cuda.reset_peak_memory_stats()
    res = {**job, "key": job_key(job)}

    if cond != "zeroshot":
        train = C.train_set(task, seed, cond, ratio)
        enc = [encode(tok, e) for e in train]
        steps = max(math.ceil(EPOCHS * len(enc) / BATCH), MIN_STEPS)
        eval_at = {round(steps * (k + 1) / N_VAL_EVALS) for k in range(N_VAL_EVALS)}
        from peft import LoraConfig, get_peft_model
        # LoRA adapters are injected in place, so `model` keeps working as the plain HF model.
        peft_model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                                                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        params = [p for p in model.parameters() if p.requires_grad]
        res["trainable_params"] = int(sum(p.numel() for p in params))
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
        warm = max(1, int(WARMUP_FRAC * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
        step, best, best_state, curve, bl = 0, float("inf"), None, [], []
        while step < steps:
            for b in batches(enc, rng):
                model.train()
                ids, att, lab = (t.cuda() for t in collate([enc[i] for i in b], tok.pad_token_id))
                train_step_backward(model, ids, att, lab)
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step in eval_at:
                    s_, c_ = nll_per_item(model, val_enc, tok.pad_token_id)
                    v = float(s_.sum() / c_.sum())
                    curve.append([step, v])
                    if v < best:
                        best, bl = v, step
                        best_state = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}
                if step >= steps:
                    break
        s_, c_ = nll_per_item(model, test_enc, tok.pad_token_id)
        res["train_seconds"] = round(time.time() - t0, 1)
        res.update(n_train=len(enc), steps=steps, epochs=steps * BATCH / len(enc), best_step=bl,
                   val_curve=curve, val_nll=best, test_nll_final=float(s_.sum() / c_.sum()),
                   train_tokens=int(sum(len(e[0]) for e in enc)))
        res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in best_state:
                    p.copy_(best_state[k].to(p.device))
        model = peft_model.merge_and_unload()   # fold the adapters into the weights for evaluation
        del best_state, opt, peft_model
    else:
        s_, c_ = nll_per_item(model, val_enc, tok.pad_token_id)
        res.update(n_train=0, steps=0, val_nll=float(s_.sum() / c_.sum()))

    model.to(torch.bfloat16)   # evaluation only: avoids re-casting fp32 weights on every decoding step
    sums, cnts = nll_per_item(model, test_enc, tok.pad_token_id)
    res["test_nll"] = float(sums.sum() / cnts.sum())
    metrics, recs = task_eval(task, model, tok, data["test"], seed)
    res.update(metrics)
    res["seconds"] = round(time.time() - t0, 1)

    del model
    torch.cuda.empty_cache()
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(C.OUT_DIR / (res["key"].replace("|", "_") + ".json.gz"), "wt") as f:
        json.dump({"key": res["key"], "nll_sum": sums.tolist(), "n_tok": cnts.tolist(), "items": recs}, f)
    with open(C.RUNS_FILE, "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


NEED_MB = 11000   # free GPU memory (MiB) required to start a run


def pick_gpu(name):
    """The GPUs are shared with other jobs whose memory use changes minute to minute, so before every
    run choose the card with the most free memory (queried through nvidia-smi, which creates no CUDA
    context) and wait while none has enough."""
    waited = 0
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        free = sorted(((int(l.split(",")[1]), int(l.split(",")[0])) for l in out.strip().splitlines()), reverse=True)
        if free and free[0][0] >= NEED_MB:
            torch.cuda.set_device(free[0][1])
            return free[0][1]
        if waited % 600 == 0:
            print(f"[{name}] waiting for GPU memory (best: {free[0][0] if free else '?'} MiB free)", flush=True)
        time.sleep(20 + random.random() * 20)
        waited += 30


def finished():
    return {json.loads(l)["key"] for l in open(C.RUNS_FILE)} if C.RUNS_FILE.exists() else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="w0")
    ap.add_argument("--filter", default=".*", help="regex on job key task|cond|ratio|seed")
    ap.add_argument("--passes", type=int, default=40, help="sweeps over the queue (failed jobs are retried)")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(C.STUDENT)
    tok.padding_side = "left"
    jobs = [j for j in all_jobs() if re.search(args.filter, job_key(j))]
    for p in range(args.passes):
        todo = [j for j in jobs if job_key(j) not in finished()]
        if not todo:
            break
        ran = False
        for j in todo:
            key = job_key(j)
            if key in finished() or not claim(key):
                continue
            try:
                gpu = pick_gpu(args.name)
                time.sleep(random.random() * 10)   # de-synchronise workers that picked the same card
                r = run(j, tok)
                ran = True
                print(f"[{args.name}] gpu{gpu} {r['key']}: n={r['n_train']} steps={r['steps']} val={r['val_nll']:.4f} "
                      f"test_nll={r['test_nll']:.4f} score={r.get('score')} train={r.get('train_seconds')}s "
                      f"total={r['seconds']}s", flush=True)
            except Exception as e:   # out-of-memory on a shared GPU, or pool not generated yet: retry later
                print(f"[{args.name}] FAILED {key}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                release(key)
                torch.cuda.empty_cache()
                time.sleep(20)
        if not ran:
            time.sleep(60)


if __name__ == "__main__":
    main()
