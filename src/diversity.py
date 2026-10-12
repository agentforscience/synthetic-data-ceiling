"""E3: does the synthetic pool's diversity saturate as it grows (the hypothesised 'mode collapse')?

For nested prefixes of each synthetic pool, and for real data of the same size, compute
  * exact-duplicate rate of the task input (normalised text) and of a short "key" (title / function name),
  * near-duplicate rate: share of examples whose cosine similarity to an EARLIER example exceeds 0.9,
  * distinct-2: unique word bigrams / total bigrams, and the raw number of unique bigrams,
  * Vendi score (exp. entropy of the eigenvalues of the cosine-similarity kernel / n): an effective
    number of distinct examples,
  * coverage of the real test distribution: mean cosine similarity from each real test example to its
    nearest neighbour in the training prefix.
Embeddings: sentence-transformers/all-MiniLM-L6-v2 on CPU. Output: results/diversity.json
"""
import json
import re
from collections import Counter

import numpy as np

import common as C

SIZES = [100, 200, 300, 500, 1000, 2000, 3000, 5000]
NEAR_DUP = 0.9


def view(task, ex):
    """Text that characterises the *content* of an example (input side, plus the story for creative)."""
    if task == "mbpp":
        return ex["text"]
    if task == "gsm8k":
        return ex["question"]
    return ex["title"] + ". " + ex["story"]


def key(task, ex):
    """A coarse identity: function name / title / leading proper name of the word problem."""
    if task == "mbpp":
        m = re.search(r"def\s+(\w+)", ex["code"])
        return m.group(1) if m else ""
    if task == "roc":
        return ex["title"].lower().strip()
    m = re.search(r"\b([A-Z][a-z]+)\b", ex["question"])
    return m.group(1) if m else ""


def norm(t):
    return " ".join(re.findall(r"[a-z0-9]+", t.lower()))


def vendi(E):
    ev = np.linalg.eigvalsh((E @ E.T).astype(np.float64) / len(E))
    ev = ev[ev > 1e-12]
    return float(np.exp(-(ev * np.log(ev)).sum()))


def stats(task, exs, E, E_test):
    out = {}
    texts = [norm(view(task, e)) for e in exs]
    keys = [key(task, e) for e in exs]
    S = E @ E.T
    prior_max = np.array([S[i, :i].max() if i else 0.0 for i in range(len(E))])   # similarity to any earlier item
    for n in SIZES:
        if n > len(exs):
            break
        bg = [b for t in texts[:n] for b in zip(t.split(), t.split()[1:])]
        kc = Counter(keys[:n])
        out[n] = {
            "exact_dup_rate": 1 - len(set(texts[:n])) / n,
            "key_dup_rate": 1 - len(kc) / n,
            "top10_key_share": sum(c for _, c in kc.most_common(10)) / n,
            "near_dup_rate": float((prior_max[:n] > NEAR_DUP).mean()),
            "distinct2": len(set(bg)) / max(1, len(bg)),
            "unique_bigrams": len(set(bg)),
            "vendi": vendi(E[:n]),
            "test_coverage": float((E_test @ E[:n].T).max(1).mean()),
        }
    return out


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    emb = lambda xs: enc.encode(xs, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    res, pool_info = {}, {}
    for task in C.TASKS:
        test = C.load_real(task)["test"]
        E_test = emb([view(task, e) for e in test])
        for seed in C.SEEDS:
            raw = C.load_pool(task, seed, only_valid=False)
            pool = [r for r in raw if r["valid"]]
            pool_info[f"{task}|{seed}"] = {
                "requests": len(raw), "parsed": sum(r["parsed"] for r in raw), "valid": len(pool),
                "exec": dict(Counter(r.get("exec") for r in raw if r.get("exec")))}
            pool = pool[:max(SIZES)]
            res[f"{task}|synth|{seed}"] = stats(task, pool, emb([view(task, e) for e in pool]), E_test)
            real = C.real_order(task, seed)[:max(SIZES)]
            res[f"{task}|real|{seed}"] = stats(task, real, emb([view(task, e) for e in real]), E_test)
            print(task, seed, "synth", res[f"{task}|synth|{seed}"].get(max(SIZES)) or list(res[f"{task}|synth|{seed}"].items())[-1])
            print(task, seed, "real ", list(res[f"{task}|real|{seed}"].items())[-1], flush=True)
            if seed == 0:   # most frequent keys: a direct look at the modes
                kc = Counter(key(task, e) for e in pool)
                pool_info[f"{task}|top_keys"] = kc.most_common(15)
    json.dump({"stats": res, "pools": pool_info}, open(C.ROOT / "results" / "diversity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
