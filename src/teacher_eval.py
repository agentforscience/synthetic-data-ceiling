"""Reference point: the teacher's own zero-shot score on the code and math test sets (same 500 items and
same scoring as the student), to document the teacher-student gap.

    CUDA_VISIBLE_DEVICES=g python src/teacher_eval.py --gpu_frac 0.3
"""
import argparse
import json

import numpy as np

import common as C
from exec_eval import check_many, extract_code
from train_eval import last_number

HINT = {"mbpp": "", "gsm8k": "\nSolve step by step and end with a final line of the form '#### <number>'."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_frac", type=float, default=0.3)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(C.TEACHER, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=args.gpu_frac, seed=0)
    tok = llm.get_tokenizer()
    out = {"model": C.TEACHER}
    for task in ["mbpp", "gsm8k"]:
        items = C.load_real(task)["test"][:C.N_GEN_TEST[task]]
        prompts = [tok.apply_chat_template([{"role": "user", "content": e["prompt"] + HINT[task]}],
                                           tokenize=False, add_generation_prompt=True) for e in items]
        gens = [o.outputs[0].text for o in llm.generate(prompts, SamplingParams(temperature=0, max_tokens=512),
                                                        use_tqdm=False)]
        if task == "mbpp":
            st = check_many([extract_code(g) for g in gens], [e["tests"] for e in items], [e["setup"] for e in items])
            out[task] = float(np.mean([s == "pass" for s in st]))
        else:
            out[task] = float(np.mean([C.gsm_equal(C.gsm_extract(g) or last_number(g), e["gold"])
                                       for g, e in zip(gens, items)]))
        print(task, out[task], flush=True)
    json.dump(out, open(C.ROOT / "results" / "teacher_baseline.json", "w"), indent=1)


if __name__ == "__main__":
    main()
