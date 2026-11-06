#!/usr/bin/env python3
"""Cross-model benchmark: same tasks, same prompts, same sampling budget.

The question it answers: in the "2B / 4-bit / local / offline" niche, how much
does the Flux Spark pipeline actually contribute — and how does that compare to
the gap between model tiers?

Three conditions, per-task records, resumable:
    s1   single greedy generation
    b3   best-of-3 (samples 2 and 3 at temperature); any pass counts
    b3r  for b3 failures, one round of "reflect against the requirement -> rewrite"

Note: b3/b3r are built on "at most 3 samples per task". If model A gets 3
samples, every model must get 3 — otherwise you measure compute, not capability.

Usage:
  python3 scripts/bench_matrix.py --model /path --out /tmp/bench_x.jsonl --limit 10
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from eval_humaneval import load_tasks, make_prompt, clean_completion, run_test

# Reflection/rewrite templates come straight from the CLI so that every model
# is judged with exactly the wording Flux Spark uses.
from fluxspark import REFLECT_TMPL, FIX_TMPL

HEPLUS_CACHE = "/tmp/heplus.json.gz"


def make_judge(use_plus):
    """Returns judge(task, full_code) -> (primary verdict, base verdict).

    use_plus=False: judge with HumanEval's own tests (historical setting).
    use_plus=True : judge with HumanEval+ augmented tests (~80x the test volume)
                    and also record the base verdict so false positives can be
                    measured on the same code.
    """
    if not use_plus:
        return (lambda task, code: (run_test(task, code), None))

    import gzip
    from evalplus_local import eq  # noqa: F401  (used inside the worker)
    with gzip.open(HEPLUS_CACHE, "rt", encoding="utf-8") as fh:
        probs = json.load(fh)
    here = os.path.dirname(os.path.abspath(__file__))

    def judge(task, full_code):
        if task["task_id"] not in probs:
            return (None, None)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(full_code)
            code_path, out_path = f.name, f.name + ".json"
        try:
            subprocess.run(
                [sys.executable, os.path.join(here, "evalplus_local.py"),
                 "worker", "--data", HEPLUS_CACHE, "--task", task["task_id"],
                 "--code", code_path, "--out", out_path],
                capture_output=True, timeout=180)
            if os.path.exists(out_path):
                d = json.load(open(out_path, encoding="utf-8"))
                return (d["plus"]["pass"], d["base"]["pass"])
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        finally:
            for p in (code_path, out_path):
                if os.path.exists(p):
                    os.unlink(p)
        return (False, None)

    return judge


def build_prompt(tokenizer, task):
    prompt = make_prompt(task, instruct=True)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/Volumes/LLM/datasets/humaneval.jsonl")
    ap.add_argument("--out", required=True, help="JSONL, appended per task, resumable")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--no-reflect", action="store_true", help="run s1/b3 only")
    ap.add_argument("--plus", action="store_true",
                    help="judge with HumanEval+ augmented tests (~80x the test volume) "
                         "and also record the base verdict to measure false positives")
    args = ap.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["task_id"])

    model, tok = load(args.model)
    tasks = load_tasks(args.data, args.limit)
    tasks = tasks[args.offset:]
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"model {os.path.basename(args.model.rstrip('/'))} | "
          f"to run {len(todo)}/{len(tasks)} tasks", flush=True)

    sampler = make_sampler(temp=args.temperature)
    judge = make_judge(args.plus)
    t0 = time.time()

    for n, task in enumerate(todo, 1):
        gen_prompt = build_prompt(tok, task)
        base = task["prompt"]

        def gen(s=None):
            raw = generate(model, tok, prompt=gen_prompt,
                           max_tokens=args.max_tokens, verbose=False,
                           **({"sampler": s} if s else {}))
            return base + clean_completion(base, raw)

        # --- s1: single greedy ---
        code1 = gen()
        s1, s1_base = judge(task, code1)

        # --- b3: at most two more samples ---
        b3, b3_base, best_code = s1, s1_base, code1
        if not s1:
            for _ in range(2):
                c = gen(sampler)
                ok, ok_base = judge(task, c)
                if ok:
                    b3, b3_base, best_code = True, ok_base, c
                    break
                best_code = c

        # --- b3r: reflect + rewrite if still failing ---
        b3r, b3r_base, fixed_code = b3, b3_base, None
        reflected = False
        if (not b3) and (not args.no_reflect):
            reflected = True
            advice = generate(
                model, tok,
                prompt=build_prompt(tok, {"prompt": REFLECT_TMPL.format(
                    req=base, code=best_code)}),
                max_tokens=args.max_tokens, verbose=False)
            fixed_code = base + clean_completion(base, generate(
                model, tok,
                prompt=build_prompt(tok, {"prompt": FIX_TMPL.format(
                    req=base, code=best_code, reflection=advice)}),
                max_tokens=args.max_tokens, verbose=False,
                sampler=sampler))
            b3r, b3r_base = judge(task, fixed_code)

        final_code = fixed_code if reflected else best_code
        rec = {"task_id": task["task_id"], "s1": bool(s1), "b3": bool(b3),
               "b3r": bool(b3r), "reflected": reflected,
               "s1_base": None if s1_base is None else bool(s1_base),
               "b3_base": None if b3_base is None else bool(b3_base),
               "b3r_base": None if b3r_base is None else bool(b3r_base),
               "code": final_code}
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if n % 5 == 0 or n == len(todo):
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            rows = [json.loads(x) for x in
                    open(args.out, encoding="utf-8") if x.strip()]
            tot = len(rows)
            rates = " ".join(
                f"{k}={100*sum(1 for r in rows if r[k])/tot:.1f}%" for k in ("s1", "b3", "b3r"))
            print(f"  ({n}/{len(todo)}) {rates}  elapsed {el/60:.1f} min  "
                  f"eta {eta/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
