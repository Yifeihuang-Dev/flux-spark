#!/usr/bin/env python3
"""HumanEval evaluation (the standard 164-task benchmark).

Key difference from the 31 homemade tasks: enough volume for a statistically
credible absolute level, and comparable with the outside world. Every task is
judged by unit tests, never by the model's self-assessment.

Usage:
  python3 scripts/eval_humaneval.py --model <path> [--adapter <path>] \
      [--chat-template] [--repeat 3] [--limit 5]
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time


def load_tasks(path, limit=0):
    tasks = [json.loads(line) for line in open(path) if line.strip()]
    return tasks[:limit] if limit > 0 else tasks


def make_prompt(task, instruct):
    """In instruct mode the model gets an explicit instruction; otherwise it is
    plain HumanEval continuation."""
    if not instruct:
        return task["prompt"]
    return (
        "Complete the following Python function.\n"
        "Output ONLY the completed code, with no explanation, no markdown fences,\n"
        "and no example usage.\n\n" + task["prompt"]
    )


def strip_fences(text):
    return re.sub(r"```(?:python|py)?[ \t]*\n?", "", text)


def clean_completion(prompt, completion):
    """Turn model output into code that can be appended to the prompt and compiled.

    The model often opens with chatter or appends examples after the code; both
    must be stripped. Method: find the start of the code, then cut back to the
    longest prefix that compiles.
    """
    comp = strip_fences(completion)
    lines = comp.split("\n")

    start = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        # Code start: a def/import keyword, or an indented line (function body)
        if s.startswith(("def ", "import ", "from ", "return", "if ", "for ", "while ")) \
                or ln[:1] in (" ", "\t"):
            start = i
            break

    cand = lines[start:]
    for n in range(len(cand), 0, -1):
        chunk = "\n".join(cand[:n])
        try:
            ast.parse(prompt + chunk)
            return chunk
        except SyntaxError:
            continue
    return "\n".join(cand)


def run_test(task, full_code, timeout=15):
    code = full_code + "\n\n" + task["test"] + f"\n\ncheck({task['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="/Volumes/LLM/datasets/humaneval.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N tasks (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--chat-template", action="store_true",
                    help="required for SFT/instruct models (otherwise it is raw continuation)")
    ap.add_argument("--repeat", type=int, default=1, help="samples per task")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    tasks = load_tasks(args.data, args.limit)

    results = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        prompt = make_prompt(task, args.chat_template)
        gen_prompt = prompt
        if args.chat_template:
            try:
                gen_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                gen_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True)

        n_pass = 0
        for k in range(args.repeat):
            kw = {}
            if k > 0 and args.temperature > 0:
                kw["sampler"] = make_sampler(temp=args.temperature)
            raw = generate(model, tokenizer, prompt=gen_prompt,
                           max_tokens=args.max_tokens, verbose=False, **kw)
            comp = clean_completion(task["prompt"], raw)
            if run_test(task, task["prompt"] + comp):
                n_pass += 1

        results.append({"task_id": task["task_id"], "passed": n_pass,
                        "repeat": args.repeat})
        if i % 10 == 0 or i == len(tasks):
            done = sum(1 for r in results if r["passed"] > 0)
            print(f"  ({i}/{len(tasks)}) passed at least once so far: {done}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    total = len(results)
    avg = sum(r["passed"] for r in results) / (total * args.repeat)
    any_pass = sum(1 for r in results if r["passed"] > 0)

    tag = args.model.rstrip("/").split("/")[-1] + ("+adapter" if args.adapter else "")
    print()
    print(f"===== HumanEval [{tag}] =====")
    print(f"  tasks {total}   samples {args.repeat}/task   "
          f"template={'on' if args.chat_template else 'raw continuation'}")
    print(f"  average pass rate   {avg:.1%}")
    print(f"  best-of-{args.repeat}      "
          f"{any_pass}/{total} = {any_pass / total:.1%}")
    print(f"  elapsed {time.time() - t0:.0f}s")

    if args.out:
        json.dump({"model": args.model, "adapter": args.adapter,
                   "repeat": args.repeat, "chat_template": args.chat_template,
                   "avg": avg, "best_of_n": any_pass / total,
                   "tasks": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
