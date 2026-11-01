#!/usr/bin/env python3
"""HumanEval 评估（164 题标准基准）。

跟手写 31 题的关键区别：题量够大，能给出统计上可信的绝对水平，
也便于对外比较。每题跑单元测试判定，不靠模型自评。

用法：
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
    """instruct 模式给模型明确指令，否则按 HumanEval 标准做纯续写。"""
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
    """把模型输出整理成能拼在 prompt 之后且能编译的代码。

    模型常会先说一段废话、或在代码后追加示例，都要剥掉。
    做法：找到代码起点后，逐行截断取"能编译的最长前缀"。
    """
    comp = strip_fences(completion)
    lines = comp.split("\n")

    start = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        # 代码起点：def/import 关键字，或缩进行（函数体）
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
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--chat-template", action="store_true",
                    help="SFT/instruct 模型必开（否则是裸续写）")
    ap.add_argument("--repeat", type=int, default=1, help="每题采样几次")
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
            print(f"  ({i}/{len(tasks)}) 至今至少过一次: {done}  [{time.time()-t0:.0f}s]",
                  flush=True)

    total = len(results)
    avg = sum(r["passed"] for r in results) / (total * args.repeat)
    any_pass = sum(1 for r in results if r["passed"] > 0)

    tag = args.model.rstrip("/").split("/")[-1] + ("+adapter" if args.adapter else "")
    print()
    print(f"===== HumanEval [{tag}] =====")
    print(f"  题数 {total}   采样 {args.repeat} 次/题   "
          f"模板={'开' if args.chat_template else '裸续写'}")
    print(f"  平均通过率      {avg:.1%}")
    print(f"  best-of-{args.repeat}    "
          f"{any_pass}/{total} = {any_pass / total:.1%}")
    print(f"  用时 {time.time() - t0:.0f}s")

    if args.out:
        json.dump({"model": args.model, "adapter": args.adapter,
                   "repeat": args.repeat, "chat_template": args.chat_template,
                   "avg": avg, "best_of_n": any_pass / total,
                   "tasks": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
