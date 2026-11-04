#!/usr/bin/env python3
"""跨模型横评：同一批题、同一套提示词、同一采样预算。

回答的问题：在「2B / 4bit / 本地 / 离线」这个赛道里，Flux Spark 的流水线
    到底贡献了多少，和各档位模型自身的差距相比，谁大。

三个条件，逐题记录，可断点续跑：
    s1  单次生成（贪婪）
    b3  best-of-3（第2、3 次带温度），任一通过即算过
    b3r b3 仍失败的题，再走一次「对照需求反思 → 重写」

注意：b3/b3r 都建立在「每题最多 3 次采样」上。给 A 模型 3 次就得给所有
模型 3 次，否则比的是算力不是能力。

用法：
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

# 反思/重写模板直接取自 CLI，保证与 Flux Spark 用的是同一套话术
from fluxspark import REFLECT_TMPL, FIX_TMPL

HEPLUS_CACHE = "/tmp/heplus.json.gz"


def make_judge(use_plus):
    """返回 judge(task, full_code) -> (主判定, base判定)。

    use_plus=False：只用 HumanEval 原生测试（历史口径）。
    use_plus=True ：用 HumanEval+ 增强测试当裁判（约 80 倍测试量），
                    同时记录 base 判定以便对比假阳性率。
    """
    if not use_plus:
        return (lambda task, code: (run_test(task, code), None))

    import gzip
    from evalplus_local import eq  # noqa: F401  (worker 内部使用)
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
    ap.add_argument("--out", required=True, help="JSONL，逐题追加，可续跑")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--no-reflect", action="store_true", help="只跑 s1/b3")
    ap.add_argument("--plus", action="store_true",
                    help="用 HumanEval+ 增强测试当裁判（约 80 倍测试量），"
                         "同时记录 base 判定以便测假阳性率")
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
    print(f"模型 {os.path.basename(args.model.rstrip('/'))} | "
          f"待跑 {len(todo)}/{len(tasks)} 题", flush=True)

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

        # --- s1：单次贪婪 ---
        code1 = gen()
        s1, s1_base = judge(task, code1)

        # --- b3：最多再采两次 ---
        b3, b3_base, best_code = s1, s1_base, code1
        if not s1:
            for _ in range(2):
                c = gen(sampler)
                ok, ok_base = judge(task, c)
                if ok:
                    b3, b3_base, best_code = True, ok_base, c
                    break
                best_code = c

        # --- b3r：仍失败则反思重写 ---
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
            print(f"  ({n}/{len(todo)}) {rates}  已用 {el/60:.1f} 分  "
                  f"剩余约 {eta/60:.1f} 分", flush=True)


if __name__ == "__main__":
    main()
