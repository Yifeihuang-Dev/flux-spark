#!/usr/bin/env python3
"""闭环验证：执行器生成 → 测试判定 → 审查器审查。

这是整套架构第一次端到端实战。要回答的不是「审查器在合成数据上有多准」，
而是一个更硬的问题：

    **面对真实生成出来的代码，审查器报警的那些题，是不是真的更容易错？**

如果报警组的失败率显著高于不报警组，说明审查器提供了**独立于测试**的增量
信号——架构里的「验证层」才算站得住。反之，如果两组失败率一样，那审查器
只是在抛硬币，再多召回率也没意义。

用法：
  python3 scripts/run_pipeline.py --model <基座> --reviewer <审查器adapter> --out <json>
"""
import argparse
import gc
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_humaneval import load_tasks, make_prompt, clean_completion, run_test

PROMPT_TMPL = '''审查以下 Python 函数，找出其中的问题。

```python
{code}
```

请列出所有问题，每行一条，格式为「类型: 描述」。如果没有任何问题，只回答「无问题」。'''

NO_ISSUE = "无问题"
CATS = ("边界条件", "空值处理", "逻辑错误")

FIX_TMPL = '''下面这个 Python 函数没有通过测试。审查意见指出：

{review}

请根据这些意见重写这个函数，修正其中的问题。
只输出修正后的完整代码，不要解释，不要 markdown 代码块标记，不要示例用法。

```python
{code}
```'''

# 对照组：完全一样的重写请求，但不给任何审查意见。
# 用来排除「单纯重写一次就有效」这个混淆项——只有 A 组明显好于 B 组，
# 才能说审查意见真的起了作用，而不是「再试一次碰巧对了」。
FIX_TMPL_BLIND = '''下面这个 Python 函数没有通过测试。

请重写这个函数，修正其中的问题。
只输出修正后的完整代码，不要解释，不要 markdown 代码块标记，不要示例用法。

```python
{code}
```'''

# 反思环节（新增的一环）：不猜「哪里像有 bug」，而是先说清这段代码实际在做什么，
# 再对照任务要求找不满足的地方。定位的是「思路偏了」，而不是「代码写错了」——
# 后者才是指望靠报错或审查意见能修好的那类问题。
REFLECT_TMPL = '''下面是一个编程任务的要求，以及一份没能通过测试的解答。

【任务要求】
{requirement}

【当前代码】
```python
{code}
```

请分三步回答：
1. 这段代码实际采用的思路是什么？（一两句话讲清它在做什么，不要复述代码）
2. 对照任务要求，它哪些地方不满足？（具体指出，不要泛泛而谈）
3. 要满足要求，应该怎么改？（给出具体的修改建议）

只回答这三部分，不要输出完整代码。'''

FIX_REFLECT_TMPL = '''下面这个 Python 函数没有通过测试。

【任务要求】
{requirement}

【对这份代码的分析】
{reflection}

请根据上面的分析重写这个函数，修正其中的问题。
只输出修正后的完整代码，不要解释，不要 markdown 代码块标记，不要示例用法。

```python
{code}
```'''


def wrap(tokenizer, prompt):
    """套 chat template。不套的话模型会退化成裸续写。"""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)


def reported_any(text):
    """宽松判断：模型到底有没有报出问题（不管用词）。"""
    if NO_ISSUE in text:
        return False
    if any(c in text for c in CATS):
        return True
    return bool(re.search(r"类型\s*[:：]|^\s*[-*]\s+|^\s*\d+[\.、)]\s*", text, re.M))


def try_fix(model, tok, spec, task_prompt, prompt_text, max_tokens):
    """按给定提示重写一次并测。三组修复共用它，保证除提示外过程完全一致。"""
    from mlx_lm import generate
    raw = generate(model, tok, prompt=wrap(tok, prompt_text),
                   max_tokens=max_tokens, verbose=False)
    return run_test(spec, task_prompt + clean_completion(task_prompt, raw))


def save_partial(path, results, summary=None):
    """中途落盘。上次跑到 160/164 被中断、结果全丢，所以每阶段结束就存一次。"""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary or {}, "results": results}, fh,
                  ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="基座模型（执行器）")
    ap.add_argument("--reviewer", default=None, help="审查器 adapter 路径")
    ap.add_argument("--data", default="/Volumes/LLM/datasets/humaneval.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--review-tokens", type=int, default=220)
    ap.add_argument("--reflect-tokens", type=int, default=320)
    ap.add_argument("--repeat", type=int, default=1,
                    help="每题采样几次（best-of-N）。叠加模式要用它先把能蒙对的题挑走")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="第 2 次及以后采样的温度（与 eval_humaneval 默认一致，保证基线可比）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm import load, generate

    tasks = load_tasks(args.data, args.limit)
    print(f"共 {len(tasks)} 题")

    # ---- 阶段 1：执行器生成 + 测试判定（用基座，不能挂审查器 adapter）----
    print("\n=== 阶段 1：生成代码，用测试判定对错 ===")
    model, tokenizer = load(args.model)
    # 第 2 次及以后改用带温度采样，保证候选之间真的不一样
    sampler = None
    if args.repeat > 1 and args.temperature > 0:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=args.temperature)

    results = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        gen_prompt = wrap(tokenizer, make_prompt(task, True))
        passed = False
        first_code = None
        # best-of-N：采样 N 次，任一次通过即通过。
        # 刻意不早停——否则实际采样次数变少，就没法和 eval_humaneval 的 best-of-3 对齐了。
        for k in range(args.repeat):
            kw = {"sampler": sampler} if (k > 0 and sampler is not None) else {}
            raw = generate(model, tokenizer, prompt=gen_prompt,
                           max_tokens=args.max_tokens, verbose=False, **kw)
            full = task["prompt"] + clean_completion(task["prompt"], raw)
            if first_code is None:
                first_code = full      # 反思环节分析这一份（首个候选）
            if run_test(task, full):
                passed = True
        results.append({
            "task_id": task["task_id"],
            "passed": passed,
            "code": first_code,        # 修复阶段要用来裁剪生成结果
            "task_prompt": task["prompt"],
        })
        if i % 10 == 0 or i == len(tasks):
            n = sum(1 for r in results if r["passed"])
            print(f"  ({i}/{len(tasks)}) 至少过一次 {n}  [{time.time()-t0:.0f}s]", flush=True)

    # 释放基座，再挂 adapter——审查器和执行器不能同时生效
    del model, tokenizer
    gc.collect()

    n_pass = sum(1 for r in results if r["passed"])
    print(f"生成完成：通过 {n_pass}/{len(results)} = {n_pass/len(results):.1%}")
    save_partial(args.out, results, {"stage": "after_generate",
                                     "pass_rate": n_pass / len(results) if results else 0})

    # ---- 阶段 2：审查器审查每一份真实生成的代码 ----
    print("\n=== 阶段 2：审查器审查真实生成的代码 ===")
    rmodel, rtok = load(args.model, adapter_path=args.reviewer)
    t1 = time.time()
    for i, r in enumerate(results, 1):
        out = generate(rmodel, rtok,
                       prompt=wrap(rtok, PROMPT_TMPL.format(code=r["code"])),
                       max_tokens=args.review_tokens, verbose=False)
        r["flagged"] = reported_any(out)
        r["review"] = out[:200]
        if i % 10 == 0 or i == len(results):
            print(f"  ({i}/{len(results)})  [{time.time()-t1:.0f}s]", flush=True)

    del rmodel, rtok
    gc.collect()
    fmodel, ftok = load(args.model)

    test_by_id = {t["task_id"]: t["test"] for t in tasks}
    entry_by_id = {t["task_id"]: t["entry_point"] for t in tasks}

    # ---- 阶段 2.5：反思环节（新增的一环）----
    # 与审查器不同：它不猜「哪里像有 bug」，而是先说清「这段代码实际在做什么」，
    # 再对照任务要求找出不满足的地方。定位的是「思路偏了」，而非「代码写错了」。
    print("\n=== 阶段 2.5：对失败题反思（总结思路 / 对照要求 / 给出建议）===")
    t15 = time.time()
    n_reflect = 0
    for i, r in enumerate(results, 1):
        r["reflection"] = None
        if r["passed"]:
            continue
        n_reflect += 1
        r["reflection"] = generate(fmodel, ftok, prompt=wrap(ftok, REFLECT_TMPL.format(
            requirement=r["task_prompt"], code=r["code"])),
            max_tokens=args.reflect_tokens, verbose=False).strip()
        if i % 10 == 0 or i == len(results):
            print(f"  ({i}/{len(results)})  [{time.time()-t15:.0f}s]", flush=True)
    print(f"  反思完成：{n_reflect} 道失败题")
    save_partial(args.out, results, {"stage": "after_reflect"})

    # ---- 阶段 3：三组对照修复 ----
    # 范围扩大到「所有失败题」，不再限于审查器报警的那些：
    # 审查器已被证伪，拿它当门禁只会把样本砍到 9 题，失去统计意义。
    print("\n=== 阶段 3：三组对照修复（盲重写 / 审查意见 / 反思建议）===")
    t2 = time.time()
    repaired = blind_fixed = reflect_fixed = attempted = 0
    for i, r in enumerate(results, 1):
        r["fixed_pass"] = r["blind_pass"] = r["reflect_pass"] = None
        if r["passed"]:
            continue
        attempted += 1
        spec = {"task_id": r["task_id"], "prompt": r["task_prompt"],
                "test": test_by_id[r["task_id"]],
                "entry_point": entry_by_id[r["task_id"]]}

        # A 组：不给任何信息，纯粹重写一次（基线）
        r["blind_pass"] = try_fix(fmodel, ftok, spec, r["task_prompt"],
                                  FIX_TMPL_BLIND.format(code=r["code"]), args.max_tokens)
        if r["blind_pass"]:
            blind_fixed += 1

        # B 组：带审查器意见（仅对审查器报了警的题，其余保持 None）
        if r.get("flagged") and r.get("review", "").strip():
            r["fixed_pass"] = try_fix(fmodel, ftok, spec, r["task_prompt"],
                                      FIX_TMPL.format(code=r["code"], review=r["review"]),
                                      args.max_tokens)
            if r["fixed_pass"]:
                repaired += 1

        # C 组：带反思建议（新增的这一环）
        if r["reflection"]:
            r["reflect_pass"] = try_fix(fmodel, ftok, spec, r["task_prompt"],
                                        FIX_REFLECT_TMPL.format(
                                            requirement=r["task_prompt"],
                                            code=r["code"], reflection=r["reflection"]),
                                        args.max_tokens)
            if r["reflect_pass"]:
                reflect_fixed += 1

        if i % 10 == 0 or i == len(results):
            print(f"  ({i}/{len(results)}) 盲重写{blind_fixed} / 审查意见{repaired} / "
                  f"反思建议{reflect_fixed}  [{time.time()-t2:.0f}s]", flush=True)

    # ---- 统计：报警到底有没有信息量 ----
    flagged = [r for r in results if r["flagged"]]
    quiet = [r for r in results if not r["flagged"]]
    n_flag, n_quiet = len(flagged), len(quiet)

    fail_flag = sum(1 for r in flagged if not r["passed"])
    fail_quiet = sum(1 for r in quiet if not r["passed"])
    base_fail = sum(1 for r in results if not r["passed"]) / len(results)

    p_flag = fail_flag / n_flag if n_flag else 0.0
    p_quiet = fail_quiet / n_quiet if n_quiet else 0.0

    print(f"\n{'='*64}")
    print(f"生成通过率              {n_pass}/{len(results)} = {n_pass/len(results):.1%}")
    print(f"审查器报警的题目        {n_flag}/{len(results)} = {n_flag/len(results):.1%}")
    print("-" * 64)
    print(f"报警组里真失败的比例    {fail_flag}/{n_flag} = {p_flag:.1%}")
    print(f"未报警组里真失败的比例  {fail_quiet}/{n_quiet} = {p_quiet:.1%}")
    print(f"（整体失败率基线        {base_fail:.1%}）")
    print("-" * 64)
    if n_quiet and p_quiet > 0:
        print(f"提升度 lift             {p_flag / p_quiet:.2f}x"
              f"   报警组失败概率 ÷ 未报警组")
    elif n_quiet:
        print("提升度 lift             n/a（未报警组里没有一题失败）")
    else:
        print("提升度 lift             n/a（没有未报警的题）")
    n_reviewed = sum(1 for r in results
                     if not r["passed"] and r.get("flagged") and r.get("review", "").strip())
    print(f"\n--- 阶段 3：三组对照修复（失败题共 {attempted} 道）---")
    if attempted:
        print(f"A 组 盲重写（基线）        {blind_fixed:>2}/{attempted} = {blind_fixed/attempted:.1%}")
        if n_reviewed:
            print(f"B 组 带审查器意见          {repaired:>2}/{n_reviewed} = {repaired/n_reviewed:.1%}"
                  f"    （仅 {n_reviewed} 道被报警）")
        print(f"C 组 带反思建议            {reflect_fixed:>2}/{attempted} = {reflect_fixed/attempted:.1%}")
        print("-" * 64)
        print(f"反思建议的净价值           {reflect_fixed - blind_fixed:+d} 题    ← 本轮要验证的")
        final = n_pass + max(blind_fixed, repaired, reflect_fixed)
        print(f"取最佳组后通过率           {final}/{len(results)} = {final/len(results):.1%}"
              f"   （修复前 {n_pass/len(results):.1%}）")
    else:
        print("没有失败题，无需修复")
    print(f"{'='*64}")

    if args.out:
        summary = {
            "n_tasks": len(results),
            "pass_rate": n_pass / len(results) if results else 0,
            "flag_rate": n_flag / len(results) if results else 0,
            "fail_rate_in_flagged": p_flag,
            "fail_rate_in_quiet": p_quiet,
            "base_fail_rate": base_fail,
            "lift": (p_flag / p_quiet) if (n_quiet and p_quiet > 0) else None,
            "repair_attempted": attempted,
            "repair_blind_fixed": blind_fixed,
            "repair_review_fixed": repaired,
            "repair_reflect_fixed": reflect_fixed,
            "final_pass_rate": (n_pass + max(blind_fixed, repaired, reflect_fixed))
                               / len(results) if results else 0,
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "results": results}, fh,
                      ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
