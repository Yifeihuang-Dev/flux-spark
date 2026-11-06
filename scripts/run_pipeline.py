#!/usr/bin/env python3
"""Closed-loop verification: executor generates -> tests judge -> reviewer reviews.

This was the architecture's first end-to-end run. The question is not "how
accurate is the reviewer on synthetic data" but a harder one:

    **On real generated code, are the tasks the reviewer flags actually more
    likely to be wrong?**

If the flagged group fails significantly more often than the unflagged group,
the reviewer provides a signal that is independent of the tests — only then
does the architecture's "verification layer" stand. If both groups fail at the
same rate, the reviewer is flipping coins, and no amount of recall helps.

Usage:
  python3 scripts/run_pipeline.py --model <base> --reviewer <reviewer adapter> --out <json>
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

# NOTE: all prompt templates below, plus NO_ISSUE and CATS, are kept in Chinese
# on purpose. They are part of the measured configuration behind the numbers in
# REPORT.md (a Chinese-heavy SFT base produced these exact outputs); translating
# them would be an untested behavioral change and break reproducibility. The
# detection keywords must match the language the reviewer actually answers in.
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

# Control arm: the exact same rewrite request, but with no review advice at all.
# It rules out the confounder "a second attempt just works" — only if arm A is
# clearly better than arm B can we say the advice mattered.
FIX_TMPL_BLIND = '''下面这个 Python 函数没有通过测试。

请重写这个函数，修正其中的问题。
只输出修正后的完整代码，不要解释，不要 markdown 代码块标记，不要示例用法。

```python
{code}
```'''

# The reflection stage: instead of guessing "where does a bug look like", state
# what the code actually does, then compare it against the task requirement.
# It targets "the approach is wrong", not "the code is buggy" — the latter is
# the only kind that error messages or review advice can fix.
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
    """Apply the chat template. Without it the model degenerates to raw continuation."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)


def reported_any(text):
    """Loose check: did the model report any issue at all (regardless of wording)?"""
    if NO_ISSUE in text:
        return False
    if any(c in text for c in CATS):
        return True
    return bool(re.search(r"类型\s*[:：]|^\s*[-*]\s+|^\s*\d+[\.、)]\s*", text, re.M))


def try_fix(model, tok, spec, task_prompt, prompt_text, max_tokens):
    """Rewrite once with the given prompt and test it. Shared by all three arms
    so everything except the prompt is identical."""
    from mlx_lm import generate
    raw = generate(model, tok, prompt=wrap(tok, prompt_text),
                   max_tokens=max_tokens, verbose=False)
    return run_test(spec, task_prompt + clean_completion(task_prompt, raw))


def save_partial(path, results, summary=None):
    """Checkpoint mid-run. An earlier run died at 160/164 and lost everything,
    so results are now written after every stage."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary or {}, "results": results}, fh,
                  ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model (the executor)")
    ap.add_argument("--reviewer", default=None, help="reviewer adapter path")
    ap.add_argument("--data", default="/Volumes/LLM/datasets/humaneval.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N tasks (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--review-tokens", type=int, default=220)
    ap.add_argument("--reflect-tokens", type=int, default=320)
    ap.add_argument("--repeat", type=int, default=1,
                    help="samples per task (best-of-N). The stacked mode uses this "
                         "to skim off tasks a lucky sample would pass")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="temperature for sample 2 onwards (matches eval_humaneval's "
                         "default so baselines stay comparable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm import load, generate

    tasks = load_tasks(args.data, args.limit)
    print(f"{len(tasks)} tasks")

    # ---- stage 1: executor generates + tests judge (base model, no adapter) ----
    print("\n=== stage 1: generate code, judge with tests ===")
    model, tokenizer = load(args.model)
    # From sample 2 onward use temperature so candidates genuinely differ
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
        # best-of-N: sample N times, any pass counts.
        # Deliberately no early stop — otherwise the effective sample count drops
        # and the numbers stop aligning with eval_humaneval's best-of-3.
        for k in range(args.repeat):
            kw = {"sampler": sampler} if (k > 0 and sampler is not None) else {}
            raw = generate(model, tokenizer, prompt=gen_prompt,
                           max_tokens=args.max_tokens, verbose=False, **kw)
            full = task["prompt"] + clean_completion(task["prompt"], raw)
            if first_code is None:
                first_code = full      # the reflection stage analyses this one (first candidate)
            if run_test(task, full):
                passed = True
        results.append({
            "task_id": task["task_id"],
            "passed": passed,
            "code": first_code,        # the repair stage trims the generation from this
            "task_prompt": task["prompt"],
        })
        if i % 10 == 0 or i == len(tasks):
            n = sum(1 for r in results if r["passed"])
            print(f"  ({i}/{len(tasks)}) passed at least once: {n}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    # Free the base before attaching the adapter — reviewer and executor cannot
    # be active at the same time
    del model, tokenizer
    gc.collect()

    n_pass = sum(1 for r in results if r["passed"])
    print(f"generation done: pass {n_pass}/{len(results)} = {n_pass/len(results):.1%}")
    save_partial(args.out, results, {"stage": "after_generate",
                                     "pass_rate": n_pass / len(results) if results else 0})

    # ---- stage 2: the reviewer reviews every piece of real generated code ----
    print("\n=== stage 2: reviewer reviews real generated code ===")
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

    # ---- stage 2.5: reflection (newly added) ----
    # Unlike the reviewer, it does not guess "where a bug looks like"; it states
    # what the code does and compares that against the requirement. It targets
    # "wrong approach", not "buggy code".
    print("\n=== stage 2.5: reflect on failures (summarize / compare / advise) ===")
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
    print(f"  reflection done: {n_reflect} failing tasks")
    save_partial(args.out, results, {"stage": "after_reflect"})

    # ---- stage 3: three controlled repair arms ----
    # Scope widened to ALL failures, no longer just the flagged ones: with the
    # reviewer falsified, gating on it would shrink the sample to 9 and destroy
    # statistical meaning.
    print("\n=== stage 3: three repair arms (blind / review / reflection) ===")
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

        # Arm A: no information at all, a pure rewrite (baseline)
        r["blind_pass"] = try_fix(fmodel, ftok, spec, r["task_prompt"],
                                  FIX_TMPL_BLIND.format(code=r["code"]), args.max_tokens)
        if r["blind_pass"]:
            blind_fixed += 1

        # Arm B: with reviewer advice (only for flagged tasks; the rest stay None)
        if r.get("flagged") and r.get("review", "").strip():
            r["fixed_pass"] = try_fix(fmodel, ftok, spec, r["task_prompt"],
                                      FIX_TMPL.format(code=r["code"], review=r["review"]),
                                      args.max_tokens)
            if r["fixed_pass"]:
                repaired += 1

        # Arm C: with reflection advice (the newly added stage)
        if r["reflection"]:
            r["reflect_pass"] = try_fix(fmodel, ftok, spec, r["task_prompt"],
                                        FIX_REFLECT_TMPL.format(
                                            requirement=r["task_prompt"],
                                            code=r["code"], reflection=r["reflection"]),
                                        args.max_tokens)
            if r["reflect_pass"]:
                reflect_fixed += 1

        if i % 10 == 0 or i == len(results):
            print(f"  ({i}/{len(results)}) blind {blind_fixed} / review {repaired} / "
                  f"reflection {reflect_fixed}  [{time.time()-t2:.0f}s]", flush=True)

    # ---- statistics: does flagging carry information at all? ----
    flagged = [r for r in results if r["flagged"]]
    quiet = [r for r in results if not r["flagged"]]
    n_flag, n_quiet = len(flagged), len(quiet)

    fail_flag = sum(1 for r in flagged if not r["passed"])
    fail_quiet = sum(1 for r in quiet if not r["passed"])
    base_fail = sum(1 for r in results if not r["passed"]) / len(results)

    p_flag = fail_flag / n_flag if n_flag else 0.0
    p_quiet = fail_quiet / n_quiet if n_quiet else 0.0

    print(f"\n{'='*64}")
    print(f"generation pass rate         {n_pass}/{len(results)} = {n_pass/len(results):.1%}")
    print(f"tasks flagged by reviewer    {n_flag}/{len(results)} = {n_flag/len(results):.1%}")
    print("-" * 64)
    print(f"true failure in flagged      {fail_flag}/{n_flag} = {p_flag:.1%}")
    print(f"true failure in unflagged    {fail_quiet}/{n_quiet} = {p_quiet:.1%}")
    print(f"(overall failure baseline    {base_fail:.1%})")
    print("-" * 64)
    if n_quiet and p_quiet > 0:
        print(f"lift                         {p_flag / p_quiet:.2f}x"
              f"   (failure odds: flagged / unflagged)")
    elif n_quiet:
        print("lift                         n/a (no failure in the unflagged group)")
    else:
        print("lift                         n/a (nothing left unflagged)")
    n_reviewed = sum(1 for r in results
                     if not r["passed"] and r.get("flagged") and r.get("review", "").strip())
    print(f"\n--- stage 3: three repair arms ({attempted} failing tasks) ---")
    if attempted:
        print(f"A blind rewrite (baseline)   {blind_fixed:>2}/{attempted} = {blind_fixed/attempted:.1%}")
        if n_reviewed:
            print(f"B reviewer advice            {repaired:>2}/{n_reviewed} = {repaired/n_reviewed:.1%}"
                  f"    (only {n_reviewed} flagged)")
        print(f"C reflection advice          {reflect_fixed:>2}/{attempted} = {reflect_fixed/attempted:.1%}")
        print("-" * 64)
        print(f"net value of reflection      {reflect_fixed - blind_fixed:+d} tasks   <- what this round tests")
        final = n_pass + max(blind_fixed, repaired, reflect_fixed)
        print(f"pass rate with best arm      {final}/{len(results)} = {final/len(results):.1%}"
              f"   (before repair {n_pass/len(results):.1%})")
    else:
        print("no failing tasks; nothing to repair")
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
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
