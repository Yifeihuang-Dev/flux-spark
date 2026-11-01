#!/usr/bin/env python3
"""精细版编码能力评估。

相比 eval_coding.py 的单次通过/失败，本脚本做四件事：
  1. 多次采样：每题跑 N 遍，看是稳定会做还是碰巧蒙对
  2. 失败分类：语法错误 / 逻辑错误 / 格式问题，而不是笼统的 FAIL
  3. 分题型统计：数组 / 字符串 / 数学 / 排序查找 / 结构转换
  4. 输出风格：代码长度、是否先说废话、是否用围栏、是否加类型注解

用法:
  python3 eval_detailed.py --model <基座路径> --repeat 3 --out base.json
  python3 eval_detailed.py --model <基座路径> --adapter <adapter目录> --repeat 3 --out tuned.json
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coding import TASKS, extract_code  # noqa: E402

# 题目分类，用于分维度统计
CATEGORY = {
    "two_sum": "数组", "dedup_keep_order": "数组", "flatten_nested": "数组",
    "max_subarray_sum": "数组", "rotate_list": "数组", "chunk_list": "数组",
    "list_intersection": "数组", "max_pair_product": "数组",
    "find_missing_number": "数组",
    "is_palindrome": "字符串", "word_count": "字符串", "reverse_string": "字符串",
    "count_vowels": "字符串", "longest_common_prefix": "字符串",
    "is_anagram": "字符串", "caesar_cipher": "字符串", "valid_parentheses": "字符串",
    "fibonacci": "数学", "factorial": "数学", "is_prime": "数学", "gcd": "数学",
    "sum_of_digits": "数学", "is_power_of_two": "数学", "climbing_stairs": "数学",
    "roman_to_int": "数学", "fizzbuzz": "数学",
    "binary_search": "排序查找", "merge_sorted": "排序查找", "sort_list": "排序查找",
    "matrix_transpose": "结构转换", "dict_invert": "结构转换",
}

# 输出风格探针
HAS_FENCE = re.compile(r"```")
HAS_TYPE_HINT = re.compile(r"->|:\s*(int|str|list|dict|bool|float)")
HAS_DOCSTRING = re.compile(r'"""|\'\'\'')


def analyse_style(raw: str, code: str) -> dict:
    """分析一次生成的输出风格。"""
    fence_pos = raw.find("```")
    # 代码之前有多少非代码文字（模型是否先说了废话）
    preamble_len = fence_pos if fence_pos > 0 else raw.find("def ")
    return {
        "raw_chars": len(raw),
        "code_chars": len(code),
        "code_lines": code.count("\n") + 1 if code else 0,
        "preamble_chars": preamble_len if preamble_len > 0 else 0,
        "has_fence": bool(HAS_FENCE.search(raw)),
        "has_type_hint": bool(HAS_TYPE_HINT.search(code)),
        "has_docstring": bool(HAS_DOCSTRING.search(code)),
    }


def classify_failure(stage: str, exc: Exception | None) -> str:
    if stage == "gen":
        return "生成失败"
    if stage == "exec":
        if isinstance(exc, SyntaxError):
            return "语法错误"
        if isinstance(exc, IndentationError):
            return "缩进错误"
        return f"执行异常:{type(exc).__name__}"
    if stage == "test":
        if isinstance(exc, AssertionError):
            return "逻辑错误"
        return f"测试异常:{type(exc).__name__}"
    return "未知"


def run_once(model, tokenizer, task, max_tokens, temperature):
    """跑一次生成 + 执行 + 测试，返回详细记录。"""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    rec = {"passed": False, "failure": None, "style": None}

    # mlx_lm 的 generate 签名是 (model, tokenizer, prompt, verbose, **kwargs)，
    # 没有 temp 参数；温度必须通过 sampler 传入。
    gen_kwargs = {}
    if temperature > 0:
        gen_kwargs["sampler"] = make_sampler(temp=temperature)

    try:
        raw = generate(
            model, tokenizer, prompt=task["prompt"],
            max_tokens=max_tokens, verbose=False, **gen_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        # 带上具体异常类型，否则失败原因会被吞掉
        rec["failure"] = f"生成失败({type(exc).__name__}: {str(exc)[:100]})"
        return rec

    code = extract_code(raw)
    rec["style"] = analyse_style(raw, code)

    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        rec["failure"] = classify_failure("exec", exc)
        return rec

    try:
        exec(task["tests"], ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        rec["failure"] = classify_failure("test", exc)
        return rec

    rec["passed"] = True
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--repeat", type=int, default=3, help="每题采样次数")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部），用于快速验证")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm import load

    tag = "微调后" if args.adapter else "基座"
    print(f"[{tag}] 模型={args.model}")
    print(f"[{tag}] adapter={args.adapter or '（无）'}  采样={args.repeat}次  temp={args.temperature}")
    model, tokenizer = load(args.model, adapter_path=args.adapter)

    tasks = TASKS if args.limit <= 0 else TASKS[: args.limit]
    all_records = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        records = []
        for r in range(args.repeat):
            # 第一次用贪心（可复现），其余按温度采样，兼顾确定性与多样性
            temp = 0.0 if r == 0 else args.temperature
            records.append(run_once(model, tokenizer, task, args.max_tokens, temp))
        passed = sum(1 for x in records if x["passed"])
        all_records.append({"name": task["name"], "category": CATEGORY.get(task["name"], "其他"),
                            "passed": passed, "repeat": args.repeat, "records": records})
        elapsed = time.time() - t0
        print(f"  ({i}/{len(tasks)}) {task['name']}: {passed}/{args.repeat}"
              f"   [{elapsed:.0f}s]", flush=True)

    # ---------- 汇总 ----------
    total_runs = sum(len(t["records"]) for t in all_records)
    total_pass = sum(t["passed"] for t in all_records)

    print(f"\n{'='*56}")
    print(f"[{tag}] 总体通过率: {total_pass}/{total_runs} = {total_pass/total_runs:.1%}")
    print(f"[{tag}] 严格通过率({args.repeat}次全对): "
          f"{sum(1 for t in all_records if t['passed']==t['repeat'])}/{len(all_records)}")
    print(f"[{tag}] 从不通过: {sum(1 for t in all_records if t['passed']==0)}/{len(all_records)}")

    # 分题型
    print(f"\n--- 分题型通过率 ({tag}) ---")
    cats: dict[str, list[int]] = {}
    for t in all_records:
        cats.setdefault(t["category"], []).append(t["passed"])
    cat_stat = {}
    for c in sorted(cats):
        got = sum(cats[c])
        tot = len(cats[c]) * args.repeat
        cat_stat[c] = got / tot
        print(f"  {c:6s} {got}/{tot} = {got/tot:.1%}")

    # 失败模式
    print(f"\n--- 失败模式分布 ({tag}) ---")
    fails: dict[str, int] = {}
    for t in all_records:
        for rec in t["records"]:
            if not rec["passed"] and rec["failure"]:
                fails[rec["failure"]] = fails.get(rec["failure"], 0) + 1
    for k, v in sorted(fails.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # 输出风格
    print(f"\n--- 输出风格 ({tag}) ---")
    styles = [r["style"] for t in all_records for r in t["records"]
              if r["style"]]
    if styles:
        n = len(styles)
        print(f"  平均原始长度: {sum(s['raw_chars'] for s in styles)/n:.0f} 字符")
        print(f"  平均代码长度: {sum(s['code_chars'] for s in styles)/n:.0f} 字符")
        print(f"  平均代码行数: {sum(s['code_lines'] for s in styles)/n:.1f} 行")
        print(f"  平均前言长度: {sum(s['preamble_chars'] for s in styles)/n:.0f} 字符  ← 越大越爱先说废话")
        print(f"  使用代码围栏: {sum(1 for s in styles if s['has_fence'])/n:.0%}")
        print(f"  带类型注解:   {sum(1 for s in styles if s['has_type_hint'])/n:.0%}")
        print(f"  带 docstring: {sum(1 for s in styles if s['has_docstring'])/n:.0%}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "tag": tag, "model": args.model, "adapter": args.adapter,
                "repeat": args.repeat, "temperature": args.temperature,
                "total_pass": total_pass, "total_runs": total_runs,
                "category": cat_stat, "failures": fails,
                "tasks": all_records,
            }, fh, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    sys.exit(main())
