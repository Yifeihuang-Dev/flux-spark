#!/usr/bin/env python3
"""汇总横评结果：同一批题、同一采样预算下，各模型在三个条件下的通过率。

用法：
  python3 scripts/summarize_bench.py /Volumes/LLM/logs/bench_*.jsonl
"""
import json
import os
import sys


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["task_id"]] = r
    return rows


def main():
    paths = sys.argv[1:]
    if not paths:
        print("用法: summarize_bench.py <jsonl...>")
        return

    data = {}
    for p in paths:
        name = os.path.basename(p).replace("bench_", "").replace(".jsonl", "")
        data[name] = load(p)

    # 只比较三个模型都跑过的题，避免缺题造成偏差
    common = set.intersection(*[set(v) for v in data.values()])
    print(f"共同题数：{len(common)}\n")

    head = f"{'模型':<22}{'单次':>8}{'best-3':>9}{'best-3+反思':>12}{'Δ采样':>9}{'Δ反思':>9}"
    print(head)
    print("-" * 70)

    for name, rows in data.items():
        sel = [rows[t] for t in common]
        n = len(sel)
        s1 = sum(r["s1"] for r in sel) / n * 100
        b3 = sum(r["b3"] for r in sel) / n * 100
        b3r = sum(r["b3r"] for r in sel) / n * 100
        print(f"{name:<22}{s1:>7.1f}%{b3:>8.1f}%{b3r:>11.1f}%"
              f"{b3-s1:>+8.1f}{b3r-b3:>+8.1f}")

    # 反思环节本身的修复率（只统计 b3 失败且真的反思过的题）
    print("\n反思环作用范围（仅 b3 失败的题）")
    print("-" * 62)
    for name, rows in data.items():
        sel = [r for r in (rows[t] for t in common) if not r["b3"]]
        if not sel:
            print(f"{name:<22}  没有失败题")
            continue
        tried = [r for r in sel if r["reflected"]]
        fixed = sum(r["b3r"] and not r["b3"] for r in sel)
        rate = fixed / len(tried) * 100 if tried else 0
        print(f"{name:<22}  失败 {len(sel):>3} 题，反思 {len(tried):>3} 题，"
              f"修好 {fixed:>2} 题 = {rate:.1f}%")


if __name__ == "__main__":
    main()
