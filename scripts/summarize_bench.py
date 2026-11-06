#!/usr/bin/env python3
"""Summarize benchmark results: pass rates for each model under three conditions,
on the same tasks and the same sampling budget.

Usage:
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
        print("usage: summarize_bench.py <jsonl...>")
        return

    data = {}
    for p in paths:
        name = os.path.basename(p).replace("bench_", "").replace(".jsonl", "")
        data[name] = load(p)

    # Compare only tasks every model ran, so missing tasks can't bias the table
    common = set.intersection(*[set(v) for v in data.values()])
    print(f"common tasks: {len(common)}\n")

    head = f"{'model':<22}{'single':>8}{'best-3':>9}{'best-3+reflect':>16}{'Δsampling':>11}{'Δreflect':>10}"
    print(head)
    print("-" * 76)

    for name, rows in data.items():
        sel = [rows[t] for t in common]
        n = len(sel)
        s1 = sum(r["s1"] for r in sel) / n * 100
        b3 = sum(r["b3"] for r in sel) / n * 100
        b3r = sum(r["b3r"] for r in sel) / n * 100
        print(f"{name:<22}{s1:>7.1f}%{b3:>8.1f}%{b3r:>15.1f}%"
              f"{b3-s1:>+10.1f}{b3r-b3:>+9.1f}")

    # Repair rate of the reflection stage itself (only b3 failures that were reflected on)
    print("\nReflection stage coverage (b3 failures only)")
    print("-" * 62)
    for name, rows in data.items():
        sel = [r for r in (rows[t] for t in common) if not r["b3"]]
        if not sel:
            print(f"{name:<22}  no failures")
            continue
        tried = [r for r in sel if r["reflected"]]
        fixed = sum(r["b3r"] and not r["b3"] for r in sel)
        rate = fixed / len(tried) * 100 if tried else 0
        print(f"{name:<22}  failed {len(sel):>3}, reflected {len(tried):>3}, "
              f"fixed {fixed:>2} = {rate:.1f}%")


if __name__ == "__main__":
    main()
