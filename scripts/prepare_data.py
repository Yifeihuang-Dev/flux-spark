#!/usr/bin/env python3
"""把 Magicoder-OSS-Instruct 数据集转成 mlx-lm 的 chat 微调格式。

输出 train.jsonl / valid.jsonl，每行形如：
    {"messages": [{"role":"user","content":"问题"}, {"role":"assistant","content":"解答"}]}

默认只保留 Python 样本——评估编码能力用的是 Python 题，
同语言数据对可测量指标的提升最直接。可用 --langs 调整。
"""

import argparse
import json
import os
import random

DEFAULT_SRC = "/Volumes/LLM/datasets/magicoder_oss.jsonl"
DEFAULT_OUT = "/Volumes/LLM/datasets/magicoder_mlx"


def build_messages(problem: str, solution: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": problem.strip()},
            {"role": "assistant", "content": solution.strip()},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--n-train", type=int, default=1400)
    ap.add_argument("--n-valid", type=int, default=100)
    ap.add_argument("--langs", default="python",
                    help="逗号分隔的语言白名单，或 all 表示不过滤")
    ap.add_argument("--max-chars", type=int, default=6000,
                    help="problem+solution 超过此长度则丢弃")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    langs = None if args.langs.strip().lower() == "all" else {
        x.strip().lower() for x in args.langs.split(",")
    }

    pool, seen = [], set()
    kept = scanned = 0
    with open(args.src, "r", encoding="utf-8") as fh:
        for line in fh:
            scanned += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            lang = str(row.get("lang", "")).lower()
            if langs is not None and lang not in langs:
                continue
            problem, solution = row.get("problem", ""), row.get("solution", "")
            if not problem or not solution:
                continue
            if len(problem) + len(solution) > args.max_chars:
                continue
            key = problem[:120]
            if key in seen:
                continue
            seen.add(key)
            pool.append(build_messages(problem, solution))
            kept += 1

    random.seed(args.seed)
    random.shuffle(pool)

    need = args.n_train + args.n_valid
    if len(pool) < need:
        print(f"警告: 符合条件的样本只有 {len(pool)} 条，少于需要的 {need} 条")
        n_valid = min(args.n_valid, max(0, len(pool) - args.n_train))
        n_train = len(pool) - n_valid
    else:
        n_train, n_valid = args.n_train, args.n_valid

    os.makedirs(args.out_dir, exist_ok=True)
    train, valid = pool[:n_train], pool[n_train:n_train + n_valid]

    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写出 {path}: {len(rows)} 条")

    print(f"扫描 {scanned} 行 -> 保留 {kept} 条 (语言: {args.langs})")


if __name__ == "__main__":
    main()
