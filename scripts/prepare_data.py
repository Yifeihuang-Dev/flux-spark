#!/usr/bin/env python3
"""Convert the Magicoder-OSS-Instruct dataset into mlx-lm chat fine-tuning format.

Writes train.jsonl / valid.jsonl, one line each:
    {"messages": [{"role":"user","content":"problem"}, {"role":"assistant","content":"solution"}]}

By default only Python samples are kept — the coding evaluation uses Python
tasks, and same-language data moves a measurable metric most directly.
Use --langs to change that.
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
                    help="comma-separated language whitelist, or 'all' to skip filtering")
    ap.add_argument("--max-chars", type=int, default=6000,
                    help="drop samples where problem+solution exceeds this length")
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
        print(f"warning: only {len(pool)} samples qualify, fewer than the {need} requested")
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
        print(f"wrote {path}: {len(rows)} rows")

    print(f"scanned {scanned} lines -> kept {kept} (langs: {args.langs})")


if __name__ == "__main__":
    main()
