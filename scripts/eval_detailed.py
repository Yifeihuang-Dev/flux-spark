#!/usr/bin/env python3
"""Detailed coding evaluation.

Unlike eval_coding.py's single pass/fail, this script does four things:
  1. Repeated sampling: run each task N times to separate stable skill from luck
  2. Failure classification: syntax / logic / format, instead of a flat FAIL
  3. Per-category stats: arrays / strings / math / sorting / structure transforms
  4. Output style: code length, preamble chatter, fences, type hints

Usage:
  python3 eval_detailed.py --model <base path> --repeat 3 --out base.json
  python3 eval_detailed.py --model <base path> --adapter <adapter dir> --repeat 3 --out tuned.json
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coding import TASKS, extract_code  # noqa: E402

# Task categories, for per-dimension statistics
CATEGORY = {
    "two_sum": "arrays", "dedup_keep_order": "arrays", "flatten_nested": "arrays",
    "max_subarray_sum": "arrays", "rotate_list": "arrays", "chunk_list": "arrays",
    "list_intersection": "arrays", "max_pair_product": "arrays",
    "find_missing_number": "arrays",
    "is_palindrome": "strings", "word_count": "strings", "reverse_string": "strings",
    "count_vowels": "strings", "longest_common_prefix": "strings",
    "is_anagram": "strings", "caesar_cipher": "strings", "valid_parentheses": "strings",
    "fibonacci": "math", "factorial": "math", "is_prime": "math", "gcd": "math",
    "sum_of_digits": "math", "is_power_of_two": "math", "climbing_stairs": "math",
    "roman_to_int": "math", "fizzbuzz": "math",
    "binary_search": "sorting/searching", "merge_sorted": "sorting/searching", "sort_list": "sorting/searching",
    "matrix_transpose": "structure", "dict_invert": "structure",
}

# Output-style probes
HAS_FENCE = re.compile(r"```")
HAS_TYPE_HINT = re.compile(r"->|:\s*(int|str|list|dict|bool|float)")
HAS_DOCSTRING = re.compile(r'"""|\'\'\'')


def analyse_style(raw: str, code: str) -> dict:
    """Analyse the output style of one generation."""
    fence_pos = raw.find("```")
    # How much non-code text precedes the code (does the model preface with chatter?)
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
        return "generation failed"
    if stage == "exec":
        if isinstance(exc, SyntaxError):
            return "syntax error"
        if isinstance(exc, IndentationError):
            return "indentation error"
        return f"exec error:{type(exc).__name__}"
    if stage == "test":
        if isinstance(exc, AssertionError):
            return "logic error"
        return f"test error:{type(exc).__name__}"
    return "unknown"


def run_once(model, tokenizer, task, max_tokens, temperature):
    """Run one generation + exec + tests, return a detailed record."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    rec = {"passed": False, "failure": None, "style": None}

    # mlx_lm's generate signature is (model, tokenizer, prompt, verbose, **kwargs) with
    # no temp argument; temperature must be passed via a sampler.
    gen_kwargs = {}
    if temperature > 0:
        gen_kwargs["sampler"] = make_sampler(temp=temperature)

    try:
        raw = generate(
            model, tokenizer, prompt=task["prompt"],
            max_tokens=max_tokens, verbose=False, **gen_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        # Include the exception type, otherwise the failure cause is swallowed
        rec["failure"] = f"generation failed({type(exc).__name__}: {str(exc)[:100]})"
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
    ap.add_argument("--repeat", type=int, default=3, help="samples per task")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N tasks (0 = all), for a quick check")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm import load

    tag = "fine-tuned" if args.adapter else "base"
    print(f"[{tag}] model={args.model}")
    print(f"[{tag}] adapter={args.adapter or '(none)'}  samples={args.repeat}  temp={args.temperature}")
    model, tokenizer = load(args.model, adapter_path=args.adapter)

    tasks = TASKS if args.limit <= 0 else TASKS[: args.limit]
    all_records = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        records = []
        for r in range(args.repeat):
            # First sample greedy (reproducible), the rest at temperature: determinism plus diversity
            temp = 0.0 if r == 0 else args.temperature
            records.append(run_once(model, tokenizer, task, args.max_tokens, temp))
        passed = sum(1 for x in records if x["passed"])
        all_records.append({"name": task["name"], "category": CATEGORY.get(task["name"], "other"),
                            "passed": passed, "repeat": args.repeat, "records": records})
        elapsed = time.time() - t0
        print(f"  ({i}/{len(tasks)}) {task['name']}: {passed}/{args.repeat}"
              f"   [{elapsed:.0f}s]", flush=True)

    # ---------- summary ----------
    total_runs = sum(len(t["records"]) for t in all_records)
    total_pass = sum(t["passed"] for t in all_records)

    print(f"\n{'='*56}")
    print(f"[{tag}] overall pass rate: {total_pass}/{total_runs} = {total_pass/total_runs:.1%}")
    print(f"[{tag}] strict pass rate (all {args.repeat} correct): "
          f"{sum(1 for t in all_records if t['passed']==t['repeat'])}/{len(all_records)}")
    print(f"[{tag}] never passed: {sum(1 for t in all_records if t['passed']==0)}/{len(all_records)}")

    # by category
    print(f"\n--- pass rate by category ({tag}) ---")
    cats: dict[str, list[int]] = {}
    for t in all_records:
        cats.setdefault(t["category"], []).append(t["passed"])
    cat_stat = {}
    for c in sorted(cats):
        got = sum(cats[c])
        tot = len(cats[c]) * args.repeat
        cat_stat[c] = got / tot
        print(f"  {c:6s} {got}/{tot} = {got/tot:.1%}")

    # failure modes
    print(f"\n--- failure mode distribution ({tag}) ---")
    fails: dict[str, int] = {}
    for t in all_records:
        for rec in t["records"]:
            if not rec["passed"] and rec["failure"]:
                fails[rec["failure"]] = fails.get(rec["failure"], 0) + 1
    for k, v in sorted(fails.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # output style
    print(f"\n--- output style ({tag}) ---")
    styles = [r["style"] for t in all_records for r in t["records"]
              if r["style"]]
    if styles:
        n = len(styles)
        print(f"  avg raw length: {sum(s['raw_chars'] for s in styles)/n:.0f} chars")
        print(f"  avg code length: {sum(s['code_chars'] for s in styles)/n:.0f} chars")
        print(f"  avg code lines: {sum(s['code_lines'] for s in styles)/n:.1f}")
        print(f"  avg preamble: {sum(s['preamble_chars'] for s in styles)/n:.0f} chars  <- higher means more chatter")
        print(f"  uses code fences: {sum(1 for s in styles if s['has_fence'])/n:.0%}")
        print(f"  has type hints:   {sum(1 for s in styles if s['has_type_hint'])/n:.0%}")
        print(f"  has docstring:    {sum(1 for s in styles if s['has_docstring'])/n:.0%}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "tag": tag, "model": args.model, "adapter": args.adapter,
                "repeat": args.repeat, "temperature": args.temperature,
                "total_pass": total_pass, "total_runs": total_runs,
                "category": cat_stat, "failures": fails,
                "tasks": all_records,
            }, fh, ensure_ascii=False, indent=2)
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
