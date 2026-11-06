#!/usr/bin/env python3
"""Re-judge existing code with HumanEval+ augmented tests.

Why not the official `evalplus.evaluate`: its sandbox sets a memory limit with
resource.setrlimit, which macOS does not support, so every test is scored as a
failure (the measuring instrument is broken on this machine). This script
bypasses the sandbox:
    expected output = the official canonical_solution run on the same input
    verdict         = candidate output compared layer by layer (floats via atol)

Usage:
  # parent: fetch data, dispatch tasks
  python3 scripts/evalplus_local.py judge --samples /tmp/samples_hist.jsonl

  # worker (invoked by the parent; not normally run by hand)
  python3 scripts/evalplus_local.py worker --data /tmp/heplus.json.gz \
      --task HumanEval/0 --code /tmp/code_0.py --out /tmp/res_0.json
"""
import argparse
import gzip
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE = "/tmp/heplus.json.gz"


def fetch_plus():
    """Fetch HumanEval+ and cache it locally so workers never re-download."""
    from evalplus.data.humaneval import get_human_eval_plus
    probs = get_human_eval_plus()
    slim = {tid: {k: p[k] for k in
                  ("prompt", "entry_point", "canonical_solution",
                   "test", "base_input", "plus_input", "atol")}
            for tid, p in probs.items()}
    with gzip.open(DATA_CACHE, "wt", encoding="utf-8") as fh:
        json.dump(slim, fh, ensure_ascii=False)
    print(f"cached {len(slim)} tasks to {DATA_CACHE} "
          f"({os.path.getsize(DATA_CACHE)/1e6:.0f} MB)", flush=True)
    return slim


def eq(a, b, atol):
    """Compare layer by layer; floats use abs_tol=atol (matching EvalPlus semantics)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0.0, abs_tol=atol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(eq(x, y, atol) for x, y in zip(a, b))
    return a == b


def run_one(problem, code):
    """Execute the candidate in this process; returns (base, plus, failure detail)."""
    ns = {}
    exec(compile(problem["prompt"] + code, "<sol>", "exec"), ns)
    fn = ns[problem["entry_point"]]
    atol = problem["atol"]

    # Expected outputs: the official reference solution on the same inputs
    ref_ns = {}
    exec(compile(problem["prompt"] + problem["canonical_solution"],
                 "<ref>", "exec"), ref_ns)
    ref = ref_ns[problem["entry_point"]]

    detail = {}
    for name, inputs in (("base", problem["base_input"]),
                         ("plus", problem["plus_input"])):
        fail = None
        for i, args in enumerate(inputs):
            try:
                got = fn(*args)
                exp = ref(*args)
            except Exception as e:
                fail = (i, f"{type(e).__name__}: {e}")
                break
            if not eq(got, exp, atol):
                fail = (i, f"output mismatch: got={str(got)[:80]} exp={str(exp)[:80]}")
                break
        detail[name] = {"pass": fail is None, "n": len(inputs), "fail": fail}
    return detail


def cmd_worker(args):
    with gzip.open(args.data, "rt", encoding="utf-8") as fh:
        probs = json.load(fh)
    p = probs[args.task]
    code = io.open(args.code, encoding="utf-8").read()
    try:
        detail = run_one(p, code)
    except Exception as e:
        # The code doesn't even run (syntax/name error) — both sets fail
        detail = {"base": {"pass": False, "n": len(p["base_input"]),
                           "fail": [0, f"{type(e).__name__}: {e}"]},
                  "plus": {"pass": False, "n": len(p["plus_input"]),
                           "fail": [0, f"{type(e).__name__}: {e}"]}}
    io.open(args.out, "w", encoding="utf-8").write(json.dumps(detail))


def cmd_judge(args):
    if not os.path.exists(DATA_CACHE):
        fetch_plus()
    with gzip.open(DATA_CACHE, "rt", encoding="utf-8") as fh:
        probs = json.load(fh)

    samples = [json.loads(x) for x in io.open(args.samples, encoding="utf-8") if x.strip()]
    print(f"re-judging {len(samples)} solutions", flush=True)

    results = {}
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        tid = s["task_id"]
        if tid not in probs:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(s["solution"])
            code_path = f.name
        out_path = code_path + ".json"
        try:
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), "worker",
                 "--data", DATA_CACHE, "--task", tid,
                 "--code", code_path, "--out", out_path],
                capture_output=True, timeout=args.timeout)
            if os.path.exists(out_path):
                results[tid] = json.load(io.open(out_path, encoding="utf-8"))
        except subprocess.TimeoutExpired:
            results[tid] = {"base": {"pass": False, "fail": ["timeout"]},
                            "plus": {"pass": False, "fail": ["timeout"]}}
        finally:
            os.unlink(code_path)
            if os.path.exists(out_path):
                os.unlink(out_path)

        if i % 20 == 0 or i == len(samples):
            b = sum(1 for r in results.values() if r["base"]["pass"])
            p_ = sum(1 for r in results.values() if r["plus"]["pass"])
            print(f"  ({i}/{len(samples)}) base {b}  plus {p_}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    n = len(results)
    nb = sum(1 for r in results.values() if r["base"]["pass"])
    nplus = sum(1 for r in results.values() if r["plus"]["pass"])
    print("\n========== HumanEval+ re-judgment ==========")
    print(f"tasks           : {n}")
    print(f"base   pass     : {nb}/{n} = {nb/n*100:.1f}%")
    print(f"plus   pass     : {nplus}/{n} = {nplus/n*100:.1f}%")
    io.open(args.out, "w", encoding="utf-8").write(
        json.dumps(results, ensure_ascii=False, indent=1))
    print(f"per-task detail -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--samples", required=True)
    j.add_argument("--out", default="/tmp/heplus_results.json")
    j.add_argument("--timeout", type=int, default=120)
    w = sub.add_parser("worker")
    w.add_argument("--data", required=True)
    w.add_argument("--task", required=True)
    w.add_argument("--code", required=True)
    w.add_argument("--out", required=True)
    args = ap.parse_args()
    {"judge": cmd_judge, "worker": cmd_worker}[args.cmd](args)


if __name__ == "__main__":
    main()
