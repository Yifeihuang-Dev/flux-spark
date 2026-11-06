#!/usr/bin/env python3
"""
Flux Spark — a local small model that writes code and self-checks/self-repairs
it against the requirement.

Usage:
    python3 fluxspark.py "write a function that checks if a list is monotonic (equal neighbors allowed)"
    python3 fluxspark.py "..." --tests 'assert f([1,2,2,4]); assert not f([1,3,2])'
    python3 fluxspark.py "..." --tests '...' --n 3      # strongest mode
    python3 fluxspark.py "..." --selftest               # let the model write its own tests first
    python3 fluxspark.py "..." --out result.py

Design notes:
    1. Tests are an optional enhancement. With tests you get best-of-N filtering
       and final verification; without them the tool degrades to "generate +
       reflect" with no way to confirm the fix actually worked.
    2. The reflection loop (SGSR) needs no tests — its only inputs are the
       requirement and the code, so it runs in every mode.
    3. Code goes to stdout, logs to stderr — redirect straight into a .py file.
"""

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time

DEFAULT_MODEL = "/Volumes/LLM/hf/MiniCPM5-2B-SFT-MLX-4bit"

GEN_TMPL = """Write a Python function that satisfies the requirement below.
Output ONLY the completed code (the function definition), with no explanation, no markdown fences, and no example usage.

Requirement:
{req}
"""

# Fallback prompt: when the model starts "explaining" instead of writing code,
# retry once with harder wording.
STRICT_TMPL = """Write ONE Python function for the requirement below.

Requirement:
{req}

CRITICAL: your reply must be a single Python function definition that STARTS with "def ".
No explanation. No markdown fences. No thinking out loud. No repeated code."""

# ---------------------------------------------------------------------------
# NOTE ON THE THREE TEMPLATES BELOW (REFLECT / FIX / SELFTEST):
# They are intentionally kept in Chinese. They are part of the *measured*
# configuration: every number in REPORT.md was produced with these exact
# prompts against a Chinese-heavy SFT base. Translating them is an untested
# behavioral change and would break reproducibility of the reported results.
# ---------------------------------------------------------------------------

REFLECT_TMPL = """下面是一个编程任务的要求，以及一份没能通过测试的解答。

【任务要求】
{req}

【当前代码】
```python
{code}
```

请分三步回答：
1. 这段代码实际采用的思路是什么？（一两句话讲清它在做什么，不要复述代码）
2. 对照任务要求，它哪些地方不满足？（具体指出，不要泛泛而谈）
3. 要满足要求，应该怎么改？（给出具体的修改建议）

只回答这三部分，不要输出完整代码。"""

FIX_TMPL = """下面这个 Python 函数没有通过测试。

【任务要求】
{req}

【对这份代码的分析】
{reflection}

请根据上面的分析重写这个函数，修正其中的问题。
只输出修正后的完整代码，不要解释，不要 markdown 代码块标记，不要示例用法。

```python
{code}
```"""

SELFTEST_TMPL = """下面是任务要求，以及一份实现。请为它写若干条单元测试，用来验证它是否正确。
只覆盖需求里明确描述的行为，不要测没提到的边界。

【任务要求】
{req}

【实现】
```python
{code}
```

只输出 assert 语句，每行一条，不要解释，不要 import，不要 def，不要 markdown 标记。
函数名用 {entry}。"""


def log(msg):
    """All logs go to stderr so stdout carries clean code only."""
    print(msg, file=sys.stderr)


# ---------- model ----------

def load_model(path):
    if not os.path.exists(path):
        sys.exit(f"Model not found: {path}\n(Is the external drive mounted? The default path lives under /Volumes/LLM/)")
    from mlx_lm import load
    return load(path)


def wrap(tok, prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def generate(model, tok, prompt, max_tokens=512, sampler=None):
    from mlx_lm import generate as _gen
    kw = {"sampler": sampler} if sampler is not None else {}
    return _gen(model, tok, prompt=wrap(tok, prompt),
                max_tokens=max_tokens, verbose=False, **kw)


# ---------- code processing ----------

def strip_fences(text):
    """Strip markdown code fences."""
    t = text.strip()
    for fence in ("```python", "```py", "```"):
        if t.startswith(fence):
            t = t[len(fence):]
            break
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _pick_code(text):
    """Find the longest prefix that starts with a def and parses via ast.

    Small models produce three kinds of dirty output:
      1) self-talk after the code ("That's it...", </think>)
      2) the whole code block indented by 4 spaces (as if quoted) → dedent first
      3) the same function printed twice
    Unified fix: dedent, then walk back from the first def to the longest
    parseable prefix that actually contains a function definition.
    """
    for s in (text, textwrap.dedent(text)):
        lines = s.split("\n")
        start = None
        for i, line in enumerate(lines):
            if line.startswith(("def ", "async def ", "import ",
                                "from ", "class ", "@")):
                start = i
                break
        if start is None:
            continue
        for end in range(len(lines), start, -1):
            cand = "\n".join(lines[start:end]).rstrip()
            if not cand.strip():
                continue
            try:
                tree = ast.parse(cand)
            except SyntaxError:
                continue
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   for n in tree.body):
                return cand
    return ""


def clean_code(raw):
    """Extract runnable pure code from a generation; empty string if none (caller retries)."""
    # Prefer the first fenced block (the model often wraps prose in fences,
    # with chatter before and after).
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", raw, re.S)
    for text in list(blocks) + [strip_fences(raw)]:
        code = _pick_code(text)
        if code:
            return code
    return ""


def entry_of(code):
    """Parse the entry function name from code (first top-level def)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


# ---------- test execution ----------

def run_tests(code, tests, entry, timeout=15):
    """Returns (passed, error summary). The f( placeholder in tests is
    rewritten to the real entry name."""
    src = tests
    if entry and "f(" in src:
        src = src.replace("f(", entry + "(")
    full = code + "\n\n" + src
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(full)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, path],
                           capture_output=True, timeout=timeout)
        if r.returncode == 0:
            return True, ""
        return False, r.stderr.decode(errors="replace")[-400:]
    except subprocess.TimeoutExpired:
        return False, f"timed out ({timeout}s)"
    finally:
        os.unlink(path)


# ---------- main flow ----------

def run_once(req, tests_cli, args, model, tok, sampler):
    """Run one task: generate → test → reflect/repair → emit. Returns final code."""
    t0 = time.time()
    if not args.quiet:
        log("=" * 56)
        log("Flux Spark")
        log("=" * 56)
        log(f"Requirement: {req}")

    if not args.quiet:
        log(f"\n[1/3] Generating code{(' (%d samples)' % args.n) if args.n > 1 else ''}")

    # ---- stage 1: generation ----
    cands = []
    for k in range(args.n):
        raw = generate(model, tok, GEN_TMPL.format(req=req),
                       max_tokens=args.max_tokens,
                       sampler=sampler if k > 0 else None)
        code = clean_code(raw)
        if not code:
            # The model is probably "explaining" — retry once with a harder instruction
            raw = generate(model, tok, STRICT_TMPL.format(req=req),
                           max_tokens=args.max_tokens,
                           sampler=sampler if k > 0 else None)
            code = clean_code(raw)
        entry = entry_of(code)
        cands.append({"code": code, "entry": entry, "passed": None})
    base = cands[0]
    entry = base["entry"]

    if not args.quiet:
        log(f"      Entry function: {entry or '(unparsed)'}")

    # ---- three-tier test-source fallback ----
    tests = tests_cli
    test_src = "user-provided"
    if not tests and args.selftest:
        if not args.quiet:
            log("\n[tests] No tests given — asking the model to write its own first")
        raw_t = generate(model, tok, SELFTEST_TMPL.format(
            req=req, code=base["code"], entry=entry or "f"),
            max_tokens=300)
        # Keep assert lines only; filter out any explanation the model smuggled in
        tests = "\n".join(l for l in raw_t.split("\n")
                          if l.strip().startswith("assert"))
        test_src = "model-generated (low confidence)"
        if not tests:
            log("      The model produced no usable tests; falling back to no-test mode")

    mode = "full loop (tests provided)" if tests else "generate + reflect only (no tests)"
    if not args.quiet:
        log(f"      Mode: {mode}")
        if tests:
            log(f"      Test source: {test_src}")

    # ---- stage 2: filter candidates with tests ----
    if tests:
        for c in cands:
            ok, err = run_tests(c["code"], tests, c["entry"])
            c["passed"] = ok
            c["err"] = err
        n_pass = sum(1 for c in cands if c["passed"])
        if not args.quiet:
            if args.n > 1:
                log(f"      best-of-{args.n}: {n_pass}/{args.n} candidates pass the tests")
            else:
                log(f"      Test result: {'PASS' if cands[0]['passed'] else 'FAIL'}")
                if not cands[0]["passed"] and cands[0].get("err"):
                    log(f"      Error summary: {cands[0]['err'].strip().splitlines()[-1][:120]}")

        good = [c for c in cands if c["passed"]]
        if good:
            best = good[0]
            if not args.quiet:
                log("\n[2/3] A candidate already passes the tests" +
                    ("; no repair needed" if len(good) == args.n else f" ({len(good)}/{args.n})"))
                log("\n[3/3] Skipping reflection (already passing; don't break it)" if len(good) == args.n
                    else "")
            if len(good) == args.n:
                return emit(best["code"], args, tz=time.time() - t0,
                            status="passed all tests")
            # Partial pass: reflect-repair only the failing candidates
            target = next(c for c in cands if not c["passed"])
        else:
            target = cands[0]
    else:
        target = cands[0]

    if args.no_reflect:
        if not args.quiet:
            log("\n[2/3] Reflection disabled; emitting the generated result as-is")
        return emit(target["code"], args, tz=time.time() - t0,
                    status="unverified (reflection off)" if not tests else "failed tests")

    # ---- stage 3: reflect + repair ----
    if not args.quiet:
        log("\n[2/3] Reflecting against the requirement (summarize / find gaps / advise)")
    reflection = generate(model, tok, REFLECT_TMPL.format(
        req=req, code=target["code"]), max_tokens=320).strip()

    if not args.quiet:
        log("\n[3/3] Rewriting per the reflection advice")
    new_raw = generate(model, tok, FIX_TMPL.format(
        req=req, code=target["code"], reflection=reflection),
        max_tokens=args.max_tokens)
    new_code = clean_code(new_raw)
    if not new_code.strip():
        new_code = target["code"]

    status = "unverified"
    if tests:
        ok, err = run_tests(new_code, tests, entry_of(new_code) or target["entry"])
        status = "fixed; passes tests" if ok else "still failing after repair"
        # Only accept improvements: if the new version fails, revert to the old code
        if not ok and target.get("passed") is False:
            if not args.quiet:
                log("      New version failed; keeping the original")
            new_code = target["code"]
            status = "failed tests (reverted to original)"

    return emit(new_code, args, tz=time.time() - t0, status=status)


def emit(code, args, tz, status):
    if not args.quiet:
        log("=" * 56)
        log(f"Status: {status}    Time: {tz:.0f}s")
        log("=" * 56)
        log("")
    # The code itself goes to stdout for easy redirection into a .py file
    print(code)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(code + "\n")
        if not args.quiet:
            log(f"\nWritten to {args.out}")
    return code


# ---------- interactive terminal ----------

HELP_TEXT = """Available commands:
  /help, /?        Show this help
  /n <number>      Set best-of-N sample count (N>=2 requires tests to filter)
  /selftest        Toggle: model writes its own tests when none given
  /reflect         Toggle: reflection self-repair (default on)
  /quiet           Toggle: code only, no process logs
  /save <file>     Save the last result's code to a file
  /quit, /q, exit  Quit
  Ctrl+D           Quit        Ctrl+C  abort current generation (session stays)

Tips:
  - Write the requirement in plain language; press Enter to run
  - Tests are optional; press Enter to skip. Use f() as the function-name
    placeholder, e.g.
        assert f([1,2,2,4]); assert not f([1,3,2])
  - The better your tests, the more trustworthy best-of-N and "confirmed fixed" become"""


def ask(prompt):
    """input() prints its prompt to stdout, which would pollute redirected code.
    So the prompt goes to stderr; stdout stays code-only."""
    sys.stderr.write(prompt)
    sys.stderr.flush()
    return input()


def print_banner(model_path, n, selftest, reflect, quiet):
    log("=" * 56)
    log(" Flux Spark    Interactive Terminal")
    log("-" * 56)
    log(f" Model: {os.path.basename(model_path)}")
    log(f" Now: samples {n} | auto self-tests {'on' if selftest else 'off'} "
        f"| reflection {'on' if reflect else 'off'} | quiet {'on' if quiet else 'off'}")
    log("-" * 56)
    log(" Type a natural-language requirement and press Enter.")
    log(" /help for commands, /quit to exit; Ctrl+D also quits.")
    log("=" * 56)


def interactive_session(model, tok, args, sampler):
    """REPL: the model is already loaded; loop on requirements, never reload."""
    try:
        import readline        # arrow-key history and line editing; fine without
    except ImportError:
        pass

    print_banner(args.model, args.n, args.selftest,
                 not args.no_reflect, args.quiet)

    last_code = None
    turn = 0
    while True:
        try:
            req = ask(f"\n[{turn + 1}] requirement> ").strip()
        except EOFError:
            log("\n(Ctrl+D) exiting")
            break
        except KeyboardInterrupt:
            log("\n(Ctrl+C) round aborted. Keep typing, or Ctrl+D / /quit to exit.")
            continue

        if not req:
            continue

        # ---- commands ----
        low = req.lower()
        if low in ("/quit", "/q", "/exit", "exit", "quit"):
            log("bye")
            break
        if low in ("/help", "/?", "/h", "help"):
            log(HELP_TEXT)
            continue
        if low.startswith("/save"):
            parts = req.split(None, 1)
            if len(parts) < 2:
                log("Usage: /save <file>")
            elif not last_code:
                log("No code produced yet — run a task first")
            else:
                path = parts[1].strip()
                try:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(last_code + "\n")
                    log(f"Written to {path}")
                except OSError as e:
                    log(f"Write failed: {e}")
            continue
        if low.startswith("/n"):
            parts = req.split(None, 1)
            if len(parts) < 2 or not parts[1].strip().isdigit():
                log("Usage: /n <number>, e.g. /n 3")
            else:
                args.n = max(1, int(parts[1].strip()))
                log(f"Sample count set to {args.n}")
                if args.n > 1:
                    log("Note: n>=2 requires tests to filter candidates (falls back to 1 otherwise)")
            continue
        if low == "/selftest":
            args.selftest = not args.selftest
            log(f"Auto self-tests: {'on' if args.selftest else 'off'}")
            continue
        if low == "/reflect":
            args.no_reflect = not args.no_reflect
            log(f"Reflection self-repair: {'off' if args.no_reflect else 'on'}")
            continue
        if low == "/quiet":
            args.quiet = not args.quiet
            log(f"Quiet mode: {'on' if args.quiet else 'off'}")
            continue
        if req.startswith("/"):
            log(f"Unknown command: {req}    Type /help for available commands")
            continue

        # ---- tests (optional) ----
        tests = None
        try:
            raw_tests = ask("    tests (optional, Enter to skip)> ").strip()
            tests = raw_tests or None
        except EOFError:
            log("\n(Ctrl+D) exiting")
            break
        except KeyboardInterrupt:
            log("\n(Ctrl+C) round cancelled")
            continue

        # best-of-N needs tests to filter; otherwise fall back to single-shot
        turn += 1

        # best-of-N without tests cannot filter: temporarily fall back to 1, restore after
        saved_n = None
        if args.n > 1 and not tests and not args.selftest:
            log(f"\nYou set n={args.n} but gave no tests and self-tests are off; candidates can't be filtered.")
            log("Running single-shot. Add an assert or /selftest for best-of-N.")
            saved_n, args.n = args.n, 1

        try:
            last_code = run_once(req, tests, args, model, tok, sampler)
        except KeyboardInterrupt:
            log("\n(Ctrl+C) generation aborted; session continues.")
            last_code = None
        except Exception as e:                      # one bad task shouldn't kill the session
            log(f"\nThis task failed: {type(e).__name__}: {e}")
            last_code = None
        finally:
            if saved_n is not None:
                args.n = saved_n


def build_arg_parser():
    ap = argparse.ArgumentParser(
        prog="fluxspark",
        description="Flux Spark: a local small model that writes code and self-checks it against the requirement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  one-shot   python3 fluxspark.py "check if a list is monotonic (equal neighbors allowed)" \\
               --tests 'assert f([1,2,2,4]); assert not f([1,3,2])'
  interactive python3 fluxspark.py -i
  save file  python3 fluxspark.py "..." --tests '...' > out.py

Code goes to stdout, logs to stderr — redirect straight into a .py file.""")
    ap.add_argument("requirement", nargs="?", default=None,
                    help="describe the function you want in plain language; omit to enter the interactive terminal")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="force the interactive terminal (model loads once; good for several questions)")
    ap.add_argument("--tests", default=None,
                    help="assert-style tests, separated by semicolons or newlines; use f() as the name placeholder")
    ap.add_argument("--selftest", action="store_true",
                    help="when no tests are given, have the model write its own first (reduced confidence)")
    ap.add_argument("--n", type=int, default=1,
                    help="how many samples (best-of-N). >1 requires tests to filter; default 1")
    ap.add_argument("--no-reflect", action="store_true",
                    help="disable reflection self-repair (for ablation: how much does reflection contribute?)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="path to the SFT base model")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=None, help="also write the result to this file")
    ap.add_argument("--quiet", action="store_true", help="code only; no process logs")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    # One-shot mode needs a requirement unless interaction was requested
    interactive = args.interactive or (not args.requirement and sys.stdin.isatty())
    if not interactive and not args.requirement:
        ap.error("Provide a requirement description, or use -i for the interactive terminal")

    model, tok = load_model(args.model)

    # best-of-N uses a temperature sampler so candidates genuinely differ
    sampler = None
    if args.n > 1:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=0.8)

    if interactive:
        interactive_session(model, tok, args, sampler)
        return

    run_once(args.requirement.strip(), args.tests, args, model, tok, sampler)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\ninterrupted")
