"""Multilingual probe: answers "is this model Python-only?"

The conclusion lives in README/REPORT; this is the reproducible instrument.

Usage:
    python3 scripts/probe_multilang.py                # syntax-check every language
    python3 scripts/probe_multilang.py --lang sql     # SQL only (with semantic assertions)
    python3 scripts/probe_multilang.py --lang js

It also demonstrates the **three things** needed to generalize fluxspark to any
language:
  1. start_markers : where the code begins (to trim thinking leaked before it)
  2. check_cmd     : syntax check with a local tool (instead of Python's ast.parse)
  3. test_cmd      : run tests (only this can judge "is the logic right", not just "does it parse")

WARNING: syntax checking alone badly overestimates capability — the JS run passed
syntax every time while the dedup function returned an empty array.
"""
import argparse
import os
import re
import sqlite3
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fluxspark as fx

TMPL = """Write {lang} code that satisfies the requirement below.

CRITICAL: reply with ONLY the code.
No analysis, no step-by-step reasoning, no explanation, no markdown fences.

Requirement:
{req}
"""

# Per language: start markers (find the real code start) + a local syntax checker.
# NOTE: the "req" strings are kept in Chinese on purpose — the reported probe
# results (e.g. SQL 3/3 semantically correct) were produced with these exact
# Chinese requirements against a Chinese-heavy SFT base. Translating them would
# be an untested behavioral change.
LANGS = {
    "js": {
        "display": "JavaScript", "suffix": ".js",
        "start": (r"^\s*(function|const|let|var|class|export|async)\b",
                  r"^\s*'use strict'"),
        "req": "写一个函数，去掉数组里的重复元素并保持原顺序",
        "check": lambda p: ["node", "--check", p],
    },
    "bash": {
        "display": "Bash", "suffix": ".sh",
        "start": (r"^#!|^[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{", r"^\s*function\b"),
        "req": "递归统计一个目录下所有 .py 文件的总行数",
        "check": lambda p: ["bash", "-n", p],
    },
    "ruby": {
        "display": "Ruby", "suffix": ".rb",
        "start": (r"^\s*(def|require|class|module)\b",),
        "req": "写一个方法，统计字符串中每个单词出现的次数",
        "check": lambda p: ["ruby", "-c", p],
    },
    "rust": {
        "display": "Rust", "suffix": ".rs",
        "start": (r"^\s*(fn|use|pub|struct|impl)\b",),
        "req": "写一个函数，计算 Vec<i32> 中所有元素的和",
        "check": lambda p: ["rustc", "--edition", "2021", "--crate-type", "lib",
                            "--emit=metadata", "-o", "/dev/null", p],
    },
    "sql": {
        "display": "SQLite SQL", "suffix": ".sql",
        "start": (r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)\b",),
        "req": "表 employees(name, department, salary)，查出每个部门工资最高的员工的姓名、部门和工资",
        "check": lambda p: ["sqlite3", ":memory:",
                            "CREATE TABLE employees(name TEXT, department TEXT,"
                            " salary INT); EXPLAIN " + open(p).read()],
    },
}


def strip_think(raw, spec):
    """Trim the self-talk the model emits before the code.

    Two cues:
      a) a closing </think> tag -> start after it
      b) otherwise find the first line that looks like a code start (start_markers)
    """
    m = re.search(r"```(?:[a-zA-Z+#]*)\s*\n(.*?)```", raw, re.S)
    text = m.group(1) if m else fx.strip_fences(raw)

    if "</think>" in text:
        text = text.split("</think>")[-1]

    lines = text.split("\n")
    for i, line in enumerate(lines):
        for pat in spec["start"]:
            if re.match(pat, line):
                return "\n".join(lines[i:]).strip()
    return text.strip()


def check_syntax(spec, code):
    fd, path = tempfile.mkstemp(suffix=spec["suffix"])
    os.write(fd, code.encode())
    os.close(fd)
    try:
        r = subprocess.run(spec["check"](path), capture_output=True, text=True,
                           timeout=60)
        if r.returncode == 0:
            return True, ""
        return False, (r.stderr or r.stdout).strip().splitlines()[-1][:120]
    except FileNotFoundError:
        return None, "this language's checker is not installed locally"
    except Exception as e:
        return False, str(e)[:120]
    finally:
        os.unlink(path)


def probe_sql_semantics(model, tok, rounds=3):
    """SQL only: syntax passing is not enough — run it on real data and check the answer."""
    schema = """
    CREATE TABLE employees (name TEXT, department TEXT, salary INT);
    INSERT INTO employees VALUES ('Alice','Eng',100), ('Bob','Eng',120),
                                 ('Carol','Sales',90), ('Dave','Sales',150);
    """
    expected = {"Bob": ("Eng", 120), "Dave": ("Sales", 150)}   # name -> (department, salary)
    hit = 0
    for i in range(rounds):
        raw = fx.generate(model, tok,
                          TMPL.format(lang="SQL", req=LANGS["sql"]["req"]),
                          max_tokens=900)
        sql = strip_think(raw, LANGS["sql"]).rstrip().rstrip(";")
        con = sqlite3.connect(":memory:")
        try:
            con.executescript(schema)
            got = {r[0]: (r[1], r[2]) for r in con.execute(sql)}
            ok = got == expected
            detail = "" if ok else "got %s" % (got,)
        except Exception as e:
            ok, detail = False, str(e)[:100]
        con.close()
        hit += ok
        print("  round %d: %s %s" % (i + 1, "semantically correct" if ok else "wrong", detail))
        print("    " + " ".join(sql.split())[:150])
    print("  -> SQL semantic accuracy %d/%d" % (hit, rounds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default=None,
                    help="run only one of: %s" % ", ".join(LANGS))
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    targets = ([args.lang] if args.lang else list(LANGS))
    model, tok = fx.load_model(fx.DEFAULT_MODEL)

    for key in targets:
        spec = LANGS[key]
        print("=" * 62)
        print("[%s] %s" % (spec["display"], spec["req"]))
        if key == "sql":
            probe_sql_semantics(model, tok, rounds=args.rounds)
            continue
        for i in range(args.rounds):
            raw = fx.generate(model, tok,
                              TMPL.format(lang=spec["display"], req=spec["req"]),
                              max_tokens=900)
            code = strip_think(raw, spec)
            ok, msg = check_syntax(spec, code)
            print("  round %d: %s %s" % (
                i + 1, {True: "syntax OK", False: "syntax FAIL",
                        None: "not verified"}[ok], msg))
            if args.rounds == 1 and code:
                for line in code.splitlines()[:8]:
                    print("    | " + line)
    print("=" * 62)
    print("Note: passing syntax != correct logic. Judging logic requires tests "
          "(which is exactly why the reflection loop exists).")


if __name__ == "__main__":
    main()
