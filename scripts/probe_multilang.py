"""多语言探针：回答「这个模型是不是只会 Python」。

结论写在 README/REPORT 里，这里是可复现的量具。

用法：
    python3 scripts/probe_multilang.py                # 全跑一遍语法验证
    python3 scripts/probe_multilang.py --lang sql     # 只跑 SQL（带语义断言）
    python3 scripts/probe_multilang.py --lang js

它同时演示了 fluxspark  generalization 到任意语言所需的**三个东西**：
  1. start_markers : 代码从哪儿开始（用来砍掉模型吐在前面的思维泄漏）
  2. check_cmd     : 用本机工具做语法检查（代替 Python 的 ast.parse）
  3. test_cmd      : 跑测试（有这个才能判断「逻辑对不对」，而不只是「能不能解析」）

⚠️ 只做语法检查会严重高估模型能力：JS 那次语法全过，但去重函数返回空数组。
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

# 每种语言：起始标记（找真正的代码开头）+ 本机可用的语法检查器
LANGS = {
    "js": {
        "display": "JavaScript", "suffix": ".js",
        "start": (r"^\s*(function|const|let|var|class|export|async)\b",
                  r"^\s*'.use strict'"),
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
    """砍掉模型吐在代码前面的自言自语。

    两条线索：
      a) 有 </think> 闭合标签 → 从它之后开始
      b) 否则找第一个「像代码开头」的行（start_markers）
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
        return None, "本机没有该语言的检查工具"
    except Exception as e:
        return False, str(e)[:120]
    finally:
        os.unlink(path)


def probe_sql_semantics(model, tok, rounds=3):
    """SQL 专属：语法过了还没完，用真数据查一遍看答案对不对。"""
    schema = """
    CREATE TABLE employees (name TEXT, department TEXT, salary INT);
    INSERT INTO employees VALUES ('Alice','Eng',100), ('Bob','Eng',120),
                                 ('Carol','Sales',90), ('Dave','Sales',150);
    """
    expected = {"Bob": ("Eng", 120), "Dave": ("Sales", 150)}   # 姓名→(部门,工资)
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
            detail = "" if ok else "得到 %s" % (got,)
        except Exception as e:
            ok, detail = False, str(e)[:100]
        con.close()
        hit += ok
        print("  第%d次：%s %s" % (i + 1, "语义正确" if ok else "不对", detail))
        print("    " + " ".join(sql.split())[:150])
    print("  → SQL 语义正确率 %d/%d" % (hit, rounds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default=None,
                    help="只跑某一种：%s" % ", ".join(LANGS))
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
            print("  第%d次：%s %s" % (
                i + 1, {True: "语法通过", False: "语法失败",
                        None: "未验证"}[ok], msg))
            if args.rounds == 1 and code:
                for line in code.splitlines()[:8]:
                    print("    | " + line)
    print("=" * 62)
    print("注意：语法通过 ≠ 逻辑正确。要判断逻辑，必须给测试（这正是反思环的前提）。")


if __name__ == "__main__":
    main()
