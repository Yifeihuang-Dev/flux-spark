#!/usr/bin/env python3
"""
Flux Spark —— 本地小模型自主写代码，并对照需求自查自纠。

用法：
    python3 fluxspark.py "写一个函数判断列表是否单调，允许相邻元素相等"
    python3 fluxspark.py "..." --tests 'assert f([1,2,2,4]); assert not f([1,3,2])'
    python3 fluxspark.py "..." --tests '...' --n 3      # 最强模式
    python3 fluxspark.py "..." --selftest               # 让模型自己先写测试
    python3 fluxspark.py "..." --out result.py

设计要点：
    1. 「测试」是可选增强。给了测试就能享受 best-of-N 筛选和最终验证；
       不给也能用，只是退化为「生成 + 反思自纠」，无法确认真的改对了。
    2. 反思环（SGSR）不依赖任何测试，输入只有「需求 + 代码」，所以任何时候都能跑。
    3. 代码输出走 stdout，过程日志走 stderr —— 可以直接重定向成 .py 文件。
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

# 备用 Prompt：当模型开始"讲思路"而不是写代码时，用更硬的措辞重试一次
STRICT_TMPL = """Write ONE Python function for the requirement below.

Requirement:
{req}

CRITICAL: your reply must be a single Python function definition that STARTS with "def ".
No explanation. No markdown fences. No thinking out loud. No repeated code."""

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
    """过程日志一律走 stderr，保证 stdout 只有干净代码。"""
    print(msg, file=sys.stderr)


# ---------- 模型 ----------

def load_model(path):
    if not os.path.exists(path):
        sys.exit(f"找不到模型：{path}\n（SSD 挂了吗？默认路径在 /Volumes/LLM/ 下）")
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


# ---------- 代码处理 ----------

def strip_fences(text):
    """去掉 markdown 代码块标记"""
    t = text.strip()
    for fence in ("```python", "```py", "```"):
        if t.startswith(fence):
            t = t[len(fence):]
            break
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _pick_code(text):
    """在一段文本里找出「以 def 开头、且能被 ast 解析的最长前缀」。

    小模型有三种典型脏输出：
      1) 代码后面接着自言自语（"That's it..."、</think>）
      2) 代码块整体带 4 空格缩进（因为在列表/引用里）→ 要先 dedent
      3) 把同一个函数重复输出两遍
    统一解法：dedent 后从第一个 def 起，反向收缩到能解析、且确实含函数定义的最长前缀。
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
    """从生成结果里取出可执行的纯代码；取不到就返回空串（交给上层重试）"""
    # 优先用第一个围栏代码块（模型常常把正文写在围栏里、正文前后还有一堆废话）
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", raw, re.S)
    for text in list(blocks) + [strip_fences(raw)]:
        code = _pick_code(text)
        if code:
            return code
    return ""


def entry_of(code):
    """从代码里解析入口函数名（第一个顶层 def）"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


# ---------- 测试执行 ----------

def run_tests(code, tests, entry, timeout=15):
    """返回 (是否通过, 报错摘要)。测试里的 f( 占位符会被替换成真实入口名。"""
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
        return False, f"执行超时（{timeout}s）"
    finally:
        os.unlink(path)


# ---------- 主流程 ----------

def run_once(req, tests_cli, args, model, tok, sampler):
    """跑一道题：生成 → 测试判定 → 反思自纠 → 输出。返回最终代码。"""
    t0 = time.time()
    if not args.quiet:
        log("=" * 56)
        log("Flux Spark")
        log("=" * 56)
        log(f"需求：{req}")

    if not args.quiet:
        log(f"\n[1/3] 生成代码{'（采样 %d 次）' % args.n if args.n > 1 else ''}")

    # ---- 阶段 1：生成 ----
    cands = []
    for k in range(args.n):
        raw = generate(model, tok, GEN_TMPL.format(req=req),
                       max_tokens=args.max_tokens,
                       sampler=sampler if k > 0 else None)
        code = clean_code(raw)
        if not code:
            # 模型大概率在"讲思路"，换个更硬的指令重试一次
            raw = generate(model, tok, STRICT_TMPL.format(req=req),
                           max_tokens=args.max_tokens,
                           sampler=sampler if k > 0 else None)
            code = clean_code(raw)
        entry = entry_of(code)
        cands.append({"code": code, "entry": entry, "passed": None})
    base = cands[0]
    entry = base["entry"]

    if not args.quiet:
        log(f"      入口函数：{entry or '（未能解析）'}")

    # ---- 测试来源三级降级 ----
    tests = tests_cli
    test_src = "用户提供"
    if not tests and args.selftest:
        if not args.quiet:
            log("\n[测试] 无现成测试，让模型自己先写测试")
        raw_t = generate(model, tok, SELFTEST_TMPL.format(
            req=req, code=base["code"], entry=entry or "f"),
            max_tokens=300)
        # 只保留 assert 行，过滤掉模型可能夹带的解释
        tests = "\n".join(l for l in raw_t.split("\n")
                          if l.strip().startswith("assert"))
        test_src = "模型自生成（置信度低）"
        if not tests:
            log("      模型没写出可用的测试，退化为无测试模式")

    mode = "完整闭环（有测试）" if tests else "仅生成 + 反思自纠（无测试）"
    if not args.quiet:
        log(f"      模式：{mode}")
        if tests:
            log(f"      测试来源：{test_src}")

    # ---- 阶段 2：先用测试筛选 ----
    if tests:
        for c in cands:
            ok, err = run_tests(c["code"], tests, c["entry"])
            c["passed"] = ok
            c["err"] = err
        n_pass = sum(1 for c in cands if c["passed"])
        if not args.quiet:
            if args.n > 1:
                log(f"      best-of-{args.n}：{n_pass}/{args.n} 个候选通过测试")
            else:
                log(f"      测试结果：{'通过' if cands[0]['passed'] else '未通过'}")
                if not cands[0]["passed"] and cands[0].get("err"):
                    log(f"      报错摘要：{cands[0]['err'].strip().splitlines()[-1][:120]}")

        good = [c for c in cands if c["passed"]]
        if good:
            best = good[0]
            if not args.quiet:
                log("\n[2/3] 已有候选通过测试" +
                    ("，无需修复" if len(good) == args.n else f"，{len(good)}/{args.n} 通过"))
                log("\n[3/3] 跳过反思（已通过，避免改坏）" if len(good) == args.n
                    else "")
            if len(good) == args.n:
                return emit(best["code"], args, tz=time.time() - t0,
                            status="通过全部测试")
            # 部分通过：只对没通过的做反思修复
            target = next(c for c in cands if not c["passed"])
        else:
            target = cands[0]
    else:
        target = cands[0]

    if args.no_reflect:
        if not args.quiet:
            log("\n[2/3] 已关闭反思，直接输出生成结果")
        return emit(target["code"], args, tz=time.time() - t0,
                    status="未验证（关闭反思）" if not tests else "未通过测试")

    # ---- 阶段 3：反思 + 修复 ----
    if not args.quiet:
        log("\n[2/3] 对照需求反思（总结思路 / 找不满足处 / 给建议）")
    reflection = generate(model, tok, REFLECT_TMPL.format(
        req=req, code=target["code"]), max_tokens=320).strip()

    if not args.quiet:
        log("\n[3/3] 按反思建议重写")
    new_raw = generate(model, tok, FIX_TMPL.format(
        req=req, code=target["code"], reflection=reflection),
        max_tokens=args.max_tokens)
    new_code = clean_code(new_raw)
    if not new_code.strip():
        new_code = target["code"]

    status = "未验证"
    if tests:
        ok, err = run_tests(new_code, tests, entry_of(new_code) or target["entry"])
        status = "修复后通过测试" if ok else "修复后仍未通过测试"
        # 只接受变好的修改：新方案不行就退回旧代码
        if not ok and target.get("passed") is False:
            if not args.quiet:
                log("      新方案未通过，保留原有版本")
            new_code = target["code"]
            status = "未通过测试（已退回原版本）"

    return emit(new_code, args, tz=time.time() - t0, status=status)


def emit(code, args, tz, status):
    if not args.quiet:
        log("=" * 56)
        log(f"状态：{status}    耗时：{tz:.0f}s")
        log("=" * 56)
        log("")
    # 代码本体走 stdout，方便重定向成 .py
    print(code)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(code + "\n")
        if not args.quiet:
            log(f"\n已写入 {args.out}")
    return code


# ---------- 交互式终端 ----------

HELP_TEXT = """可用命令：
  /help, /?        显示本帮助
  /n <数字>        设置采样次数 best-of-N（>=2 时需要提供测试来筛选）
  /selftest        开关：没有测试时让模型自己先写测试
  /reflect         开关：反思自纠（默认开）
  /quiet           开关：只输出代码，不打过程日志
  /save <文件名>    把上一次的结果代码存到文件
  /quit, /q, exit  退出
  Ctrl+D           退出        Ctrl+C  中断当前生成（不退出会话）

提示：
  - 需求用自然语言写就行，回车即开始跑
  - 测试可选，直接回车跳过；函数名用 f() 占位，例如
        assert f([1,2,2,4]); assert not f([1,3,2])
  - 测试越准，best-of-N 和「确认修好了」越有把握"""


def ask(prompt):
    """input() 的提示语默认打到 stdout，会把重定向出来的代码弄脏。
    所以把提示语写进 stderr，代码依旧独占 stdout。"""
    sys.stderr.write(prompt)
    sys.stderr.flush()
    return input()


def print_banner(model_path, n, selftest, reflect, quiet):
    log("=" * 56)
    log(" Flux Spark    交互式终端")
    log("-" * 56)
    log(f" 模型：{os.path.basename(model_path)}")
    log(f" 当前：采样 {n} 次 | 自动生成测试 {'开' if selftest else '关'} "
        f"| 反思自纠 {'开' if reflect else '关'} | 安静 {'开' if quiet else '关'}")
    log("-" * 56)
    log(" 输入自然语言需求，回车开始。")
    log(" /help 看命令，/quit 退出，Ctrl+D 也能退出。")
    log("=" * 56)


def interactive_session(model, tok, args, sampler):
    """REPL：模型已经加载好了，循环接需求，不再重复加载。"""
    try:
        import readline        # 支持上下箭头翻历史、行内编辑；没有也无所谓
    except ImportError:
        pass

    print_banner(args.model, args.n, args.selftest,
                 not args.no_reflect, args.quiet)

    last_code = None
    turn = 0
    while True:
        try:
            req = ask(f"\n[{turn + 1}] 需求> ").strip()
        except EOFError:
            log("\n（Ctrl+D）已退出")
            break
        except KeyboardInterrupt:
            log("\n（Ctrl+C）中断一轮。继续输入需求，或 Ctrl+D / /quit 退出。")
            continue

        if not req:
            continue

        # ---- 命令 ----
        low = req.lower()
        if low in ("/quit", "/q", "/exit", "exit", "quit"):
            log("已退出")
            break
        if low in ("/help", "/?", "/h", "help"):
            log(HELP_TEXT)
            continue
        if low.startswith("/save"):
            parts = req.split(None, 1)
            if len(parts) < 2:
                log("用法：/save <文件名>")
            elif not last_code:
                log("还没有产出过代码，先跑一道题吧")
            else:
                path = parts[1].strip()
                try:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(last_code + "\n")
                    log(f"已写入 {path}")
                except OSError as e:
                    log(f"写入失败：{e}")
            continue
        if low.startswith("/n"):
            parts = req.split(None, 1)
            if len(parts) < 2 or not parts[1].strip().isdigit():
                log("用法：/n <数字>，例如 /n 3")
            else:
                args.n = max(1, int(parts[1].strip()))
                log(f"采样次数已设为 {args.n}")
                if args.n > 1:
                    log("注意：n>=2 时需要给测试，否则无法筛选候选（会提示降到 1）")
            continue
        if low == "/selftest":
            args.selftest = not args.selftest
            log(f"自动生成测试：{'开' if args.selftest else '关'}")
            continue
        if low == "/reflect":
            args.no_reflect = not args.no_reflect
            log(f"反思自纠：{'关' if args.no_reflect else '开'}")
            continue
        if low == "/quiet":
            args.quiet = not args.quiet
            log(f"安静模式：{'开' if args.quiet else '关'}")
            continue
        if req.startswith("/"):
            log(f"未知命令：{req}    输入 /help 看可用命令")
            continue

        # ---- 测试（可选）----
        tests = None
        try:
            raw_tests = ask("    测试(可选，直接回车跳过)> ").strip()
            tests = raw_tests or None
        except EOFError:
            log("\n（Ctrl+D）已退出")
            break
        except KeyboardInterrupt:
            log("\n（Ctrl+C）已取消这一轮")
            continue

        # best-of-N 必须有测试才能筛选，否则退化为单次
        turn += 1

        # best-of-N 没有测试就无法筛选，临时退回单次，跑完再复原
        saved_n = None
        if args.n > 1 and not tests and not args.selftest:
            log(f"\n本次设了 n={args.n}，但没给测试也没开自动生成，无法筛候选。")
            log("按单次处理。想要 best-of-N 就补一条 assert，或 /selftest 打开自动生成。")
            saved_n, args.n = args.n, 1

        try:
            last_code = run_once(req, tests, args, model, tok, sampler)
        except KeyboardInterrupt:
            log("\n（Ctrl+C）已中断这一轮生成，会话继续。")
            last_code = None
        except Exception as e:                      # 单题出错不该拖垮整个会话
            log(f"\n这一题出错了：{type(e).__name__}: {e}")
            last_code = None
        finally:
            if saved_n is not None:
                args.n = saved_n


def build_arg_parser():
    ap = argparse.ArgumentParser(
        prog="fluxspark",
        description="Flux Spark：本地小模型写代码 + 对照需求自查自纠",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""用法举例：
  单次  python3 fluxspark.py "写一个函数判断列表是否单调，允许相等" \\
            --tests 'assert f([1,2,2,4]); assert not f([1,3,2])'
  交互  python3 fluxspark.py -i
  存文件 python3 fluxspark.py "..." --tests '...' > out.py

代码走 stdout，过程日志走 stderr，所以可以直接重定向成 .py 文件。""")
    ap.add_argument("requirement", nargs="?", default=None,
                    help="用自然语言描述你要的函数；不给则进入交互式终端")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="强制进入交互式终端（模型只加载一次，适合连着问）")
    ap.add_argument("--tests", default=None,
                    help="assert 形式的测试，多条用分号或换行隔开；函数名用 f() 占位")
    ap.add_argument("--selftest", action="store_true",
                    help="没有现成测试时，让模型先写测试（置信度降级）")
    ap.add_argument("--n", type=int, default=1,
                    help="采样几次（best-of-N）。>1 必须有测试才能筛选，默认 1")
    ap.add_argument("--no-reflect", action="store_true",
                    help="关掉反思自纠（用于对照：看反思到底起了多大作用）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SFT 基座模型路径")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=None, help="同时写入指定文件")
    ap.add_argument("--quiet", action="store_true", help="只输出代码，不打过程日志")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    # 单次模式不能既没需求又没显式要求交互
    interactive = args.interactive or (not args.requirement and sys.stdin.isatty())
    if not interactive and not args.requirement:
        ap.error("请给出需求描述，或用 -i 进入交互式终端")

    model, tok = load_model(args.model)

    # best-of-N 用带温度的采样器，保证候选之间真的不一样
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
        log("\n已中断")
