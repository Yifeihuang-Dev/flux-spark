#!/usr/bin/env python3
"""审查器能力评估：召回率 + 误报率。

因为测试集里的 bug 是我们自己注入的，标签 100% 正确，
所以这两个指标可以精确计算，不需要任何人工判断。

- 召回率：注入的 bug 里，被模型报出了几个
- 误报率：干净代码里，被模型乱报了几个

用法:
  # 基线
  python3 eval_reviewer.py --model /Volumes/LLM/hf/openbmb__MiniCPM5-2B-MLX
  # 微调后
  python3 eval_reviewer.py --model /Volumes/LLM/hf/openbmb__MiniCPM5-2B-MLX \
      --adapter /Volumes/LLM/adapters/reviewer-v1
"""

import argparse
import json
import re
from collections import Counter

CATS = ["边界条件", "逻辑错误", "空值处理"]
NO_ISSUE = "无问题"


def parse_types(text):
    """提取模型报出的问题类型（按出现次数计）。"""
    if NO_ISSUE in text:
        return []
    found = []
    for cat in CATS:
        found.extend([cat] * len(re.findall(re.escape(cat), text)))
    return found


def reported_any(text):
    """宽松判断：模型到底有没有报出问题（不管它用什么词、归到哪一类）。

    只看类型词会低估基线——未微调的模型抓得到 bug，但不知道我们的分类词汇，
    会把「边界条件」说成「逻辑错误」。这个指标衡量的是"有没有看出问题"。
    """
    if NO_ISSUE in text:
        return False
    if any(cat in text for cat in CATS):
        return True
    # 输出了条目化的内容（"类型:" 开头，或 1. / - / * 列表）
    return bool(re.search(r"类型\s*[:：]|^\s*[-*]\s+|^\s*\d+[\.、)]\s*", text, re.M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="/Volumes/LLM/datasets/reviewer_mlx/test.jsonl")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，用于快速验证")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="保存逐条原始输出到文件")
    args = ap.parse_args()

    from mlx_lm import generate, load

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit > 0:
        rows = rows[: args.limit]

    tag = "微调后" if args.adapter else "基座"
    print(f"[{tag}] 模型={args.model}")
    print(f"[{tag}] adapter={args.adapter or '（无）'}")
    print(f"[{tag}] 测试集 {len(rows)} 条")

    model, tokenizer = load(args.model, adapter_path=args.adapter)

    inj_total = 0        # 注入的 bug 总数
    rec_total = 0        # 被报出的 bug 数（按类型配对，不多算）
    reported_total = 0   # 模型总共报出的条数
    neg_total = 0        # 干净样本数
    neg_false = 0        # 干净样本里被误报的
    pos_samples = 0      # 有 bug 的样本数
    perfect = 0          # 所有 bug 都抓到的样本数
    loose_rec = 0        # 宽松：只要报了问题就算，不论分类用词

    dumps = []
    for i, row in enumerate(rows, 1):
        prompt = row["messages"][0]["content"]
        truth = row["meta"]["types"]
        # 关键：generate 传字符串不会自动套 chat template，模型会退化成裸续写
        # （表现为复读提示词）。必须手动套模板，并关掉 thinking——
        # 训练数据教的是"直接给问题列表"，推理时不关 thinking 格式就对不上。
        try:
            fmt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            fmt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)

        try:
            out = generate(model, tokenizer, prompt=fmt,
                           max_tokens=args.max_tokens, verbose=False)
        except Exception as exc:  # noqa: BLE001
            out = f"[生成失败] {type(exc).__name__}: {exc}"

        rep = parse_types(out)
        inj, rpt = Counter(truth), Counter(rep)
        recalled = sum(min(inj[c], rpt[c]) for c in inj)

        inj_total += len(truth)
        rec_total += recalled
        reported_total += len(rep)

        if truth:
            pos_samples += 1
            if recalled == len(truth):
                perfect += 1
            if reported_any(out):
                loose_rec += 1
        else:
            neg_total += 1
            # 用宽松口径判误报：模型说了任何问题都算，不能只认我们的类型词
            if reported_any(out):
                neg_false += 1

        dumps.append({"truth": truth, "reported": rep, "raw": out})
        if i % 8 == 0:
            print(f"  已跑 {i}/{len(rows)}", flush=True)

    def pct(a, b):
        return a / b if b else 0.0

    print(f"\n{'=' * 62}")
    print(f"[{tag}] 严格召回  {rec_total}/{inj_total} = {pct(rec_total, inj_total):6.1%}"
          f"   按我们的分类词匹配（未微调时天然吃亏）")
    print(f"[{tag}] 宽松召回  {loose_rec}/{pos_samples} = {pct(loose_rec, pos_samples):6.1%}"
          f"   只要看出有问题就算，不论用词")
    print(f"[{tag}] 误报率    {neg_false}/{neg_total} = {pct(neg_false, neg_total):6.1%}"
          f"   干净代码被乱报的比例")
    print(f"[{tag}] 完全报对  {perfect}/{pos_samples} = {pct(perfect, pos_samples):6.1%}"
          f"   所有 bug 都抓到的样本")
    print(f"[{tag}] 报出/实际 {reported_total}/{inj_total} = "
          f"{reported_total / max(inj_total, 1):.2f}      >1 说明乱报")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "tag": tag, "model": args.model, "adapter": args.adapter,
                "recall": pct(rec_total, inj_total), "recalled": rec_total,
                "injected": inj_total,
                "false_alarm": pct(neg_false, neg_total),
                "neg_false": neg_false, "neg_total": neg_total,
                "perfect": perfect, "pos_samples": pos_samples,
                "reported": reported_total,
            }, fh, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.out}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            for d in dumps:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"逐条输出已写入 {args.dump}")


if __name__ == "__main__":
    main()
