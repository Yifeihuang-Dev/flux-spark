#!/usr/bin/env python3
"""Reviewer capability evaluation: recall + false-positive rate.

Because the test-set bugs are injected by us, the labels are 100% correct, so both
metrics are exact — no human judgement needed.

- recall: how many injected bugs the model reported
- false-positive rate: how many clean samples it wrongly flagged

Usage:
  # baseline
  python3 eval_reviewer.py --model /Volumes/LLM/hf/openbmb__MiniCPM5-2B-MLX
  # fine-tuned
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
    """Extract the issue types the model reported (counted per occurrence)."""
    if NO_ISSUE in text:
        return []
    found = []
    for cat in CATS:
        found.extend([cat] * len(re.findall(re.escape(cat), text)))
    return found


def reported_any(text):
    """Loose check: did the model report any issue at all (regardless of wording or category)?

    Counting only our category words underrates the baseline — an untuned model can
    spot a bug but does not know our vocabulary and calls a boundary issue a "logic
    error". This metric measures "did it notice something".
    """
    if NO_ISSUE in text:
        return False
    if any(cat in text for cat in CATS):
        return True
    # Bulleted output (lines starting with "类型:", or 1. / - / * lists)
    return bool(re.search(r"类型\s*[:：]|^\s*[-*]\s+|^\s*\d+[\.、)]\s*", text, re.M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="/Volumes/LLM/datasets/reviewer_mlx/test.jsonl")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N rows (quick check)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="dump per-row raw output to a file")
    args = ap.parse_args()

    from mlx_lm import generate, load

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit > 0:
        rows = rows[: args.limit]

    tag = "fine-tuned" if args.adapter else "base"
    print(f"[{tag}] model={args.model}")
    print(f"[{tag}] adapter={args.adapter or '(none)'}")
    print(f"[{tag}] test set: {len(rows)} rows")

    model, tokenizer = load(args.model, adapter_path=args.adapter)

    inj_total = 0        # total injected bugs
    rec_total = 0        # bugs reported (matched by type, no double counting)
    reported_total = 0   # total items the model reported
    neg_total = 0        # clean samples
    neg_false = 0        # clean samples wrongly flagged
    pos_samples = 0      # samples with bugs
    perfect = 0          # samples where every bug was caught
    loose_rec = 0        # loose: any reported issue counts, regardless of category wording

    dumps = []
    for i, row in enumerate(rows, 1):
        prompt = row["messages"][0]["content"]
        truth = row["meta"]["types"]
        # Critical: passing a string to generate() does not apply the chat template and
        # the model degenerates into raw continuation (parroting the prompt). Template it
        # by hand and disable thinking — training taught "answer with an issue list", so
        # leaving thinking on makes the format mismatch.
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
            out = f"[generation failed] {type(exc).__name__}: {exc}"

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
            # Count false positives loosely: any reported issue counts, not just our vocabulary
            if reported_any(out):
                neg_false += 1

        dumps.append({"truth": truth, "reported": rep, "raw": out})
        if i % 8 == 0:
            print(f"  ran {i}/{len(rows)}", flush=True)

    def pct(a, b):
        return a / b if b else 0.0

    print(f"\n{'=' * 62}")
    print(f"[{tag}] strict recall {rec_total}/{inj_total} = {pct(rec_total, inj_total):6.1%}"
          f"   matched on our category words (untuned models are disadvantaged here)")
    print(f"[{tag}] loose recall  {loose_rec}/{pos_samples} = {pct(loose_rec, pos_samples):6.1%}"
          f"   any noticed issue counts, regardless of wording")
    print(f"[{tag}] false pos.    {neg_false}/{neg_total} = {pct(neg_false, neg_total):6.1%}"
          f"   clean code wrongly flagged")
    print(f"[{tag}] all caught   {perfect}/{pos_samples} = {pct(perfect, pos_samples):6.1%}"
          f"   samples with every bug found")
    print(f"[{tag}] reported/true {reported_total}/{inj_total} = "
          f"{reported_total / max(inj_total, 1):.2f}      >1 means over-reporting")

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
        print(f"\nresults written to {args.out}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            for d in dumps:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"per-row output written to {args.dump}")


if __name__ == "__main__":
    main()
