#!/usr/bin/env python3
"""生成「空值处理」专用测试集。

背景：主测试集里空值类 bug 只有 4 个（占 6%），导致 v5（空值增强版）
的效果根本测不出来——那 4 个全对也只影响 6 个百分点。

这里用 20 个**全新的、带空值守卫的**种子函数造一个只测空值的测试集。
这些函数从未出现在任何训练集中（SEEDS 里没有它们），所以测的是泛化能力。

用法：
  python3 scripts/make_null_test.py --out /Volumes/LLM/datasets/null_test/test.jsonl
"""
import argparse
import json
import os

PROMPT_TMPL = '''审查以下 Python 函数，找出其中的问题。

```python
{code}
```

请列出所有问题，每行一条，格式为「类型: 描述」。如果没有任何问题，只回答「无问题」。'''

NO_ISSUE = "无问题"

# 20 个带空值守卫的新种子。全部是全新函数名，不出现在任何训练集里。
SEEDS = {
    "get_head": '''def get_head(items):
    if not items:
        return None
    return items[0]''',

    "pick_last": '''def pick_last(arr):
    if len(arr) == 0:
        return None
    return arr[-1]''',

    "mean_of": '''def mean_of(values):
    if not values:
        return 0.0
    return sum(values) / len(values)''',

    "scale_all": '''def scale_all(data):
    if not data:
        return []
    m = max(data)
    out = []
    for x in data:
        out.append(x / m)
    return out''',

    "fetch_value": '''def fetch_value(table, key, default):
    if table is None:
        return default
    return table.get(key, default)''',

    "glue_text": '''def glue_text(chunks):
    if not chunks:
        return ""
    out = ""
    for c in chunks:
        out += c
    return out''',

    "biggest_of": '''def biggest_of(arr):
    if arr is None:
        return None
    best = arr[0]
    for x in arr:
        if x > best:
            best = x
    return best''',

    "size_of": '''def size_of(seq):
    if seq is None:
        return 0
    return len(seq)''',

    "lead_word": '''def lead_word(text):
    if not text:
        return ""
    return text.split()[0]''',

    "row_totals": '''def row_totals(grid):
    if not grid:
        return []
    out = []
    for row in grid:
        out.append(sum(row))
    return out''',

    "take_first": '''def take_first(pairs):
    if not pairs:
        return None
    return pairs[0][0]''',

    "mid_value": '''def mid_value(nums):
    if not nums:
        return None
    s = sorted(nums)
    return s[len(s) // 2]''',

    "blend": '''def blend(a, b):
    if not a or not b:
        return []
    out = []
    for i in range(min(len(a), len(b))):
        out.append(a[i] + b[i])
    return out''',

    "ratio_of": '''def ratio_of(num, den):
    if den == 0:
        return 0.0
    return num / den''',

    "unique_keep": '''def unique_keep(items):
    if items is None:
        return []
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out''',

    "zip_up": '''def zip_up(keys, vals):
    if keys is None:
        return {}
    out = {}
    for i in range(min(len(keys), len(vals))):
        out[keys[i]] = vals[i]
    return out''',

    "trim_edges": '''def trim_edges(text):
    if not text:
        return ""
    return text.strip()''',

    "to_int_safe": '''def to_int_safe(text, fallback):
    if text is None:
        return fallback
    return int(text)''',

    "deep_fetch": '''def deep_fetch(root, path):
    if not path:
        return root
    cur = root
    for p in path:
        cur = cur[p]
    return cur''',

    "chunk_up": '''def chunk_up(items, size):
    if not items:
        return []
    out = []
    for i in range(0, len(items), size):
        out.append(items[i:i + size])
    return out''',
}

# (种子名, 要替换的原文, 替换成, 类型, 描述)
# 变异按种子逐个指定，避免全局字符串匹配误伤其他函数。
MUTATIONS = [
    ("get_head", "    if not items:\n        return None\n", "",
     "缺少空列表守卫，items 为空时访问 items[0] 会抛 IndexError"),
    ("pick_last", "    if len(arr) == 0:\n        return None\n", "",
     "缺少空列表守卫，arr 为空时访问 arr[-1] 会抛 IndexError"),
    ("mean_of", "    if not values:\n        return 0.0\n", "",
     "缺少空列表守卫，空列表会导致除零错误"),
    ("scale_all", "    if not data:\n        return []\n", "",
     "缺少空列表守卫，max() 对空序列会抛 ValueError"),
    ("fetch_value", "    if table is None:\n        return default\n", "",
     "缺少 None 判断，table 为 None 时调用 .get 会抛 AttributeError"),
    ("glue_text", "    if not chunks:\n        return \"\"\n", "",
     "缺少空列表守卫，空输入时行为不确定"),
    ("biggest_of", "    if arr is None:\n        return None\n", "",
     "缺少 None 判断，arr 为 None 时访问 arr[0] 会抛 TypeError"),
    ("size_of", "    if seq is None:\n        return 0\n", "",
     "缺少 None 判断，len(None) 会抛 TypeError"),
    ("lead_word", "    if not text:\n        return \"\"\n", "",
     "缺少空串守卫，text 为空时 split()[0] 会抛 IndexError"),
    ("row_totals", "    if not grid:\n        return []\n", "",
     "缺少空矩阵守卫，未显式处理空输入"),
    ("take_first", "    if not pairs:\n        return None\n", "",
     "缺少空列表守卫，pairs 为空时访问 pairs[0] 会抛 IndexError"),
    ("mid_value", "    if not nums:\n        return None\n", "",
     "缺少空列表守卫，nums 为空时访问 s[0] 会抛 IndexError"),
    ("blend", "    if not a or not b:\n        return []\n", "",
     "缺少空输入守卫，任一列表为空时结果异常"),
    ("ratio_of", "    if den == 0:\n        return 0.0\n", "",
     "缺少除零守卫，den 为 0 时会抛 ZeroDivisionError"),
    ("unique_keep", "    if items is None:\n        return []\n", "",
     "缺少 None 判断，遍历 None 会抛 TypeError"),
    ("zip_up", "    if keys is None:\n        return {}\n", "",
     "缺少 None 判断，len(None) 会抛 TypeError"),
    ("trim_edges", "    if not text:\n        return \"\"\n", "",
     "缺少空串守卫，未显式处理空输入"),
    ("to_int_safe", "    if text is None:\n        return fallback\n", "",
     "缺少 None 判断，int(None) 会抛 TypeError"),
    ("deep_fetch", "    if not path:\n        return root\n", "",
     "缺少空路径守卫，path 为空时行为异常"),
    ("chunk_up", "    if not items:\n        return []\n", "",
     "缺少空列表守卫，未显式处理空输入"),
]

# 把 None 检查改写成等价性检查（None 会漏过去）
REWRITES = [
    ("biggest_of", "    if arr is None:", "    if arr == []:",
     "None 判断写成了空列表判断，None 会漏过去导致抛异常"),
    ("size_of", "    if seq is None:", "    if seq == []:",
     "None 判断写成了空列表判断，None 会漏过去"),
    ("fetch_value", "    if table is None:", "    if table == {}:",
     "None 判断写成了空字典判断，None 会漏过去"),
    ("unique_keep", "    if items is None:", "    if items == []:",
     "None 判断写成了空列表判断，None 会漏过去"),
    ("zip_up", "    if keys is None:", "    if keys == []:",
     "None 判断写成了空列表判断，None 会漏过去"),
    ("to_int_safe", "    if text is None:", "    if text == \"\":",
     "None 判断写成了空串判断，None 会漏过去"),
]


def to_messages(code, issues):
    """issues 为空表示干净代码，答案是「无问题」。"""
    if issues:
        answer = "\n".join(f"空值处理: {d}" for d in issues)
    else:
        answer = NO_ISSUE
    return {
        "messages": [
            {"role": "user", "content": PROMPT_TMPL.format(code=code)},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "seed": None,   # 由调用方填
            "n_issues": len(issues),
            "types": ["空值处理"] * len(issues),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Volumes/LLM/datasets/null_test/test.jsonl")
    ap.add_argument("--neg-ratio", type=float, default=0.25,
                    help="干净代码（负样本）在测试集中的占比，用于测误报")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    samples = []
    n_bug = 0

    # 正样本：删掉守卫
    for name, search, replace, desc in MUTATIONS:
        code = SEEDS[name]
        if search not in code:
            raise SystemExit(f"[错误] 种子 {name} 里找不到要删的守卫:\n{search!r}")
        mutated = code.replace(search, replace, 1)
        s = to_messages(mutated, [desc])
        s["meta"]["seed"] = name
        samples.append(s)
        n_bug += 1

    # 正样本：把 None 检查改写错
    for name, search, replace, desc in REWRITES:
        code = SEEDS[name]
        if search not in code:
            raise SystemExit(f"[错误] 种子 {name} 里找不到要改写的判断:\n{search!r}")
        mutated = code.replace(search, replace, 1)
        s = to_messages(mutated, [desc])
        s["meta"]["seed"] = name
        samples.append(s)
        n_bug += 1

    # 负样本：干净代码
    n_neg = int(n_bug * args.neg_ratio / (1 - args.neg_ratio))
    seen = set()
    neg = []
    for name in sorted(SEEDS):
        if len(neg) >= n_neg:
            break
        s = to_messages(SEEDS[name], [])
        s["meta"]["seed"] = name
        if name in seen:
            continue
        seen.add(name)
        neg.append(s)
    samples.extend(neg)

    with open(args.out, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"已生成 {args.out}")
    print(f"  有 bug 的样本: {n_bug} 条（全部是空值处理类，其中删守卫 {len(MUTATIONS)} / 改写判断 {len(REWRITES)}）")
    print(f"  干净代码样本: {len(neg)} 条（用于测误报）")
    print(f"  合计: {len(samples)} 条")


if __name__ == "__main__":
    main()
