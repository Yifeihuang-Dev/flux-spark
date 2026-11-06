#!/bin/bash
# Null-handling specific evaluation — does v5 (null-augmented) actually help?
#
# Background: the main test set contains only 4 null-type bugs (6%), so even a
#             perfect score there moves the total by 6 points. That makes the
#             earlier "v5 worse than v3" conclusion unsupported.
# This builds a null-specific test set from 26 brand-new functions (never seen
# in any training set) and compares six configurations: baseline + v1..v5.
#
# Prerequisite: the SSD must be mounted (models and data live on /Volumes/LLM).
# Usage: bash scripts/run_null_test.sh      (~20-30 minutes)
cd "$(cd "$(dirname "$0")/.." && pwd)"

SFT=/Volumes/LLM/hf/MiniCPM5-2B-SFT-MLX-4bit
DATA=/Volumes/LLM/datasets/null_test/test.jsonl

if [ ! -d "/Volumes/LLM" ]; then
    echo "error: SSD not mounted (/Volumes/LLM missing); models and data unavailable."
    exit 1
fi

echo "########## [0/6] building the null-specific test set ##########"
python3 scripts/make_null_test.py --out "$DATA" --neg-ratio 0.40

echo
echo "########## [1/6] baseline (SFT, no fine-tuning) ##########"
python3 scripts/eval_reviewer.py --model "$SFT" --data "$DATA" \
    --out /Volumes/LLM/logs/null_baseline.json

echo
echo "########## [2/6] v1 (25% negatives / 300 steps) ##########"
python3 scripts/eval_reviewer.py --model "$SFT" \
    --adapter /Volumes/LLM/adapters/reviewer-v1 --data "$DATA" \
    --out /Volumes/LLM/logs/null_v1.json

echo
echo "########## [3/6] v2 (40% negatives / 300 steps) ##########"
python3 scripts/eval_reviewer.py --model "$SFT" \
    --adapter /Volumes/LLM/adapters/reviewer-v2 --data "$DATA" \
    --out /Volumes/LLM/logs/null_v2.json

echo
echo "########## [4/6] v3 (40% negatives / 500 steps, current best) ##########"
python3 scripts/eval_reviewer.py --model "$SFT" \
    --adapter /Volumes/LLM/adapters/reviewer-v3 --data "$DATA" \
    --out /Volumes/LLM/logs/null_v3.json

echo
echo "########## [5/6] v4 (25% negatives / 500 steps) ##########"
python3 scripts/eval_reviewer.py --model "$SFT" \
    --adapter /Volumes/LLM/adapters/reviewer-v4 --data "$DATA" \
    --out /Volumes/LLM/logs/null_v4.json

echo
echo "########## [6/6] v5 (null-augmented / 500 steps) ##########"
python3 scripts/eval_reviewer.py --model "$SFT" \
    --adapter /Volumes/LLM/adapters/reviewer-v5 --data "$DATA" \
    --out /Volumes/LLM/logs/null_v5.json

echo
echo "########## all done ##########"
echo "Summary:"
python3 - <<'PY'
import glob, json, os
for f in sorted(glob.glob("/Volumes/LLM/logs/null_*.json")):
    d = json.load(open(f, encoding="utf-8"))
    print(f"{os.path.basename(f):<22} recall {d.get('recall', 0):6.1%}   "
          f"false alarms {d.get('false_alarm', 0):6.1%}")
PY
