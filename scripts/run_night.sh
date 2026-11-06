#!/bin/bash
# Unattended overnight run (~2.5 hours, comfortably inside a 5-hour budget)
#
# Ordering: fast and critical first, so an interruption still leaves the core
# numbers in hand.
#   1-2  HumanEval single-sample   -> the executor's true level (~10 min each)
#   3-4  Reviewer v2 / v3          -> push false positives down, test the negative ratio
#   5-6  HumanEval best-of-3       -> how much "sample more + filter" adds
cd "$(cd "$(dirname "$0")/.." && pwd)"

SFT=/Volumes/LLM/hf/MiniCPM5-2B-SFT-MLX-4bit
BASE=/Volumes/LLM/hf/openbmb__MiniCPM5-2B-MLX

echo "########## [1/6] HumanEval 164: SFT (templated) ##########"
python3 scripts/eval_humaneval.py --model $SFT --chat-template \
    --out /Volumes/LLM/logs/he_sft1.json

echo
echo "########## [2/6] HumanEval 164: Base (raw continuation, control) ##########"
python3 scripts/eval_humaneval.py --model $BASE \
    --out /Volumes/LLM/logs/he_base1.json

echo
echo "########## [3/6] Reviewer v2: negatives 25%->40%, 300 steps ##########"
python3 -m mlx_lm lora --config scripts/train_reviewer_v2.yaml
python3 scripts/eval_reviewer.py --model $SFT \
    --adapter /Volumes/LLM/adapters/reviewer-v2 \
    --out /Volumes/LLM/logs/rev_v2.json

echo
echo "########## [4/6] Reviewer v3: same data, 500 steps (more epochs) ##########"
python3 -m mlx_lm lora --config scripts/train_reviewer_v3.yaml
python3 scripts/eval_reviewer.py --model $SFT \
    --adapter /Volumes/LLM/adapters/reviewer-v3 \
    --out /Volumes/LLM/logs/rev_v3.json

echo
echo "########## [5/6] HumanEval best-of-3: SFT ##########"
python3 scripts/eval_humaneval.py --model $SFT --chat-template --repeat 3 \
    --out /Volumes/LLM/logs/he_sft3.json

echo
echo "########## [6/6] HumanEval best-of-3: Base ##########"
python3 scripts/eval_humaneval.py --model $BASE --repeat 3 \
    --out /Volumes/LLM/logs/he_base3.json

echo
echo "########## overnight run finished ##########"
