#!/bin/bash
# Short experiment run (~75 minutes)
cd "$(cd "$(dirname "$0")/.." && pwd)"
SFT=/Volumes/LLM/hf/MiniCPM5-2B-SFT-MLX-4bit

echo "########## [1/2] Reviewer v4: 25% negatives + 500 steps (factor isolation) ##########"
python3 -m mlx_lm lora --config scripts/train_reviewer_v4.yaml
echo "----- v4 evaluation -----"
python3 scripts/eval_reviewer.py --model $SFT \
    --adapter /Volumes/LLM/adapters/reviewer-v4 \
    --out /Volumes/LLM/logs/rev_v4.json

echo
echo "########## [2/2] HumanEval best-of-5: SFT (see the gain curve) ##########"
python3 scripts/eval_humaneval.py --model $SFT --chat-template --repeat 5 \
    --out /Volumes/LLM/logs/he_sft5.json

echo
echo "########## short run finished ##########"
