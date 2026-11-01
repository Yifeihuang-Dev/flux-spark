#!/bin/bash
# 短时实验（约 75 分钟）
cd "$(cd "$(dirname "$0")/.." && pwd)"
SFT=/Volumes/LLM/hf/MiniCPM5-2B-SFT-MLX-4bit

echo "########## [1/2] 审查器 v4：负样本25% + 500步（因果拆解）##########"
python3 -m mlx_lm lora --config scripts/train_reviewer_v4.yaml
echo "----- v4 评估 -----"
python3 scripts/eval_reviewer.py --model $SFT \
    --adapter /Volumes/LLM/adapters/reviewer-v4 \
    --out /Volumes/LLM/logs/rev_v4.json

echo
echo "########## [2/2] HumanEval best-of-5：SFT 版（看收益曲线）##########"
python3 scripts/eval_humaneval.py --model $SFT --chat-template --repeat 5 \
    --out /Volumes/LLM/logs/he_sft5.json

echo
echo "########## 短时实验全部完成 ##########"
