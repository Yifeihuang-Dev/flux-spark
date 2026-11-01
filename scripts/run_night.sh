#!/bin/bash
# 今晚挂机任务（约 2.5 小时，5 小时预算内留足余量）
#
# 排序原则：快且关键的先跑，万一中断也先拿到核心数字。
#   1-2  HumanEval 单采样  → 确认执行器真实水平（各约 10 分钟）
#   3-4  审查器 v2 / v3    → 压误报，验证负样本比例的效果
#   5-6  HumanEval best-of-3 → 验证「多采样+筛选」能再加多少
cd "$(cd "$(dirname "$0")/.." && pwd)"

SFT=/Volumes/LLM/hf/MiniCPM5-2B-SFT-MLX-4bit
BASE=/Volumes/LLM/hf/openbmb__MiniCPM5-2B-MLX

echo "########## [1/6] HumanEval 164题：SFT 版（套模板）##########"
python3 scripts/eval_humaneval.py --model $SFT --chat-template \
    --out /Volumes/LLM/logs/he_sft1.json

echo
echo "########## [2/6] HumanEval 164题：Base 版（裸续写，对照）##########"
python3 scripts/eval_humaneval.py --model $BASE \
    --out /Volumes/LLM/logs/he_base1.json

echo
echo "########## [3/6] 审查器 v2：负样本 25%→40%，300 步 ##########"
python3 -m mlx_lm lora --config scripts/train_reviewer_v2.yaml
python3 scripts/eval_reviewer.py --model $SFT \
    --adapter /Volumes/LLM/adapters/reviewer-v2 \
    --out /Volumes/LLM/logs/rev_v2.json

echo
echo "########## [4/6] 审查器 v3：同数据，500 步（更多 epoch）##########"
python3 -m mlx_lm lora --config scripts/train_reviewer_v3.yaml
python3 scripts/eval_reviewer.py --model $SFT \
    --adapter /Volumes/LLM/adapters/reviewer-v3 \
    --out /Volumes/LLM/logs/rev_v3.json

echo
echo "########## [5/6] HumanEval best-of-3：SFT 版 ##########"
python3 scripts/eval_humaneval.py --model $SFT --chat-template --repeat 3 \
    --out /Volumes/LLM/logs/he_sft3.json

echo
echo "########## [6/6] HumanEval best-of-3：Base 版 ##########"
python3 scripts/eval_humaneval.py --model $BASE --repeat 3 \
    --out /Volumes/LLM/logs/he_base3.json

echo
echo "########## 今晚任务全部完成 ##########"
