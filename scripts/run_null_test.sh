#!/bin/bash
# 空值处理专项评测 —— 验证 v5（空值增强版）到底有没有用
#
# 背景：主测试集里空值类 bug 只有 4 个（占 6%），就算全对也只影响 6 个百分点，
#       所以上一轮 v5「不如 v3」的结论证据不足。
# 这里用 26 个全新函数（从未出现在任何训练集）造的空值专用测试集，
# 对比 6 个配置：基线（未微调）+ v1~v5。
#
# 前提：SSD 必须挂载（模型和数据都在 /Volumes/LLM 上）
# 用法：bash scripts/run_null_test.sh      （约 20~30 分钟）
cd "$(cd "$(dirname "$0")/.." && pwd)"

SFT=/Volumes/LLM/hf/MiniCPM5-2B-SFT-MLX-4bit
DATA=/Volumes/LLM/datasets/null_test/test.jsonl

if [ ! -d "/Volumes/LLM" ]; then
    echo "错误：SSD 未挂载（/Volumes/LLM 不存在），模型和数据都取不到，无法运行。"
    exit 1
fi

echo "########## [0/6] 生成空值专用测试集 ##########"
python3 scripts/make_null_test.py --out "$DATA" --neg-ratio 0.40

echo
echo "########## [1/6] 基线（SFT 未微调）##########"
python3 scripts/eval_reviewer.py --model "$SFT" --data "$DATA" \
    --out /Volumes/LLM/logs/null_baseline.json

echo
echo "########## [2/6] v1（负样本25% / 300步）##########"
python3 scripts/eval_reviewer.py --model "$SFT" \
    --adapter /Volumes/LLM/adapters/reviewer-v1 --data "$DATA" \
    --out /Volumes/LLM/logs/null_v1.json

echo
echo "########## [3/6] v2（负样本40% / 300步）##########"
python3 scripts/eval_reviewer.py --model "$SFT" \
    --adapter /Volumes/LLM/adapters/reviewer-v2 --data "$DATA" \
    --out /Volumes/LLM/logs/null_v2.json

echo
echo "########## [4/6] v3（负样本40% / 500步，当前最佳）##########"
python3 scripts/eval_reviewer.py --model "$SFT" \
    --adapter /Volumes/LLM/adapters/reviewer-v3 --data "$DATA" \
    --out /Volumes/LLM/logs/null_v3.json

echo
echo "########## [5/6] v4（负样本25% / 500步）##########"
python3 scripts/eval_reviewer.py --model "$SFT" \
    --adapter /Volumes/LLM/adapters/reviewer-v4 --data "$DATA" \
    --out /Volumes/LLM/logs/null_v4.json

echo
echo "########## [6/6] v5（空值增强 / 500步）##########"
python3 scripts/eval_reviewer.py --model "$SFT" \
    --adapter /Volumes/LLM/adapters/reviewer-v5 --data "$DATA" \
    --out /Volumes/LLM/logs/null_v5.json

echo
echo "########## 全部完成 ##########"
echo "结果汇总："
grep -h -E "严格召回|误报率" /Volumes/LLM/logs/null_*.json 2>/dev/null || \
    echo "  （见 /Volumes/LLM/logs/null_*.json）"
