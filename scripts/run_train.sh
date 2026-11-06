#!/usr/bin/env bash
# 无人值守 LoRA 微调启动器
#   - caffeinate 阻止睡眠
#   - nohup 保证终端关闭后继续
#   - 异常崩溃自动用最新 checkpoint 续训（最多 5 次）
# 用法: nohup caffeinate -dimsu ./run_train.sh &
set -u

CONFIG=""$(cd "$(dirname "$0")" && pwd)/train_lora.yaml""
ADAPTER_DIR="/Volumes/LLM/adapters/executor-v1"
LOG="/Volumes/LLM/logs/train.log"
MAX_RETRY=5

mkdir -p "$ADAPTER_DIR" /Volumes/LLM/logs
echo "[$(date '+%F %T')] ==== 启动无人值守微调 ====" >> "$LOG"

latest_adapter() {
  # mlx-lm 的 save_every 会写出形如 0000050_adapters.safetensors 的 checkpoint
  ls -t "$ADAPTER_DIR"/*adapters*.safetensors 2>/dev/null | head -1
}

for attempt in $(seq 1 "$MAX_RETRY"); do
  RESUME_ARGS=""
  ckpt="$(latest_adapter)"
  if [ -n "$ckpt" ]; then
    RESUME_ARGS="--resume-adapter-file $ckpt"
    echo "[$(date '+%F %T')] 第 $attempt 轮：从 $ckpt 续训" >> "$LOG"
  else
    echo "[$(date '+%F %T')] 第 $attempt 轮：从头开始" >> "$LOG"
  fi

  # shellcheck disable=SC2086
  python3 -m mlx_lm lora --config "$CONFIG" $RESUME_ARGS >> "$LOG" 2>&1
  code=$?

  if [ "$code" -eq 0 ]; then
    echo "[$(date '+%F %T')] 训练正常结束 (exit 0)" >> "$LOG"
    break
  fi

  echo "[$(date '+%F %T')] 异常退出 code=$code，30 秒后重试" >> "$LOG"
  sleep 30
done

echo "[$(date '+%F %T')] ==== 全部结束 ====" >> "$LOG"
