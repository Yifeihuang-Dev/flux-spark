#!/usr/bin/env bash
# Unattended LoRA fine-tuning launcher
#   - caffeinate keeps the machine awake
#   - nohup keeps it alive after the terminal closes
#   - on a crash, resumes automatically from the latest checkpoint (up to 5 tries)
# Usage: nohup caffeinate -dimsu ./run_train.sh &
set -u

CONFIG="$(cd "$(dirname "$0")" && pwd)/train_lora.yaml"
ADAPTER_DIR="/Volumes/LLM/adapters/executor-v1"
LOG="/Volumes/LLM/logs/train.log"
MAX_RETRY=5

mkdir -p "$ADAPTER_DIR" /Volumes/LLM/logs
echo "[$(date '+%F %T')] ==== starting unattended fine-tuning ====" >> "$LOG"

latest_adapter() {
  # mlx-lm's save_every writes checkpoints named like 0000050_adapters.safetensors
  ls -t "$ADAPTER_DIR"/*adapters*.safetensors 2>/dev/null | head -1
}

for attempt in $(seq 1 "$MAX_RETRY"); do
  RESUME_ARGS=""
  ckpt="$(latest_adapter)"
  if [ -n "$ckpt" ]; then
    RESUME_ARGS="--resume-adapter-file $ckpt"
    echo "[$(date '+%F %T')] attempt $attempt: resuming from $ckpt" >> "$LOG"
  else
    echo "[$(date '+%F %T')] attempt $attempt: starting from scratch" >> "$LOG"
  fi

  # shellcheck disable=SC2086
  python3 -m mlx_lm lora --config "$CONFIG" $RESUME_ARGS >> "$LOG" 2>&1
  code=$?

  if [ "$code" -eq 0 ]; then
    echo "[$(date '+%F %T')] training finished cleanly (exit 0)" >> "$LOG"
    break
  fi

  echo "[$(date '+%F %T')] crashed with code=$code; retrying in 30s" >> "$LOG"
  sleep 30
done

echo "[$(date '+%F %T')] ==== all done ====" >> "$LOG"
