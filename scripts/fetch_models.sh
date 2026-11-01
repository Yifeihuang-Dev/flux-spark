#!/usr/bin/env bash
# 从 HuggingFace 下载模型权重到外接 SSD（/Volumes/LLM）
# 不依赖任何 Python 包，纯 curl，支持断点续传。
# 用法:
#   ./fetch_models.sh router      # 路由器 Qwen3.5-0.8B-4bit (0.61GB)
#   ./fetch_models.sh executor    # 执行器 MiniCPM5-2B-MLX  (1.33GB)
#   ./fetch_models.sh reviewer    # 审查器 Qwen3.5-4B-4bit  (2.85GB)
#   ./fetch_models.sh fallback    # 兜底   Qwen3.5-9B-4bit  (~6GB)
#   ./fetch_models.sh all         # 全部
set -u

DEST="${DEST:-/Volumes/LLM/hf}"

fetch() {
  local repo="$1" dir="$2"; shift 2
  mkdir -p "$DEST/$dir"
  for f in "$@"; do
    local url="https://huggingface.co/$repo/resolve/main/$f"
    local out="$DEST/$dir/$f"
    if [ -s "$out" ]; then
      printf '  跳过(已存在) %s/%s\n' "$dir" "$f"
      continue
    fi
    printf '  下载 %s/%s ...\n' "$dir" "$f"
    if curl -L -C - --retry 3 --retry-delay 2 --max-time 3600 \
         -o "$out" "$url"; then
      printf '  完成 %s/%s  (%s)\n' "$dir" "$f" "$(du -h "$out" | cut -f1)"
    else
      printf '  失败 %s/%s\n' "$dir" "$f"
      rm -f "$out"
    fi
  done
}

router() {
  echo "== 路由器: Qwen3.5-0.8B-4bit =="
  fetch mlx-community/Qwen3.5-0.8B-4bit mlx-community__Qwen3.5-0.8B-4bit \
    config.json model.safetensors model.safetensors.index.json \
    tokenizer.json tokenizer_config.json chat_template.jinja vocab.json
}

executor() {
  echo "== 执行器: MiniCPM5-2B-MLX =="
  fetch openbmb/MiniCPM5-2B-MLX openbmb__MiniCPM5-2B-MLX \
    config.json generation_config.json model.safetensors \
    model.safetensors.index.json tokenizer.json tokenizer_config.json chat_template.jinja
}

reviewer() {
  echo "== 审查器: Qwen3.5-4B-MLX-4bit =="
  fetch mlx-community/Qwen3.5-4B-MLX-4bit mlx-community__Qwen3.5-4B-MLX-4bit \
    config.json model.safetensors model.safetensors.index.json \
    tokenizer.json tokenizer_config.json chat_template.jinja vocab.json
}

fallback() {
  echo "== 兜底: Qwen3.5-9B-4bit =="
  fetch mlx-community/Qwen3.5-9B-4bit mlx-community__Qwen3.5-9B-4bit \
    config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
    model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json
}

case "${1:-all}" in
  router)   router ;;
  executor) executor ;;
  reviewer) reviewer ;;
  fallback) fallback ;;
  all)      router; executor; reviewer ;;
  *) echo "未知参数: $1"; exit 1 ;;
esac

echo "== 当前 SSD 占用 =="
du -sh "$DEST" 2>/dev/null
df -h /Volumes/LLM | tail -1
