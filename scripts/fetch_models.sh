#!/usr/bin/env bash
# Download model weights from HuggingFace to the external SSD (/Volumes/LLM).
# No Python packages needed — plain curl with resumable downloads.
# Usage:
#   ./fetch_models.sh router      # router   Qwen3.5-0.8B-4bit (0.61GB)
#   ./fetch_models.sh executor    # executor MiniCPM5-2B-MLX  (1.33GB)
#   ./fetch_models.sh reviewer    # reviewer Qwen3.5-4B-4bit  (2.85GB)
#   ./fetch_models.sh fallback    # fallback Qwen3.5-9B-4bit  (~6GB)
#   ./fetch_models.sh all         # everything
set -u

DEST="${DEST:-/Volumes/LLM/hf}"

fetch() {
  local repo="$1" dir="$2"; shift 2
  mkdir -p "$DEST/$dir"
  for f in "$@"; do
    local url="https://huggingface.co/$repo/resolve/main/$f"
    local out="$DEST/$dir/$f"
    if [ -s "$out" ]; then
      printf '  skip (exists) %s/%s\n' "$dir" "$f"
      continue
    fi
    printf '  downloading %s/%s ...\n' "$dir" "$f"
    if curl -L -C - --retry 3 --retry-delay 2 --max-time 3600 \
         -o "$out" "$url"; then
      printf '  done %s/%s  (%s)\n' "$dir" "$f" "$(du -h "$out" | cut -f1)"
    else
      printf '  failed %s/%s\n' "$dir" "$f"
      rm -f "$out"
    fi
  done
}

router() {
  echo "== router: Qwen3.5-0.8B-4bit =="
  fetch mlx-community/Qwen3.5-0.8B-4bit mlx-community__Qwen3.5-0.8B-4bit \
    config.json model.safetensors model.safetensors.index.json \
    tokenizer.json tokenizer_config.json chat_template.jinja vocab.json
}

executor() {
  echo "== executor: MiniCPM5-2B-MLX =="
  fetch openbmb/MiniCPM5-2B-MLX openbmb__MiniCPM5-2B-MLX \
    config.json generation_config.json model.safetensors \
    model.safetensors.index.json tokenizer.json tokenizer_config.json chat_template.jinja
}

reviewer() {
  echo "== reviewer: Qwen3.5-4B-MLX-4bit =="
  fetch mlx-community/Qwen3.5-4B-MLX-4bit mlx-community__Qwen3.5-4B-MLX-4bit \
    config.json model.safetensors model.safetensors.index.json \
    tokenizer.json tokenizer_config.json chat_template.jinja vocab.json
}

fallback() {
  echo "== fallback: Qwen3.5-9B-4bit =="
  fetch mlx-community/Qwen3.5-9B-4bit mlx-community__Qwen3.5-9B-4bit \
    config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
    model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json
}

# Main model: the one the CLI uses by default.
# It is not downloaded but quantized locally from the public base
# openbmb/MiniCPM5-2B-SFT, which is why the repo carries no weights —
# one command reproduces it (architectures match item by item, 4.501 bits/weight).
base() {
  echo "== main: MiniCPM5-2B-SFT-MLX-4bit (local quantization of the public base) =="
  local out="$DEST/MiniCPM5-2B-SFT-MLX-4bit"
  if [ -s "$out/model.safetensors" ]; then
    echo "  skip (exists) $out"
    return
  fi
  if ! command -v mlx_lm.convert >/dev/null 2>&1; then
    echo "  mlx_lm missing: pip install mlx-lm first"
    return 1
  fi
  mlx_lm.convert --hf-path openbmb/MiniCPM5-2B-SFT --mlx-path "$out" \
    -q --q-bits 4 --q-group-size 64 --dtype bfloat16
}

case "${1:-all}" in
  base)     base ;;
  router)   router ;;
  executor) executor ;;
  reviewer) reviewer ;;
  fallback) fallback ;;
  all)      base; router; executor; reviewer ;;
  *) echo "unknown argument: $1"; exit 1 ;;
esac

echo "== SSD usage =="
du -sh "$DEST" 2>/dev/null
df -h /Volumes/LLM | tail -1
