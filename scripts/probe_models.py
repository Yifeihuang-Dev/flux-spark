"""Query the real size and file list of candidate models on HuggingFace.

Purpose: estimate disk usage before downloading, so we know what fits on the
external SSD.
Note: this machine's Python lacks a complete SSL chain, so certificate
verification is disabled here (read-only metadata queries only).
"""

import json
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

MODELS = [
    "openbmb/MiniCPM5-2B",
    "openbmb/MiniCPM5-2B-MLX",
    "mlx-community/MiniCPM5-2B-8bit",
    "Qwen/Qwen3.5-2B",
    "mlx-community/Qwen3.5-0.8B-4bit",
    "mlx-community/Qwen3.5-2B-4bit",
    "mlx-community/Qwen3.5-4B-MLX-4bit",
]


def probe(model_id: str) -> None:
    url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        print(f"{model_id}  query failed: {exc}")
        return

    files = [(s.get("rfilename"), s.get("size") or 0) for s in data.get("siblings", [])]
    total = sum(size for _, size in files)
    print(f"\n{data.get('id')}   total {total / 1024**3:.2f} GB")
    for name, size in sorted(files, key=lambda x: -x[1])[:6]:
        if size > 5 * 1024**2:
            print(f"    {size / 1024**3:6.2f} GB  {name}")


if __name__ == "__main__":
    for m in MODELS:
        probe(m)
