"""查询 HuggingFace 上候选模型的实际体积与文件清单。

用途：在下载前评估磁盘占用，决定哪些模型放外接 SSD。
注意：本机 Python 的 SSL 证书链不完整，需要放行证书校验（仅用于只读查询）。
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
        print(f"{model_id}  查询失败: {exc}")
        return

    files = [(s.get("rfilename"), s.get("size") or 0) for s in data.get("siblings", [])]
    total = sum(size for _, size in files)
    print(f"\n{data.get('id')}   总计 {total / 1024**3:.2f} GB")
    for name, size in sorted(files, key=lambda x: -x[1])[:6]:
        if size > 5 * 1024**2:
            print(f"    {size / 1024**3:6.2f} GB  {name}")


if __name__ == "__main__":
    for m in MODELS:
        probe(m)
