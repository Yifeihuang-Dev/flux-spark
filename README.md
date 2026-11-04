# Flux Spark

本地小模型自主写代码、并对照需求自查自纠的闭环。跑在 Apple Silicon 上，全部推理在本机完成。

> **Flux** = 代码在「生成 → 验证 → 反思 → 修复」的流里持续收敛。

## 这条流水线做了什么

```
SFT 基座 + chat template
  → best-of-3 采样 + 测试筛选      72.6% → 82.9%
  → 对仍失败的题做反思 + 重写             → 86.6%
```

全程**零 LoRA、零训练**。中途训过五个版本的审查器 LoRA，但严格 A/B 对照测出它们给出的修复意见**净价值为 +0**，已弃用（配置保留在 `scripts/train_reviewer_v*.yaml` 作对照证据）。

真正有效的是「反思环」：输入只有**任务要求**和**自己刚写的代码**，两样东西在生成那一刻就齐了，所以不依赖任何外部报错或审查器。

## 在 2B / 4bit / 本地赛道里的位置（2026-11-04 横评）

同一批 164 题、同一采样预算，三个模型对跑（报告第十章）：

| 模型 | 单次 | best-3 | 最终 |
|---|---|---|---|
| 本地 4bit（本项目，1.3 GB） | 72.6% | 84.8% | **87.8%** |
| 官方 4bit（1.3 GB） | 72.0% | 79.9% | **87.8%** |
| bf16 全精度（4.7 GB） | **84.1%** | 89.0% | **90.2%** |

读法：量化让单次能力掉 11.5 pp（显著），但走完闭环后与全精度**统计上已不可区分**（p=0.344）——**少用 3.6 倍内存，最终成绩基本不掉**。且流水线增益随基线递减（72% 基线 +15 pp，84% 基线只 +6 pp）：**它是给弱模型雪中送炭的，不是给强模型锦上添花的。**

## 快速开始

依赖：Python 3 + `pip install mlx-lm`（Apple Silicon）

```bash
# 交互式终端（推荐）：模型只加载一次，可连续提问
python3 fluxspark.py -i

# 单次调用
python3 fluxspark.py "写一个函数判断列表是否单调，允许相邻元素相等" \
  --tests 'assert f([1,2,2,4]); assert not f([1,3,2])'

# 最强模式：自带测试 + 采样 3 次 + 反思修复
python3 fluxspark.py "..." --tests 'assert f([1,2,2,4])' --n 3

# 不给测试也能跑：让模型先自己写测试（输出里会标注置信度降级）
python3 fluxspark.py "..." --selftest

# 代码走 stdout，过程日志走 stderr，可直接重定向成 .py
python3 fluxspark.py "..." --tests '...' > out.py
```

测试里函数名统一用 `f()` 占位，CLI 会自动替换成真实函数名。

### 交互终端里的命令

`/help` `/n 3` `/selftest` `/reflect` `/quiet` `/save out.py` `/quit`；
`Ctrl+D` 退出，`Ctrl+C` 只中断当前生成。

## 三级设计（最值得记住的一点）

**测试是可选增强**，不是必需：

| 情况 | 能享受什么 |
|---|---|
| 给了测试 | 完整闭环：best-of-N 筛选 + 最终验证；失败会触发反思重写，且**只接受变好的修改** |
| `--selftest` | 让模型先写测试（置信度降级，输出里明确标注） |
| 都不给 | 退化为「生成 + 反思自纠」，**它改没改对没人知道** |

**给测试和不给测试是两个档次。** 想认真用就顺手写几条断言，很值。

## 目录

| 路径 | 说明 |
|---|---|
| `fluxspark.py` | 命令行入口（单次 + 交互式终端） |
| `scripts/run_pipeline.py` | 批量评测流水线：生成 → 测试 → 反思 → 修复 → 再测，含三组对照 |
| `scripts/probe_multilang.py` | 多语言探针：验证模型并非只会 Python |
| `scripts/eval_*.py` | 各阶段评测脚本 |
| `scripts/bench_matrix.py` | 跨模型横评：同题同预算跑三条件，逐题落盘、可断点续跑 |
| `scripts/summarize_bench.py` | 横评结果汇总表 |
| `scripts/train_reviewer_v*.yaml` | 审查器 LoRA 训练配置（已证无效，留作对照） |
| `REPORT.md` | 完整实验报告，含每一步的方法论教训 |

## 当前边界

- **CLI 目前只支持 Python**。实测模型其实会写 SQL（3/3 语义正确，用的标准窗口函数）、Ruby、Bash、Rust——限制在包装器而非模型。多语言改造方案见 `scripts/probe_multilang.py`。
- 默认模型路径 ` /Volumes/LLM/hf/MiniCPM5-2B-SFT-MLX-4bit` 是本机外置 SSD，换机器请用 `--model` 指定自己的 SFT 基座——流水线对基座是中立的。权重**不入库**，原因见下。

## 模型权重为什么不入库

用的是 `MiniCPM5-2B-SFT` 的 **4-bit MLX 量化版本**（1.3 GB）。它没被放进这个仓库，也不需要：

1. **放不下。** GitHub 单文件硬上限 100 MB，1.3 GB 的权重会直接被拒。走 Git LFS 也不行：免费额度仅 1 GB 存储 + 1 GB/月带宽，这一个文件就撑爆了。
2. **它不是本项目的产物。** 权重是从公开基座 `openbmb/MiniCPM5-2B-SFT` 量化而来——架构超参逐项一致（`hidden_size 2048` / `42 层` / `16 头` / `vocab 130560` / `131072 上下文`），本地只是多了 `bits=4, group_size=64` 的量化配置；转换日志写着 `Quantized model with 4.501 bits per weight`。
3. **一行就能复现。** 与其让别人下载 1.3 GB，不如本地转换：

```bash
mlx_lm.convert --hf-path openbmb/MiniCPM5-2B-SFT \
               --mlx-path ./MiniCPM5-2B-SFT-MLX-4bit \
               -q --q-bits 4 --q-group-size 64 --dtype bfloat16

./scripts/fetch_models.sh base        # 脚本里已内置这一步
```

> 若确实想发布一份可直接下载的权重（比如让别人免去做本地转换），正确的去处是 **Hugging Face Hub**：免费、容量不限、专为权重设计，还带断点续传。`mlx_lm.convert` 自带 `--upload-repo` 参数，一条命令就能推上去。GitHub 是放代码的地方，HuggingFace 才是放模型的地方。
