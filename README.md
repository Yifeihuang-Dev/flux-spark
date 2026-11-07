# Flux Spark

A local small-model pipeline that writes code autonomously, then self-checks and self-repairs against the requirement. Runs on Apple Silicon, all inference on-device.

> **Flux** = code continuously converging through the loop: *generate → verify → reflect → repair*.

## What the pipeline does

```
SFT base + chat template
  → best-of-3 sampling + test filtering        72.6% → 82.9%
  → reflection + rewrite on remaining failures         → 86.6%
```

**Zero LoRA, zero training** end to end. Five versions of a reviewer LoRA were trained along the way, but strict A/B ablations showed their repair advice had a **net value of exactly +0**, so they were dropped (configs kept in `scripts/train_reviewer_v*.yaml` as evidence).

What actually works is the **reflection loop**: its only inputs are the **requirement** and the **model's own code** — both available at generation time, so it depends on no external errors and no reviewer.

## Where it stands in the 2B / 4-bit / local niche (cross-model eval, 2026-11-04)

Same 164 tasks, same sampling budget, three models head-to-head (report, Chapter 10):

| Model | Single | best-3 | Final |
|---|---|---|---|
| Local 4-bit (this project, 1.3 GB) | 72.6% | 84.8% | **87.8%** |
| Official 4-bit (1.3 GB) | 72.0% | 79.9% | **87.8%** |
| bf16 full precision (4.7 GB) | **84.1%** | 89.0% | **90.2%** |

How to read it: quantization costs 11.5 pp on single-shot (significant), but after the full loop the 1.3 GB 4-bit model is **statistically indistinguishable** from full precision (p=0.344) — **3.6× less memory, virtually no final-score loss**. Also, pipeline gains shrink as the baseline rises (+15 pp on a 72% baseline, +6 pp on 84%): **this loop is a lifeline for weak models, not a polish for strong ones.**

Under strict judging (HumanEval+, ~80× more tests), the loop's final score is **79.9%** — the same level as Gemini 1.5 Pro's *single-shot* 79.3% on the same benchmark (see report, Chapter 11).

Head-to-head against the strongest open ~1.5B coder, **same harness, same tasks, same budget** (2026-11-06): Qwen2.5-Coder-1.5B-Instruct-4bit scores 59.8% single / 72.0% final, vs 72.6% / 87.8% here — **+15.8 pp final, paired McNemar p=0.00001** (31 wins, 5 losses). Two side findings: the loop **transfers across model families** (it bought Qwen +12.2 pp too), and its repair yield depends on the model (20% here vs 8% on Qwen).

## Quick start

Requires Python 3 + `pip install mlx-lm` (Apple Silicon).

```bash
# Interactive terminal (recommended): model loads once, ask repeatedly
python3 fluxspark.py -i

# One-shot
python3 fluxspark.py "write a function that checks if a list is monotonic (equal neighbors allowed)" \
  --tests 'assert f([1,2,2,4]); assert not f([1,3,2])'

# Strongest mode: tests + 3 samples + reflection repair
python3 fluxspark.py "..." --tests 'assert f([1,2,2,4])' --n 3

# No tests? The model can write its own first (output is flagged with reduced confidence)
python3 fluxspark.py "..." --selftest

# Code goes to stdout, logs to stderr — redirect straight into a .py file
python3 fluxspark.py "..." --tests '...' > out.py
```

In tests, use `f()` as a placeholder for the function name; the CLI rewrites it to the real name automatically.

### Commands inside the interactive terminal

`/help` `/n 3` `/selftest` `/reflect` `/quiet` `/save out.py` `/quit`;
`Ctrl+D` quits, `Ctrl+C` only aborts the current generation.

## Three-tier design (the thing worth remembering)

**Tests are an optional enhancement**, not a requirement:

| Situation | What you get |
|---|---|
| Tests provided | Full loop: best-of-N filtering + final verification; failures trigger reflection rewrites, and **only improvements are accepted** |
| `--selftest` | The model writes its own tests first (reduced confidence, clearly flagged in output) |
| Neither | Degrades to "generate + reflect", with **no way to know if the fix actually worked** |

**With tests and without tests are two different products.** If you care about correctness, write a few asserts — it's worth it.

## Layout

| Path | What it is |
|---|---|
| `fluxspark.py` | CLI entry point (one-shot + interactive terminal) |
| `scripts/run_pipeline.py` | Batch evaluation pipeline: generate → test → reflect → repair → re-test, with three controlled groups |
| `scripts/probe_multilang.py` | Multilingual probe: evidence the model is not Python-only |
| `scripts/eval_*.py` | Stage evaluation scripts |
| `scripts/bench_matrix.py` | Cross-model benchmarking: same tasks, same budget, three conditions; per-task JSONL, resumable |
| `scripts/summarize_bench.py` | Summary table for benchmark results |
| `scripts/evalplus_local.py` | Local HumanEval+ re-judge (official harness breaks on macOS; see report Ch. 11) |
| `scripts/train_reviewer_v*.yaml` | Reviewer LoRA training configs (proven useless, kept as evidence) |
| `REPORT.md` | Full experiment report, including the methodological lessons of every stage |

## Current limits

- **The CLI currently supports Python only.** In probes the model did write correct SQL (3/3, standard window functions), Ruby, Bash and Rust — the limit is the wrapper, not the model. A multi-language plan lives in `scripts/probe_multilang.py`.
- The pipeline is base-agnostic: pass `--model` with any MLX model directory (see [Getting the weights](#getting-the-weights)).

## Getting the weights

The model is a **4-bit MLX quantization** of `MiniCPM5-2B-SFT` (1.3 GB). Two ways to get it:

**Option A — download from Hugging Face** (fastest):

```bash
hf download Issingggggg/MiniCPM5-2B-SFT-MLX-4bit --local-dir ./MiniCPM5-2B-SFT-MLX-4bit
python3 fluxspark.py -i --model ./MiniCPM5-2B-SFT-MLX-4bit
```

The HF repo ships a full model card: quantization details, measured benchmarks, limitations, and license/attribution (Apache-2.0, inherited from the base).

**Option B — convert it yourself** (one command, fully reproducible):

```bash
mlx_lm.convert --hf-path openbmb/MiniCPM5-2B-SFT \
               --mlx-path ./MiniCPM5-2B-SFT-MLX-4bit \
               -q --q-bits 4 --q-group-size 64 --dtype bfloat16

./scripts/fetch_models.sh base        # built into the script
```

Both produce byte-identical weights (local conversion log: `Quantized model with 4.501 bits per weight`; all architecture hyperparameters match the base item by item).

> Why not commit weights to GitHub: files over 100 MB are rejected, and Git LFS's free tier (1 GB storage + 1 GB/month) is exhausted by a single weight. GitHub is for code; Hugging Face is for models.
