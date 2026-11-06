# Code-LLM Fine-Tuning Experiment Report

**Project: Flux Spark** — a local small-model loop that writes code autonomously and self-checks/self-repairs against the requirement; CLI entry point `fluxspark.py`

**Dates**: 2026-10-26 ~ 2026-10-27
**Round 1 (executor)**: does the "write code" path work at all via fine-tuning? → **No effect**
**Round 2 (reviewer)**: is "find bugs" worth fine-tuning? → **Clearly effective**

> **Results at a glance**
>
> | Round | Role | Method | Outcome |
> |---|---|---|---|
> | Round 1 | Executor | LoRA fine-tune | No effect; three evaluations point in different directions, all within noise |
> | Round 2 | Reviewer | LoRA fine-tune | **Clearly effective: strict recall 26.5% → 76.5% (+50 pp)** |
> | **Round 3** | **Executor** | **Swap to SFT base (no fine-tuning)** | **+58 pp: HumanEval 14.6% → 72.6%** |
>
> Same base, same LoRA recipe, completely different outcomes by role — the difference is whether the capability was already there: teaching a model to write code restyles a skill it already has; teaching it to review fills a real gap.
>
> Round 3 carried the bigger lesson: **under the right yardstick (HumanEval), swapping the base bought +58 pp — dwarfing the ±3 pp we got from a dozen hours of fine-tuning.** Every earlier judgment about the executor rested on a too-soft homemade yardstick.

---

# Round 1: Executor fine-tuning

## Conclusion first

**Capability: fine-tuning is statistically indistinguishable from zero. Style: fine-tuning changed the model dramatically — it became verbose.**

Three evaluations, three different directions:

| Evaluation | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| 8 tasks × 1 (greedy) | 75.0% (6/8) | 62.5% (5/8) | **−12.5 pp** |
| 31 tasks × 1 (greedy) | 67.7% (21/31) | 71.0% (22/31) | **+3.3 pp** |
| **31 tasks × 3 (1 greedy + 2 sampled)** | **66.7% (62/93)** | **63.4% (59/93)** | **−3.2 pp** |

Three measurements, three directions — that itself is the strongest evidence: **the true effect size is near zero, and any directional result is noise.**

A finer test confirms it — across the 31 tasks, **10 improved and 11 regressed**, almost perfectly symmetric; the count of "3-for-3" tasks is **11/31 on both sides**, identical. That is the signature of random fluctuation, not a systematic capability change.

**But style did change, dramatically** (see Section 5): raw output length grew from 423 to 769 characters (**+82%**), while the code itself shrank from 237 to 204 characters. **The extra content is not in the code, not in the preamble — it comes after the code**: the model learned Magicoder's "code + usage example + explanation" output shape.

> **Correction note**: the first draft of this report judged "negative return" from the 8-task run, then "slightly better" from the 31-task greedy run — both were over-readings. Merged across three runs, the capability conclusion converged to **no significant difference**.

---

## 1. Environment

| Item | Detail |
|---|---|
| Machine | Apple Silicon (arm64), 16 GB unified memory |
| External SSD | `/Volumes/LLM`, 232 GB free, measured write 437 MB/s |
| Framework | MLX + mlx-lm 0.32.2 |
| GPU | Metal available, measured ~**1.93 TFLOPS** |
| Python | 3.13.0, **native arm64** (`proc_translated=0`, not Rosetta) |

> The system Python is a universal binary and picked arm64 automatically, so MLX and Metal acceleration both work.

## 2. Model and data

**Base model**: `openbmb/MiniCPM5-2B-MLX` (official MLX build, 1.33 GB, Q4 quantized)

**Training data**: `Magicoder-OSS-Instruct-75K`
- 75,197 raw samples, filtered to 36,894 Python samples
- Actually used: 5,000 train + 200 validation

**Training config**:

| Param | Value |
|---|---|
| Method | LoRA (rank 16, scale 32, on 16 layers) |
| Trainable params | 9.568 M / 2.517 B (**0.38%**) |
| Steps / batch | 1,000 / batch 4 |
| Seq length | 1,024 |
| LR | 1.0e-5 (Adam) |
| Gradient checkpointing | On |
| Loss mask | `mask_prompt` — loss computed on answers only |

## 3. Training results

Training finished normally (**2026-10-27 03:07**, exit 0), 1,000/1,000 steps.

| Metric | Start | End | Change |
|---|---|---|---|
| val loss | 0.628 | **0.351** | ↓ 44% |
| train loss | 0.476 (step 10) | 0.34–0.40 range | ↓ |

**All 21 checkpoints saved** (every 50 steps, 37 MB each).
Throughput measured **39–51 tokens/s**, ~6 hours total.

> **val loss fell 44%, yet coding capability did not improve.** This contrast matters: falling loss only means the model fits the training distribution better — and Magicoder's distribution does not overlap much with real coding tasks.

## 4. Evaluation method

Give the model a function signature description → it writes the implementation → extract code and `exec` → run unit tests with `assert`.

The 31 tasks cover: arrays/lists (9), strings (8), math (8), sorting/searching (3), structure transforms (3).

### Per-task results (31 tasks)

| Task | Baseline | Fine-tuned |
|---|---|---|
| two_sum | PASS | PASS |
| dedup_keep_order | PASS | PASS |
| flatten_nested | PASS | PASS |
| max_subarray_sum | FAIL | FAIL |
| rotate_list | PASS | PASS |
| chunk_list | PASS | PASS |
| list_intersection | PASS | PASS |
| max_pair_product | PASS | PASS |
| find_missing_number | PASS | PASS |
| is_palindrome | FAIL | PASS |
| word_count | FAIL | FAIL |
| reverse_string | PASS | PASS |
| count_vowels | PASS | PASS |
| longest_common_prefix | PASS | PASS |
| is_anagram | PASS | PASS |
| caesar_cipher | FAIL | PASS |
| valid_parentheses | PASS | PASS |
| fibonacci | PASS | PASS |
| factorial | PASS | PASS |
| is_prime | PASS | PASS |
| gcd | PASS | PASS |
| sum_of_digits | PASS | PASS |
| is_power_of_two | PASS | PASS |
| climbing_stairs | FAIL | FAIL |
| roman_to_int | FAIL | PASS |
| fizzbuzz | PASS | PASS |
| binary_search | PASS | PASS |
| merge_sorted | PASS | PASS |
| sort_list | FAIL | FAIL |
| matrix_transpose | FAIL | FAIL |
| dict_invert | PASS | PASS |
| **Total** | **21/31 (67.7%)** | **22/31 (71.0%)** |

Net change for the fine-tuned model: 3 fixed (is_palindrome, caesar_cipher, roman_to_int), and **no PASS→FAIL regressions** — healthier than the "broke 3 tasks" picture from the 8-task run.

### Fine test (31 tasks × 3 samples)

To remove single-sample luck, each task was sampled 3 times (1 greedy + 2 at temperature 0.8) — 93 generations per model.

**Overall**

| Metric | Base | Fine-tuned |
|---|---|---|
| Overall pass rate | 62/93 = 66.7% | 59/93 = 63.4% |
| Tasks passing 3/3 | 11/31 | **11/31** |
| Tasks failing 3/3 | 4/31 | 3/31 |

**By category**

| Category | Base | Fine-tuned | Delta |
|---|---|---|---|
| Strings | 62.5% | 62.5% | 0.0 pp |
| Sorting/searching | 55.6% | 66.7% | +11.1 pp |
| Math | 74.1% | 66.7% | −7.4 pp |
| Arrays | 59.3% | 55.6% | −3.7 pp |
| Structure transforms | 100.0% | 83.3% | −16.7 pp |

**Per-task delta distribution**: 10 up, 11 down, 10 flat. Near-perfect symmetry — random fluctuation, not systematic change.

**Failure modes**

| Failure type | Base | Fine-tuned |
|---|---|---|
| Logic errors | 22 | 20 |
| Syntax errors | 5 | 3 |
| NameError | 2 | 5 |
| TypeError | 2 | 3 |
| Other | 0 | 3 |

Failures are dominated by **logic errors** (~2/3 of all failures); syntax errors slightly decreased — fine-tuning did not break syntax; the problem is task logic.

## 5. Analysis

### 1. The small-sample trap (this round's biggest lesson)

With 8 tasks, one task = 12.5 pp — enough to flip "improved" into "regressed". With 31 tasks, one task is still 3.2 pp. **Reliably detecting a few percentage points requires 100+ tasks** (HumanEval's 164, MBPP's 500).

**Before the sample size is sufficient, any "it improved / it regressed" claim is self-deception.**

### 2. Style changed, capability didn't follow (hard evidence from the fine test)

| Style metric | Base | Fine-tuned | Change |
|---|---|---|---|
| Raw output length | 423 chars | **769 chars** | **+82%** |
| Code length | 237 chars | 204 chars | −33 |
| Code lines | 9.0 | 6.2 | −2.8 |
| Preamble length | 94 chars | 97 chars | +3 |
| Outputs over 800 chars | 12% | **38%** | 3× |
| Uses code fences | 95% | 96% | flat |

**Output doubled in length while the code itself got shorter.** The extra ~380 characters are neither in the code nor in the preamble (94→97 barely moved) — they come **after the code**: the model learned Magicoder's "code + usage example + explanation" shape.

This is direct evidence for the "style mismatch" hypothesis: **the fine-tuning did take effect** (behavior changed significantly), but what changed was **expression style**, not **problem-solving ability**. The appended examples and explanations do nothing for passing unit tests — they only slow generation and burn tokens.

### 3. Why capability wouldn't move

- **Baseline already strong**: stock MiniCPM5-2B scores 67.7% out of the box — it has seen plenty of code. Pushing further with style-mismatched, uneven synthetic data is hard.
- **Diluted effect**: LoRA touches 0.38% of parameters, LR 1e-5, 1,000 steps — enough to change output style, not enough to systematically improve underlying problem-solving.
- **Data–objective mismatch**: training data is "complex project → long code"; the objective is "short function, correct implementation". Not the same capability.

### 4. What deserves credit

- The fine-tuned model shows **no capability regression** (no PASS→FAIL among the 31) — the training process itself is stable and didn't break the model.
- val loss 0.628 → 0.351; the whole training pipeline (configs, checkpointing) is verified working.
- The full loop (download → prepare data → train → evaluate) runs end to end and is reusable for the next round.

### 5. Capability profile and best usage (this round's most valuable finding)

By behavior across 3 samples, the 31 tasks fall into three tiers:

| Tier | Count | Representative tasks |
|---|---|---|
| Consistently can (3/3) | 4 | is_power_of_two, is_prime, sum_of_digits, matrix_transpose |
| **Can but unstable (1–2/3)** | **16** | dedup, rotate, FizzBuzz, palindrome, fibonacci, merge sorted lists, valid parentheses… |
| Consistently can't (0/3) | 4 | roman_to_int, longest_common_prefix, sort_list, max_pair_product |

**More than half the tasks sit in "can do but unstable"** — the model knows what to write; it just often writes it wrong.

This points directly at correct usage:

| Usage | Pass rate |
|---|---|
| Single greedy shot | 67.7% (base) / 71.0% (tuned) |
| **3 samples + filter by unit tests** | **87.1% (base) / 90.3% (tuned)** |

**Same model, different usage: usable rate jumps from 67% to 87%.** That gain (~+20 pp) dwarfs fine-tuning's (±3 pp, within noise).

Conclusion: **the key to usability is not more fine-tuning but "sample more + filter by tests" — i.e., the verification layer of the architecture.** Fine-tuning is the most expensive and least certain link; the verification layer costs zero training and pays immediately.

## 6. Conclusions

1. **Capability: fine-tuning is indistinguishable from zero.** Three runs, three directions (−12.5 / +3.3 / −3.2 pp); the fine test shows 10 up / 11 down with identical "3-for-3" counts (11 vs 11). Random noise, not signal.
2. **Style: fine-tuning clearly worked.** Output length +82% while code got shorter — the model did learn Magicoder's shape. The problem isn't "nothing was learned"; it's that what was learned doesn't help problem-solving.
3. **Loss is not a proxy for capability.** val loss fell 44% (0.628→0.351) while capability didn't move.
4. **Small-sample evaluation cannot drive decisions.** Three rounds of measurement, three directions, all within noise.
5. The full pipeline (download→prepare→train→evaluate) is verified working; **the bottleneck is data selection, not engineering**.
6. **Usage > fine-tuning**: same model, 67% single-shot vs 87% with 3-sample + test filtering. Before spending on more fine-tuning, harvest the zero-cost verification-layer gain first.

## 7. Next steps (by priority)

| Priority | Action | Rationale |
|---|---|---|
| **P0 (first)** | **Build "sample N + filter by unit tests"**: generate 3 candidates per task, run tests, return the first pass | Zero training cost; usability 67%→87% (+20 pp), far beyond fine-tuning. This is the verification layer made real |
| **P0** | Switch benchmarks: HumanEval (164) or MBPP (500) | 31 homemade tasks can't support decisions; standard benchmarks have statistical power and enable external comparison |
| **P1** | Change data: `evol-codealpaca`-style "short instruction → concise code", or filter Magicoder for short samples | Directly targets the style mismatch |
| **P1+** | **During data cleaning, strip everything after the code in solutions** (usage examples, explanations); train on "pure code output" only | The fine test proved this bloat is the cause of the +82% output growth. Cheapest, most direct fix |
| **P2** | Tuning: LR 5e-6, fewer steps, mix in general instructions to prevent forgetting | Reduce style pollution, keep existing capability |
| **P3** | Improve eval scripts: larger `max_tokens`, tolerate "explanation before code" outputs | Current extractor is brittle with explanatory outputs and may underestimate the model |
| — | Router, reviewer | **On hold.** The executor path hasn't shown positive returns; stacking roles on top is pointless |

## 8. Assets

| Type | Path |
|---|---|
| Base model | `/Volumes/LLM/hf/openbmb__MiniCPM5-2B-MLX` |
| Training data | `/Volumes/LLM/datasets/magicoder_mlx/{train,valid}.jsonl` |
| Raw dataset | `/Volumes/LLM/datasets/magicoder_oss.jsonl` (193 MB) |
| LoRA weights | `/Volumes/LLM/adapters/executor-v1/` (21 checkpoints) |
| Training log | `/Volumes/LLM/logs/train.log` |
| Eval results | `/Volumes/LLM/logs/eval30_{base,tuned}.json` |
| Eval script | `scripts/eval_coding.py` (31 tasks) |
| Data prep | `scripts/prepare_data.py` |
| Training config | `scripts/train_lora.yaml` |
| Unattended launcher | `scripts/run_train.sh` |
| Model download | `scripts/fetch_models.sh` |

### Common commands

```bash
# Evaluate baseline
python3 scripts/eval_coding.py --model /Volumes/LLM/hf/openbmb__MiniCPM5-2B-MLX

# Evaluate fine-tuned
python3 scripts/eval_coding.py --model /Volumes/LLM/hf/openbmb__MiniCPM5-2B-MLX \
    --adapter /Volumes/LLM/adapters/executor-v1

# Retrain (after config changes)
nohup caffeinate -dimsu bash scripts/run_train.sh &
```

---

# Round 2: Reviewer fine-tuning (a clearly positive result)

## 1. Why switch targets

Round 1's executor fine-tuning yielded nothing (three runs, three directions, all within noise). The post-mortem concluded: **the role worth fine-tuning is not the executor but the reviewer.**

1. Reviewing is a **discriminative task** (find problems in existing code), far easier than generation (construct a correct solution from scratch).
2. Review labels are **100% automatically verifiable** — inject known bugs into clean functions; the label is exactly what we injected. No manual annotation. This unties Round 1's "data is hard to build and hard to verify" knot.
3. It directly serves the architecture's verification layer.

## 2. Key discovery: we had been using the Base model all along

While debugging the reviewer task we found `mlx_lm.generate` with a raw string **does not apply the chat template** — the model degenerates to raw continuation (parroting the prompt). Further digging confirmed:

**The `MiniCPM5-2B-MLX` we had been using is the Base version.** The official `MiniCPM5-2B-SFT` (the true instruction-tuned model, 4.7 GB) exists but had no ready MLX build.

| Symptom | Explanation |
|---|---|
| Executor scored 67% | "Writing code" is fundamentally continuation; base models are good at that |
| Executor fine-tuning useless | Base was already at its continuation ceiling; restyling it just adds noise |
| Reviewer baseline only spammed | Base cannot follow instructions to produce structured output |

Downloaded the SFT version and converted to MLX 4-bit (1.3 GB) as the new base.

> **Side effect**: Round 1's 67.7% was measured in raw-continuation mode. Re-measured with the SFT model + proper templating, the baseline could be different — a to-do at the time.

## 3. Data construction

- 72 clean seed functions × 60+ mutation operators (boundary conditions / null handling / logic errors)
- Mutations applied as exact string replacements, with controllable injection sites
- Output: 177 train / 16 validation / 64 test, 194 injected bugs in total
- **Zero seed overlap across the three splits**: functions in the test set never appeared in training — we measure reviewing skill, not memorization

## 4. Results (64-task test set)

| Metric | Baseline (no FT) | 50 steps | **300 steps (best)** |
|---|---|---|---|
| Strict recall | 26.5% (18/68) | 48.5% | **76.5% (52/68)** |
| Loose recall | 95.8% | 70.8% | 93.8% |
| False positive rate | 56.2% (9/16) | 18.8% | **31.2% (5/16)** |
| Exactly correct reports | 16.7% (8/48) | 35.4% | **66.7% (32/48)** |
| Reported/actual | 1.51 (spamming) | 0.60 (under-reporting) | **0.99 (calibrated)** |

**Strict recall +50 pp, exactly-correct reports ×4, false positives −25 pp.**

`reported/actual = 0.99` is the prettiest number: 68 bugs injected, 67 reported — counts line up with ground truth; neither spamming nor silence.

## 5. Two counter-intuitive findings

**1. The checkpoint with the lowest val loss was not the best.**

Step 50: val loss 0.842. Step 300: 1.111 (+32%) — yet step 300's recall is 28 pp higher. Step 50's problem is **under-reporting** — reported/actual only 0.60; too conservative, it only dares flag 60%.

Same lesson as Round 1's "loss is not a capability proxy", but starker: **val loss rose 32% while capability rose 28 pp.**

**2. The baseline's 95.8% "loose recall" is an illusion.**

It must be read together with the 56.2% false-positive rate: the model isn't "spotting problems", it's **flagging every piece of code it sees**. The opposite extreme of parroting — one outputs nothing, the other outputs indiscriminately.

## 6. Why the reviewer worked and the executor didn't

| | Executor | Reviewer |
|---|---|---|
| Task nature | Continue code | Follow instructions + structured output |
| Built-in base ability | Strong (via continuation) | None (only spams) |
| Fine-tuning headroom | Almost none | Huge |
| Measured | Within noise (±3 pp) | **+50 pp recall** |

The base already "knows how to write code but can't review". Teaching it to review **fills a gap**, so it works; teaching it to write restyles a skill it already has, so it's useless and adds noise.

**The role-selection hypothesis is confirmed: the reviewer is the role most worth fine-tuning in this architecture.**

---

## Update log

- **2026-10-27 03:07**: 1,000-step training completed normally, val loss 0.628 → 0.351.
- **2026-10-27 morning (8-task run)**: 75% → 62.5%; briefly judged "negative return".
- **2026-10-27 morning (31-task greedy)**: 67.7% → 71.0%; direction flipped; corrected to "indistinguishable from zero".
- **2026-10-27 (31 × 3 fine test)**: 66.7% → 63.4%; third direction change.
  - Capability: 10 up / 11 down, 3-for-3 counts 11 vs 11 → **confirmed: no significant capability difference**.
  - Style: output length 423 → 769 chars (+82%), code length 237 → 204 → **confirmed: style significantly changed, extra content comes after the code**.
- **2026-10-27 evening (Round 2: reviewer)**:
  - Found `mlx_lm.generate` doesn't apply chat templates to raw strings, and that `MiniCPM5-2B-MLX` is the **Base version**; switched to official `MiniCPM5-2B-SFT` (converted to MLX 4-bit, 1.3 GB).
  - Built review data: 72 seeds × 60+ mutations, 177/16/64 splits with zero function overlap.
  - Baseline: strict recall 26.5%, FPR 56.2%, reported/actual 1.51 (spamming).
  - After 300 steps: **strict recall 76.5% (+50 pp), FPR 31.2% (−25 pp), exactly-correct 66.7% (×4), reported/actual 0.99**.
  - Counter-intuitive: **the lowest-val-loss checkpoint (step 50) was worse** (under-reporting, only flags 60%) — again, val loss is not a capability metric.
- **2026-10-28 morning (Round 3: HumanEval + reviewer tuning)**:
  - HumanEval 164 tasks: **Base 14.6% vs SFT 72.6% (+58 pp)**; best-of-3 brings SFT to **82.9%**.
  - **Major correction**: the 31 homemade tasks badly overestimated Base (67.7% measured vs 14.6% true) — HumanEval's "continue the function body" format defeats Base, while homemade "write a function" tasks just need one complete function.
  - Reviewer v3 (500 steps): FPR 31.2% → **18.8%**, precision 77.6% → **88.2%**, at the cost of recall 76.5% → 66.2%.
  - **Hypothesis falsified**: raising the negative-sample ratio (25%→40%) does **not** reduce false positives (v2 FPR still 31.2%); what works is training more epochs.
  - Overall: **swap the base (+58 pp) >> best-of-3 (+10 pp) >> fine-tuning (±3 pp, noise)**.
  - Added v4 (25% negatives + 500 steps): FPR down to **6.2%**, recall 55.9% → **confirms steps are the dominant factor**; negative ratio barely matters.
  - Added v5 (null augmentation: +15 guarded seeds, null share 6%→15%): precision 75.4%, F1 68.8% — **worse than v3**.
    - But the test set had only 4 null bugs (6%); **it has no power to measure null-specific improvements** — needs a dedicated test set.
  - **Current best reviewer: v3** (F1 75.6%, FPR 18.8%, precision 88.2%).
- **2026-10-31 (Round 4: null-specific eval + closed-loop verification)**:
  - Built a null-specific test set (26 null bugs + 17 clean functions, all unseen functions) and **overturned the previous conclusion**:
    v5 (null-augmented) recall **23.1%** vs v3's 7.7% (**3× gap**), with v5 at **zero false positives** — null augmentation works.
    Round 3's "v5 worse than v3" was a misjudgment caused by only 4 null samples in the main set.
  - **Methodological lesson (hit twice now)**: verifying "did capability X improve" requires a test set dedicated to X; otherwise aggregate metrics hide the improvement.
  - **First full closed loop** (executor generates → tests judge → reviewer reviews; HumanEval 164): generation pass rate 72.6%;
    flagged tasks fail at 37.5% vs 25.7% for unflagged — **lift only 1.46×** — positive but weak signal.
  - Conclusion: **reviewing skill trained on synthetic data does not transfer to real generated code.** Next step: train on "real generated wrong code".
  - **Added the repair stage with an A/B control** (rewrite with reviewer advice vs blind rewrite with no advice, 9 tasks each):
    both fixed exactly 1 task (11.1%) — **reviewer advice net value +0**; pass rate 72.6% → 73.2%.
  - Root cause: the model **cannot do these tasks at all** (rewrite success 11.1%); review-and-repair only helps "can do but wrote it wrong".
  - **The framework's core mechanism (verification-layer-driven improvement) fails when the driving signal is reviewer advice.**
  - **Added the "reflection loop" (summarize the algorithm → compare against the requirement → give advice; zero training), proven effective in a three-arm controlled trial**:
    blind rewrite 4/45 (8.9%) / reviewer advice 1/9 (11.1%) / **reflection advice 11/45 (24.4%, 2.7×)**;
    7 of the 11 were "fixed only by reflection"; McNemar exact test **p≈0.016 (significant)**.
    Loop pass rate **72.6% → 79.3%** (+6.7 pp), approaching best-of-3's 82.9% — and the mechanisms differ, so they **compose: 3f's stacked experiment reached 86.6%**.
  - Why it works: it catches **conceptual misunderstanding** (e.g. implementing strict monotonicity where non-strict was asked). Such errors raise no exception and leave no code smell; only comparing behavior against the requirement can find them.
  - **Zero training cost** (base model as-is) — the highest ROI step in the whole project.

---

# The overnight run (night of 2026-10-27)

Budget 5 hours; actual ~2.5 hours — HumanEval's 164 tasks take ~8 minutes, far faster than expected.

| # | Task | Purpose |
|---|---|---|
| 1 | HumanEval 164: SFT (templated) | The executor's **true absolute level** |
| 2 | HumanEval 164: Base (raw continuation) | Control: how much does swapping to SFT buy? |
| 3 | Reviewer v2: negatives 25%→40%, 300 steps | **Push FPR down** (31.2% → target <20%) |
| 4 | Reviewer v3: same data, 500 steps | Do more epochs keep helping? |
| 5 | HumanEval best-of-3: SFT | The increment of "sample more + filter" |
| 6 | HumanEval best-of-3: Base | Control |

Ordering: **fast and critical first** — if interrupted, the core numbers are already in.

Log: `/Volumes/LLM/logs/night.log`

## Results

### 1. HumanEval: Base's true level is only 14.6%

| | HumanEval 164 | Homemade 31 |
|---|---|---|
| Base (raw continuation) | **14.6%** (24/164) | 67.7% |
| SFT (templated) | **72.6%** (119/164) | 90.3% |
| Gap | **+58 pp** | +22.7 pp |

**The homemade tasks massively overestimated Base.** Why: their prompt is "Write a function that..." — producing one complete function suffices. HumanEval gives a **signature + docstring and asks to continue the body**; Base doesn't know where to stop and drifts.

> **Methodological lesson**: the prompt format of a benchmark can swing conclusions wildly. Measuring coding ability with "instruction-style + short function" tasks systematically overestimates models that can't follow instructions. **Every executor judgment from the first two rounds rested on a too-soft yardstick.**

### 2. best-of-3 still works, but gains shrink as the baseline rises

| | Single | best-of-3 | Gain |
|---|---|---|---|
| SFT | 72.6% | **82.9%** (136/164) | +10.3 pp |
| Base | 14.6% | 34.8% (57/164) | +20.2 pp |

The higher the baseline, the smaller the marginal return of more sampling (+20 → +10 pp), as expected. 82.9% is a strong number at **zero training cost**.

### 3. Reviewer: buying precision costs recall

| Version | Recall | Precision | F1 | FPR |
|---|---|---|---|---|
| v1 (25% negatives, 300 steps) | **76.5%** | 77.6% | **77.0%** | 31.2% |
| v2 (40% negatives, 300 steps) | 63.2% | 79.6% | 70.5% | 31.2% |
| **v3 (40% negatives, 500 steps)** | 66.2% | 88.2% | 75.6% | 18.8% |
| v4 (25% negatives, 500 steps) | 55.9% | **88.4%** | 68.5% | **6.2%** |
| v5 (null-augmented, 500 steps) | 63.2% | 75.4% | 68.8% | 12.5% |

**Factor decomposition (v4 is the key cell)**: v1's negative ratio + v3's 500 steps = v4 — recall falls further to 55.9%, FPR further to 6.2%. Now confirmed:

- **Steps are the dominant factor**: 300→500 makes the model noticeably more conservative; FPR falls monotonically (31.2% → 18.8% / 6.2%), recall falls along with it.
- **Raising the negative ratio doesn't work (hypothesis falsified)**: v2 moved negatives 25%→40%; FPR didn't budge (31.2%) while recall dropped 13 pp.

v1 and v3 are the classic trade-off: **v1 catches more but spams; v3 is precise but misses**. On F1, v1 wins (77.0%) — but a reviewer that cries wolf loses the user's trust quickly. **Balancing precision and credibility, v3 stays.**

### 3b. Null augmentation: validated by a dedicated test set — v5 dominates on nulls

"Null handling" bugs were only **6%** of the training set — root cause: across 71 seeds there were only **7 guards to delete**, and null bugs can only be created by removing existing guards. We added 15 **new seeds with null guards** + 13 matching mutations, raising the null share to **15%** (34/219).

**The first conclusion was wrong.** On the main test set v5's F1 was 68.8%, below v3's 75.6%, and we judged "null augmentation doesn't work". But that set had only **4** null bugs (6%) — even catching all of them moves the total by 6 points. **It simply cannot measure null-specific changes.**

So we built a null-specific test set (`scripts/make_null_test.py`): **26 null bugs** (20 guards deleted + 6 `None` checks rewritten into equivalence checks) + 17 clean functions, all using **functions never seen in any training set**.

| Version | Strict recall | Loose recall | FPR |
|---|---|---|---|
| Baseline (no FT) | 0.0% | 57.7% | 76.5% |
| v1 (25% / 300 steps) | 7.7% | 26.9% | 29.4% |
| v2 (40% / 300 steps) | 3.8% | 23.1% | 11.8% |
| v3 (40% / 500 steps) | 7.7% | 11.5% | 11.8% |
| v4 (25% / 500 steps) | 3.8% | 3.8% | 5.9% |
| **v5 (null-augmented / 500 steps)** | **23.1%** | 23.1% | **0.0%** |

**v5's recall is 3× v3's, with zero false positives on the 17 clean functions.** Null augmentation does work — the earlier "v5 worse than v3" was an artifact of an underpowered test set.

> **Methodological lesson (second time hitting this)**: when a category is a tiny share of the test set, aggregate metrics systematically hide category-targeted improvements. **To verify "did category-X ability improve", you need a category-X test set** — otherwise you're watching other categories' noise. Same error family as "homemade tasks overestimate Base": **wrong yardstick, unreliable conclusion**.

**But the absolute level is still low**: even v5 catches only 23.1% of null bugs, far below its 60%+ on logic/boundary classes. Null handling remains a clear weakness (34 training samples is still too few).

Note: baseline loose recall 57.7% with 76.5% FPR is still the "flag everything" mode.

### 3c. Closed-loop verification: weak signal on real generated code

All three parts exist (executor, reviewer, tests) — time for the first end-to-end run (`scripts/run_pipeline.py`):
the SFT executor answers HumanEval's 164 tasks → tests judge them → reviewer v3 reviews every piece of **really generated code**.

The question is not "how accurate is the reviewer on synthetic data" but the harder one: **are the flagged tasks actually more likely to be wrong?**

| Metric | Value |
|---|---|
| Generation pass rate | 119/164 = 72.6% (matches the standalone measurement — mutual validation) |
| Tasks flagged by the reviewer | 24/164 = 14.6% |
| True failure rate among flagged | **9/24 = 37.5%** |
| True failure rate among unflagged | 36/140 = 25.7% |
| Overall failure baseline | 45/164 = 27.4% |
| **Lift** | **1.46×** |

**The signal is positive but weak.** Flagged tasks do fail more (37.5% vs 25.7%) — the reviewer isn't flipping coins. But 1.46× is far from usable: 62.5% of flagged tasks actually pass (false alarms), and 25.7% of unflagged ones are wrong (misses).

**Root cause: training distribution badly misaligned with reality.** The reviewer learned "simple injected bugs in small functions" (delete a guard, off-by-one a boundary); real failures are "the algorithm is simply wrong" (misread the task, wrong approach). The former is a local flaw, the latter a global error — the surface patterns don't transfer.

**Conclusion: high synthetic metrics (F1 75%) say nothing about real-world capability.** For the reviewer to filter for real, training data must be **really generated wrong code**, not "clean code with injected fake bugs". That's the next high-value direction.

### 3d. Repair stage: the net value of reviewer advice is 0 (decisive evidence)

We added the architecture's missing last link — **rewriting failed code per reviewer advice** (`run_pipeline.py`, stage 3). For every task that both **failed tests and was flagged**, two repairs were run:

- **Group A**: feed the reviewer's advice to the executor and have it rewrite accordingly
- **Group B**: the exact same rewrite request, **with no advice at all** (control)

The control exists to rule out the confounder "a second attempt just happens to pass".

| Metric | Result |
|---|---|
| Repairable samples (failed and flagged) | 9 tasks |
| Group A, with advice | 1/9 = 11.1% |
| Group B, blind rewrite | 1/9 = 11.1% |
| **Net value of reviewer advice** | **+0 tasks** |
| Framework pass rate | 120/164 = 73.2% (from 72.6%) |

**Groups A and B are identical.** The one fix came from "a second try happened to work", not from the advice.

**The root cause is not reviewer inaccuracy — it's the nature of the failures.** With rewrite success at 11.1% regardless of advice, the model **cannot do these tasks at all** — what's missing is the solution itself, not a flaw in the code. The review-and-repair paradigm only helps "can do but wrote it slightly wrong"; for "can't do it at all", no amount of advice helps.

> **Conclusion: the framework's core mechanism (verification-layer-driven improvement) does not hold in this configuration.** The synthetic-data reviewer neither filters effectively (lift 1.46×) nor guides repair (net value +0).
>
> To matter in the real world, training data must satisfy two conditions: **the errors are real**, and **they concentrate on "can do but wrote wrong"** — not "can't do at all".

### 3e. The reflection loop: a controlled trial proves it works

Since "guess where the bug is" is a dead end (3d), change the approach — don't guess. First **state what the code actually does**, then **compare against the task requirement** to find where it falls short, and only then give a fix. This needs no training; the base model can do it as-is.

Run under three strict arms (`run_pipeline.py` stages 2.5 + 3). Note the repair scope widened from "the 9 flagged tasks" to **all 45 failures** — with the reviewer falsified, using it as a gate would shrink the sample below statistical meaning.

| Arm | Method | Fixed | Success rate |
|---|---|---|---|
| A (baseline) | Nothing given; pure rewrite | 4/45 | 8.9% |
| B | With fine-tuned reviewer advice | 1/9 | 11.1% (only 9 flagged) |
| **C** | **With reflection advice (summarize algorithm / compare requirement / suggest)** | **11/45** | **24.4%** |

**Arm C is 2.7× arm A; net value +7 tasks; overall pass rate 72.6% → 79.3% (+6.7 pp).**

Crucially, of the 11 fixed tasks, **7 were "fixed only by reflection"** (blind rewrite failed on them too); the other 4 both arms fixed — luck. McNemar exact test: **p ≈ 0.016** — statistically significant, not luck.

**Why it works — a real case** (HumanEval/57, list monotonicity):

The reflection output correctly pointed out: the code uses `l[i] < l[i+1]`, which is **strict** monotonicity; the task asks for **non-strict** (equal neighbors allowed), so `[1, 2, 2, 4]` would be wrongly rejected. Suggested changing `<` to `<=`.

This is a **conceptual misunderstanding** — it raises no exception (tests just report AssertionError), and the code itself looks flawless (so no reviewer can catch it). The only way to catch it is to compare the code's actual behavior against the requirement itself. That is precisely what this stage does.

> **Conclusion: what drives repair is not how much information you give, but whether it points at the true cause.**
>
> - A runtime error says "expected 3, got 5" — describes the symptom, not the cause.
> - A reviewer guesses "this line looks suspicious" — doesn't even pin the symptom (net value +0).
> - **Reflection compares against the requirement and finds "strict where non-strict was meant" — it hits the cause.**
>
> And this stage **needs no fine-tuning at all** — the base model as-is, marginal cost one extra inference. The highest-ROI step in the project.

### 3f. Stacking best-of-3 and reflection: genuinely complementary — 86.6%

The reflection loop needs no errors; best-of-3 needs test filtering. Different mechanisms, so they should stack: **use best-of-3 to pick off "unlucky sampling" tasks, then reflect on the rest**. The generator gained best-of-N support (`--repeat`; deliberately no early stop; samples 2 and 3 use `temperature=0.8` to align strictly with known baselines).

Measured on HumanEval's 164 tasks (all four stages ~50 minutes):

| Stage | Result |
|---|---|
| Stage 1 generation (best-of-3) | 136/164 = **82.9%** (fully reproduces the standalone baseline) |
| Still failing | 28 tasks |
| Arm A blind rewrite (baseline) | 1/28 = **3.6%** |
| Arm B with reviewer advice | 1/7 = 14.3% (only 7 flagged; too small to conclude) |
| **Arm C with reflection advice** | **6/28 = 21.4%** |
| Reflection's net value | **+5 tasks** |
| **Final pass rate** | **142/164 = 86.6%** (from 82.9%, +3.7 pp) |

Three conclusions:

1. **They really are complementary.** best-of-3 picks off "could do, unlucky sampling" tasks; reflection attacks "misunderstood the requirement". Five more fixed, 82.9% → **86.6%**. The framework's strongest configuration to date.

2. **Marginal repair difficulty rises — don't extrapolate single points.** Comparing the two experiments:

   | | Failures after single-shot | Failures after best-of-3 |
   |---|---|---|
   | Failure count | 45 | 28 |
   | Arm A blind rewrite success | 8.9% | **3.6%** |
   | Arm C reflection success | 24.4% | **21.4%** |

   best-of-3 already skimmed the "get lucky on a rewrite" easy tasks, leaving reflection only hard ones — hence blind rewrite collapsing hardest (8.9% → 3.6%) and reflection easing from 24.4% to 21.4%. **Exactly as predicted before stacking.**

3. **The reviewer remains useless.** In the stacked setting only 7 tasks got flagged and 1 was fixed — far below statistical meaning; combined with 3d, the reviewer should not stay in the framework.

**The framework's strongest configuration** (no LoRA anywhere):

```
SFT base + chat template
  → best-of-3 sampling + test filtering          (72.6% → 82.9%)
  → reflection (SGSR) + rewrite on the rest       (        → 86.6%)
```

The executor LoRA (executor-v1) is useless; the reviewer LoRAs (reviewer-v1~v5) fail to transfer to real code with zero repair value. Both are retired.

### 4. This round's most important conclusion: swap the base >> fine-tune

| Method | Gain (HumanEval) |
|---|---|
| **Swap base (Base → SFT)** | **+58 pp** (14.6% → 72.6%) |
| best-of-3 + test filtering | +10.3 pp (72.6% → 82.9%) |
| Two rounds of LoRA (executor) | ±3 pp, **all within noise** |

**What the executor needed was never fine-tuning: the right base, correct prompting, more samples with filtering.**

Stack the reflection loop (SGSR) on top for tasks best-of-3 still misses, and 82.9% becomes **86.6%** (see 3f) — neither step requires training, and together they are the strongest configuration on this 2B model.

The dozen hours spent on fine-tuning across two rounds returned less than the single act of swapping the base. That also explains the early walls: we were optimizing the wrong thing against a too-soft yardstick.

---

## 9. Productization: the Flux Spark CLI

The findings crystallized into `fluxspark.py` (pure CLI, no GUI). It needs only the SFT base — **no LoRA at all**.

### Design points

- **Code to stdout, logs to stderr** — redirect straight into a `.py` file.
- **Tests are an optional enhancement**, three-tier fallback:
  1. `--tests` given → full loop (best-of-N filtering + reflection + regression check)
  2. `--selftest` → the model writes its own tests first (reduced confidence, flagged in output)
  3. Neither → "generate + reflect" only; no way to confirm the fix actually worked
- **The reflection loop (SGSR) needs no tests** — inputs are just "requirement + code", so it runs in every mode.
- **Only improvements are accepted**: a reflected rewrite must pass the tests to replace the original; otherwise the original is kept (never make it worse).

### Usage

```bash
# Strongest mode: tests + 3 samples + reflection repair
python3 fluxspark.py "check if a list is monotonic (equal neighbors allowed)" \
  --tests 'assert f([1,2,2,4]); assert not f([1,3,2])' --n 3

# Let the model write its own tests
python3 fluxspark.py "..." --selftest

# Save to file (stdout redirect, or --out)
python3 fluxspark.py "..." --n 3 --tests '...' > out.py
```

### Interactive terminal

Beyond one-shot questions, there's a **persistent interactive terminal**:

```bash
python3 fluxspark.py -i        # explicit
python3 fluxspark.py           # auto-enters when no requirement argument is given
```

```
========================================================
 Flux Spark    Interactive Terminal
--------------------------------------------------------
 Model: MiniCPM5-2B-SFT-MLX-4bit
 Now: samples 1 | self-tests off | reflection on | quiet off
--------------------------------------------------------
 Type a natural-language requirement and press Enter.
 /help for commands, /quit to exit, Ctrl+D also quits.
========================================================

[1] requirement> write a function that checks if a list is monotonic (equal neighbors allowed)
    tests (optional, Enter to skip)> assert f([1,2,2,4]); assert not f([1,3,2])
```

**Its real value is "the model loads once".** In one-shot mode, every question re-reads the weights into memory (~20–30 s);
in interactive mode you pay that once, then each question costs only generation (~10 s). Ask several questions in a row and the difference is dramatic.

Settings can change mid-session without restarting:

| Command | Effect |
|---|---|
| `/n 3` | Set samples to 3 (best-of-N; needs tests to filter) |
| `/selftest` | Toggle: model writes its own tests when none given |
| `/reflect` | Toggle: reflection self-repair (default on) |
| `/quiet` | Toggle: code only, no logs |
| `/save out.py` | Save the last answer's code |
| `/help` | Command list |
| `/quit`, `Ctrl+D` | Exit (`Ctrl+C` aborts only the current generation) |

Verified live: two consecutive questions (including the no-test reflection path), `/help`, `/save` to disk, `/quit` — all working.

### Real-run verification

| Requirement | Result |
|---|---|
| List monotonicity (equal allowed) | Passes all tests, 13 s, non-strict `<=` / `>=` implementation |
| Median of a list | Passes all tests, averaging for even lengths |
| **Second-largest element (dedup; None if fewer than two)** | **First attempt failed → reflection triggered → fixed and passed** |
| Max of a list (None for empty) | First try; `--out` file clean and runnable |
| Reverse words in a string | First try; outputs `' '.join(reversed(s.split()))` |

That third row is a real self-repair worth highlighting: the model's first output wasn't even parseable, so the tests couldn't run; after reflection it rewrote with `set(nums)` dedup + sort + `[-2]`, and correctly handled "None when fewer than two distinct elements" — **exactly the three edge cases (duplicate max, single element, all-equal) the requirement stated and the first implementation missed.**

It confirms 3e: **no external errors, no reviewer — just comparing behavior against the requirement recovers missed edge cases.** In the product this becomes a practical property: the user can ignore the process entirely and receive the corrected result.

### Two small-model pitfalls we hit

1. **The model loves to "explain first"**: given a natural-language requirement, it outputs paragraphs of (often Chinese) reasoning instead of code.
   Fix: the prompt demands "Write a Python function..." explicitly, and on extraction failure retry once with a harder instruction.
2. **Indented code blocks**: the model indents whole code blocks by 4 spaces (as if quoting), so `ast.parse` raises IndentationError.
   Fix: `textwrap.dedent` first, then walk back from the first `def` to find the longest prefix that parses and contains a function definition —
   this also strips duplicated second code blocks and trailing self-talk.

---

# 10. Cross-model eval: what 4-bit costs, and how much of it the loop pays back (2026-11-04)

Every conclusion so far concerned a single model. This chapter answers a question we couldn't answer before:

> **In the "2B / 4-bit / local / fully offline" niche, what is this pipeline worth? And how much does quantization itself cost?**

## Design

One yardstick, three models (`scripts/bench_matrix.py`; per-task JSONL, resumable):

| Code | Model | Size | Note |
|---|---|---|---|
| A | MiniCPM5-2B-SFT-MLX-4bit | 1.3 GB | Local quantization (project default) |
| B | openbmb/MiniCPM5-2B-MLX | 1.3 GB | Official 4-bit quantization |
| C | openbmb/MiniCPM5-2B-SFT | 4.7 GB | bf16 full precision — the ceiling of quantization loss |

Three conditions, all 164 tasks: **single greedy** → **best-3** (samples 2–3 at temperature 0.8, any pass counts) → **reflection rewrite for best-3 failures**. Prompts, templates and `max_tokens=400` identical to all prior experiments.

**Yardstick validation**: model A's single-shot score of 72.6% matches the Round-4 historical number **exactly** — same model, same greedy decoding, same result. The ruler is trustworthy.

## Results

| Model | Single | best-3 | best-3 + reflection | Δsampling | Δreflection |
|---|---|---|---|---|---|
| A local 4-bit | 72.6% | 84.8% | **87.8%** | +12.2 | +3.0 |
| B official 4-bit | 72.0% | 79.9% | **87.8%** | +7.9 | +7.9 |
| C bf16 | 84.1% | 89.0% | **90.2%** | +4.9 | +1.2 |

Paired McNemar exact tests (same-task comparison):

| Comparison | Single | best-3 | Final |
|---|---|---|---|
| A vs B | p=1.000 | p=0.134 | p=1.000 |
| A vs C | **p=0.002** | p=0.143 | p=0.344 |
| B vs C | **p=0.001** | **p=0.003** | p=0.388 |

## Three conclusions

**1. The real cost of 4-bit: −11.5 pp single-shot (significant), only −2.4 pp final (not significant).**

On single-shot, bf16 is clearly stronger (p<0.003) — that's quantization's real cost. But after "best-3 + reflection", the 1.3 GB 4-bit model and the 4.7 GB bf16 are **statistically indistinguishable** (87.8% vs 90.2%, p=0.344).

**The loop pays quantization's bill down from 11.5 pp to 2.4 pp.** In deployment terms: **3.6× less memory, virtually no final-score loss.**

**2. Local quantization = official quantization.** A vs B single-shot p=1.0 (11 wins vs 10). Quantizing yourself with `mlx_lm.convert` loses nothing — one more reason weights don't belong in the repo.

**3. Pipeline gains shrink as the baseline rises — third confirmation of the same law.**

| Baseline (single-shot) | Total pipeline gain |
|---|---|
| 72.6% (A) | +15.2 pp |
| 72.0% (B) | +15.8 pp |
| 84.1% (C) | +6.1 pp |

Perfectly isomorphic to Round 4's "higher baseline, smaller sampling marginal" (+20 → +10 pp). **The pipeline's position is now clear: it's a lifeline for weak models, not a polish for strong ones** — the weaker the baseline, the more points it buys back.

## Honest limitations

- **Reflection repair rates 39.4% (B) > 20.0% (A) > 11.1% (C) are mostly selection bias, not B being smarter**: the stronger the best-3 filter, the harder the tasks it leaks through, and the harder they are to reflect-fix. Read the final pass rates; don't over-read that middle column.
- **p=0.344 not significant ≠ proof of equality.** 164 tasks have limited power; a true 3–4 pp gap would go undetected. The correct statement is "the gap is smaller than 164 tasks can resolve".
- A's best-3 (84.8%) is above its historical 82.9% — sampling randomness (only 2 temperature-0.8 draws per task), normal variance.
- All three models converge to 88–90% — **HumanEval's ceiling probably lives around here**; more budget should buy little.

## Reproduce

```bash
python3 scripts/bench_matrix.py --model /path/to/model --out bench_x.jsonl
python3 scripts/summarize_bench.py bench_*.jsonl
```

Per-task results: `/Volumes/LLM/logs/bench_{A_ours_2b_4bit,B_official_2b_4bit,C_2b_bf16}.jsonl`.

---

# 11. Re-judged on HumanEval+: how much of 86.6% is real skill (2026-11-04)

All pass rates so far were judged by **HumanEval's own tests** — famously sparse. This chapter re-judges everything under **HumanEval+** (same 164 tasks, ~80× more tests), asking one question:

> **Of the numbers in this report, how much is real skill, and how much merely fooled a sparse test suite?**

## The yardstick

- The official `evalplus.evaluate` is unusable on macOS (its sandbox sets `resource.setrlimit`, which explodes and scores everything 0) — yet another case of "the measuring instrument breaks first".
- Replaced with a local re-judge, `scripts/evalplus_local.py`: expected output = the official `canonical_solution` run on the same input; floats compared with atol.
- **Cross-validation**: its verdicts on the base tests match our `run_test` exactly (119/164, identical).

## Design

Same model (A, local 4-bit), same generation procedure — the only change is the **judge**: `bench_matrix.py --plus`. Judging, filtering and reflection verification all switch to plus tests; every piece of code is recorded under both yardsticks. Generated code is now saved to disk, so future re-judging needs no regeneration.

## Results

| Stage | HumanEval verdict | HumanEval+ verdict | False positives |
|---|---|---|---|
| Single | 72.6% (119) | **68.9%** (113) | 6 tasks (5.0%) |
| best-3 | 81.1% (133) | **78.0%** (128) | 5 tasks (3.8%) |
| + reflection | 82.3% (135) | **79.9%** (131) | 4 tasks (3.0%) |

## Four conclusions

**1. Single-shot shrinks 3.7 pp; 5.0% of "passes" are false positives.** One in every 20 passing solutions collapses under the stricter tests.

**2. The false-positive share falls along the pipeline (5.0% → 3.8% → 3.0%), and so do the absolute counts (6 → 5 → 4).** The most surprising finding: **test-based filtering and reflection repair do not pander to the sparse suite — they incidentally cleaned up 2 false positives**. The chosen candidates and repaired code happened to survive the stricter tests too. The loop does not amplify the yardstick's blind spots.

**3. Most of the pipeline's gain survives strict judging.** 68.9% → 79.9%, still +11.0 pp (vs +9.7~15.2 pp under the lenient ruler, with cross-run sampling noise). **"Gains come from compute + verification" does not depend on how lenient the test suite is.**

**4. SGSR's repair rate shrinks under strict judging: 20% → 8.3% (3/36).** But with only 36 failing tasks, a 3-task difference is entirely within noise — we can say "lower", not "lowered to what".

## Honest notes

- **Cross-run sampling noise is ±3 pp** (same model, same script; base-judged best-3 was 84.8% once and 81.1% another time). Read direction and magnitude, not decimal points.
- The 68.9% single-shot figure **changes the judge but not the generation** — the temperature sampling drew different random numbers than the earlier run, so 68.9 vs 72.6 mixes "stricter tests" (~−3.7) with sampling noise.
- Frontier models also sit around 90% on HumanEval+ — the benchmark itself is saturating. 79.9% (2B / 4-bit / 1.3 GB) still holds up on this ruler, but the gap to the frontier is wider than the HumanEval lens suggests.

## Reproduce

```bash
python3 scripts/evalplus_local.py judge --samples samples.jsonl   # re-judge existing code
python3 scripts/bench_matrix.py --model <path> --plus --out x.jsonl  # full loop under the strict judge
```

Per-task detail: `/Volumes/LLM/logs/benchplus_A.jsonl` (base/plus dual verdicts plus final code).
