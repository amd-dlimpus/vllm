# multi_turn_lcb — LongCodeBench as a multi-turn benchmark

Restructures LongCodeBench-LQA so that **every question about the same
repository is asked in one chat session**, in order. Turn 1 sends the full
LCB prompt (`prompt_goal + repo_text + question_1`); turns 2..N send only
the next question — the model retains the codebase via chat history.

This exposes prefix caching to a workload where the *first* turn is a
cold ~25K-token prefill and turns 2..N share that prefix verbatim. The
benchmark records per-turn cache hit rate (server-reported
`cached_tokens`) alongside accuracy, so you can watch both move
together as KV scheme / pool size / context length change.

## What it captures (per turn)

| Metric                | Source                                                |
|-----------------------|-------------------------------------------------------|
| `is_correct`          | `LongCodeQAEvaluator._parse_final_answer` (inlined)   |
| `predicted_letter`    | same                                                  |
| `cached_tokens`       | server `usage.prompt_tokens_details.cached_tokens`    |
| `server_prompt_tokens`| server `usage.prompt_tokens`                          |
| `completion_tokens`   | server `usage.completion_tokens`                      |
| `ttft_ms`             | time from POST to first non-empty content/reasoning   |
| `tpot_ms`             | mean inter-chunk delay over the response stream       |
| `latency_ms`          | end-to-end                                            |
| `content_chars`,      | char counts on `delta.content` and                    |
| `reasoning_chars`     | `delta.reasoning_content` respectively                |

The accuracy parser is byte-identical to the upstream
`LongCodeQAEvaluator._parse_final_answer`, so this benchmark's accuracy
is directly comparable to the existing single-turn LCB runs in
`experiments/results/{35b,gptoss,m25}/lcb_aditi_pr/`.

## Server requirements

Three vLLM flags are required for this benchmark to produce meaningful data:

| Flag | Why |
|---|---|
| `--enable-prefix-caching`         | otherwise no cache hits exist at all |
| `--enable-prompt-tokens-details`  | otherwise `cached_tokens` is missing from the SSE usage chunk and every record reports `-1` |
| `--reasoning-parser <name>`       | required for reasoning models. Without it the model's CoT goes into `delta.content`, the trailing-letter regex picks the wrong letter from CoT noise, and accuracy degrades by ~15–30pp depending on CoT length |

`boot_server_multiturn.sh` sets all three. Reasoning parser is the 6th
positional arg; canonical names per model:

| Model            | Reasoning parser  |
|------------------|-------------------|
| MiniMax-M2.5     | `minimax_m2`      |
| Qwen3.5-35B-A3B  | `qwen3`           |
| GPT-OSS-120b     | `openai_gptoss`   |

### Boot commands per model

```bash
# MiniMax-M2.5 (TP=2, fp8 KV)
bash vllm-kv-eval/_orchestrator/boot_server_multiturn.sh \
    "2,3" 9311 fp8 /tmp/lcb_servers \
    /group/amdneuralopt/huggingface/pretrained_models/MiniMaxAI/MiniMax-M2.5 \
    minimax_m2 \
    131072

# Qwen3.5-35B-A3B-FP8 (TP=2, fp8 KV)
bash vllm-kv-eval/_orchestrator/boot_server_multiturn.sh \
    "2,3" 9311 fp8 /tmp/lcb_servers \
    /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3.5-35B-A3B-FP8 \
    qwen3 \
    131072

# GPT-OSS-120b (TP=1 typical; tq schemes need SKIP_LAYERS + ENFORCE_EAGER)
SKIP_LAYERS="0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34" \
ENFORCE_EAGER=1 \
bash vllm-kv-eval/_orchestrator/boot_server_multiturn.sh \
    "5" 9311 tq44 /tmp/lcb_servers \
    /group/amdneuralopt/huggingface/pretrained_models/openai/gpt-oss-120b \
    openai_gptoss \
    131072
```

## Turn structure (the design choice)

```
turn 1 user:      prompt_goal + repo_text + question_1
turn 1 assistant: <model response, content only>
turn 2 user:      question_2
turn 2 assistant: <model response, content only>
turn N user:      question_N
turn N assistant: <model response, content only>
```

Two non-obvious decisions worth flagging:

1. **`repo_text` is sent only on turn 1.** Repeating it on later turns
   would force a fresh ~25K-token prefix-cache miss every turn for zero
   accuracy benefit; the model already has it in chat history.
2. **`reasoning_content` is never fed back into chat history**, only
   `content`. Reasoning is captured in telemetry (`reasoning_chars`)
   and used as a fallback for *answer parsing* when `content` is empty
   (model busted budget mid-CoT), but feeding it into the next turn
   would (a) blow up context size and (b) prime the model with its own
   prior scratchwork on later questions.

## Output

```
{output_dir}/{tag}__{ctx}__{ts}.turns.jsonl    one row per turn, streamed
{output_dir}/{tag}__{ctx}__{ts}.summary.json   aggregate + per-turn-index
```

`turns.jsonl` is flushed after every turn so a mid-run crash leaves usable
data behind. Each row is a `TurnRecord` (see source for fields).

`summary.json` includes:

- Headline: overall `accuracy`, `overall_cache_hit_rate`, mean
  `ttft_ms` / `tpot_ms` / `latency_ms`, `wall_sec`.
- `per_turn_index`: one row per turn index (0, 1, 2, …) aggregating
  `n`, `accuracy`, `cache_hit_rate`, mean timings, mean
  `completion_tokens`. This is the headline curve — turn 0 should be a
  full cache miss (cold prefill), turns 1+ should be near-100% cache
  hits when the KV pool fits the session.

## Sessions and concurrency

- One **session = one repo cluster.** Sessions are sorted by descending
  cluster size, so `decorator` (25 turns at 32K), `tomlkit` (19 at
  128K), etc. land first.
- Within a session: turns run strictly sequentially (turn N+1 cannot
  start until turn N's response has been appended to chat history).
- Between sessions: `--concurrency` controls how many sessions run in
  parallel.
  - `--concurrency 1` (default): clean per-turn-index curves, no
    inter-session eviction noise. Use for the headline numbers.
  - `--concurrency >1`: stress mode. Multiple ~25K-token codebases
    contend for the KV pool; eviction will start to clip per-turn
    cache hit rate on later turns. Useful for exposing scheme-level
    differences in pool capacity.

## Usage

```bash
cd /scratch/dlimpus/vllm-profiling/vllm-aditi/benchmarks/multi_turn_lcb

python benchmark_lcb_multiturn.py \
    --lcb-file /scratch/dlimpus/vllm-profiling/LongCodeBench/data/LQA/32K.json \
    --model    /group/amdneuralopt/huggingface/pretrained_models/MiniMaxAI/MiniMax-M2.5 \
    --url      http://127.0.0.1:9311 \
    --output-dir /scratch/dlimpus/vllm-profiling/experiments/results/m25/lcb_multiturn/fp8_32K \
    --tag fp8 --ctx 32K \
    --max-model-len 131072 --max-tokens-cap 16384 \
    --concurrency 1
```

For a smoke test (largest 2 clusters, 4 questions each):

```bash
python benchmark_lcb_multiturn.py \
    --lcb-file ... --model ... --url ... --output-dir /tmp/smoke \
    --tag smoke --ctx 32K --max-model-len 131072 \
    --max-repos 2 --max-questions-per-repo 4
```

### Long-context buckets (256K, 512K, 1M)

LongCodeBench ships LQA buckets at 32K / 64K / 128K / 256K / 512K / 1M
(`LongCodeBench/data/LQA/{32K,64K,128K,256K,512K,1M}.json`). The driver is
agnostic to context — pass any bucket via `--lcb-file` and a matching
`--max-model-len`. Three things that change at long context:

1. **Completion budget**: `max_tokens` is computed as
   `max_model_len - n_prompt - safety_margin`, capped at `--max-tokens-cap`.
   At 1M context with `--max-model-len 1048576` and a ~1M-token prompt, the
   budget collapses to 64 — useless. **Set `--max-model-len` above the largest
   prompt + completion budget + safety margin.** For LCB 1M the longest
   prompt is ~1M tokens, so `--max-model-len 1064960` (1M + 16K + 384) is a
   reasonable floor. The driver now warns when the per-turn budget falls
   below `--min-completion-budget` (default 512).
2. **Request timeout**: 1M turn-1 prefill on a TP=4 MoE can exceed 30 minutes.
   `--request-timeout-sec` defaults to 7200 (2h) so this doesn't bite; drop
   to 1800 for ≤128K runs to fail-fast on hung servers.
3. **Server `--max-model-len`**: must be set on the vLLM server too (via
   `MAX_MODEL_LEN` arg to `boot_server_multiturn.sh`). The driver's
   `max_model_len` is used to compute completion budgets, but the server's
   value is what gates the actual request.

#### M2.5 at 256K

M2.5's native context cap is 256K, so this is the largest single-instance
bucket. TP=2 fits with `gpu_memory_utilization=0.85` (fp8) / 0.90 (tq).

```bash
# Server (FP8 baseline, TP=2)
bash vllm-kv-eval/_orchestrator/boot_server_multiturn.sh \
    "2,3" 9311 fp8 /tmp/lcb_servers \
    /group/amdneuralopt/huggingface/pretrained_models/MiniMaxAI/MiniMax-M2.5 \
    minimax_m2 \
    273408   # 256K + 16K completion + 384 margin

# Driver
python benchmark_lcb_multiturn.py \
    --lcb-file /scratch/dlimpus/vllm-profiling/LongCodeBench/data/LQA/256K.json \
    --model    /group/amdneuralopt/huggingface/pretrained_models/MiniMaxAI/MiniMax-M2.5 \
    --url      http://127.0.0.1:9311 \
    --output-dir /scratch/dlimpus/vllm-profiling/experiments/results/m25/lcb_multiturn/fp8_256K \
    --tag fp8 --ctx 256K \
    --max-model-len 273408 --max-tokens-cap 16384 \
    --concurrency 1 --request-timeout-sec 3600
```

#### DeepSeek-V4-Flash at 1M

DSv4-Flash supports 1M natively. TP=4 is recommended; 1M-token KV at TQ44
is roughly 8 GB / session, comfortable on 4× MI300X (768 GB pool). Boot
script does not need any changes — `MAX_MODEL_LEN` accepts the larger
value.

```bash
# Server (FP8 baseline, TP=4)
bash vllm-kv-eval/_orchestrator/boot_server_multiturn.sh \
    "0,1,2,3" 9311 fp8 /tmp/lcb_servers \
    /group/amdneuralopt/huggingface/pretrained_models/deepseek-ai/DeepSeek-V4-Flash \
    deepseek_v3 \
    1064960   # 1M + 16K completion + 384 margin

# Driver
python benchmark_lcb_multiturn.py \
    --lcb-file /scratch/dlimpus/vllm-profiling/LongCodeBench/data/LQA/1M.json \
    --model    /group/amdneuralopt/huggingface/pretrained_models/deepseek-ai/DeepSeek-V4-Flash \
    --url      http://127.0.0.1:9311 \
    --output-dir /scratch/dlimpus/vllm-profiling/experiments/results/dsv4_flash/lcb_multiturn/fp8_1M \
    --tag fp8 --ctx 1M \
    --max-model-len 1064960 --max-tokens-cap 16384 \
    --concurrency 1 --request-timeout-sec 7200
```

The DSv4 reasoning parser name (`deepseek_v3` above) is a placeholder —
verify against `vllm/reasoning/` when booting; existing parsers include
`deepseek_v3_reasoning_parser.py` and `deepseek_r1_reasoning_parser.py`.

## Comparing to the single-turn LCB baseline

Same model, same scheme, same context bucket → expect:

- **Accuracy** within ±2pp of the single-turn run on the same item set
  (the prompt content is bit-identical on turn 1; turns 2..N differ
  only in that the model sees its own prior answers, which mostly
  doesn't change the answer). A larger gap means model behavior is
  affected by chat history → that's a real finding worth investigating.
- **Cache hit rate**: single-turn LCB will report ~0% hits across
  items (every request is a fresh single-message prompt). This benchmark
  should report 0% on turn 0 and rising rapidly to ~99% on turn 1+
  while the session fits in the KV pool.

## What this benchmark is **not**

- Not a replacement for the single-turn LCB accuracy eval. The accuracy
  number here is over a different prompt distribution (chat history
  accumulates, model can self-condition). Quote both.
- Not a multi-conversation pressure test like
  `vllm-aditi/benchmarks/multi_turn/`. That driver is built for
  many concurrent random-prefix conversations under load. This driver
  is built for clean per-turn cache/accuracy curves on real eval
  questions, with structural prefix reuse.
- Not a kernel-level micro-benchmark. The TPOT signal here is
  end-to-end through the OpenAI-API HTTP layer.
