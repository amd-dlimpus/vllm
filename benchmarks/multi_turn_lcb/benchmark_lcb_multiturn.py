#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""benchmark_lcb_multiturn.py — Multi-turn LongCodeBench LQA driver.

Restructures LongCodeBench LQA items into multi-turn chat sessions: every
question about a given repository is asked in a single conversation, in
order. Turn 1 sends the full LCB prompt (prompt_goal + repo_text +
question_1). Turns 2..N send only the next question; the model retains
the codebase via chat history.

Why this is interesting for KV-cache studies:
  * Turn 1 is a cold prefill of ~repo_text tokens (e.g. ~25K at 32K).
  * Turns 2..N see a chat-history prefix that already contains
    repo_text + prior Q/A. With prefix caching enabled and a KV pool
    big enough to retain the session, those turns hit ~100% on the
    server-reported cached_tokens metric — until/unless eviction
    kicks in. The shape of the per-turn cache-hit curve is the
    headline signal.

Captures, per turn:
  * accuracy (LongCodeBench's _parse_final_answer, exact-match)
  * server-reported cached_tokens / prompt_tokens
    (requires --enable-prompt-tokens-details on the vLLM server)
  * TTFT, mean inter-chunk TPOT, end-to-end latency
  * predicted letter, correct letter

Output:
  {output_dir}/{tag}__{ctx}__{ts}.turns.jsonl   one row per turn,
                                                streamed (crash-safe)
  {output_dir}/{tag}__{ctx}__{ts}.summary.json  aggregate metrics
                                                + per-turn-index breakdown

Usage:
  python benchmark_lcb_multiturn.py \\
      --lcb-file /scratch/dlimpus/vllm-profiling/LongCodeBench/data/LQA/32K.json \\
      --model /path/to/MiniMax-M2.5 \\
      --url http://127.0.0.1:9311 \\
      --output-dir results/m25_fp8_32K \\
      --tag fp8 --ctx 32K \\
      --max-model-len 131072 --max-tokens-cap 16384 \\
      --concurrency 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Optional

import aiohttp  # type: ignore
from transformers import AutoTokenizer  # type: ignore


# ---- LongCodeBench answer parsing (kept identical to LongCodeQAEvaluator) ----

_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*([ABCD])", re.IGNORECASE)
_TRAILING_LETTER_RE = re.compile(r"\b([ABCD])\s*$", re.IGNORECASE)


def parse_final_answer(response: str) -> Optional[str]:
    """Match LongCodeQAEvaluator._parse_final_answer exactly so accuracy is
    directly comparable to existing LCB runs in experiments/results/."""
    if not response:
        return None
    m = _FINAL_ANSWER_RE.search(response)
    if m:
        return m.group(1).upper()
    m = _TRAILING_LETTER_RE.search(response.strip())
    if m:
        return m.group(1).upper()
    return None


# ------------------------------ Dataset ------------------------------


@dataclass
class LcbItem:
    lcb_index: int        # original index in the LCB jsonl
    repo: str
    prompt: str           # full LCB prompt (prompt_goal + repo_text + question)
    question: str         # just the question portion (used on turns 2..N)
    correct_letter: str   # 'A'|'B'|'C'|'D'


def load_and_group(lcb_file: Path) -> list[tuple[str, list[LcbItem]]]:
    """Load an LCB LQA file and group items by repo, preserving original
    item order within each cluster.

    Returns [(repo, items), ...] sorted by descending cluster size, then
    repo name for determinism. Biggest clusters first means the most
    interesting cache curves land first if a run is interrupted.
    """
    with lcb_file.open() as f:
        raw = json.load(f)
    by_repo: dict[str, list[LcbItem]] = defaultdict(list)
    for i, x in enumerate(raw):
        by_repo[x["repo"]].append(
            LcbItem(
                lcb_index=i,
                repo=x["repo"],
                prompt=x["prompt"],
                question=x["question"],
                correct_letter=x["correct_letter"].strip().upper(),
            )
        )
    return sorted(by_repo.items(), key=lambda kv: (-len(kv[1]), kv[0]))


# --------------------------- Telemetry record ---------------------------


@dataclass
class TurnRecord:
    session_id: str
    repo: str
    turn_index: int          # 0-indexed within session
    lcb_index: int           # original LCB-file index, for joining to baselines
    correct_letter: str
    predicted_letter: Optional[str]
    is_correct: bool
    parse_failed: bool
    # Token counts (server-reported; -1 means usage missing on this response)
    server_prompt_tokens: int = -1
    cached_tokens: int = -1
    completion_tokens: int = -1
    # Driver-side text sizes (chars, not tokens — cheap sanity-check field)
    content_chars: int = 0
    reasoning_chars: int = 0
    # Timing
    ttft_ms: float = -1.0
    tpot_ms: float = -1.0
    latency_ms: float = -1.0
    start_ns: int = 0
    # Failure mode (None = ok)
    error: Optional[str] = None


# --------------------------- Streaming request ---------------------------


async def stream_chat(
    session: aiohttp.ClientSession,
    chat_url: str,
    served_model: str,
    messages: list[dict],
    max_tokens: int,
    timeout_sec: float,
) -> dict:
    """One streaming chat completion. Returns content, reasoning, timings,
    and server-reported usage (prompt_tokens, completion_tokens,
    cached_tokens). Mirrors benchmark_serving_multi_turn.send_request,
    but also accumulates delta.reasoning_content so reasoning models
    work end-to-end."""
    payload = {
        "model": served_model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
    }
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    ttft_ns: Optional[int] = None
    last_chunk_ns: Optional[int] = None
    inter_chunk_delays_ns: list[int] = []
    prompt_tokens = -1
    cached_tokens = -1
    completion_tokens = -1
    error: Optional[str] = None

    start_ns = time.perf_counter_ns()
    try:
        async with session.post(
            chat_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                error = f"HTTP {resp.status}: {body[:500]}"
            else:
                async for raw in resp.content:
                    raw = raw.strip()
                    if not raw:
                        continue
                    line = raw.decode("utf-8")
                    if not line.startswith("data:"):
                        continue
                    line = line.removeprefix("data:").strip()
                    if line == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Terminal usage chunk: empty `choices`, populated `usage`.
                    # vLLM also sometimes piggybacks `usage` on the final
                    # delta chunk — capture it either way.
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get(
                            "prompt_tokens", prompt_tokens
                        )
                        completion_tokens = usage.get(
                            "completion_tokens", completion_tokens
                        )
                        details = usage.get("prompt_tokens_details") or {}
                        cached_tokens = details.get(
                            "cached_tokens", cached_tokens
                        )

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    chunk_content = delta.get("content")
                    # vLLM reasoning parsers use different streaming field
                    # names. minimax_m2 emits delta.reasoning; qwen3 and
                    # openai_gptoss emit delta.reasoning_content. Read both
                    # so this driver works across all three target models
                    # (matches the non-streaming fallback in
                    # vllm-kv-eval/lcb_run_native.py).
                    chunk_reasoning = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                    )

                    if not (chunk_content or chunk_reasoning):
                        continue

                    now_ns = time.perf_counter_ns()
                    if chunk_content:
                        content_parts.append(chunk_content)
                    if chunk_reasoning:
                        reasoning_parts.append(chunk_reasoning)

                    if ttft_ns is None:
                        ttft_ns = now_ns - start_ns
                    else:
                        assert last_chunk_ns is not None
                        inter_chunk_delays_ns.append(now_ns - last_chunk_ns)
                    last_chunk_ns = now_ns

    except asyncio.TimeoutError:
        error = error or "timeout"
    except Exception as e:  # noqa: BLE001
        error = error or f"{type(e).__name__}: {e}"

    end_ns = time.perf_counter_ns()
    latency_ns = end_ns - start_ns
    if ttft_ns is None:
        ttft_ns = latency_ns
    tpot_ns = mean(inter_chunk_delays_ns) if inter_chunk_delays_ns else 0.0

    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "ttft_ms": ttft_ns / 1e6,
        "tpot_ms": tpot_ns / 1e6,
        "latency_ms": latency_ns / 1e6,
        "start_ns": start_ns,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
    }


# ---------------------------- Per-session driver ----------------------------


async def run_session(
    session_id: str,
    repo: str,
    items: list[LcbItem],
    http: aiohttp.ClientSession,
    args: argparse.Namespace,
    tokenizer,
    out_jsonl,
    out_lock: asyncio.Lock,
) -> list[TurnRecord]:
    """Run all items in one repo cluster as a single multi-turn chat session.
    Each turn streams to disk before the next one starts so a mid-session
    crash still leaves usable telemetry."""
    chat_url = f"{args.url.rstrip('/')}/v1/chat/completions"
    served_model = args.served_model_name or args.model
    messages: list[dict] = []
    records: list[TurnRecord] = []

    for turn_idx, item in enumerate(items):
        # Turn 1 sends the full LCB prompt (prompt_goal + repo_text +
        # question). Turns 2..N send only the next question; the codebase
        # is already in chat history, so repeating it would be a guaranteed
        # cache miss on ~25K tokens for zero accuracy benefit.
        user_content = item.prompt if turn_idx == 0 else item.question
        messages.append({"role": "user", "content": user_content})

        # Size max_tokens for this turn against the live chat history.
        # Falls back to a per-message char count if the tokenizer lacks a
        # chat template (rare, but possible for raw HF tokenizers).
        try:
            n_prompt = len(
                tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                )
            )
        except Exception:
            n_prompt = sum(
                len(tokenizer.encode(m["content"], add_special_tokens=False))
                for m in messages
            )
        budget = max(64, args.max_model_len - n_prompt - args.safety_margin)
        max_tokens = min(budget, args.max_tokens_cap)
        # At extreme contexts (256K/1M), it's easy to set --max-model-len too
        # tight and silently floor max_tokens at 64. Surface that loudly so a
        # 1M run doesn't waste hours producing 64-token answers.
        if budget < args.min_completion_budget:
            print(
                f"[multiturn-lcb] WARNING session={session_id} turn={turn_idx} "
                f"n_prompt={n_prompt} max_model_len={args.max_model_len} "
                f"budget={budget} < min_completion_budget="
                f"{args.min_completion_budget}. "
                f"Bump --max-model-len above n_prompt + completion_budget "
                f"+ safety_margin or expect truncated answers."
            )

        result = await stream_chat(
            http, chat_url, served_model,
            messages, max_tokens, args.request_timeout_sec,
        )

        content = result["content"].strip()
        reasoning = result["reasoning"].strip()
        # Score off content (LCB convention). Fall back to reasoning only
        # when content is empty — matches the rationale in
        # vllm-kv-eval/lcb_run_native.py: appending reasoning to content
        # would push the real answer letter off the trailing-letter regex.
        scoring_text = content if content else reasoning
        predicted = parse_final_answer(scoring_text)

        rec = TurnRecord(
            session_id=session_id,
            repo=repo,
            turn_index=turn_idx,
            lcb_index=item.lcb_index,
            correct_letter=item.correct_letter,
            predicted_letter=predicted,
            is_correct=(predicted == item.correct_letter),
            parse_failed=(predicted is None),
            server_prompt_tokens=result["prompt_tokens"],
            cached_tokens=result["cached_tokens"],
            completion_tokens=result["completion_tokens"],
            content_chars=len(content),
            reasoning_chars=len(reasoning),
            ttft_ms=result["ttft_ms"],
            tpot_ms=result["tpot_ms"],
            latency_ms=result["latency_ms"],
            start_ns=result["start_ns"],
            error=result["error"],
        )
        records.append(rec)

        # Append assistant message (content only; never feed reasoning back).
        # Empty content can occur when the model busts its budget mid-CoT;
        # we still append it so the chat structure stays well-formed and
        # turn-counting downstream is consistent. The empty turn is flagged
        # via content_chars=0 in the record.
        messages.append({"role": "assistant", "content": content})

        async with out_lock:
            out_jsonl.write(json.dumps(asdict(rec)) + "\n")
            out_jsonl.flush()

        if result["error"]:
            print(
                f"[{session_id} turn {turn_idx}] error: "
                f"{result['error']!r} — aborting session"
            )
            break

    return records


# -------------------------------- Summary --------------------------------


def build_summary(
    records: list[TurnRecord], args: argparse.Namespace, wall_sec: float
) -> dict:
    n_correct = sum(1 for r in records if r.is_correct)
    n_parse_failed = sum(1 for r in records if r.parse_failed and r.error is None)
    n_errored = sum(1 for r in records if r.error is not None)

    has_cache = any(r.cached_tokens >= 0 for r in records)
    cache_records = [r for r in records if r.cached_tokens >= 0]
    total_prompt = sum(max(r.server_prompt_tokens, 0) for r in cache_records)
    total_cached = sum(max(r.cached_tokens, 0) for r in cache_records)
    overall_cache_hit_rate = (
        total_cached / total_prompt
        if (has_cache and total_prompt > 0) else None
    )

    by_turn: dict[int, list[TurnRecord]] = defaultdict(list)
    for r in records:
        by_turn[r.turn_index].append(r)

    per_turn = []
    for t in sorted(by_turn.keys()):
        bucket = by_turn[t]
        scored = [r for r in bucket if r.error is None]
        cache_bucket = [r for r in bucket if r.cached_tokens >= 0]
        prompt_sum = sum(max(r.server_prompt_tokens, 0) for r in cache_bucket)
        cached_sum = sum(max(r.cached_tokens, 0) for r in cache_bucket)
        cache_hit = (cached_sum / prompt_sum) if prompt_sum > 0 else None
        per_turn.append({
            "turn": t,
            "n": len(bucket),
            "n_correct": sum(1 for r in scored if r.is_correct),
            # Accuracy denominator follows LCB convention: counts every
            # request that returned (errors excluded), with parse_failed
            # treated as wrong.
            "accuracy": (
                sum(1 for r in scored if r.is_correct) / len(scored)
                if scored else 0.0
            ),
            "cache_hit_rate": cache_hit,
            "prompt_tokens_sum": prompt_sum,
            "cached_tokens_sum": cached_sum,
            "ttft_ms_mean": (
                mean([r.ttft_ms for r in scored]) if scored else 0.0
            ),
            "tpot_ms_mean": (
                mean([r.tpot_ms for r in scored]) if scored else 0.0
            ),
            "latency_ms_mean": (
                mean([r.latency_ms for r in scored]) if scored else 0.0
            ),
            "completion_tokens_mean": (
                mean([max(r.completion_tokens, 0) for r in scored])
                if scored else 0.0
            ),
        })

    valid = [r for r in records if r.error is None]
    return {
        "tag": args.tag,
        "ctx": args.ctx,
        "model": args.model,
        "served_model_name": args.served_model_name,
        "lcb_file": str(args.lcb_file),
        "url": args.url,
        "concurrency": args.concurrency,
        "max_model_len": args.max_model_len,
        "max_tokens_cap": args.max_tokens_cap,
        "wall_sec": wall_sec,
        "n_sessions": len({r.session_id for r in records}),
        "n_turns": len(records),
        "n_correct": n_correct,
        "n_parse_failed": n_parse_failed,
        "n_errored": n_errored,
        # Headline accuracy: correct / total turns, parse-fails counted wrong.
        # Same denominator the LCB scorer uses, so directly comparable to
        # the single-turn baselines in experiments/results/*/lcb_aditi_pr/.
        "accuracy": (n_correct / len(records)) if records else 0.0,
        "overall_cache_hit_rate": overall_cache_hit_rate,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "ttft_ms_mean": mean([r.ttft_ms for r in valid]) if valid else 0.0,
        "tpot_ms_mean": mean([r.tpot_ms for r in valid]) if valid else 0.0,
        "latency_ms_mean": (
            mean([r.latency_ms for r in valid]) if valid else 0.0
        ),
        "per_turn_index": per_turn,
    }


# ---------------------------------- main ----------------------------------


async def amain(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = f"{args.tag}__{args.ctx}__{ts}"
    turns_path = out_dir / f"{base}.turns.jsonl"
    summary_path = out_dir / f"{base}.summary.json"

    print(f"[multiturn-lcb] tag={args.tag} ctx={args.ctx} url={args.url}")
    print(f"[multiturn-lcb] lcb_file={args.lcb_file}")
    print(f"[multiturn-lcb] turns   -> {turns_path}")
    print(f"[multiturn-lcb] summary -> {summary_path}")

    sessions = load_and_group(Path(args.lcb_file))
    if args.max_repos:
        sessions = sessions[: args.max_repos]
    if args.max_questions_per_repo:
        sessions = [
            (r, items[: args.max_questions_per_repo]) for r, items in sessions
        ]
    n_sessions = len(sessions)
    n_turns = sum(len(items) for _, items in sessions)
    longest = max(len(items) for _, items in sessions)
    print(
        f"[multiturn-lcb] {n_sessions} sessions, {n_turns} total turns "
        f"(longest session = {longest} turns)"
    )

    print(f"[multiturn-lcb] loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    all_records: list[TurnRecord] = []
    t0 = time.time()

    async with aiohttp.ClientSession() as http:
        with turns_path.open("w") as out_jsonl:

            async def run_one(session_id: str, repo: str, items):
                async with sem:
                    print(
                        f"[multiturn-lcb] start session={session_id} "
                        f"({len(items)} turns)"
                    )
                    recs = await run_session(
                        session_id, repo, items, http, args, tokenizer,
                        out_jsonl, out_lock,
                    )
                    n_ok = sum(1 for r in recs if r.error is None)
                    n_correct = sum(1 for r in recs if r.is_correct)
                    print(
                        f"[multiturn-lcb] done  session={session_id} "
                        f"ok={n_ok}/{len(recs)} correct={n_correct}/{len(recs)}"
                    )
                    return recs

            tasks = [
                asyncio.create_task(
                    run_one(repo.replace("/", "__"), repo, items)
                )
                for repo, items in sessions
            ]

            for fut in asyncio.as_completed(tasks):
                all_records.extend(await fut)

    wall_sec = time.time() - t0

    summary = build_summary(all_records, args, wall_sec)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"[multiturn-lcb] DONE in {wall_sec / 60:.1f} min")
    print(
        f"[multiturn-lcb] accuracy = {summary['accuracy']:.4f} "
        f"({summary['n_correct']}/{summary['n_turns']}, "
        f"parse_fail={summary['n_parse_failed']}, "
        f"errored={summary['n_errored']})"
    )
    if summary.get("overall_cache_hit_rate") is not None:
        print(
            f"[multiturn-lcb] cache_hit_rate = "
            f"{summary['overall_cache_hit_rate']:.4f} "
            f"(cached {summary['total_cached_tokens']:,} / "
            f"prompt {summary['total_prompt_tokens']:,})"
        )
    else:
        print(
            "[multiturn-lcb] cache_hit_rate = (no usage reported by server; "
            "did you launch vLLM with --enable-prompt-tokens-details?)"
        )
    print(
        f"[multiturn-lcb] ttft_ms mean={summary['ttft_ms_mean']:.1f}  "
        f"tpot_ms mean={summary['tpot_ms_mean']:.2f}  "
        f"latency_ms mean={summary['latency_ms_mean']:.1f}"
    )
    print()
    print("Per-turn-index breakdown:")
    print(
        f"  {'turn':>4}  {'n':>4}  {'cache_hit':>9}  "
        f"{'acc':>6}  {'ttft_ms':>9}  {'tpot_ms':>8}  {'latency_ms':>11}"
    )
    for row in summary["per_turn_index"]:
        ch = (
            f"{row['cache_hit_rate']:.3f}"
            if row["cache_hit_rate"] is not None else "  --   "
        )
        print(
            f"  {row['turn']:>4d}  {row['n']:>4d}  {ch:>9}  "
            f"{row['accuracy']:>6.3f}  {row['ttft_ms_mean']:>9.1f}  "
            f"{row['tpot_ms_mean']:>8.2f}  {row['latency_ms_mean']:>11.1f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        prog="benchmark_lcb_multiturn",
        description=(
            "Multi-turn LongCodeBench LQA driver. Treats same-repo questions "
            "as one chat session so prefix caching has structural reuse to "
            "exploit. Captures cache hit rate alongside accuracy."
        ),
    )
    p.add_argument(
        "--lcb-file", required=True,
        help="Path to LongCodeBench LQA jsonl, "
             "e.g. .../LongCodeBench/data/LQA/32K.json",
    )
    p.add_argument(
        "--model", required=True,
        help="Tokenizer path/name (HF). Also used as the served-model id "
             "in API requests unless --served-model-name is given.",
    )
    p.add_argument(
        "--served-model-name", default=None,
        help="If set, use this in the API request body. --model is still "
             "used for the tokenizer.",
    )
    p.add_argument(
        "--url", required=True,
        help="vLLM base URL, e.g. http://127.0.0.1:9311",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--tag", required=True,
        help="Scheme tag for output filenames (fp8|tq44|tq34|tq42|...).",
    )
    p.add_argument(
        "--ctx", required=True,
        help="Context tag for output filenames (e.g. 32K). Free-form; the "
             "actual context budget comes from --max-model-len.",
    )
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--max-tokens-cap", type=int, default=16384)
    p.add_argument("--safety-margin", type=int, default=256)
    p.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent sessions in flight. Default 1 keeps "
             "per-turn cache curves clean (no inter-session eviction noise).",
    )
    p.add_argument(
        "--request-timeout-sec", type=float, default=7200.0,
        help="Per-request timeout. Default 7200s (2h) accommodates 1M-context "
             "turn-1 prefills on slower TP configs. Drop to 1800 for <=128K.",
    )
    p.add_argument(
        "--min-completion-budget", type=int, default=512,
        help="Minimum tokens that must be available for a completion before "
             "the driver warns. Trips at extreme contexts when --max-model-len "
             "is set too tight. Default 512.",
    )
    p.add_argument(
        "--max-repos", type=int, default=None,
        help="Smoke-test cap: only run the largest N repo clusters.",
    )
    p.add_argument(
        "--max-questions-per-repo", type=int, default=None,
        help="Smoke-test cap: only run the first N questions of each cluster.",
    )
    args = p.parse_args()

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
