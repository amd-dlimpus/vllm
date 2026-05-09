# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-pool KV cache allocator for prefix-tier mixed-precision KV (Phase 1E).

This is the scheduler-side scaffolding for Option E of the prefill-tier
mixed-precision KV plan. It defines the data structures and policy needed
to route blocks of a request into one of two physical KV cache pools per
attention layer:

* **Pool A — high-precision (TQ84 / `turboquant_k8v4_nc`):**
  holds the *prefix* of every request — the first prefill chunk of turn 1
  in a multi-turn LCB session, which is the codebase prefix that is
  reused across all subsequent turns and dominates attention mass.

* **Pool B — low-precision (TQ44 / `turboquant_4bit_nc`):**
  holds *dialogue / decode* blocks — continuation prefill (turn 2..N user
  messages) and decode-time growth — which are smaller, more recent, and
  more error-tolerant per the empirical motivation
  (`experiments/results/m25/lcb_aditi_pr/compare_lcb_accuracy_by_prompt_quartile_all_schemes_orange.png`).

Allocation policy (deterministic, no per-block branching at attention time):
  * If a block is allocated during the **first prefill chunk** of a request
    (i.e. ``cached_len == 0`` and the chunk is_prefill on the request's
    very first scheduler step), it goes to Pool A.
  * All later blocks (continuation prefill chunks, decode growth) go to
    Pool B.
  * Prefix caching: cached prefix blocks already encode their pool_id, so
    a hit returns blocks tagged with their original pool. No recomputation
    is needed — that's what makes Option E cheap on TPOT vs. mass-based
    promotion.

Status: **scaffolding only**. The data structures and policy here are the
public interface that:
  * the v3_split Triton kernel iterates over (two homogeneous phases
    per request: read pool A blocks first, then pool B blocks, single
    online-softmax accumulator);
  * the TQ continuation-prefill code reads to know which dequant codec
    to apply per chunk.

What is **NOT** wired up yet (left as TODOs with explicit comments at
each integration point):
  * Actual physical allocation of two distinct cache tensors per layer
    (currently `KVCacheManager` allocates one tensor per layer).
  * Block-table emission to the GPU per pool (the runner today builds
    a single block table per layer; pool routing requires two).
  * Prefix caching block reuse across pool boundaries (current hash key
    doesn't include pool_id).
  * Cudagraph capture: the kernel needs to run with two block tables.

These are tracked under todo `phase1e_two_pool_alloc` and will be filled
in by follow-up work. The env flag `VLLM_TQ_PREFIX_TIER` gates everything;
when the flag is off (the default) this module is a no-op and the
existing single-pool allocator path is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum

from vllm.logger import init_logger

logger = init_logger(__name__)


# --- Public env flag ------------------------------------------------------


def is_prefix_tier_enabled() -> bool:
    """Whether two-pool prefix-tier allocation is enabled.

    Reads VLLM_TQ_PREFIX_TIER on each call so tests can toggle it. The
    parser accepts ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Anything else (including unset) is treated as off.
    """
    val = os.environ.get("VLLM_TQ_PREFIX_TIER", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


# --- Pool identification --------------------------------------------------


class PoolID(IntEnum):
    """Pool identifier. Encoded into block descriptors and block tables.

    Two-pool allocation MUST use exactly these values: the v3_split kernel
    iterates pool A first, then pool B, in lock-step with these IDs.
    """

    PREFIX = 0  # Pool A — TQ84/high-precision, prefix blocks.
    DIALOGUE = 1  # Pool B — TQ44/low-precision, decode/continuation blocks.


# --- Per-pool dtype configuration ----------------------------------------


@dataclass(frozen=True)
class TwoPoolDtypeConfig:
    """Resolved dtype for each of the two physical KV pools.

    The scheduler uses this to decide which pool a newly-allocated block
    goes into, and the runner uses it to choose the right slot size when
    allocating each pool's underlying tensor.

    Defaults match the plan's primary configuration:
      Pool A (PREFIX)   = turboquant_k8v4_nc  (TQ84 with NC for symmetry)
      Pool B (DIALOGUE) = turboquant_4bit_nc  (TQ44, the Aditi default)
    """

    pool_a_dtype: str = "turboquant_k8v4_nc"
    pool_b_dtype: str = "turboquant_4bit_nc"

    def dtype_for(self, pool_id: PoolID) -> str:
        if pool_id == PoolID.PREFIX:
            return self.pool_a_dtype
        return self.pool_b_dtype


# --- Block descriptor -----------------------------------------------------


@dataclass
class PooledBlockDescriptor:
    """A KV cache block tagged with the pool it lives in.

    This wraps the existing `KVCacheBlock` rather than replacing it, so the
    rest of the scheduler that doesn't care about pools sees the same
    `block_id` it always saw. The pool tag is read by the runner at
    block-table emission time and by the v3_split kernel at attention
    time.

    NOTE: This descriptor is intentionally lightweight. The plan calls for
    the block table to store ``(pool_id, block_id)`` packed; in this
    scaffolding we keep them as separate fields and rely on the runner to
    emit two block-table tensors (one per pool) at materialization. The
    packed representation is an optimization to consider if block-table
    transfer becomes a TPOT cost, which it shouldn't given block tables
    are <1KB per request.
    """

    block_id: int
    pool_id: PoolID = PoolID.DIALOGUE  # default = TQ44 = current behavior


# --- Per-request split-point bookkeeping ---------------------------------


@dataclass
class TwoPoolBlockTable:
    """Per-request, per-layer block table split across the two pools.

    The kernel reads ``pool_a_block_ids`` first (TQ84 dequant) and then
    ``pool_b_block_ids`` (TQ44 dequant) within a single online-softmax
    accumulator pass. ``split_token_index`` is the absolute token offset
    where pool A ends and pool B begins; the kernel uses it to compute
    causal masks correctly across the boundary.

    Construction invariant: ``split_token_index ==
    sum(block_size for blk in pool_a_block_ids)``. This is enforced by
    `TwoPoolBlockTableBuilder`.
    """

    pool_a_block_ids: list[int] = field(default_factory=list)
    pool_b_block_ids: list[int] = field(default_factory=list)
    split_token_index: int = 0


# --- Allocation policy ----------------------------------------------------


def select_pool(
    *,
    is_first_chunk_prefill: bool,
    cached_len: int,
) -> PoolID:
    """Decide which pool a newly-allocated block goes into.

    The plan's static prefix-tier policy: blocks created while serving the
    very first prefill chunk of a request (no cached prefix yet) go into
    pool A (TQ84/high-precision). Everything else — continuation prefill
    chunks, decode growth, recovery from a partial prefix hit — goes into
    pool B (TQ44/low-precision).

    Args:
      is_first_chunk_prefill: True iff the current scheduler step is
        running the first prefill chunk for this request.
      cached_len: Number of tokens already covered by a prefix-cache hit
        for this request. Strictly > 0 means the prefix is already
        materialised and any newly allocated block is for tokens past
        the cached prefix.

    Returns:
      `PoolID.PREFIX` if and only if the block is being allocated for
      brand-new prefill tokens (no cache hit), else `PoolID.DIALOGUE`.
    """
    if is_first_chunk_prefill and cached_len == 0:
        return PoolID.PREFIX
    return PoolID.DIALOGUE


# --- Block-table builder --------------------------------------------------


class TwoPoolBlockTableBuilder:
    """Helper for building `TwoPoolBlockTable` from a sequence of
    `PooledBlockDescriptor` instances.

    Use:
        builder = TwoPoolBlockTableBuilder(block_size=16)
        for desc in pooled_blocks:
            builder.append(desc)
        table = builder.build()

    Validates the plan's structural invariant: pool A blocks must form a
    contiguous prefix of the request — no interleaving with pool B. The
    v3_split kernel relies on this so it can iterate pool A in one phase
    and pool B in the next without per-block dispatch.
    """

    def __init__(self, block_size: int):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self._block_size = block_size
        self._pool_a: list[int] = []
        self._pool_b: list[int] = []
        self._seen_b = False

    def append(self, desc: PooledBlockDescriptor) -> None:
        if desc.pool_id == PoolID.PREFIX:
            if self._seen_b:
                raise ValueError(
                    "Pool A (PREFIX) block appended after a Pool B (DIALOGUE) "
                    "block. The two-pool layout requires pool A blocks to "
                    "form a contiguous prefix of the request."
                )
            self._pool_a.append(desc.block_id)
        else:
            self._seen_b = True
            self._pool_b.append(desc.block_id)

    def build(self) -> TwoPoolBlockTable:
        return TwoPoolBlockTable(
            pool_a_block_ids=list(self._pool_a),
            pool_b_block_ids=list(self._pool_b),
            split_token_index=len(self._pool_a) * self._block_size,
        )


# --- Integration hook for kv_cache_manager.py ----------------------------


def maybe_tag_block_pool(
    block_id: int,
    *,
    is_first_chunk_prefill: bool,
    cached_len: int,
) -> PooledBlockDescriptor:
    """Wrap a freshly-allocated `block_id` with its pool tag.

    Called from `KVCacheManager` when the env flag is on. Returns a
    `PooledBlockDescriptor` that downstream code (block-table emission,
    kernel) consumes. When the env flag is off, callers should not invoke
    this — the existing single-pool path returns bare ints.
    """
    pool_id = select_pool(
        is_first_chunk_prefill=is_first_chunk_prefill,
        cached_len=cached_len,
    )
    return PooledBlockDescriptor(block_id=block_id, pool_id=pool_id)
