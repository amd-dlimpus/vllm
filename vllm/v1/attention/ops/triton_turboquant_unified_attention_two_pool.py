# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1F: Fused two-pool TurboQuant attention (v4).

Replaces the Phase 1E ``v3_split`` host-orchestrated "two single-pool launches +
host-side LSE merge" with a single fused Triton kernel that:

  1. Loads Q once.
  2. Walks Pool A blocks (non-causal — all pool-A tokens are causally before
     any Q in the current chunk by construction of the prefix/dialogue split).
  3. Walks Pool B blocks (causal — pool B holds tokens at/after the split,
     including the current Q chunk's own keys).
  4. Maintains ONE ``(M, L, acc)`` accumulator in registers/SMEM across both
     passes. No HBM round-trip for per-pool intermediate outputs / LSE; no
     second kernel launch; no host-side merge.

Equivalence: identical to running single-pool TQ attention over
``concat(pool_a_seq, pool_b_seq)`` (which IS what we want — the two pools are
just different bytes-per-slot storage views of the same conceptual KV-cache
sequence). Online softmax is associative and commutative across block
iteration order, so visiting Pool A then Pool B with one running ``(M, L, acc)``
state is numerically equivalent to visiting them interleaved.

SMEM math at head_dim=256 (MI300's per-CTA SMEM limit is 64 KB):
  Two-accumulator v3_split was forced through the single-pool kernel at
  prefill ``BLOCK_M=128``, which requests ~256 KB SMEM for output+Q tiles and
  OOMs. The fused kernel uses ONE accumulator and a SMEM-aware ``BLOCK_M``
  heuristic (cap=64 at head_dim=128, cap=16 at head_dim>=256), which fits in
  ~50 KB.

Public API:
  ``triton_turboquant_unified_attention_two_pool_fused(...)`` is a drop-in
  replacement for ``triton_turboquant_unified_attention_split`` from the Phase
  1E module. Same call signature for the backend; same numerical contract;
  faster (one launch) and unblocks head_dim>=256.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_turboquant_decode import _use_fp8_e4b15
from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
    _find_seq_idx,
    _get_layout,
    _get_pair_lut,
    _tq_fuse_q_rotation,
    _tq_load_k_tile,
    _tq_load_v_tile,
)

_is_hip = current_platform.is_rocm()


# ---------------------------------------------------------------------------
# Pool-pass helper: runs the standard online-softmax loop over ONE pool's
# blocks, updating the shared (M, L, acc) state in place. The two pools
# differ only in their TQ codec constants, KV cache pointer, block table,
# and causal flag — everything else (Q, query positions, masks) is shared.
# ---------------------------------------------------------------------------


@triton.jit
def _tq_pool_pass(
    Q,                          # [BLOCK_M, HEAD_SIZE_PADDED] — rotated Q
    M,                          # [BLOCK_M] fp32 — running max
    L,                          # [BLOCK_M] fp32 — running denom
    acc,                        # [BLOCK_M, HEAD_SIZE_PADDED] fp32 — running num
    # Pool-specific KV cache + metadata
    KV_cache_ptr,
    KV_cache_u16_ptr,
    Centroids_ptr,
    Pair_lut_ptr,
    block_tables_ptr,           # [num_seqs, max_blocks_per_seq] int32
    pool_seq_len,               # int32 — tokens this pool holds for this seq
    block_table_offset,         # int64 — seq_idx * block_table_stride
    stride_cache_block: tl.int64,
    # Query geometry (shared across pools)
    cur_batch_query_len,        # int32
    context_len_in_pool,        # int32 — for causal mask: tokens in THIS pool
    #                            already cached before the current Q chunk
    query_pos,                  # [BLOCK_M] int32 — relative q position
    query_mask_0,               # [BLOCK_M] int1
    query_mask_1,               # [BLOCK_M] int1
    dim_mask,                   # [HEAD_SIZE_PADDED] int1
    scale,                      # fp32
    kv_head_idx,                # int32
    offs_d,                     # [HEAD_SIZE_PADDED]
    offs_t,                     # [TILE_SIZE]
    # Pool-specific codec constants (this is what differs between A and B)
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    USE_PAIR_LUT: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    META_REGION_OFFSET: tl.constexpr,
    NUM_SOA_FIELDS: tl.constexpr,
    SOA_K_NORM: tl.constexpr,
    SOA_V_SCALE: tl.constexpr,
    SOA_V_ZERO: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    FP8_E4B15: tl.constexpr,
    # Shared geometry constexprs
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Run one TQ attention pass over ``pool_seq_len`` tokens, updating
    ``(M, L, acc)`` via online softmax. Returns updated state.

    Causal contract:
      * CAUSAL=0  →  every Q row attends to every token in this pool (used
        for the prefix pool, whose tokens are all causally before any Q).
      * CAUSAL=1  →  standard causal mask using positions relative to this
        pool's start (``context_len_in_pool + query_pos``). Used for the
        dialogue pool, which contains the current Q chunk's own keys.
    """
    # Short-circuit if this pool is empty for this seq. ``pool_seq_len``
    # may legally be 0 (e.g. Phase 1E continuation prefill where the
    # current chunk is the first one — pool A is empty).
    if pool_seq_len <= 0:
        return M, L, acc

    DATA_BYTES_PER_SLOT: tl.constexpr = KEY_DATA_BYTES + VAL_DATA_BYTES

    # Longest key prefix any Q row in this q-block can attend to within
    # THIS pool. Non-causal pools always see the full pool. Causal pools
    # see up to (context_len_in_pool + position-of-last-q-in-this-block).
    if CAUSAL:
        max_seq_prefix_len = (
            context_len_in_pool
            + tl.max(query_pos)
            + 1
        )
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, pool_seq_len)
    else:
        max_seq_prefix_len = pool_seq_len

    num_tiles = tl.cdiv(max_seq_prefix_len, TILE_SIZE)
    query_abs_pos = context_len_in_pool + query_pos[:, None]
    dummy_tile_mask = tl.full([TILE_SIZE], 1, tl.int1)

    # --- Main loop: tiles [0, num_tiles-1) are fully within the pool.
    for j in range(0, num_tiles - 1):
        seq_offset = j * TILE_SIZE + offs_t
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)
        slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases = (
            block_base
            + slot_within_block * (NUM_KV_HEADS * DATA_BYTES_PER_SLOT)
            + tl.cast(kv_head_idx, tl.int64) * DATA_BYTES_PER_SLOT
        )
        val_bases = data_bases + KEY_DATA_BYTES
        head_meta_u16_base = (
            (block_base + META_REGION_OFFSET) // 2
            + tl.cast(kv_head_idx, tl.int64) * (NUM_SOA_FIELDS * BLOCK_SIZE)
        )
        knorm_u16_addrs = (
            head_meta_u16_base + SOA_K_NORM * BLOCK_SIZE + slot_within_block
        )
        vscale_u16_addrs = (
            head_meta_u16_base + SOA_V_SCALE * BLOCK_SIZE + slot_within_block
        )
        vzero_u16_addrs = (
            head_meta_u16_base + SOA_V_ZERO * BLOCK_SIZE + slot_within_block
        )
        K_T = _tq_load_k_tile(
            KV_cache_ptr, KV_cache_u16_ptr, data_bases, knorm_u16_addrs,
            offs_d, dim_mask, dummy_tile_mask, Centroids_ptr, Pair_lut_ptr,
            OUT_DTYPE=Q.dtype, HEAD_DIM=HEAD_SIZE, BLOCK_D=HEAD_SIZE_PADDED,
            MSE_BITS=MSE_BITS, N_CENTROIDS=N_CENTROIDS, KEY_FP8=KEY_FP8,
            USE_PAIR_LUT=USE_PAIR_LUT, NORM_CORRECTION=NORM_CORRECTION,
            FP8_E4B15=FP8_E4B15, TILE_SIZE=TILE_SIZE, UNMASKED=True,
        )
        V = _tq_load_v_tile(
            KV_cache_ptr, KV_cache_u16_ptr, val_bases, vscale_u16_addrs,
            vzero_u16_addrs, offs_d, dim_mask, dummy_tile_mask,
            OUT_DTYPE=Q.dtype, HEAD_DIM=HEAD_SIZE, VQB=VQB, UNMASKED=True,
        )
        if USE_BF16_DOT:
            S = scale * tl.dot(Q.to(tl.bfloat16), K_T.to(tl.bfloat16))
        else:
            S = scale * tl.dot(Q, K_T)
        if CAUSAL:
            seq_mask = seq_offset[None, :] <= query_abs_pos
            S = tl.where(
                query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
                S, float("-inf"),
            )
        else:
            S = tl.where(
                query_mask_1[:, None] & query_mask_0[:, None],
                S, float("-inf"),
            )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        if USE_BF16_DOT:
            acc += tl.dot(P.to(tl.bfloat16), V.to(tl.bfloat16))
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # --- Tail tile: last tile may be partial.
    if num_tiles > 0:
        j = num_tiles - 1
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)
        slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases = (
            block_base
            + slot_within_block * (NUM_KV_HEADS * DATA_BYTES_PER_SLOT)
            + tl.cast(kv_head_idx, tl.int64) * DATA_BYTES_PER_SLOT
        )
        val_bases = data_bases + KEY_DATA_BYTES
        head_meta_u16_base = (
            (block_base + META_REGION_OFFSET) // 2
            + tl.cast(kv_head_idx, tl.int64) * (NUM_SOA_FIELDS * BLOCK_SIZE)
        )
        knorm_u16_addrs = (
            head_meta_u16_base + SOA_K_NORM * BLOCK_SIZE + slot_within_block
        )
        vscale_u16_addrs = (
            head_meta_u16_base + SOA_V_SCALE * BLOCK_SIZE + slot_within_block
        )
        vzero_u16_addrs = (
            head_meta_u16_base + SOA_V_ZERO * BLOCK_SIZE + slot_within_block
        )
        K_T = _tq_load_k_tile(
            KV_cache_ptr, KV_cache_u16_ptr, data_bases, knorm_u16_addrs,
            offs_d, dim_mask, tile_mask, Centroids_ptr, Pair_lut_ptr,
            OUT_DTYPE=Q.dtype, HEAD_DIM=HEAD_SIZE, BLOCK_D=HEAD_SIZE_PADDED,
            MSE_BITS=MSE_BITS, N_CENTROIDS=N_CENTROIDS, KEY_FP8=KEY_FP8,
            USE_PAIR_LUT=USE_PAIR_LUT, NORM_CORRECTION=NORM_CORRECTION,
            FP8_E4B15=FP8_E4B15, TILE_SIZE=TILE_SIZE, UNMASKED=False,
        )
        V = _tq_load_v_tile(
            KV_cache_ptr, KV_cache_u16_ptr, val_bases, vscale_u16_addrs,
            vzero_u16_addrs, offs_d, dim_mask, tile_mask,
            OUT_DTYPE=Q.dtype, HEAD_DIM=HEAD_SIZE, VQB=VQB, UNMASKED=False,
        )
        if USE_BF16_DOT:
            S = scale * tl.dot(Q.to(tl.bfloat16), K_T.to(tl.bfloat16))
        else:
            S = scale * tl.dot(Q, K_T)
        if CAUSAL:
            seq_mask = seq_offset[None, :] <= query_abs_pos
            S = tl.where(
                query_mask_1[:, None] & query_mask_0[:, None]
                & seq_mask & tile_mask[None, :],
                S, float("-inf"),
            )
        else:
            # Non-causal tail: tile_mask gates the trailing lanes whose K/V
            # loads returned zero. Without this, those lanes would
            # contribute exp(0)=1 to the softmax denominator.
            S = tl.where(
                query_mask_1[:, None] & query_mask_0[:, None]
                & tile_mask[None, :],
                S, float("-inf"),
            )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        if USE_BF16_DOT:
            acc += tl.dot(P.to(tl.bfloat16), V.to(tl.bfloat16))
        else:
            acc += tl.dot(P.to(V.dtype), V)

    return M, L, acc


# ---------------------------------------------------------------------------
# Fused two-pool 2D kernel. Single (M, L, acc) accumulator carried across
# Pool A (non-causal) and Pool B (causal) passes.
# ---------------------------------------------------------------------------


@triton.jit
def kernel_tq_two_pool_fused_attention_2d(
    output_ptr,                  # [num_tokens, Hq, D] — query.dtype
    query_ptr,                   # [num_tokens, Hq, D] — raw if FUSE_Q_ROT
    # ---- Pool A inputs (TQ84 by convention; high precision, non-causal) ----
    KV_cache_a_ptr,
    KV_cache_a_u16_ptr,
    Centroids_a_ptr,
    Pair_lut_a_ptr,
    block_tables_a_ptr,
    seq_lens_a_ptr,
    stride_cache_block_a: tl.int64,
    block_table_stride_a: tl.int64,
    # ---- Pool B inputs (TQ44 by convention; low precision, causal) ----
    KV_cache_b_ptr,
    KV_cache_b_u16_ptr,
    Centroids_b_ptr,
    Pair_lut_b_ptr,
    block_tables_b_ptr,
    seq_lens_b_ptr,
    stride_cache_block_b: tl.int64,
    block_table_stride_b: tl.int64,
    # ---- Shared inputs ----
    PiT_ptr,                     # [D, D] fp32 — only dereferenced when FUSE_Q_ROT
    query_start_len_ptr,
    sinks_ptr,                   # [Hq] fp32 — only dereferenced when USE_SINKS
    scale,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    pit_stride_0: tl.int64,
    pit_stride_1: tl.int64,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    num_seqs: tl.int32,
    # ---- Pool A codec constants ----
    MSE_BITS_A: tl.constexpr,
    MSE_BYTES_A: tl.constexpr,
    VQB_A: tl.constexpr,
    VAL_DATA_BYTES_A: tl.constexpr,
    N_CENTROIDS_A: tl.constexpr,
    KEY_FP8_A: tl.constexpr,
    USE_PAIR_LUT_A: tl.constexpr,
    KEY_DATA_BYTES_A: tl.constexpr,
    META_REGION_OFFSET_A: tl.constexpr,
    NUM_SOA_FIELDS_A: tl.constexpr,
    SOA_K_NORM_A: tl.constexpr,
    SOA_V_SCALE_A: tl.constexpr,
    SOA_V_ZERO_A: tl.constexpr,
    NORM_CORRECTION_A: tl.constexpr,
    FP8_E4B15_A: tl.constexpr,
    # ---- Pool B codec constants ----
    MSE_BITS_B: tl.constexpr,
    MSE_BYTES_B: tl.constexpr,
    VQB_B: tl.constexpr,
    VAL_DATA_BYTES_B: tl.constexpr,
    N_CENTROIDS_B: tl.constexpr,
    KEY_FP8_B: tl.constexpr,
    USE_PAIR_LUT_B: tl.constexpr,
    KEY_DATA_BYTES_B: tl.constexpr,
    META_REGION_OFFSET_B: tl.constexpr,
    NUM_SOA_FIELDS_B: tl.constexpr,
    SOA_K_NORM_B: tl.constexpr,
    SOA_V_SCALE_B: tl.constexpr,
    SOA_V_ZERO_B: tl.constexpr,
    NORM_CORRECTION_B: tl.constexpr,
    FP8_E4B15_B: tl.constexpr,
    # ---- Shared flags ----
    FUSE_Q_ROT: tl.constexpr = 0,
    USE_SINKS: tl.constexpr = 0,
    USE_BF16_DOT: tl.constexpr = 0,
):
    """Fused two-pool TQ attention.

    Pool A is treated as non-causal (every Q row attends to every pool A
    token). Pool B is causal. Sinks fold into Pool A only (initial M
    = sink logit; Pool B inherits Pool A's M/L). This mirrors the
    convention of the Phase 1E v3_split launcher exactly.
    """
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = (
        tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    )
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = (
        cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = (
        kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    )
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    if FUSE_Q_ROT:
        Q = _tq_fuse_q_rotation(
            Q, PiT_ptr, pit_stride_0, pit_stride_1, dim_mask,
            HEAD_SIZE_PADDED,
        )

    # --- Online softmax state. Sinks fold into Pool A only. ---
    if USE_SINKS:
        M = tl.load(
            sinks_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")
        ).to(tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # --- Pool A pass (non-causal). All pool-A tokens are causally before
    #     any Q in the current chunk. Total seq len in this pool for this
    #     seq is just pool_a_seq_lens[seq_idx]; context_len_in_pool=0
    #     because no pool-A tokens follow the current chunk.
    pool_a_seq_len = tl.load(seq_lens_a_ptr + seq_idx)
    block_table_offset_a = seq_idx * block_table_stride_a
    M, L, acc = _tq_pool_pass(
        Q, M, L, acc,
        KV_cache_a_ptr, KV_cache_a_u16_ptr,
        Centroids_a_ptr, Pair_lut_a_ptr,
        block_tables_a_ptr,
        pool_a_seq_len,
        block_table_offset_a,
        stride_cache_block_a,
        cur_batch_query_len,
        0,  # context_len_in_pool: non-causal, irrelevant
        query_pos, query_mask_0, query_mask_1, dim_mask,
        scale, kv_head_idx, offs_d, offs_t,
        MSE_BITS=MSE_BITS_A, MSE_BYTES=MSE_BYTES_A,
        VQB=VQB_A, VAL_DATA_BYTES=VAL_DATA_BYTES_A,
        N_CENTROIDS=N_CENTROIDS_A, KEY_FP8=KEY_FP8_A,
        USE_PAIR_LUT=USE_PAIR_LUT_A,
        KEY_DATA_BYTES=KEY_DATA_BYTES_A,
        META_REGION_OFFSET=META_REGION_OFFSET_A,
        NUM_SOA_FIELDS=NUM_SOA_FIELDS_A,
        SOA_K_NORM=SOA_K_NORM_A, SOA_V_SCALE=SOA_V_SCALE_A,
        SOA_V_ZERO=SOA_V_ZERO_A,
        NORM_CORRECTION=NORM_CORRECTION_A, FP8_E4B15=FP8_E4B15_A,
        BLOCK_SIZE=BLOCK_SIZE, TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=HEAD_SIZE, HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        BLOCK_M=BLOCK_M, NUM_KV_HEADS=NUM_KV_HEADS,
        USE_BF16_DOT=USE_BF16_DOT, CAUSAL=0,
    )

    # --- Pool B pass (causal). Pool B contains tokens at-and-after the
    #     prefix split, including the current Q chunk's own keys.
    #     ``context_len_in_pool`` for the causal mask is the number of
    #     pool-B tokens that exist BEFORE the current Q chunk, i.e.
    #     pool_b_seq_len - cur_batch_query_len.
    pool_b_seq_len = tl.load(seq_lens_b_ptr + seq_idx)
    context_len_b = pool_b_seq_len - cur_batch_query_len
    block_table_offset_b = seq_idx * block_table_stride_b
    M, L, acc = _tq_pool_pass(
        Q, M, L, acc,
        KV_cache_b_ptr, KV_cache_b_u16_ptr,
        Centroids_b_ptr, Pair_lut_b_ptr,
        block_tables_b_ptr,
        pool_b_seq_len,
        block_table_offset_b,
        stride_cache_block_b,
        cur_batch_query_len,
        context_len_b,
        query_pos, query_mask_0, query_mask_1, dim_mask,
        scale, kv_head_idx, offs_d, offs_t,
        MSE_BITS=MSE_BITS_B, MSE_BYTES=MSE_BYTES_B,
        VQB=VQB_B, VAL_DATA_BYTES=VAL_DATA_BYTES_B,
        N_CENTROIDS=N_CENTROIDS_B, KEY_FP8=KEY_FP8_B,
        USE_PAIR_LUT=USE_PAIR_LUT_B,
        KEY_DATA_BYTES=KEY_DATA_BYTES_B,
        META_REGION_OFFSET=META_REGION_OFFSET_B,
        NUM_SOA_FIELDS=NUM_SOA_FIELDS_B,
        SOA_K_NORM=SOA_K_NORM_B, SOA_V_SCALE=SOA_V_SCALE_B,
        SOA_V_ZERO=SOA_V_ZERO_B,
        NORM_CORRECTION=NORM_CORRECTION_B, FP8_E4B15=FP8_E4B15_B,
        BLOCK_SIZE=BLOCK_SIZE, TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=HEAD_SIZE, HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        BLOCK_M=BLOCK_M, NUM_KV_HEADS=NUM_KV_HEADS,
        USE_BF16_DOT=USE_BF16_DOT, CAUSAL=1,
    )

    # --- Epilogue: normalize and store. Both pools empty (mask/padding
    #     rows): L = 1.0 (initial), acc = 0 → output = 0. Correct.
    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# ---------------------------------------------------------------------------
# Python launcher.
# ---------------------------------------------------------------------------


def _smem_aware_block_m(
    head_dim: int,
    kv_group_size: int,
    is_prefill_like: bool,
) -> int:
    """SMEM-aware BLOCK_M for the fused two-pool kernel.

    The single-pool kernel's prefill heuristic of BLOCK_M=128 OOMs at
    head_dim>=256 (256 KB of fp32 output tile + Q tile + K/V tile vs MI300's
    64 KB SMEM per CTA). We use ONE accumulator instead of two but the tile
    sizes are otherwise the same, so we cap BLOCK_M to fit:

      head_dim<=128, prefill:  BLOCK_M=128 (matches single-pool optimum)
      head_dim<=128, decode :  BLOCK_M=16
      head_dim==192, prefill:  BLOCK_M=64
      head_dim>=256, prefill:  BLOCK_M=32
      head_dim>=256, decode :  BLOCK_M=16

    These caps are conservative — head_dim=256 with one fp32 acc =
    BLOCK_M * 256 * 4 B = BLOCK_M * 1 KB; even BLOCK_M=64 fits comfortably
    once you remove the v3_split's two-accumulator pressure. The 32 cap
    is the empirical sweet spot where Triton's autoscheduler is robust.
    """
    if is_prefill_like:
        if head_dim <= 128:
            base = 128
        elif head_dim <= 192:
            base = 64
        else:
            base = 32
    else:
        base = 16
    return max(base, triton.next_power_of_2(kv_group_size))


def triton_turboquant_unified_attention_two_pool_fused(
    query: torch.Tensor,            # [num_tokens, Hq, D]
    pool_a_kv_cache: torch.Tensor,  # [n_blocks_a, bs, Hk, slot_a] uint8
    pool_b_kv_cache: torch.Tensor,  # [n_blocks_b, bs, Hk, slot_b] uint8
    pool_a_block_table: torch.Tensor,  # [num_seqs, max_a_blocks] int32
    pool_b_block_table: torch.Tensor,  # [num_seqs, max_b_blocks] int32
    pool_a_seq_lens: torch.Tensor,  # int32
    pool_b_seq_lens: torch.Tensor,  # int32
    query_start_loc: torch.Tensor,  # [num_seqs+1] int32
    *,
    pool_a_Pi: torch.Tensor,
    pool_a_centroids: torch.Tensor,
    pool_a_mse_bits: int,
    pool_a_key_packed_size: int,
    pool_a_value_quant_bits: int,
    pool_a_value_packed_size: int,
    pool_a_key_fp8: bool,
    pool_a_norm_correction: bool,
    pool_a_PiT: torch.Tensor | None = None,
    pool_b_Pi: torch.Tensor,
    pool_b_centroids: torch.Tensor,
    pool_b_mse_bits: int,
    pool_b_key_packed_size: int,
    pool_b_value_quant_bits: int,
    pool_b_value_packed_size: int,
    pool_b_key_fp8: bool,
    pool_b_norm_correction: bool,
    pool_b_PiT: torch.Tensor | None = None,
    scale: float,
    output: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    max_pool_a_seq_len: int | None = None,
    max_pool_b_seq_len: int | None = None,
    max_query_len: int | None = None,
    fuse_q_rot: bool = True,
    block_m_override: int | None = None,
    tile_size_override: int | None = None,
) -> torch.Tensor:
    """v4 fused two-pool TQ attention launcher.

    Drop-in replacement for ``triton_turboquant_unified_attention_split``:
    same call signature, same numerical contract, but a single fused
    Triton kernel instead of two single-pool launches + host-side merge.

    Wins vs v3_split:
      * One kernel launch instead of two (-1 launch overhead per attn).
      * One (M, L, acc) accumulator instead of two (lower SMEM pressure).
      * No HBM round-trip for per-pool intermediate outputs / LSE.
      * No host-side fp32 merge tensor op.
      * **Unblocks head_dim>=256** (the v3_split SMEM OOM at MI300).

    Causal contract (same as v3_split):
      * Pool A is non-causal — every Q row attends to every pool-A token.
        Used for the prefix pool whose tokens are all causally before any
        Q in the current chunk.
      * Pool B is causal — standard causal mask. Used for the dialogue
        pool, which contains the current Q chunk's own keys.

    Edge cases:
      * ``pool_a_seq_lens[i] == 0``: pool-A pass is a no-op, accumulator
        passes through unchanged into pool-B.
      * ``pool_b_seq_lens[i] == 0``: symmetric — pool-B pass is a no-op.
      * Both empty: L stays at 1.0, acc stays 0 → output = 0.
      * ``sinks`` is folded into pool A by initializing M = sink logit
        and L = 1.0 (== exp(sink-sink)). Pool B inherits.
    """
    if query.dim() != 3:
        raise ValueError(f"query must be [N, Hq, D]; got {query.shape}")
    if pool_a_kv_cache.dim() != 4 or pool_b_kv_cache.dim() != 4:
        raise ValueError(
            "pool_{a,b}_kv_cache must be 4-D "
            f"[num_blocks, block_size, Hk, slot]; "
            f"got A={pool_a_kv_cache.shape} B={pool_b_kv_cache.shape}"
        )
    a_bs, a_Hk = pool_a_kv_cache.shape[1], pool_a_kv_cache.shape[2]
    b_bs, b_Hk = pool_b_kv_cache.shape[1], pool_b_kv_cache.shape[2]
    if a_bs != b_bs or a_Hk != b_Hk:
        raise ValueError(
            f"pool A/B must share block_size and Hk; got "
            f"A=(bs={a_bs}, Hk={a_Hk}) B=(bs={b_bs}, Hk={b_Hk})"
        )
    if pool_a_block_table.shape[0] != pool_b_block_table.shape[0]:
        raise ValueError(
            "pool_a_block_table and pool_b_block_table must have same "
            f"num_seqs; got A={pool_a_block_table.shape[0]} "
            f"B={pool_b_block_table.shape[0]}"
        )
    if pool_a_seq_lens.shape != pool_b_seq_lens.shape:
        raise ValueError(
            f"pool_a_seq_lens vs pool_b_seq_lens shape mismatch: "
            f"{pool_a_seq_lens.shape} vs {pool_b_seq_lens.shape}"
        )

    num_tokens, Hq, D = query.shape
    Hk = a_Hk
    block_size = a_bs
    kv_group_size = Hq // Hk
    num_seqs = int(query_start_loc.shape[0] - 1)
    device = query.device

    cfg_a = _get_layout(D, pool_a_mse_bits, pool_a_value_quant_bits)
    cfg_b = _get_layout(D, pool_b_mse_bits, pool_b_value_quant_bits)
    _ = pool_a_value_packed_size
    _ = pool_b_value_packed_size

    # ---- Codec-derived SoA layout constants per pool ----
    def _codec_layout(cfg, key_fp8: bool):
        mse_bytes = cfg["mse_bytes"]
        val_data_bytes = cfg["val_data_bytes"]
        key_data_bytes = D if key_fp8 else mse_bytes
        data_bytes_per_slot = key_data_bytes + val_data_bytes
        meta_region_offset = block_size * Hk * data_bytes_per_slot
        num_soa_fields = 2 if key_fp8 else 3
        soa_k_norm = 0  # unused for FP8; harmless constant
        soa_v_scale = 0 if key_fp8 else 1
        soa_v_zero = 1 if key_fp8 else 2
        return {
            "mse_bytes": mse_bytes,
            "val_data_bytes": val_data_bytes,
            "key_data_bytes": key_data_bytes,
            "meta_region_offset": meta_region_offset,
            "num_soa_fields": num_soa_fields,
            "soa_k_norm": soa_k_norm,
            "soa_v_scale": soa_v_scale,
            "soa_v_zero": soa_v_zero,
        }

    lay_a = _codec_layout(cfg_a, pool_a_key_fp8)
    lay_b = _codec_layout(cfg_b, pool_b_key_fp8)
    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)

    # ---- Q rotation: same path for both pools (Pi shared per layer) ----
    # SMEM guard: in-kernel Q rotation requires loading a HEAD_SIZE_PADDED ×
    # HEAD_SIZE_PADDED PiT tile (bf16 = 2B). At head_dim=256 that's 128 KB,
    # well over MI300X's 64 KB per-CTA SMEM cap. Disable the fused rotation
    # path and fall back to a launcher-side rocBLAS GEMM in that regime.
    fuse_rot_smem_ok = triton.next_power_of_2(D) <= 128
    if pool_a_key_fp8 and pool_b_key_fp8:
        q_rot = query.contiguous()
        apply_fuse_q_rot = False
        PiT_f32 = pool_a_centroids  # harmless dummy
        pit_stride_0 = 0
        pit_stride_1 = 0
    else:
        # Use pool A's PiT by convention. v3_split assumes Pi/PiT are
        # the same per layer (only the codec preset differs); we keep
        # that assumption explicit.
        PiT_src = pool_a_PiT if pool_a_PiT is not None else pool_a_Pi.T
        PiT_src = PiT_src.contiguous()
        apply_fuse_q_rot = bool(fuse_q_rot) and fuse_rot_smem_ok
        if apply_fuse_q_rot:
            q_rot = query.contiguous()
        else:
            q_rot = (
                query.float() @ PiT_src
            ).to(query.dtype).contiguous()
        PiT_f32 = (
            PiT_src if PiT_src.dtype == torch.float32
            else PiT_src.to(torch.float32)
        )
        if not PiT_f32.is_contiguous():
            PiT_f32 = PiT_f32.contiguous()
        pit_stride_0 = PiT_f32.stride(0)
        pit_stride_1 = PiT_f32.stride(1)

    # ---- Sinks ----
    if sinks is not None:
        sinks_f32 = (
            sinks if sinks.dtype == torch.float32
            else sinks.to(torch.float32)
        )
        if not sinks_f32.is_contiguous():
            sinks_f32 = sinks_f32.contiguous()
        assert sinks_f32.numel() == Hq, (
            f"sinks must have shape [Hq={Hq}], got numel={sinks_f32.numel()}"
        )
        use_sinks = True
    else:
        sinks_f32 = pool_a_centroids  # harmless dummy
        use_sinks = False

    if output is None:
        output = torch.empty_like(query)

    # ---- BLOCK_M heuristic (SMEM-aware) ----
    if max_query_len is not None:
        is_prefill_like = max_query_len > 1
    else:
        is_prefill_like = num_tokens > num_seqs
    BLOCK_M = (
        block_m_override
        if block_m_override is not None
        else _smem_aware_block_m(D, kv_group_size, is_prefill_like)
    )
    BLOCK_Q = BLOCK_M // kv_group_size
    total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs

    # ---- Pair-LUT fast path per pool ----
    use_pair_lut_a = (not pool_a_key_fp8) and (pool_a_mse_bits == 4)
    use_pair_lut_b = (not pool_b_key_fp8) and (pool_b_mse_bits == 4)
    pair_lut_a = (
        _get_pair_lut(pool_a_centroids) if use_pair_lut_a
        else pool_a_centroids
    )
    pair_lut_b = (
        _get_pair_lut(pool_b_centroids) if use_pair_lut_b
        else pool_b_centroids
    )

    # ---- TILE_SIZE (same heuristic as single-pool: 16 for decode, 32
    #      for prefill) ----
    if tile_size_override is not None:
        TILE_SIZE = tile_size_override
    elif is_prefill_like:
        TILE_SIZE = 32
    else:
        TILE_SIZE = 16

    HEAD_SIZE_PADDED = triton.next_power_of_2(D)

    # ---- uint16 views for SoA metadata loads ----
    from vllm.v1.attention.ops.triton_turboquant_decode import kv_cache_flat_u16
    pool_a_kv_u16 = kv_cache_flat_u16(pool_a_kv_cache)
    pool_b_kv_u16 = kv_cache_flat_u16(pool_b_kv_cache)

    # Centroids per pool (may differ).
    cent_a = pool_a_centroids.to(torch.float32).contiguous()
    cent_b = pool_b_centroids.to(torch.float32).contiguous()

    # ---- Stride pre-fetch ----
    stride_cache_block_a = pool_a_kv_cache.stride(0)
    stride_cache_block_b = pool_b_kv_cache.stride(0)
    block_table_stride_a = pool_a_block_table.stride(0)
    block_table_stride_b = pool_b_block_table.stride(0)

    # ---- Use BF16 dot only when query is bf16 AND on ROCm. Matches the
    #      single-pool path's guard exactly. ----
    use_bf16_dot = 1 if (_is_hip and query.dtype == torch.bfloat16) else 0
    num_stages = int(
        os.environ.get("VLLM_TQ_NUM_STAGES_3D", "1" if _is_hip else "2")
    )

    grid = (total_num_q_blocks, Hk)

    kernel_tq_two_pool_fused_attention_2d[grid](
        output,
        q_rot,
        # Pool A
        pool_a_kv_cache,
        pool_a_kv_u16,
        cent_a,
        pair_lut_a,
        pool_a_block_table,
        pool_a_seq_lens,
        stride_cache_block_a,
        block_table_stride_a,
        # Pool B
        pool_b_kv_cache,
        pool_b_kv_u16,
        cent_b,
        pair_lut_b,
        pool_b_block_table,
        pool_b_seq_lens,
        stride_cache_block_b,
        block_table_stride_b,
        # Shared
        PiT_f32,
        query_start_loc,
        sinks_f32,
        float(scale),
        q_rot.stride(0),
        q_rot.stride(1),
        output.stride(0),
        output.stride(1),
        pit_stride_0,
        pit_stride_1,
        num_query_heads=Hq,
        num_queries_per_kv=kv_group_size,
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=D,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        NUM_KV_HEADS=Hk,
        num_seqs=num_seqs,
        # Pool A codec
        MSE_BITS_A=pool_a_mse_bits,
        MSE_BYTES_A=lay_a["mse_bytes"],
        VQB_A=pool_a_value_quant_bits,
        VAL_DATA_BYTES_A=lay_a["val_data_bytes"],
        N_CENTROIDS_A=int(cent_a.numel()),
        KEY_FP8_A=1 if pool_a_key_fp8 else 0,
        USE_PAIR_LUT_A=1 if use_pair_lut_a else 0,
        KEY_DATA_BYTES_A=lay_a["key_data_bytes"],
        META_REGION_OFFSET_A=lay_a["meta_region_offset"],
        NUM_SOA_FIELDS_A=lay_a["num_soa_fields"],
        SOA_K_NORM_A=lay_a["soa_k_norm"],
        SOA_V_SCALE_A=lay_a["soa_v_scale"],
        SOA_V_ZERO_A=lay_a["soa_v_zero"],
        NORM_CORRECTION_A=1 if pool_a_norm_correction else 0,
        FP8_E4B15_A=fp8_e4b15,
        # Pool B codec
        MSE_BITS_B=pool_b_mse_bits,
        MSE_BYTES_B=lay_b["mse_bytes"],
        VQB_B=pool_b_value_quant_bits,
        VAL_DATA_BYTES_B=lay_b["val_data_bytes"],
        N_CENTROIDS_B=int(cent_b.numel()),
        KEY_FP8_B=1 if pool_b_key_fp8 else 0,
        USE_PAIR_LUT_B=1 if use_pair_lut_b else 0,
        KEY_DATA_BYTES_B=lay_b["key_data_bytes"],
        META_REGION_OFFSET_B=lay_b["meta_region_offset"],
        NUM_SOA_FIELDS_B=lay_b["num_soa_fields"],
        SOA_K_NORM_B=lay_b["soa_k_norm"],
        SOA_V_SCALE_B=lay_b["soa_v_scale"],
        SOA_V_ZERO_B=lay_b["soa_v_zero"],
        NORM_CORRECTION_B=1 if pool_b_norm_correction else 0,
        FP8_E4B15_B=fp8_e4b15,
        # Shared flags
        FUSE_Q_ROT=1 if apply_fuse_q_rot else 0,
        USE_SINKS=1 if use_sinks else 0,
        USE_BF16_DOT=use_bf16_dot,
        num_warps=4,
        num_stages=num_stages,
    )
    return output
