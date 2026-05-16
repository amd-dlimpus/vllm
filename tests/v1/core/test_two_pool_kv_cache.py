# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Phase 1E two-pool KV cache scaffolding.

These tests cover the scheduler-side data structures and policy. They do
NOT exercise the kernel or runner integration (those land in subsequent
todos).

Run:
  .venv/bin/python -m pytest tests/v1/core/test_two_pool_kv_cache.py -v
"""

import os

import pytest

from vllm.v1.core.two_pool_kv_cache import (
    PoolID,
    PooledBlockDescriptor,
    TwoPoolBlockTableBuilder,
    TwoPoolDtypeConfig,
    is_prefix_tier_enabled,
    maybe_tag_block_pool,
    partition_block_table_by_split_token,
    select_pool,
)


# Env flag ------------------------------------------------------------------


def test_env_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_TQ_PREFIX_TIER", raising=False)
    assert is_prefix_tier_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_env_flag_truthy_values(monkeypatch, val):
    monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", val)
    assert is_prefix_tier_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "garbage"])
def test_env_flag_falsy_values(monkeypatch, val):
    monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", val)
    assert is_prefix_tier_enabled() is False


# Allocation policy ---------------------------------------------------------


def test_select_pool_first_chunk_no_cache_hits_prefix_pool():
    assert (
        select_pool(is_first_chunk_prefill=True, cached_len=0)
        is PoolID.PREFIX
    )


def test_select_pool_continuation_prefill_goes_to_dialogue():
    assert (
        select_pool(is_first_chunk_prefill=False, cached_len=0)
        is PoolID.DIALOGUE
    )


def test_select_pool_first_chunk_with_cache_hit_goes_to_dialogue():
    """If we hit the prefix cache, the cached blocks already carry their
    pool_id from the original request that built them. Newly-allocated
    blocks here are for the un-cached *suffix*, which is dialogue."""
    assert (
        select_pool(is_first_chunk_prefill=True, cached_len=4096)
        is PoolID.DIALOGUE
    )


def test_select_pool_decode_goes_to_dialogue():
    """Decode steps allocate one block at a time as the sequence grows."""
    assert (
        select_pool(is_first_chunk_prefill=False, cached_len=8192)
        is PoolID.DIALOGUE
    )


# Block table builder -------------------------------------------------------


def test_block_table_builder_basic():
    """Pool A blocks (prefix) come first, then Pool B (dialogue)."""
    builder = TwoPoolBlockTableBuilder(block_size=16)
    builder.append(PooledBlockDescriptor(block_id=10, pool_id=PoolID.PREFIX))
    builder.append(PooledBlockDescriptor(block_id=11, pool_id=PoolID.PREFIX))
    builder.append(PooledBlockDescriptor(block_id=20, pool_id=PoolID.DIALOGUE))
    builder.append(PooledBlockDescriptor(block_id=21, pool_id=PoolID.DIALOGUE))
    table = builder.build()
    assert table.pool_a_block_ids == [10, 11]
    assert table.pool_b_block_ids == [20, 21]
    assert table.split_token_index == 32  # 2 * 16


def test_block_table_builder_rejects_pool_a_after_pool_b():
    """The plan's structural invariant: pool A must form a contiguous
    prefix. Interleaving is illegal because the v3_split kernel iterates
    pool A first then pool B as homogeneous phases."""
    builder = TwoPoolBlockTableBuilder(block_size=16)
    builder.append(PooledBlockDescriptor(block_id=10, pool_id=PoolID.PREFIX))
    builder.append(PooledBlockDescriptor(block_id=20, pool_id=PoolID.DIALOGUE))
    with pytest.raises(ValueError, match="contiguous prefix"):
        builder.append(
            PooledBlockDescriptor(block_id=11, pool_id=PoolID.PREFIX)
        )


def test_block_table_builder_all_pool_a():
    builder = TwoPoolBlockTableBuilder(block_size=8)
    for bid in range(5):
        builder.append(
            PooledBlockDescriptor(block_id=bid, pool_id=PoolID.PREFIX)
        )
    table = builder.build()
    assert table.pool_a_block_ids == list(range(5))
    assert table.pool_b_block_ids == []
    assert table.split_token_index == 40


def test_block_table_builder_all_pool_b():
    builder = TwoPoolBlockTableBuilder(block_size=16)
    for bid in range(3):
        builder.append(
            PooledBlockDescriptor(block_id=bid, pool_id=PoolID.DIALOGUE)
        )
    table = builder.build()
    assert table.pool_a_block_ids == []
    assert table.pool_b_block_ids == list(range(3))
    assert table.split_token_index == 0


def test_block_table_builder_rejects_invalid_block_size():
    with pytest.raises(ValueError):
        TwoPoolBlockTableBuilder(block_size=0)
    with pytest.raises(ValueError):
        TwoPoolBlockTableBuilder(block_size=-1)


# Dtype config --------------------------------------------------------------


def test_dtype_config_defaults():
    cfg = TwoPoolDtypeConfig()
    assert cfg.dtype_for(PoolID.PREFIX) == "turboquant_k8v4_nc"
    assert cfg.dtype_for(PoolID.DIALOGUE) == "turboquant_4bit_nc"


def test_dtype_config_custom():
    cfg = TwoPoolDtypeConfig(
        pool_a_dtype="turboquant_k8v4",
        pool_b_dtype="turboquant_k3v4_nc",
    )
    assert cfg.dtype_for(PoolID.PREFIX) == "turboquant_k8v4"
    assert cfg.dtype_for(PoolID.DIALOGUE) == "turboquant_k3v4_nc"


# Tagging integration helper ------------------------------------------------


def test_maybe_tag_block_pool_first_chunk():
    desc = maybe_tag_block_pool(
        block_id=42, is_first_chunk_prefill=True, cached_len=0
    )
    assert desc.block_id == 42
    assert desc.pool_id is PoolID.PREFIX


def test_maybe_tag_block_pool_decode():
    desc = maybe_tag_block_pool(
        block_id=99, is_first_chunk_prefill=False, cached_len=8192
    )
    assert desc.block_id == 99
    assert desc.pool_id is PoolID.DIALOGUE


# Block-table partitioning --------------------------------------------------


def test_partition_split_zero_all_pool_b():
    """split_token=0 → no prefix tagged → all tokens go to pool B
    (this is the default for a request without a first-allocation tag)."""
    a, b, la, lb = partition_block_table_by_split_token(
        block_ids=[10, 11, 12], seq_len=40, split_token=0, block_size=16,
    )
    assert a == []
    assert b == [10, 11, 12]
    assert la == 0
    assert lb == 40


def test_partition_split_ge_seq_len_all_pool_a():
    """split_token >= seq_len (i.e. all tokens are part of the first
    prefill chunk) → all blocks go to pool A."""
    a, b, la, lb = partition_block_table_by_split_token(
        block_ids=[10, 11, 12], seq_len=40, split_token=40, block_size=16,
    )
    assert a == [10, 11, 12]
    assert b == []
    assert la == 40
    assert lb == 0


def test_partition_split_aligned_clean_cut():
    """split_token block-aligned → clean cut, no shared block."""
    a, b, la, lb = partition_block_table_by_split_token(
        block_ids=[10, 11, 12, 13], seq_len=64, split_token=32, block_size=16,
    )
    assert a == [10, 11]
    assert b == [12, 13]
    assert la == 32
    assert lb == 32


def test_partition_split_mid_block_shared_boundary():
    """split_token lands mid-block → boundary block appears in BOTH lists.
    The v3_split kernel clips each pool by its seq_len, so this is the
    correct representation."""
    a, b, la, lb = partition_block_table_by_split_token(
        block_ids=[10, 11, 12, 13], seq_len=60, split_token=20, block_size=16,
    )
    # split_token=20 lands in block index 1 (block 11), tokens [16..32).
    # Pool A keeps blocks 10, 11 (covering [0..32) but clipped to 20).
    # Pool B starts at block 11 (the boundary) and continues 12, 13.
    assert a == [10, 11]
    assert b == [11, 12, 13]
    assert la == 20
    assert lb == 40


def test_partition_split_token_clamped_to_seq_len():
    """split_token > seq_len gets clamped to seq_len (defensive)."""
    a, b, la, lb = partition_block_table_by_split_token(
        block_ids=[10, 11], seq_len=24, split_token=999, block_size=16,
    )
    assert a == [10, 11]
    assert b == []
    assert la == 24
    assert lb == 0


def test_partition_ignores_block_table_padding():
    """The runner allocates block-table tensors at a max width; the unused
    tail is zero-padded. partition() must not include those padded zeros."""
    a, b, la, lb = partition_block_table_by_split_token(
        # 3 real blocks; tail is padding (would be 0s from torch.zeros).
        block_ids=[10, 11, 12, 0, 0, 0, 0],
        seq_len=40, split_token=0, block_size=16,
    )
    assert a == []
    assert b == [10, 11, 12]  # padding zeros must not appear
    assert la == 0
    assert lb == 40


def test_partition_rejects_insufficient_blocks():
    with pytest.raises(ValueError, match="insufficient"):
        partition_block_table_by_split_token(
            block_ids=[10], seq_len=40, split_token=0, block_size=16,
        )


def test_partition_rejects_negative_inputs():
    with pytest.raises(ValueError):
        partition_block_table_by_split_token(
            block_ids=[10], seq_len=-1, split_token=0, block_size=16,
        )
    with pytest.raises(ValueError):
        partition_block_table_by_split_token(
            block_ids=[10], seq_len=16, split_token=-1, block_size=16,
        )
    with pytest.raises(ValueError):
        partition_block_table_by_split_token(
            block_ids=[10], seq_len=16, split_token=0, block_size=0,
        )


def test_partition_invariant_seq_len_sum():
    """Property test: pool_a_seq_len + pool_b_seq_len == seq_len, for
    every split point including the edges."""
    block_size = 16
    seq_len = 128
    block_ids = list(range(100, 100 + (seq_len // block_size)))
    for split_token in range(0, seq_len + 1):
        _, _, la, lb = partition_block_table_by_split_token(
            block_ids=block_ids, seq_len=seq_len,
            split_token=split_token, block_size=block_size,
        )
        assert la + lb == seq_len, f"split_token={split_token}: {la}+{lb}!={seq_len}"
