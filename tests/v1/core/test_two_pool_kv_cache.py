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
