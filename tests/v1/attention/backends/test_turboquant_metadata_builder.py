# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1E (prefix-tier) metadata-builder wiring tests.

Verifies that ``TurboQuantMetadataBuilder.build()`` correctly populates
the two-pool fields on ``TurboQuantMetadata`` when:
  - ``VLLM_TQ_PREFIX_TIER=1`` is set, AND
  - ``CommonAttentionMetadata.prefix_tier_split_tokens_cpu`` is non-None
    with at least one tagged request (split != -1).

When the env flag is off OR the split-tokens field is None, the builder
must produce legacy single-pool metadata (``is_two_pool()`` is False).

These tests are CPU-only — they construct a synthetic
``CommonAttentionMetadata`` directly and exercise the builder logic.
They do NOT touch the runner or the GPU kernel.

Run:
  PYTHONPATH=. python -m pytest \\
      tests/v1/attention/backends/test_turboquant_metadata_builder.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


# Builder construction ------------------------------------------------------


def _make_builder():
    """Build a TurboQuantMetadataBuilder with the minimum mocked deps."""
    from vllm.v1.attention.backends.turboquant_attn import (
        TurboQuantMetadataBuilder,
    )

    # The base class stores kv_cache_spec/layer_names/vllm_config/device.
    # _init_reorder_batch_threshold reads vllm_config.parallel_config so
    # we have to provide that attribute.
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
    )
    kv_cache_spec = SimpleNamespace(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
    )
    builder = TurboQuantMetadataBuilder.__new__(TurboQuantMetadataBuilder)
    builder.kv_cache_spec = kv_cache_spec
    builder.layer_names = ["layer.0"]
    builder.vllm_config = vllm_config
    builder.device = torch.device("cpu")
    builder.reorder_batch_threshold = 1
    return builder


def _make_cam(
    *,
    seq_lens: list[int],
    block_tables: list[list[int]],
    split_tokens: list[int] | None,
) -> CommonAttentionMetadata:
    """Construct a synthetic CommonAttentionMetadata for one decode step.

    Decode-only batch: each request gets a single query token.
    """
    num_reqs = len(seq_lens)
    max_blocks = max((len(bt) for bt in block_tables), default=0)
    padded_bt = np.zeros((num_reqs, max_blocks), dtype=np.int32)
    for i, bt in enumerate(block_tables):
        padded_bt[i, : len(bt)] = bt

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32)

    cam = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens_t,
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.from_numpy(padded_bt),
        slot_mapping=torch.zeros(num_reqs, dtype=torch.int32),
        _seq_lens_cpu=seq_lens_t,
    )
    if split_tokens is not None:
        cam.prefix_tier_split_tokens_cpu = np.array(split_tokens, dtype=np.int64)
    return cam


# Flag off / no split tokens ------------------------------------------------


def test_flag_off_no_two_pool_fields(monkeypatch):
    """Env flag off → builder produces legacy single-pool metadata even
    if the runner happened to set prefix_tier_split_tokens_cpu (defensive)."""
    monkeypatch.delenv("VLLM_TQ_PREFIX_TIER", raising=False)
    builder = _make_builder()
    cam = _make_cam(
        seq_lens=[40],
        block_tables=[[10, 11, 12]],
        split_tokens=[20],  # would be a valid split, but flag is off
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=cam)
    assert md.is_two_pool() is False
    assert md.pool_a_block_table is None
    assert md.pool_b_block_table is None


def test_flag_on_but_split_tokens_none(monkeypatch):
    """Env flag on but runner didn't pass split_tokens → legacy path."""
    monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", "1")
    builder = _make_builder()
    cam = _make_cam(
        seq_lens=[40],
        block_tables=[[10, 11, 12]],
        split_tokens=None,
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=cam)
    assert md.is_two_pool() is False


def test_flag_on_but_all_untagged(monkeypatch):
    """Env flag on, split_tokens passed, but every entry is -1 (no
    request tagged) → legacy single-pool path. This is the safe
    behavior during the transition window before the runner fully
    populates split_tokens."""
    monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", "1")
    builder = _make_builder()
    cam = _make_cam(
        seq_lens=[40, 32],
        block_tables=[[10, 11, 12], [20, 21]],
        split_tokens=[-1, -1],
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=cam)
    assert md.is_two_pool() is False


# Flag on, tagged batch ----------------------------------------------------


def test_flag_on_one_tagged_request(monkeypatch):
    """Single tagged request with a clean (block-aligned) split."""
    monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", "1")
    builder = _make_builder()
    cam = _make_cam(
        seq_lens=[64],  # 4 blocks of 16
        block_tables=[[10, 11, 12, 13]],
        split_tokens=[32],  # split at end of block 1 (block-aligned)
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=cam)
    assert md.is_two_pool() is True

    # Pool A: blocks [10, 11], seq_len = 32
    # Pool B: blocks [12, 13], seq_len = 32
    pa_bt = md.pool_a_block_table[0, :2].tolist()
    pb_bt = md.pool_b_block_table[0, :2].tolist()
    assert pa_bt == [10, 11]
    assert pb_bt == [12, 13]
    assert md.pool_a_seq_lens[0].item() == 32
    assert md.pool_b_seq_lens[0].item() == 32
    assert md.max_pool_a_seq_len == 32
    assert md.max_pool_b_seq_len == 32


def test_flag_on_mid_block_split(monkeypatch):
    """Mid-block split → boundary block appears in BOTH pool tables.
    Kernel clips by per-pool seq_lens."""
    monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", "1")
    builder = _make_builder()
    cam = _make_cam(
        seq_lens=[60],  # 4 blocks (last is partial)
        block_tables=[[10, 11, 12, 13]],
        split_tokens=[20],  # mid block 1 (tokens [16..32))
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=cam)
    assert md.is_two_pool() is True
    # Pool A: blocks [10, 11], seq_len = 20 (covers [0..20))
    # Pool B: blocks [11, 12, 13], seq_len = 40 (covers [20..60); 11 shared)
    assert md.pool_a_block_table[0, :2].tolist() == [10, 11]
    assert md.pool_b_block_table[0, :3].tolist() == [11, 12, 13]
    assert md.pool_a_seq_lens[0].item() == 20
    assert md.pool_b_seq_lens[0].item() == 40


def test_flag_on_mixed_batch_tagged_and_untagged(monkeypatch):
    """Mixed batch: req 0 tagged, req 1 untagged. The untagged request
    must put all its tokens into pool B (uniform-codec equivalent for
    that row), while the tagged request splits normally."""
    monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", "1")
    builder = _make_builder()
    cam = _make_cam(
        seq_lens=[64, 48],
        block_tables=[[10, 11, 12, 13], [20, 21, 22]],
        split_tokens=[32, -1],
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=cam)
    assert md.is_two_pool() is True

    # Tagged row 0: split=32 → pool A [10,11], pool B [12,13]
    assert md.pool_a_block_table[0, :2].tolist() == [10, 11]
    assert md.pool_b_block_table[0, :2].tolist() == [12, 13]
    assert md.pool_a_seq_lens[0].item() == 32
    assert md.pool_b_seq_lens[0].item() == 32

    # Untagged row 1: all → pool B
    assert md.pool_a_block_table[1].sum().item() == 0  # pool A empty
    assert md.pool_a_seq_lens[1].item() == 0
    assert md.pool_b_block_table[1, :3].tolist() == [20, 21, 22]
    assert md.pool_b_seq_lens[1].item() == 48


def test_flag_on_all_pool_a(monkeypatch):
    """Edge: split_token >= seq_len → all tokens are pool A (the very
    first prefill step for a fresh-prompt request)."""
    monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", "1")
    builder = _make_builder()
    cam = _make_cam(
        seq_lens=[48],
        block_tables=[[10, 11, 12]],
        split_tokens=[48],
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=cam)
    assert md.is_two_pool() is True
    assert md.pool_a_seq_lens[0].item() == 48
    assert md.pool_b_seq_lens[0].item() == 0
    assert md.pool_a_block_table[0, :3].tolist() == [10, 11, 12]
    assert md.pool_b_block_table[0].sum().item() == 0


def test_flag_on_all_pool_b_split_zero(monkeypatch):
    """Edge: split_token == 0 → all tokens are pool B. This is the
    "tagged with empty prefix" case (rare but legal)."""
    monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", "1")
    builder = _make_builder()
    cam = _make_cam(
        seq_lens=[48],
        block_tables=[[10, 11, 12]],
        split_tokens=[0],
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=cam)
    # split=0 means "tagged, but no pool-A content" — by policy this is
    # still a two-pool batch (any_two_pool=True), even though pool A is
    # empty for this row. Kernel handles pool_a_seq_lens=0 by skipping
    # the pool A phase entirely.
    assert md.is_two_pool() is True
    assert md.pool_a_seq_lens[0].item() == 0
    assert md.pool_b_seq_lens[0].item() == 48


# Legacy fields unchanged --------------------------------------------------


def test_legacy_single_pool_fields_unchanged(monkeypatch):
    """Both env-flag-off and env-flag-on paths must produce identical
    legacy fields (seq_lens, block_table, etc.) — the only difference
    is whether the two-pool fields are populated."""
    monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", "1")
    builder = _make_builder()
    cam = _make_cam(
        seq_lens=[64],
        block_tables=[[10, 11, 12, 13]],
        split_tokens=[32],
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=cam)
    assert torch.equal(md.seq_lens, cam.seq_lens)
    assert torch.equal(md.block_table, cam.block_table_tensor)
    assert md.num_actual_tokens == 1
    assert md.max_seq_len == 64
