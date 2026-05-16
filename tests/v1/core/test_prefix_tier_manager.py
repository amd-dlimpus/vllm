# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1E (prefix-tier) manager-side wiring tests.

Verifies that ``KVCacheManager.allocate_slots`` correctly populates
``Request.prefix_tier_split_token`` when ``VLLM_TQ_PREFIX_TIER=1``, and
leaves it alone when the flag is off.

These tests live alongside ``test_prefix_caching.py`` and reuse its
``make_request`` / ``make_kv_cache_config`` helpers.

Run:
  PYTHONPATH=. python -m pytest tests/v1/core/test_prefix_tier_manager.py -v
"""

from __future__ import annotations

import pytest

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash

from .test_prefix_caching import make_kv_cache_config, make_request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _make_manager(
    monkeypatch,
    *,
    enabled: bool,
    block_size: int = 16,
    num_blocks: int = 32,
) -> KVCacheManager:
    if enabled:
        monkeypatch.setenv("VLLM_TQ_PREFIX_TIER", "1")
    else:
        monkeypatch.delenv("VLLM_TQ_PREFIX_TIER", raising=False)
    return KVCacheManager(
        kv_cache_config=make_kv_cache_config(block_size, num_blocks),
        max_model_len=2048,
        hash_block_size=block_size,
        enable_caching=True,
    )


def _make_req(rid: str, prompt_len: int, block_size: int = 16):
    return make_request(
        request_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        block_size=block_size,
        hash_fn=sha256,
    )


# Flag-off path: prefix_tier_split_token must stay None ------------------- #


def test_flag_off_split_token_remains_none(monkeypatch):
    """When the env flag is off, allocate_slots must NEVER touch
    request.prefix_tier_split_token. Legacy behavior is bitwise unchanged."""
    mgr = _make_manager(monkeypatch, enabled=False)
    req = _make_req("r0", prompt_len=40)
    assert req.prefix_tier_split_token is None

    blocks = mgr.allocate_slots(req, num_new_tokens=40)
    assert blocks is not None
    assert req.prefix_tier_split_token is None  # untouched


# Flag-on, no cache hit: split_token = num_prompt_tokens ------------------ #


def test_fresh_prompt_split_token_equals_prompt_length(monkeypatch):
    """Turn-1 / first-time fresh prompt with no cache hit:
    split_token = num_prompt_tokens. All prompt blocks → pool A;
    decode growth → pool B."""
    mgr = _make_manager(monkeypatch, enabled=True)
    req = _make_req("r_fresh", prompt_len=40)
    assert req.prefix_tier_split_token is None

    mgr.allocate_slots(req, num_new_tokens=40)

    assert req.prefix_tier_split_token == 40


# Flag-on, with cache hit: split_token = cache-hit length ----------------- #


def test_turn2_with_cache_hit_split_token_equals_cache_hit(monkeypatch):
    """Turn-2+ where the codebase prefix is already in the cache from a
    prior turn: split_token = number of cache-hit tokens. The inherited
    prefix blocks form pool A; the new dialogue delta forms pool B."""
    mgr = _make_manager(monkeypatch, enabled=True)
    block_size = 16

    # Turn 1 — primes the prefix cache with a 32-token codebase.
    req1 = _make_req("r_turn1", prompt_len=32, block_size=block_size)
    mgr.allocate_slots(req1, num_new_tokens=32)
    mgr.free(req1)  # release blocks back to the pool as cached

    # Turn 2 — same 32-token prefix + new 16-token user message.
    req2 = _make_req("r_turn2", prompt_len=48, block_size=block_size)
    cached_blocks, num_cache_hit_tokens = mgr.get_computed_blocks(req2)
    assert num_cache_hit_tokens > 0, "test prerequisite: turn-1 prefix hit"

    # Drop the last block from the hit if necessary so allocate_slots has
    # something to allocate (mgr requires num_new_tokens > 0).
    new_tokens = 48 - num_cache_hit_tokens
    mgr.allocate_slots(
        req2,
        num_new_tokens=new_tokens,
        num_new_computed_tokens=num_cache_hit_tokens,
        new_computed_blocks=cached_blocks,
    )

    assert req2.prefix_tier_split_token == num_cache_hit_tokens


# Flag-on: split_token is immutable once set ------------------------------ #


def test_split_token_set_once_does_not_change_on_decode_step(monkeypatch):
    """The split token is set on the FIRST allocate_slots call and must
    never change on subsequent (decode) calls — the prefix-tier policy
    is static. allocate_slots is called once per scheduler step."""
    mgr = _make_manager(monkeypatch, enabled=True)
    req = _make_req("r_static", prompt_len=40)

    mgr.allocate_slots(req, num_new_tokens=40)
    first_value = req.prefix_tier_split_token
    assert first_value == 40

    # Simulate a subsequent decode step: num_computed_tokens advances,
    # then allocate one new token.
    req.num_computed_tokens = 40
    mgr.allocate_slots(req, num_new_tokens=1)
    assert req.prefix_tier_split_token == first_value, (
        "split_token must be immutable after first allocation"
    )

    # And once more.
    req.num_computed_tokens = 41
    mgr.allocate_slots(req, num_new_tokens=1)
    assert req.prefix_tier_split_token == first_value
