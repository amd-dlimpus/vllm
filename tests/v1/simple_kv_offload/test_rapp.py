# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for RAPP (Reasoning-Aware Predictive Prefetch) on the
SimpleCPUOffloadScheduler.

Strategy: reuse the helper machinery in test_scheduler.py to build a
real scheduler + GPU block pool, drive it through a full
store-load-evict-prefetch cycle while VLLM_RAPP=1, and verify the
warm queue / prefetch emission / GPU prefix-cache promotion all
behave correctly.

These tests are CPU-only — they exercise scheduler bookkeeping and
do not invoke any actual H2D/D2H transfers (the worker is mocked).
"""

from __future__ import annotations

import os

import pytest

# Make sure RAPP is enabled before manager.py is imported the first
# time. The env var is read in __init__; setting it inside a test
# only affects schedulers created after the import is fresh, so we
# must reload the module if it was previously imported.
os.environ["VLLM_RAPP"] = "1"

from vllm.v1.outputs import KVConnectorOutput  # noqa: E402
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata  # noqa: E402

from .test_scheduler import (  # noqa: E402
    BLOCK_SIZE,
    _alloc_and_register,
    make_request,
    make_scheduler,
    make_scheduler_output,
    simulate_load_completion,
    simulate_store_completion,
)


def _force_rapp_on(sched) -> None:
    """Force RAPP on for tests that imported the manager before
    VLLM_RAPP was set in the environment.

    Test pools are tiny (16 GPU blocks), so we cap per-step prefetches
    and zero-out the floor so the gate doesn't reject every step."""
    sched._rapp_enabled = True
    sched._rapp_max_blocks_per_step = 4
    sched._rapp_min_gpu_free_blocks = 0
    sched._rapp_warm_queue_capacity = 8192


def _simulate_prefetch_completion(scheduler, event_idx: int) -> None:
    """Simulate worker reporting a prefetch (no-req) load event completion."""
    output = KVConnectorOutput(
        finished_sending=set(),
        finished_recving=set(),
        kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
            completed_store_events={},
            completed_load_events={event_idx: scheduler._expected_worker_count},
        ),
    )
    scheduler.update_connector_output(output)


def _evict_from_gpu_cache(gpu_block_pool, block) -> None:
    """Forcibly remove a GPU block from the prefix cache (simulates LRU
    eviction by another request) without actually returning it to the
    free pool. We only need its hash to disappear from
    cached_block_hash_to_block so RAPP sees it as 'CPU has, GPU doesn't'."""
    bhash = block.block_hash
    if bhash is None:
        return
    gpu_block_pool.cached_block_hash_to_block.pop(bhash, block.block_id)


# ─────────────── tests ───────────────


def _flat_block_ids(kv_blocks) -> list[int]:
    """Flatten KVCacheBlocks across groups into a single list of int IDs."""
    ids: list[int] = []
    for group in kv_blocks.blocks:
        ids.extend(b.block_id for b in group)
    return ids


def test_rapp_off_is_byte_identical_when_no_env_flag() -> None:
    """When VLLM_RAPP is not set, request_finished must not touch the
    warm queue and build_connector_meta must produce a metadata object
    with no extra prefetch load entries."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    sched._rapp_enabled = False  # explicitly off

    req = make_request(num_blocks=2)
    kv_blocks = _alloc_and_register(fix, req, num_blocks=2)
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    sched_out = make_scheduler_output(
        {req.request_id: 2 * BLOCK_SIZE},
        new_reqs={req.request_id: kv_blocks.get_block_ids()},
    )
    meta = sched.build_connector_meta(sched_out)
    simulate_store_completion(sched, meta.store_event)

    # Finish the request: warm queue should NOT receive entries.
    sched.request_finished(req, block_ids=_flat_block_ids(kv_blocks))
    tel = sched.rapp_telemetry()
    assert tel["warm_queue_len"] == 0
    assert tel["emitted_prefetches"] == 0


def test_rapp_warm_queue_populated_on_request_finish() -> None:
    """With RAPP on, finishing a request whose blocks have hashes adds
    those hashes to the warm queue."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    _force_rapp_on(sched)

    req = make_request(num_blocks=3)
    kv_blocks = _alloc_and_register(fix, req, num_blocks=3)
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)

    # Finish request — block hashes should land in warm queue.
    sched.request_finished(req, block_ids=_flat_block_ids(kv_blocks))
    tel = sched.rapp_telemetry()
    # The full prefix block hashes are pushed (each non-None hash).
    assert tel["warm_queue_len"] >= 3, (
        f"expected >=3 warm queue entries, got {tel['warm_queue_len']}"
    )


def test_rapp_emits_prefetch_when_demand_idle() -> None:
    """End-to-end: store a request's blocks to CPU, evict them from GPU
    prefix cache, finish the request, then run an idle scheduler step
    and verify a prefetch load event is emitted with the right block
    pairs.
    """
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    _force_rapp_on(sched)

    # Allocate + store + complete.
    req = make_request(num_blocks=3)
    kv_blocks = _alloc_and_register(fix, req, num_blocks=3)
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    block_ids = kv_blocks.get_block_ids()
    sched_out = make_scheduler_output(
        {req.request_id: 3 * BLOCK_SIZE},
        new_reqs={req.request_id: block_ids},
    )
    meta = sched.build_connector_meta(sched_out)
    assert meta.store_event >= 0
    n_stored = len(meta.store_gpu_blocks)
    simulate_store_completion(sched, meta.store_event)
    sched.request_finished(req, block_ids=_flat_block_ids(kv_blocks))

    # Now evict the request's blocks from GPU prefix cache (simulates
    # later requests pushing them out under pressure).
    for blk in kv_blocks.blocks[0]:
        _evict_from_gpu_cache(fix.gpu_block_pool, blk)

    # Verify pre-conditions: warm queue has entries, GPU cache misses,
    # CPU cache hits.
    tel = sched.rapp_telemetry()
    assert tel["warm_queue_len"] > 0, "warm queue should have entries"
    cpu_pool = sched.cpu_block_pool
    gpu_pool = fix.gpu_block_pool
    for bhash in list(sched._rapp_warm_queue.keys()):
        assert cpu_pool.cached_block_hash_to_block.get_one_block(bhash) is not None
        assert gpu_pool.cached_block_hash_to_block.get_one_block(bhash) is None

    # Idle scheduler step (no new requests).
    idle_out = make_scheduler_output({})
    meta2 = sched.build_connector_meta(idle_out)
    assert meta2.load_event >= 0, "expected a prefetch load event"
    # Prefetch event has empty req_id list — that's how the worker
    # decides to report it via completed_load_events.
    assert sched._load_event_to_reqs.get(meta2.load_event) == []
    assert len(meta2.load_gpu_blocks) == len(meta2.load_cpu_blocks) > 0
    assert len(meta2.load_gpu_blocks) <= n_stored

    tel2 = sched.rapp_telemetry()
    assert tel2["emitted_prefetches"] == len(meta2.load_gpu_blocks)
    assert tel2["pending_inserts"] == len(meta2.load_gpu_blocks)


def test_rapp_skipped_when_demand_load_present() -> None:
    """When a demand load is being emitted this step, RAPP must NOT
    emit a prefetch (it would compete with the request's own H2D)."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    _force_rapp_on(sched)

    # Stage 1: store + finish a request, populate CPU cache and warm queue.
    req1 = make_request(num_blocks=3)
    kv1 = _alloc_and_register(fix, req1, num_blocks=3)
    sched.update_state_after_alloc(req1, kv1, num_external_tokens=0)
    sched_out = make_scheduler_output(
        {req1.request_id: 3 * BLOCK_SIZE},
        new_reqs={req1.request_id: kv1.get_block_ids()},
    )
    meta = sched.build_connector_meta(sched_out)
    simulate_store_completion(sched, meta.store_event)
    sched.request_finished(req1, block_ids=_flat_block_ids(kv1))
    # Evict from GPU cache so prefetch candidates exist.
    for blk in kv1.blocks[0]:
        _evict_from_gpu_cache(fix.gpu_block_pool, blk)

    # Stage 2: a NEW request arrives. Use the same prompt token IDs as
    # req1 so its block hashes match (CPU cache hit).
    req2 = make_request(num_blocks=3)
    # Force same block hashes by sharing prompt_token_ids — easiest:
    # call get_num_new_matched_tokens directly on the original hashes.
    # For simplicity, just use req1's hashes by constructing a new
    # request with matching prompt tokens. We rely on the existing
    # block_hasher determinism in make_request — instead, exercise
    # only the ALREADY-warm-queue prefetch case for this test by
    # *manually* injecting a demand load entry to confirm the gate.
    sched._reqs_to_load["fake_req"] = type(  # minimal stand-in
        "X",
        (),
        {"load_event": None, "transfer_meta": type("Y", (), {
            "gpu_block_ids": [0],
            "cpu_block_ids": [0],
        })(), "request": req2, "finished": False},
    )()

    idle_out = make_scheduler_output({})
    meta2 = sched.build_connector_meta(idle_out)
    # The demand load should have been emitted; RAPP should have been
    # skipped this step.
    assert meta2.load_event >= 0
    # The load_event_to_reqs entry should have a real req_id (not the
    # empty list that signals prefetch).
    assert sched._load_event_to_reqs.get(meta2.load_event) == ["fake_req"]
    # Telemetry should record the skip.
    assert sched.rapp_telemetry()["skipped_demand_busy"] >= 1
    assert sched.rapp_telemetry()["emitted_prefetches"] == 0


def test_rapp_completion_promotes_blocks_to_gpu_prefix_cache() -> None:
    """After the prefetch load event completes, the freshly-loaded
    blocks must appear in the GPU prefix cache (discoverable for
    future demand requests)."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    _force_rapp_on(sched)

    # Setup: store, finish, evict from GPU cache.
    req = make_request(num_blocks=3)
    kv = _alloc_and_register(fix, req, num_blocks=3)
    sched.update_state_after_alloc(req, kv, num_external_tokens=0)
    sched_out = make_scheduler_output(
        {req.request_id: 3 * BLOCK_SIZE},
        new_reqs={req.request_id: kv.get_block_ids()},
    )
    meta = sched.build_connector_meta(sched_out)
    simulate_store_completion(sched, meta.store_event)
    sched.request_finished(req, block_ids=_flat_block_ids(kv))
    expected_hashes = list(sched._rapp_warm_queue.keys())
    for blk in kv.blocks[0]:
        _evict_from_gpu_cache(fix.gpu_block_pool, blk)

    # Idle step → prefetch emitted
    idle_out = make_scheduler_output({})
    meta2 = sched.build_connector_meta(idle_out)
    assert meta2.load_event >= 0
    n_emitted = len(meta2.load_gpu_blocks)
    assert n_emitted > 0

    # Pre-condition: hashes are NOT in GPU prefix cache yet.
    gpu_pool = fix.gpu_block_pool
    for bhash in expected_hashes[:n_emitted]:
        assert gpu_pool.cached_block_hash_to_block.get_one_block(bhash) is None

    # Worker reports prefetch completion.
    _simulate_prefetch_completion(sched, meta2.load_event)

    # Now the hashes MUST be in the GPU prefix cache.
    promoted = 0
    for bhash in expected_hashes:
        if gpu_pool.cached_block_hash_to_block.get_one_block(bhash) is not None:
            promoted += 1
    assert promoted >= n_emitted, (
        f"expected >={n_emitted} promoted blocks, got {promoted}"
    )

    tel = sched.rapp_telemetry()
    assert tel["completed_prefetches"] == n_emitted
    assert tel["pending_inserts"] == 0


def test_rapp_skips_already_resident_hashes() -> None:
    """If a hash in the warm queue is ALREADY in the GPU prefix cache
    (e.g. another request just pulled it back), RAPP should drop the
    entry from the warm queue and increment the skipped_already_resident
    counter, without issuing a redundant H2D."""
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler
    _force_rapp_on(sched)

    req = make_request(num_blocks=3)
    kv = _alloc_and_register(fix, req, num_blocks=3)
    sched.update_state_after_alloc(req, kv, num_external_tokens=0)
    sched_out = make_scheduler_output(
        {req.request_id: 3 * BLOCK_SIZE},
        new_reqs={req.request_id: kv.get_block_ids()},
    )
    meta = sched.build_connector_meta(sched_out)
    simulate_store_completion(sched, meta.store_event)
    sched.request_finished(req, block_ids=_flat_block_ids(kv))

    # Don't evict — the blocks remain in the GPU prefix cache. RAPP
    # should detect this and drop entries from the warm queue.
    pre_skipped = sched.rapp_telemetry()["skipped_already_resident"]

    idle_out = make_scheduler_output({})
    meta2 = sched.build_connector_meta(idle_out)
    # No prefetch should be emitted (everything already on GPU).
    assert len(meta2.load_gpu_blocks) == 0

    post_skipped = sched.rapp_telemetry()["skipped_already_resident"]
    assert post_skipped > pre_skipped, (
        "expected skipped_already_resident counter to advance"
    )
    # Warm queue should be drained.
    assert sched.rapp_telemetry()["warm_queue_len"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
