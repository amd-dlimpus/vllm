# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for SimpleCPUOffloadConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

INVALID_JOB_ID = -1


@dataclass
class SimpleCPUOffloadMetadata(KVConnectorMetadata):
    """
    Metadata passed from scheduler to worker for CPU offload operations.

    The worker receives flat block lists keyed by a monotonic event_idx.
    Job->req_id translation is handled by the scheduler-side manager
    (via inverse maps), so the worker never knows about request identities.
    """

    # Load event per step. INVALID_JOB_ID means no blocks to load this step.
    load_event: int = INVALID_JOB_ID
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_cpu_blocks: list[int] = field(default_factory=list)
    # Reverse map: load_event->req_ids, for tracking requests with finished load events
    load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)

    # Store event per step. INVALID_JOB_ID means no blocks to store this step.
    store_event: int = INVALID_JOB_ID
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_cpu_blocks: list[int] = field(default_factory=list)

    # Whether any requests were preempted this step and need flush pending transfers.
    need_flush: bool = False


@dataclass
class SimpleCPUOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store events, and (when
    RAPP/predictive prefetch is active) for completed load events that do
    not correspond to any request.

    Each worker reports {event_idx: 1} for newly completed stores.
    ``aggregate()`` sums counts across workers within a step.
    The scheduler-side manager accumulates across steps and processes
    a store completion only when count reaches ``world_size``.

    ``completed_load_events`` follows the same convention but is only
    populated for load events whose ``load_event_to_reqs`` mapping is
    empty — i.e. RAPP prefetches, where there is no waiting request to
    notify via ``finished_recving``. Demand loads (with req_ids) continue
    to report via the existing finished_recving channel.
    """

    completed_store_events: dict[int, int]
    completed_load_events: dict[int, int] = field(default_factory=dict)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, SimpleCPUOffloadWorkerMetadata)
        merged_stores = dict(self.completed_store_events)
        for k, v in other.completed_store_events.items():
            merged_stores[k] = merged_stores.get(k, 0) + v
        merged_loads = dict(self.completed_load_events)
        for k, v in other.completed_load_events.items():
            merged_loads[k] = merged_loads.get(k, 0) + v
        return SimpleCPUOffloadWorkerMetadata(
            completed_store_events=merged_stores,
            completed_load_events=merged_loads,
        )
