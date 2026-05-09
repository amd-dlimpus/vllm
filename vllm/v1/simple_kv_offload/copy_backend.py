# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers.

Two execution modes:

- NVIDIA fast path: ``cuMemcpyBatchAsync`` is invoked from a background
  thread to hide ~5ms submission overhead behind GPU compute. The raw
  driver call accepts an explicit stream handle and does not depend on
  the calling thread's CUDA stream context.

- ROCm portable path: ``ops.swap_blocks_batch`` is invoked synchronously
  on the *caller's* thread (the same thread that created the streams).
  HIP's per-thread stream context is fragile across threads, so the
  background-thread design used by the NVIDIA fast path crashes the
  worker process under sustained load. Doing the submission inline is
  ~5ms slower per launch but stable. The actual copy still runs
  asynchronously on the GPU stream.
"""

from __future__ import annotations

import queue
import threading

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload import cuda_mem_ops
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


class DmaCopyBackend:
    """cuMemcpyBatchAsync (NVIDIA) or swap_blocks_batch (portable) backend."""

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._inline: bool = False

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream

        self._store_params = build_params(gpu_caches, cpu_caches, store_stream)
        self._load_params = build_params(cpu_caches, gpu_caches, load_stream)

        self._inline = cuda_mem_ops.using_portable_backend()
        if self._inline:
            logger.info(
                "DmaCopyBackend: inline (caller-thread) submission mode "
                "(portable swap_blocks_batch backend)."
            )
            return

        # NVIDIA fast path: background thread for submission overlap.
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None
        if self._inline:
            # Submit on the caller's thread. The streams were created on
            # this thread and the HIP per-thread context is the one
            # ``ops.swap_blocks_batch`` will see.
            stream = self._store_stream if is_store else self._load_stream
            assert stream is not None
            copy_blocks(src_blocks, dst_blocks, params)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))
            return

        assert self._queue is not None
        self._queue.put(
            (src_blocks, dst_blocks, params, is_store, event_idx, events_list)
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            src_blocks, dst_blocks, params, is_store, event_idx, events_list = item
            copy_blocks(src_blocks, dst_blocks, params)
            stream = store_stream if is_store else load_stream
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))
