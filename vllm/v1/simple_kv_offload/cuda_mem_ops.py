# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level CUDA memory helpers: pinning and batch DMA transfers.

Two backends are supported:

  (a) NVIDIA fast path — ``cuMemcpyBatchAsync`` via ``cuda.bindings``.
      Lowest submission overhead; preferred when available.

  (b) Portable fallback — ``vllm._custom_ops.swap_blocks_batch``.
      Works on both CUDA and ROCm. Used automatically when
      ``cuda.bindings`` is not importable (e.g. ROCm builds).
"""

import ctypes
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor via cudaHostRegister.

    This bypasses PyTorch's CUDACachingHostAllocator which rounds
    every ``pin_memory=True`` allocation up to the next power of 2
    (e.g. 100 GB becomes 128 GB).
    """
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_uint),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


_BATCH_MEMCPY_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_uint,  # CUresult
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
)

# Resolved lazily on first use. ``False`` means "tried and failed —
# use the portable fallback path instead".
_batch_memcpy_fn: Any = None
_resolution_attempted: bool = False

# Maximum number of (src, dst, size) entries to submit in a single
# ``swap_blocks_batch`` call on the portable path. Tunable via env
# var; the default keeps the C++ submission loop under ~10 ms even
# on slow hosts. M2.5 (~90 layers) × 1024 blocks = ~92k entries,
# which crashes the worker watchdog when submitted in one call.
import os as _os  # noqa: E402

_PORTABLE_BATCH_CHUNK: int = int(_os.environ.get("VLLM_KV_OFFLOAD_CHUNK", "4096"))


def _resolve_batch_memcpy():
    """Resolve cuMemcpyBatchAsync via cuGetProcAddress (one-time).

    Returns the FFI-typed function pointer on NVIDIA. Returns ``None``
    on ROCm or any other build where ``cuda.bindings`` is unavailable
    or the symbol cannot be resolved — the caller will fall back to
    ``ops.swap_blocks_batch``.
    """
    try:
        from cuda.bindings import driver as drv  # type: ignore[import-not-found]
    except ImportError:
        logger.info(
            "cuda.bindings unavailable; SimpleCPUOffload will use the "
            "portable swap_blocks_batch backend (ROCm-compatible)."
        )
        return None

    try:
        err, ptr, _ = drv.cuGetProcAddress(b"cuMemcpyBatchAsync", 12080, 0)
    except Exception as exc:  # pragma: no cover  (defensive)
        logger.info(
            "cuGetProcAddress(cuMemcpyBatchAsync) raised %r; using "
            "portable swap_blocks_batch fallback.",
            exc,
        )
        return None
    if err != drv.CUresult.CUDA_SUCCESS:
        logger.info(
            "cuGetProcAddress(cuMemcpyBatchAsync) failed (%s); using "
            "portable swap_blocks_batch fallback.",
            err,
        )
        return None
    return _BATCH_MEMCPY_FUNC_TYPE(ptr)


def _ensure_resolved() -> None:
    global _batch_memcpy_fn, _resolution_attempted
    if _resolution_attempted:
        return
    _resolution_attempted = True
    _batch_memcpy_fn = _resolve_batch_memcpy()


def using_portable_backend() -> bool:
    """Return True if we'll use ops.swap_blocks_batch (ROCm path)."""
    _ensure_resolved()
    return _batch_memcpy_fn is None


class BatchMemcpyParams(NamedTuple):
    src_bases: np.ndarray  # [num_layers] uint64 — data_ptr per layer
    dst_bases: np.ndarray  # [num_layers] uint64
    bpb: np.ndarray  # [num_layers] uint64 — bytes per block
    num_layers: int
    attrs: _CUmemcpyAttributes
    attrs_idx: ctypes.c_size_t
    # NOTE: cuMemcpyBatchAsync_v2() removed fail_idx field, but we use
    # cuMemcpyBatchAsync() with fail_idx for backward compatibility
    fail_idx: ctypes.c_size_t
    stream_handle: int  # raw cudaStream_t / CUstream
    # Portable-fallback fields. Only consulted when the cuMemcpyBatchAsync
    # path is unavailable (ROCm). Kept off the hot path otherwise.
    stream: Any  # torch.cuda.Stream (also covers torch.hip stream wrapper)


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> BatchMemcpyParams:
    _ensure_resolved()

    assert list(src_caches.keys()) == list(dst_caches.keys())
    src_tensors = list(src_caches.values())
    dst_tensors = list(dst_caches.values())

    src_bases, dst_bases, bpb = [], [], []
    for s, d in zip(src_tensors, dst_tensors):
        s_bpb = s.stride(0) * s.element_size()
        assert s_bpb == d.stride(0) * d.element_size()
        src_bases.append(s.data_ptr())
        dst_bases.append(d.data_ptr())
        bpb.append(s_bpb)

    # Refer to https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6f1ff58e3065df3eb4b573dba77ad31f for details.  # noqa: E501
    attrs = _CUmemcpyAttributes(srcAccessOrder=3)  # ANY

    return BatchMemcpyParams(
        src_bases=np.array(src_bases, dtype=np.uint64),
        dst_bases=np.array(dst_bases, dtype=np.uint64),
        bpb=np.array(bpb, dtype=np.uint64),
        num_layers=len(src_tensors),
        attrs=attrs,
        attrs_idx=ctypes.c_size_t(0),
        fail_idx=ctypes.c_size_t(0),
        stream_handle=stream.cuda_stream,
        stream=stream,
    )


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
) -> None:
    """Copy blocks via cuMemcpyBatchAsync (NVIDIA) or, when that is
    unavailable (ROCm), via vllm._custom_ops.swap_blocks_batch."""
    n = len(src_block_ids)
    if n == 0:
        return

    src_ids = np.array(src_block_ids, dtype=np.uint64)
    dst_ids = np.array(dst_block_ids, dtype=np.uint64)

    src_all = (
        params.src_bases[:, None] + src_ids[None, :] * params.bpb[:, None]
    ).ravel()
    dst_all = (
        params.dst_bases[:, None] + dst_ids[None, :] * params.bpb[:, None]
    ).ravel()
    sz_all = np.repeat(params.bpb, n)

    if _batch_memcpy_fn is None:
        # Portable path. ``swap_blocks_batch`` on ROCm loops
        # ``cudaMemcpyAsync`` once per pair on the C++ side. With
        # M2.5-class models (~90 layers) and full-pool stores
        # (~1k blocks), a single call submits ~90k async copies and
        # the C++ loop runs long enough that vLLM's worker watchdog
        # declares the worker dead. Chunk to cap the per-call work
        # at a safe size; each chunk runs on the same stream and is
        # therefore ordered with the rest of the transfer.
        from vllm import _custom_ops as ops  # local import: avoids
        # paying the import cost on the NVIDIA fast path.

        src_arr = src_all.astype(np.int64, copy=False)
        dst_arr = dst_all.astype(np.int64, copy=False)
        sz_arr = sz_all.astype(np.int64, copy=False)
        total = src_arr.shape[0]

        chunk = _PORTABLE_BATCH_CHUNK
        with torch.cuda.stream(params.stream):
            for off in range(0, total, chunk):
                end = min(off + chunk, total)
                src_t = torch.from_numpy(src_arr[off:end])
                dst_t = torch.from_numpy(dst_arr[off:end])
                sz_t = torch.from_numpy(sz_arr[off:end])
                ops.swap_blocks_batch(src_t, dst_t, sz_t)
        return

    total = n * params.num_layers
    err = _batch_memcpy_fn(
        dst_all.ctypes.data,
        src_all.ctypes.data,
        sz_all.ctypes.data,
        total,
        ctypes.addressof(params.attrs),
        ctypes.byref(params.attrs_idx),
        1,
        ctypes.byref(params.fail_idx),
        params.stream_handle,
    )
    if err != 0:
        raise RuntimeError(
            f"cuMemcpyBatchAsync failed: err={err} failIdx={params.fail_idx.value}"
        )
