# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for vLLM.

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K and store K+V into combined cache slot.
Decode:  Compute TQ attention scores from compressed cache,
         unpack FP16 values, softmax + weighted sum.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""

import functools
import math
import os
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_decode_v2 import (
    triton_turboquant_decode_attention_v2,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
    triton_turboquant_decode_attention_v3,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

# Opt-in flag to dispatch decode path to the v2 Triton kernel.
# v1 remains the default. Set VLLM_TQ_DECODE_V2=1 to enable v2.
# Set VLLM_TQ_DECODE_V3=1 to enable v3 (unified prefill+decode kernel with
# 2D/3D split-KV dispatch and BLOCK_M=128 prefill heuristic). v3 supersedes
# v2 when enabled.
_USE_TQ_V2 = os.environ.get("VLLM_TQ_DECODE_V2", "0") == "1"
_USE_TQ_V3 = os.environ.get("VLLM_TQ_DECODE_V3", "0") == "1"

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

logger.info_once(
    "TurboQuant has flash attn: %s, decode kernel: %s",
    _HAS_FLASH_ATTN,
    "v3" if _USE_TQ_V3 else "v2" if _USE_TQ_V2 else "v1",
)
# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
_CONTINUATION_DECODE_THRESHOLD = 128


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), cached per (d, device).

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    # Normalize device string so "cuda" and "cuda:0" hit the same cache entry.
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
        "turboquant_k4v2_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Standard attention backends use (2, num_blocks, block_size, num_kv_heads,
        head_dim) with a leading 2 to separate K and V. TurboQuant packs K+V
        into a single interleaved slot per head per position, so the cache is:

            (num_blocks, block_size, num_kv_heads, slot_size_aligned)

        Each slot = [key_packed | value_packed | padding].
        This is safe because TQ has its own get_kv_cache_shape override and
        never shares cache tensors with other backends. Layers that fall back
        to native dtype via kv_cache_dtype_skip_layers get their own
        standard-shaped cache allocation.

        head_size is the model's real head_dim. slot_size_aligned is computed
        from the TQ config to ensure correct cache allocation for all head dims.
        """
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("turboquant_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0

    @classmethod
    def supports_sink(cls) -> bool:
        """Return True to indicate TurboQuant supports sink tokens.

        Sink tokens provide stable attention anchors at the start of context.
        The TQ decode kernel initializes online softmax with pre-computed
        sink attention logits for proper attention distribution.
        """
        return True


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    seq_lens: torch.Tensor  # (num_reqs,) — total context length per request
    slot_mapping: torch.Tensor  # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — cu_seqlens for queries
    num_actual_tokens: int = 0  # actual tokens (excluding padding)
    max_query_len: int = 0  # longest query in batch
    max_seq_len: int = 0  # longest context in batch
    is_prefill: bool = False
    num_decodes: int = 0  # number of decode requests (first in batch)
    num_decode_tokens: int = 0  # tokens from decode requests

    # Phase 1E (prefix-tier mixed-precision) two-pool fields. All None
    # means single-pool behavior (the legacy path). When all are
    # populated, `forward()` dispatches to the v3_split kernel (decode)
    # and `_continuation_prefill_split` (prefill chunks 2+).
    #
    # Populated by the runner when ``VLLM_TQ_PREFIX_TIER=1`` is active
    # AND the request has a non-empty pool-A prefix; the metadata builder
    # signals this via these fields rather than a separate flag so the
    # forward dispatch can branch on data presence (cleaner CUDA-graph
    # specialization).
    pool_a_block_table: torch.Tensor | None = None
    pool_b_block_table: torch.Tensor | None = None
    pool_a_seq_lens: torch.Tensor | None = None  # tokens served from pool A per req
    pool_b_seq_lens: torch.Tensor | None = None  # tokens served from pool B per req
    # Optional pool-A KV cache reference. When None and two_pool fields
    # above are set, ``forward()`` re-uses the same ``kv_cache`` for both
    # pools (i.e. same-codec test/dev mode); production wiring populates
    # this with the TQ84 cache buffer.
    pool_a_kv_cache: torch.Tensor | None = None
    pool_b_kv_cache: torch.Tensor | None = None
    # Per-pool max seq len for the kernel dispatch heuristic. Inferred from
    # block-table shape if None.
    max_pool_a_seq_len: int = 0
    max_pool_b_seq_len: int = 0

    def is_two_pool(self) -> bool:
        """True iff the two-pool dispatch path should be taken."""
        return (
            self.pool_a_block_table is not None
            and self.pool_b_block_table is not None
            and self.pool_a_seq_lens is not None
            and self.pool_b_seq_lens is not None
        )


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # Set seq_lens to 1 so CUDA graph capture is fast
        # (real seq_lens are filled at replay time).
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build TurboQuantMetadata from common attention metadata."""
        cam = common_attn_metadata

        # With reorder_batch_threshold=1, the model runner guarantees
        # decodes come first in the batch. split_decodes_and_prefills
        # finds the boundary (operates on CPU tensors — no GPU sync).
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )


class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
    """TurboQuant attention implementation.

    Vectorized PyTorch: batch quantize/store, vectorized bit-unpack
    decode with einsum scores and value gather.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Pre-compute kernel constants from config (avoid repeated arithmetic)
        cfg = self.tq_config
        self._mse_bytes = (
            math.ceil(head_size * cfg.key_mse_bits / 8)
            if not cfg.key_fp8
            else head_size
        )
        self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
        self._n_centroids = cfg.n_centroids if not cfg.key_fp8 else 1

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

        # Sink tokens support: store reference from kwargs if provided.
        # Sinks are pre-computed attention logits [Hq] that anchor attention
        # to sink tokens at the start of context (used by GPT-OSS and similar).
        # Note: sinks is passed directly via **extra_impl_args spread, not nested.
        self.sinks = kwargs.get("sinks")

    def _ensure_on_device(self, layer, device, q_dtype: torch.dtype | None = None):
        """One-time derivation of TQ buffers (rotation matrix, midpoints).

        The Hadamard rotation is shared across all layers: random sign
        flips do not improve Lloyd-Max quantization quality because the
        quantizer is symmetric around zero (sign-flipping a coordinate
        maps it to the mirror centroid with identical distortion).
        """
        if not hasattr(layer, "_tq_cached"):
            D = self.head_size

            # Pure Hadamard: orthonormal + symmetric (H = H^T), enabling
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            # fp16 copy for rotation in continuation prefill path
            layer._tq_Pi_half = H.to(torch.float16)

            # k4v2 (and any future ≤2-bit value preset): rotate values before
            # uniform quantization to spread quantization error across
            # coordinates. Reuses the same Hadamard since H = H^T = H^{-1};
            # caller applies the inverse on the attention output.
            #
            # Cache an extra runtime-dtype copy (`_tq_VRot_q`) so the
            # post-attention inverse GEMM avoids mixed-precision matmul
            # when the model runs in bf16. Defaulted to fp16 if q_dtype
            # isn't known yet — refreshed lazily on the first forward.
            if self.tq_config.value_quant_bits == 2:
                layer._tq_VRot = H
                layer._tq_VRot_half = layer._tq_Pi_half
                _qd = q_dtype if q_dtype is not None else torch.float16
                layer._tq_VRot_q = H.to(_qd)
                layer._tq_VRot_q_dtype = _qd
            else:
                layer._tq_VRot = None
                layer._tq_VRot_half = None
                layer._tq_VRot_q = None
                layer._tq_VRot_q_dtype = None

            # Centroids for Lloyd-Max quantization.
            layer._tq_centroids = get_centroids(D, self.tq_config.centroid_bits).to(
                device=device, dtype=torch.float32
            )

            c_sorted, _ = layer._tq_centroids.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache.

        Called as a separate custom op (unified_kv_cache_update) BEFORE
        the attention forward, matching FlashAttention's split pattern.
        slot_mapping is already sliced to num_actual_tokens by the caller.
        """
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._store_kv(k, v, kv_cache, slot_mapping, layer)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        # Get TQ buffers, ensure on device (one-time migration).
        # Use Any-typed alias for dynamic _tq_* attrs set by _ensure_on_device.
        tq_layer: Any = layer
        device = q.device
        self._ensure_on_device(tq_layer, device, q.dtype)
        # Lazy refresh: if model dtype differs from the cached one (e.g. first
        # call was during _store_kv with key.dtype, now we have q.dtype),
        # rebuild the runtime-dtype rotation. One-time per layer.
        if (
            tq_layer._tq_VRot is not None
            and tq_layer._tq_VRot_q_dtype != q.dtype
        ):
            tq_layer._tq_VRot_q = tq_layer._tq_VRot.to(q.dtype)
            tq_layer._tq_VRot_q_dtype = q.dtype
        Pi = tq_layer._tq_Pi
        PiT = tq_layer._tq_PiT
        centroids = tq_layer._tq_centroids

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        # k4v2: values are stored in cache pre-rotated by H_v. The decode/
        # unified kernels return attention output in *rotated value space*
        # (output = softmax(QK^T) @ (V_orig @ H_v) = (orig_output) @ H_v).
        # We invert with one GEMM after attention. Continuation-prefill
        # handles its own inverse internally (cached V is rotated back
        # before flash_attn) so its output is already in original space.
        # `_tq_VRot_q` is pre-cast to q.dtype to keep this matmul on a
        # single precision (avoids mixed fp16/bf16 matmul + extra .to() in
        # the CUDA-graph-captured region).
        v_inv_rot = tq_layer._tq_VRot_q  # None when value_quant_bits != 2

        # Phase 1E (prefix-tier) two-pool dispatch. Activated only when the
        # metadata builder populated the pool-A/pool-B fields (which itself
        # is gated by VLLM_TQ_PREFIX_TIER=1 in the runner). This branch is
        # specialized to pure decode and pure continuation prefill — mixed
        # batches under two-pool aren't in MVP scope; if the metadata
        # builder ever sets two-pool fields on a mixed batch we fall
        # through to single-pool below to avoid silent miscompiles.
        if attn_metadata.is_two_pool() and num_decodes == N:
            # Pure decode under two-pool — use v3_split kernel.
            attn_out = self._decode_attention_split(
                q, attn_metadata, Pi, centroids, PiT, layer,
            )
            if v_inv_rot is not None:
                attn_out = (attn_out.reshape(-1, self.head_size) @ v_inv_rot).reshape(
                    attn_out.shape
                )
            if output.ndim == 3:
                output[:N] = attn_out.to(output.dtype)
            else:
                output[:N] = attn_out.reshape(N, -1).to(output.dtype)
            return output

        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
            if v_inv_rot is not None:
                attn_out = (attn_out.reshape(-1, self.head_size) @ v_inv_rot).reshape(
                    attn_out.shape
                )
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q,
                k,
                v,
                kv_cache,
                attn_metadata,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            attn_out = torch.zeros(
                N, self.num_heads, self.head_size, device=device, dtype=q.dtype
            )

            # --- Decode portion (first num_decodes requests) ---
            # Use full-batch max_seq_len as safe upper bound (no GPU sync).
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            decode_out = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, Pi, centroids, PiT, layer
            )
            if v_inv_rot is not None:
                decode_out = (
                    decode_out.reshape(-1, self.head_size) @ v_inv_rot
                ).reshape(decode_out.shape)
            attn_out[:num_decode_tokens] = decode_out

            # --- Prefill portion (remaining requests) ---
            # CRITICAL: use prefill-specific max_seq_len so flash_attn's
            # fast path (max_query_len == max_seq_len) triggers for
            # first-chunk prefills. Using full-batch max_seq_len breaks
            # this because decode requests inflate max_seq_len.
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            # Use CPU-side max to avoid GPU→CPU sync from .item()
            prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )

        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache (vectorized)                         #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        """Quantize + store via fused Triton kernel."""
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            centroids=layer._tq_centroids,
            norm_correction=self.tq_config.norm_correction,
            value_rotation=layer._tq_VRot,
        )

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape
        Hk = key.shape[1]

        # Fast path: first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # When sinks are present, skip this fast path because the paged attention
        # block table setup is incorrect for batched sequences (all sequences
        # would incorrectly attend to concatenated K/V from all requests).
        if (
            self.sinks is None
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
            and _HAS_FLASH_ATTN
        ):
            return flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                causal=True,
            )

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Convert to Python lists once (single CPU-GPU sync) instead of
        # per-request .item() calls that each force a sync.
        qsl = query_start_loc.tolist()
        seq_lens_list = attn_metadata.seq_lens.tolist()

        # Pre-allocate cu_seqlens for single-request flash_attn calls
        # to avoid per-request host→device tensor creation.
        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        # Cache arange on self (avoid per-call kernel launch).
        _max_seq = attn_metadata.max_seq_len
        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = _ac
        _arange_cache: torch.Tensor = _ac

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if self.sinks is not None:
                    # Use unified attention for sink support
                    out = torch.empty_like(q_seq)
                    k_cache = k_seq.unsqueeze(0)  # [1, q_len, Hk, D]
                    v_cache = v_seq.unsqueeze(0)  # [1, q_len, Hk, D]
                    cu_single = torch.tensor(
                        [0, q_len], dtype=torch.int32, device=query.device
                    )
                    seq_lens_single = torch.tensor(
                        [q_len], dtype=torch.int32, device=query.device
                    )
                    block_table_single = torch.zeros(
                        (1, 1), dtype=torch.int32, device=query.device
                    )
                    unified_attention(
                        q=q_seq,
                        k=k_cache,
                        v=v_cache,
                        out=out,
                        cu_seqlens_q=cu_single,
                        max_seqlen_q=q_len,
                        seqused_k=seq_lens_single,
                        max_seqlen_k=q_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=(-1, -1),
                        block_table=block_table_single,
                        softcap=0.0,
                        q_descale=None,
                        k_descale=None,
                        v_descale=None,
                        sinks=self.sinks,
                    )
                elif _HAS_FLASH_ATTN:
                    self._cu_2[1] = q_len
                    cu = self._cu_2
                    out = flash_attn_varlen_func(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                        softmax_scale=self.scale,
                        causal=True,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to TQ cache
                # by do_kv_cache_update. Use decode kernel directly to
                # avoid O(cached_len) full-dequant per continuation.
                # For large continuations, fall back to _continuation_prefill.
                cached_len = seq_len - q_len
                if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    # Fast path: treat each query as a decode request
                    # with incremental seq_lens for causal masking.
                    # Slice from pre-built arange (no kernel launch)
                    synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]
                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                    if _USE_TQ_V3:
                        out = triton_turboquant_decode_attention_v3(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            value_packed_size=self.tq_config.value_packed_size,
                            max_seq_len=int(seq_len),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                            sinks=self.sinks,
                        )
                    elif _USE_TQ_V2:
                        # v2 kernel does not support sinks yet; sink plumbing
                        # lives on v1 (and soon v3). v2 is opt-in for perf
                        # experiments only.
                        out = triton_turboquant_decode_attention_v2(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            value_packed_size=self.tq_config.value_packed_size,
                            max_seq_len=int(seq_len),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                        )
                    else:
                        out = triton_turboquant_decode_attention(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                            sinks=self.sinks,
                        )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    out = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output

    @staticmethod
    def _dequant_pool_into_buf(
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        centroids: torch.Tensor,
        cached_len: int,
        Hk: int,
        D: int,
        block_size: int,
        BLOCK_D: int,
        # Per-pool TQ layout fields (from a `TurboQuantConfig`).
        mse_bits: int,
        mse_bytes: int,
        val_data_bytes: int,
        value_quant_bits: int,
        key_fp8: bool,
        norm_correction: bool,
        # Pre-allocated workspace buffers (shape [1, Hk, alloc_len, D]).
        k_buf: torch.Tensor,
        v_buf: torch.Tensor,
        device_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequant one TQ pool into the pre-allocated fp16 buffers.

        Factored out of `_continuation_prefill` so the two-pool variant
        (`_continuation_prefill_split`) can call this once per pool with
        its own layout constants. Returns the same two buffers, sliced
        to ``cached_len`` (caller is responsible for any further slicing).
        """
        alloc_len = math.ceil(cached_len / block_size) * block_size
        k_cached = k_buf[:, :, :alloc_len, :]
        v_cached = v_buf[:, :, :alloc_len, :]

        key_data_bytes = D if key_fp8 else mse_bytes
        data_bytes_per_slot = key_data_bytes + val_data_bytes
        meta_region_offset = block_size * Hk * data_bytes_per_slot
        num_soa_fields = 2 if key_fp8 else 3
        soa_v_scale = 0 if key_fp8 else 1
        soa_v_zero = 1 if key_fp8 else 2
        kv_cache_u16 = kv_cache.view(torch.uint16)

        grid = (alloc_len, 1 * Hk)
        _tq_full_dequant_kv[grid](
            kv_cache,
            kv_cache_u16,
            block_table,
            centroids,
            k_cached,
            v_cached,
            k_cached.stride(0),
            k_cached.stride(1),
            k_cached.stride(2),
            v_cached.stride(0),
            v_cached.stride(1),
            v_cached.stride(2),
            kv_cache.stride(0),
            block_table.stride(0),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=Hk,
            MSE_BYTES=mse_bytes,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=mse_bits,
            KEY_FP8=1 if key_fp8 else 0,
            KEY_DATA_BYTES=key_data_bytes,
            META_REGION_OFFSET=meta_region_offset,
            NUM_SOA_FIELDS=num_soa_fields,
            SOA_K_NORM=0,  # MSE-only (FP8 ignores)
            SOA_V_SCALE=soa_v_scale,
            SOA_V_ZERO=soa_v_zero,
            BLOCK_D=BLOCK_D,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(device_index),
            num_warps=4,
        )
        return k_cached, v_cached

    def _continuation_prefill_split(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D) — current chunk (raw)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        pool_a_kv_cache: torch.Tensor,
        pool_b_kv_cache: torch.Tensor,
        pool_a_block_table: torch.Tensor,
        pool_b_block_table: torch.Tensor,
        pool_a_cached_len: int,
        pool_b_cached_len: int,
        seq_len: int,
        # Pool A codec (TQ84 in production: key_fp8=True, val_quant_bits=4).
        pool_a_centroids: torch.Tensor,
        pool_a_mse_bits: int,
        pool_a_mse_bytes: int,
        pool_a_val_data_bytes: int,
        pool_a_value_quant_bits: int,
        pool_a_key_fp8: bool,
        pool_a_norm_correction: bool,
        # Pool B codec (TQ44 in production: key_fp8=False, val_quant_bits=4).
        pool_b_centroids: torch.Tensor,
        pool_b_mse_bits: int,
        pool_b_mse_bytes: int,
        pool_b_val_data_bytes: int,
        pool_b_value_quant_bits: int,
        pool_b_key_fp8: bool,
        pool_b_norm_correction: bool,
    ) -> torch.Tensor:
        """Phase 1E: continuation prefill over two TQ pools.

        Mirrors `_continuation_prefill` but reads cached K/V from two
        physical pools (pool A = prefix tier; pool B = dialogue tier),
        each with its own TQ codec. The output is assembled as

            [pool_a_dequant_K; pool_b_dequant_K; chunk_raw_K]

        and analogously for V, then attended via the standard
        flash_attn_varlen_func causal kernel.

        Per-pool codec assumptions match the prefix-tier plan:
          * Pool A (prefix): typically TQ84_nc — key_fp8=True, val_q=4.
          * Pool B (dialogue): typically TQ44_nc — key_fp8=False, val_q=4.

        The Pi/VRot inverse-rotation matrices are layer-level (derived
        from the Hadamard for ``head_dim``), so both pools share
        ``layer._tq_Pi_half`` (used only for MSE keys) and
        ``layer._tq_VRot_half`` (V inverse rotation when V was stored
        rotated). FP8 keys bypass K inverse rotation entirely.

        Algebraic correctness: this method produces a numerically
        equivalent attention output to running the v3_split decode
        kernel on the same data, modulo the dequant→fp16→flash_attn
        round-trip vs in-place TQ attention. We only invoke it for
        prefill (q_len > 1); decode goes through the v3_split kernel.
        """
        if pool_a_cached_len + pool_b_cached_len + key_chunk.shape[0] != seq_len:
            raise ValueError(
                f"two-pool seq-length invariant violated: "
                f"pool_a_cached_len={pool_a_cached_len} + "
                f"pool_b_cached_len={pool_b_cached_len} + "
                f"q_len={key_chunk.shape[0]} != seq_len={seq_len}"
            )

        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        # Both pools must share block_size and Hk (validated in the kernel
        # launcher; we re-check here for the dequant-grid sizing).
        block_size = pool_a_kv_cache.shape[1]
        if block_size != pool_b_kv_cache.shape[1]:
            raise ValueError(
                f"pool A/B block_size mismatch: A={block_size} "
                f"B={pool_b_kv_cache.shape[1]}"
            )
        BLOCK_D = triton.next_power_of_2(D)
        device_index = device.index or 0

        # ----- Allocate dequant workspace for BOTH pools in one call. -----
        # `WorkspaceManager.get_simultaneous` packs all returned views into
        # a single contiguous buffer; calling it twice would re-use the same
        # bytes and clobber pool-A while we dequant pool-B. So we allocate
        # all four (k_a, v_a, k_b, v_b) up front.
        pool_a_alloc_len = math.ceil(max(pool_a_cached_len, 1) / block_size) * block_size
        pool_b_alloc_len = math.ceil(max(pool_b_cached_len, 1) / block_size) * block_size

        ws = current_workspace_manager()
        k_buf_a, v_buf_a, k_buf_b, v_buf_b = ws.get_simultaneous(
            ((1, Hk, pool_a_alloc_len, D), torch.float16),
            ((1, Hk, pool_a_alloc_len, D), torch.float16),
            ((1, Hk, pool_b_alloc_len, D), torch.float16),
            ((1, Hk, pool_b_alloc_len, D), torch.float16),
        )

        if pool_a_cached_len > 0:
            k_a, v_a = self._dequant_pool_into_buf(
                pool_a_kv_cache,
                pool_a_block_table,
                pool_a_centroids,
                pool_a_cached_len,
                Hk, D, block_size, BLOCK_D,
                pool_a_mse_bits, pool_a_mse_bytes,
                pool_a_val_data_bytes, pool_a_value_quant_bits,
                pool_a_key_fp8, pool_a_norm_correction,
                k_buf_a, v_buf_a, device_index,
            )
        else:
            k_a = v_a = None

        if pool_b_cached_len > 0:
            k_b, v_b = self._dequant_pool_into_buf(
                pool_b_kv_cache,
                pool_b_block_table,
                pool_b_centroids,
                pool_b_cached_len,
                Hk, D, block_size, BLOCK_D,
                pool_b_mse_bits, pool_b_mse_bytes,
                pool_b_val_data_bytes, pool_b_value_quant_bits,
                pool_b_key_fp8, pool_b_norm_correction,
                k_buf_b, v_buf_b, device_index,
            )
        else:
            k_b = v_b = None

        # ----- Inverse rotations per pool (matches _continuation_prefill). -----
        # K inverse rotation is needed only for MSE-key pools; FP8 pools
        # store keys in original space already.
        Pi_half = layer._tq_Pi_half  # may be None; only used when MSE key
        v_inv_rot = layer._tq_VRot_half  # may be None; used when V was rotated

        def _post_dequant_inverse(k_cached, v_cached, cached_len, key_fp8):
            if cached_len == 0:
                return None, None
            # K
            if key_fp8 or Pi_half is None:
                k_trim = k_cached[0, :, :cached_len, :].transpose(0, 1)
            else:
                k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)
                k_flat = k_flat @ Pi_half
                k_trim = k_flat.reshape(Hk, cached_len, D).transpose(0, 1)
            # V
            if v_inv_rot is not None:
                v_flat = v_cached[0, :, :cached_len, :].reshape(-1, D) @ v_inv_rot
                v_trim = v_flat.reshape(Hk, cached_len, D).transpose(0, 1)
            else:
                v_trim = v_cached[0, :, :cached_len, :].transpose(0, 1)
            return k_trim, v_trim

        k_a_trim, v_a_trim = _post_dequant_inverse(
            k_a, v_a, pool_a_cached_len, pool_a_key_fp8,
        )
        k_b_trim, v_b_trim = _post_dequant_inverse(
            k_b, v_b, pool_b_cached_len, pool_b_key_fp8,
        )

        # ----- Concatenate [pool_a, pool_b, chunk] into k_full / v_full. -----
        qdtype = query.dtype
        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        offset = 0
        if k_a_trim is not None:
            k_full[offset : offset + pool_a_cached_len] = k_a_trim.to(qdtype)
            v_full[offset : offset + pool_a_cached_len] = v_a_trim.to(qdtype)
            offset += pool_a_cached_len
        if k_b_trim is not None:
            k_full[offset : offset + pool_b_cached_len] = k_b_trim.to(qdtype)
            v_full[offset : offset + pool_b_cached_len] = v_b_trim.to(qdtype)
            offset += pool_b_cached_len
        # Current chunk (always present for a continuation call).
        k_full[offset:] = key_chunk
        v_full[offset:] = val_chunk
        cached_len_total = pool_a_cached_len + pool_b_cached_len

        # ----- flash_attn_varlen / SDPA fallback (identical to single-pool). -----
        if _HAS_FLASH_ATTN:
            cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
            cu_seqlens_k = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
            return flash_attn_varlen_func(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
            )
        else:
            q_t = query.transpose(0, 1).unsqueeze(0)
            k_t = k_full.transpose(0, 1).unsqueeze(0)
            v_t = v_full.transpose(0, 1).unsqueeze(0)
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len_total
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=mask,
                scale=self.scale,
                enable_gqa=(Hk < Hq),
            )
            return out[0].transpose(0, 1)

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Dequants previously cached K/V, concatenates with the current
        chunk's raw K/V, then runs flash_attn with causal masking.
        """
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        mse_bytes = self._mse_bytes
        val_data_bytes = self._val_data_bytes

        # Dequant cached K/V from TQ cache
        # Allocate slightly over to align to block_size for the grid.
        # Reuse cached buffers to avoid per-call allocation (~16MB at 8K).
        alloc_len = math.ceil(cached_len / block_size) * block_size
        buf_shape = (1, Hk, alloc_len, D)
        # Use WorkspaceManager for dequant buffers.
        # Shared across all layers — saves 60× memory at long context.
        # Required for CUDA Graph capture (per-layer growth incompatible with CG).
        k_buf, v_buf = current_workspace_manager().get_simultaneous(
            (buf_shape, torch.float16),
            (buf_shape, torch.float16),
        )
        # Skip .zero_() — kernel writes all positions up to cached_len,
        # and we only read [:cached_len] afterwards.
        k_cached = k_buf[:, :, :alloc_len, :]
        v_cached = v_buf[:, :, :alloc_len, :]

        # Opt#3 SoA layout constants (must match store-side computation).
        key_fp8 = self.tq_config.key_fp8
        key_data_bytes = D if key_fp8 else mse_bytes
        data_bytes_per_slot = key_data_bytes + val_data_bytes
        meta_region_offset = block_size * Hk * data_bytes_per_slot
        num_soa_fields = 2 if key_fp8 else 3
        soa_k_norm = 0
        soa_v_scale = 0 if key_fp8 else 1
        soa_v_zero = 1 if key_fp8 else 2
        kv_cache_u16 = kv_cache.view(torch.uint16)

        grid = (alloc_len, 1 * Hk)
        _tq_full_dequant_kv[grid](
            kv_cache,
            kv_cache_u16,
            block_table,
            centroids,
            k_cached,
            v_cached,
            k_cached.stride(0),
            k_cached.stride(1),
            k_cached.stride(2),
            v_cached.stride(0),
            v_cached.stride(1),
            v_cached.stride(2),
            kv_cache.stride(0),
            block_table.stride(0),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=Hk,
            MSE_BYTES=mse_bytes,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=self.tq_config.key_mse_bits,
            KEY_FP8=1 if key_fp8 else 0,
            KEY_DATA_BYTES=key_data_bytes,
            META_REGION_OFFSET=meta_region_offset,
            NUM_SOA_FIELDS=num_soa_fields,
            SOA_K_NORM=soa_k_norm,
            SOA_V_SCALE=soa_v_scale,
            SOA_V_ZERO=soa_v_zero,
            BLOCK_D=BLOCK_D,
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(device.index or 0),
            num_warps=4,
        )

        # Inverse-rotate MSE keys back to original space
        if not self.tq_config.key_fp8:
            # fp16 matmul for rotation (2× less bandwidth, uses fp16 tensor cores)
            Pi_half = layer._tq_Pi_half
            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)
            k_flat = k_flat @ Pi_half
            k_cached_trim = k_flat.reshape(Hk, cached_len, D).transpose(
                0, 1
            )  # (cached_len, Hk, D) — already fp16
        else:
            k_cached_trim = k_cached[0, :, :cached_len, :].transpose(
                0, 1
            )  # (cached_len, Hk, D)

        # k4v2: values were stored rotated; inverse-rotate cached V back to
        # original space so it can be concatenated with the raw current chunk
        # and consumed by flash_attn directly. Mirrors the K-side pattern
        # above. Output of this path is in original space → no post-attention
        # inverse needed in forward().
        v_inv_rot = layer._tq_VRot_half
        if v_inv_rot is not None:
            v_flat = v_cached[0, :, :cached_len, :].reshape(-1, D) @ v_inv_rot
            v_cached_trim = v_flat.reshape(Hk, cached_len, D).transpose(0, 1)
        else:
            # Skip .contiguous() — the copy into k_full/v_full handles layout
            v_cached_trim = v_cached[0, :, :cached_len, :].transpose(0, 1)

        # Concatenate cached + current chunk K/V (match query dtype)
        # Pre-allocate full K/V buffer, copy into slices (no cat alloc)
        qdtype = query.dtype
        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk

        # Attention: q_len queries attending to seq_len K/V with causal mask
        if _HAS_FLASH_ATTN:
            cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
            cu_seqlens_k = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
            return flash_attn_varlen_func(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
            )
        else:
            # SDPA fallback: expand KV for GQA, build causal mask
            q_t = query.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)
            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            # Build causal mask: query position p can attend to K position j
            # where j <= cached_len + p (p is 0-indexed within chunk)
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos  # (q_len, seq_len)
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=mask,
                scale=self.scale,
                enable_gqa=(Hk < Hq),
            )  # (1, Hq, q_len, D)
            return out[0].transpose(0, 1)  # (q_len, Hq, D)

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # Acquire shared decode scratch buffers from WorkspaceManager.
        # Layers execute sequentially so one set of buffers is sufficient.
        # Falls back to kernel-internal allocation if workspace unavailable.
        B = query.shape[0]
        D = self.head_size
        S = self.max_num_kv_splits
        Hq = self.num_heads
        mid_o_buf = output_buf = lse_buf = None
        if is_workspace_manager_initialized():
            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.
            mid_o_buf, output_buf, lse_buf = (
                current_workspace_manager().get_simultaneous(
                    ((B, Hq, S, D + 1), torch.float32),
                    ((B, Hq, D), query.dtype),
                    ((B, Hq), torch.float32),
                )
            )

        if _USE_TQ_V3:
            result = triton_turboquant_decode_attention_v3(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                value_packed_size=self.tq_config.value_packed_size,
                max_seq_len=attn_metadata.max_seq_len,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
                sinks=self.sinks,
            )
        elif _USE_TQ_V2:
            # v2 kernel does not support sinks yet; sink plumbing lives on v1
            # (and soon v3). v2 is opt-in for perf experiments only.
            result = triton_turboquant_decode_attention_v2(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                value_packed_size=self.tq_config.value_packed_size,
                max_seq_len=attn_metadata.max_seq_len,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
            )
        else:
            result = triton_turboquant_decode_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
                sinks=self.sinks,
            )
        return result

    def _decode_attention_split(
        self,
        query: torch.Tensor,  # (B, Hq, D) — one query per seq for pure decode
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Phase 1E two-pool decode dispatch.

        Adapts the decode-style ``(B, Hq, D)`` query to the unified
        ``triton_turboquant_unified_attention_split`` signature (which
        expects ``(num_tokens, Hq, D)`` plus a ``query_start_loc``). For
        pure decode, num_tokens == B and ``query_start_loc[i] == i``.

        Codec: pool A and pool B share ``self.tq_config`` in this MVP.
        Codec-mixing (pool A = TQ84, pool B = TQ44) requires per-pool
        ``tq_config`` plumbing that's tracked separately.
        """
        from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
            triton_turboquant_unified_attention_split,
        )

        B = query.shape[0]
        device = query.device
        # Build query_start_loc = [0, 1, 2, ..., B] for one-token-per-seq.
        # Re-use attn_metadata.query_start_loc when it already matches
        # (the runner passes a CPU-side prefix-sum that's correct for
        # decode); allocate fresh otherwise.
        qsl = attn_metadata.query_start_loc
        if qsl is None or qsl.shape[0] != B + 1:
            qsl = torch.arange(B + 1, device=device, dtype=torch.int32)

        cfg = self.tq_config
        pool_a_kv = (
            attn_metadata.pool_a_kv_cache
            if attn_metadata.pool_a_kv_cache is not None
            else attn_metadata.pool_b_kv_cache  # fallback: same cache for both pools
        )
        pool_b_kv = (
            attn_metadata.pool_b_kv_cache
            if attn_metadata.pool_b_kv_cache is not None
            else pool_a_kv
        )
        if pool_a_kv is None or pool_b_kv is None:
            raise RuntimeError(
                "two-pool decode requires pool_a_kv_cache and pool_b_kv_cache "
                "(or one of them) to be set on TurboQuantMetadata; got both None"
            )

        return triton_turboquant_unified_attention_split(
            query=query,
            pool_a_kv_cache=pool_a_kv,
            pool_b_kv_cache=pool_b_kv,
            pool_a_block_table=attn_metadata.pool_a_block_table,
            pool_b_block_table=attn_metadata.pool_b_block_table,
            pool_a_seq_lens=attn_metadata.pool_a_seq_lens,
            pool_b_seq_lens=attn_metadata.pool_b_seq_lens,
            query_start_loc=qsl,
            pool_a_Pi=Pi,
            pool_a_centroids=centroids,
            pool_a_mse_bits=cfg.key_mse_bits,
            pool_a_key_packed_size=cfg.key_packed_size,
            pool_a_value_quant_bits=cfg.effective_value_quant_bits,
            pool_a_value_packed_size=cfg.value_packed_size,
            pool_a_key_fp8=cfg.key_fp8,
            pool_a_norm_correction=cfg.norm_correction,
            pool_a_PiT=PiT,
            pool_b_Pi=Pi,
            pool_b_centroids=centroids,
            pool_b_mse_bits=cfg.key_mse_bits,
            pool_b_key_packed_size=cfg.key_packed_size,
            pool_b_value_quant_bits=cfg.effective_value_quant_bits,
            pool_b_value_packed_size=cfg.value_packed_size,
            pool_b_key_fp8=cfg.key_fp8,
            pool_b_norm_correction=cfg.norm_correction,
            pool_b_PiT=PiT,
            scale=self.scale,
            max_query_len=1,
            max_pool_a_seq_len=attn_metadata.max_pool_a_seq_len or None,
            max_pool_b_seq_len=attn_metadata.max_pool_b_seq_len or None,
            sinks=self.sinks,
        )
