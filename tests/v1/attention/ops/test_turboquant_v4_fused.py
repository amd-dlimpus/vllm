"""Phase 1F v4 fused two-pool kernel — numerical equivalence tests.

The fused launcher implements two-pool TurboQuant attention by running ONE
kernel that walks Pool A (non-causal) then Pool B (causal) with a single
shared ``(M, L, acc)`` accumulator. No HBM round-trip for per-pool partials,
no host-side LSE merge.

Test strategy: hold the codec constant (same TQ44 on both pools) and assert
numerical equivalence to (a) the single-pool kernel run over the concatenated
data, AND (b) the existing v3_split launcher. This pins the algebra of the
fused path independent of codec mixing — exactly what we want at this layer.

A separate test (``test_split_vs_fused_mixed_precision``) verifies the fused
kernel runs cleanly when Pool A uses one codec and Pool B uses another
(TQ84 + TQ44 — the actual production mixed-precision config). We can only
sanity-check that the result is finite and shaped correctly, because no
single-pool reference exists for the mixed-precision case (that's the whole
reason this kernel exists).

Test surface mirrors test_turboquant_v3_split.py:
  * test_fused_equiv_to_single_pool   — equivalence to single-pool v3 ref
  * test_fused_equiv_to_v3_split      — equivalence to v3_split for same input
  * test_fused_uneven                  — tiny pool A, large pool B (and vice versa)
  * test_fused_pool_a_empty            — pool A has 0 tokens
  * test_fused_pool_b_empty            — pool B has 0 tokens (degenerates)
  * test_fused_decode_q1               — Q=1 (decode-style) path
  * test_fused_head_dim_256            — head_dim=256, what v3_split couldn't do
  * test_fused_mixed_precision         — TQ84 pool A + TQ44 pool B (sanity)
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    solve_lloyd_max,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)


GPGPU_AVAILABLE = torch.cuda.is_available()
DEVICE_TYPE = "cuda"


def _build_hadamard(d: int, device_type: str) -> torch.Tensor:
    assert (d & (d - 1)) == 0, f"d must be a power of 2, got {d}"
    h = torch.tensor([[1.0]], device=device_type)
    while h.shape[0] < d:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return (h / math.sqrt(d)).to(torch.float32)


def _build_and_store_tq_cache(
    preset: str,
    Hk: int,
    D: int,
    seq_len: int,
    block_size: int,
    seed: int,
    raw_k: torch.Tensor | None = None,
    raw_v: torch.Tensor | None = None,
):
    """Allocate a TQ-packed KV cache holding ``seq_len`` random tokens.

    If ``raw_k`` / ``raw_v`` are provided, store those bytes instead of
    drawing fresh randoms — used when we want two caches sharing the same
    underlying data but in different codec presets.
    """
    from vllm.v1.attention.ops.triton_turboquant_store import (
        triton_turboquant_store,
    )

    device = torch.device(DEVICE_TYPE)
    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=D)
    H = _build_hadamard(D, DEVICE_TYPE)
    Pi = PiT = H
    centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
    centroids = centroids.float().to(device)
    c_sorted, _ = centroids.sort()
    midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

    if raw_k is None or raw_v is None:
        torch.manual_seed(seed)
        raw_k = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        raw_v = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)

    num_blocks = (seq_len + block_size - 1) // block_size + 1
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        Hk,
        cfg.slot_size_aligned,
        device=device,
        dtype=torch.uint8,
    )
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
    triton_turboquant_store(
        raw_k,
        raw_v,
        kv_cache,
        slot_mapping,
        PiT,
        midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
        centroids=c_sorted,
        norm_correction=cfg.norm_correction,
    )
    return cfg, Pi, PiT, c_sorted, midpoints, kv_cache, num_blocks, raw_k, raw_v


def _run_single_pool_v3(
    cfg, Pi, PiT, centroids, kv_cache, query, seq_len, block_size,
    fuse_q_rot: bool = True,
):
    from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
        triton_turboquant_unified_attention,
    )

    device = query.device
    Q = query.shape[0]
    num_blocks_used = (seq_len + block_size - 1) // block_size
    block_table = (
        torch.arange(num_blocks_used, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .contiguous()
    )
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, Q], device=device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(query.shape[-1])
    return triton_turboquant_unified_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        Pi=Pi,
        centroids=centroids,
        scale=scale,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        value_packed_size=cfg.value_packed_size,
        key_fp8=cfg.key_fp8,
        norm_correction=cfg.norm_correction,
        PiT=PiT,
        max_query_len=Q,
        max_seq_len=seq_len,
        force_2d=True,
        fuse_q_rot=fuse_q_rot,
    )


def _run_fused(
    cfg_a, Pi_a, PiT_a, cent_a, kv_cache_a,
    cfg_b, Pi_b, PiT_b, cent_b, kv_cache_b,
    query, total_seq_len, split_token, block_size,
    pool_b_block_offset: int | None = None,
):
    """Run the fused two-pool launcher.

    ``pool_b_block_offset`` controls where pool B's block IDs start:
      * ``None`` (default): if kv_cache_a IS kv_cache_b (same physical
        tensor, used by equivalence tests), pool B starts at block
        ``blocks_a`` of the shared cache (mirrors v3_split's test setup);
        otherwise pool B starts at block 0 of its own cache.
      * explicit int: caller takes responsibility for the layout.
    """
    from vllm.v1.attention.ops.triton_turboquant_unified_attention_two_pool import (  # noqa: E501
        triton_turboquant_unified_attention_two_pool_fused,
    )

    device = query.device
    Q = query.shape[0]
    assert split_token % block_size == 0, "split_token must align to block_size"
    blocks_a = split_token // block_size
    blocks_b = (total_seq_len - split_token + block_size - 1) // block_size

    if pool_b_block_offset is None:
        # If the same physical cache backs both pools, pool B's blocks live
        # AFTER pool A's blocks in the shared cache. Otherwise (production
        # Phase 1F two-pool storage), each pool indexes its own cache from
        # block 0.
        shared = kv_cache_a.data_ptr() == kv_cache_b.data_ptr()
        pool_b_block_offset = blocks_a if shared else 0

    pool_a_bt = (
        torch.arange(blocks_a, device=device, dtype=torch.int32)
        .unsqueeze(0).contiguous()
        if blocks_a > 0
        else torch.zeros((1, 1), device=device, dtype=torch.int32)
    )
    pool_b_bt = (
        torch.arange(
            pool_b_block_offset,
            pool_b_block_offset + blocks_b,
            device=device, dtype=torch.int32,
        ).unsqueeze(0).contiguous()
        if blocks_b > 0
        else torch.zeros((1, 1), device=device, dtype=torch.int32)
    )

    pool_a_seq_lens = torch.tensor(
        [split_token], device=device, dtype=torch.int32
    )
    pool_b_seq_lens = torch.tensor(
        [total_seq_len - split_token], device=device, dtype=torch.int32
    )
    query_start_loc = torch.tensor([0, Q], device=device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(query.shape[-1])

    return triton_turboquant_unified_attention_two_pool_fused(
        query=query,
        pool_a_kv_cache=kv_cache_a,
        pool_b_kv_cache=kv_cache_b,
        pool_a_block_table=pool_a_bt,
        pool_b_block_table=pool_b_bt,
        pool_a_seq_lens=pool_a_seq_lens,
        pool_b_seq_lens=pool_b_seq_lens,
        query_start_loc=query_start_loc,
        pool_a_Pi=Pi_a,
        pool_a_centroids=cent_a,
        pool_a_mse_bits=cfg_a.key_mse_bits,
        pool_a_key_packed_size=cfg_a.key_packed_size,
        pool_a_value_quant_bits=cfg_a.effective_value_quant_bits,
        pool_a_value_packed_size=cfg_a.value_packed_size,
        pool_a_key_fp8=cfg_a.key_fp8,
        pool_a_norm_correction=cfg_a.norm_correction,
        pool_a_PiT=PiT_a,
        pool_b_Pi=Pi_b,
        pool_b_centroids=cent_b,
        pool_b_mse_bits=cfg_b.key_mse_bits,
        pool_b_key_packed_size=cfg_b.key_packed_size,
        pool_b_value_quant_bits=cfg_b.effective_value_quant_bits,
        pool_b_value_packed_size=cfg_b.value_packed_size,
        pool_b_key_fp8=cfg_b.key_fp8,
        pool_b_norm_correction=cfg_b.norm_correction,
        pool_b_PiT=PiT_b,
        scale=scale,
        max_query_len=Q,
        max_pool_a_seq_len=split_token,
        max_pool_b_seq_len=total_seq_len - split_token,
    )


def _run_v3_split_same_codec(
    cfg, Pi, PiT, centroids, kv_cache, query, seq_len, split_token, block_size,
):
    """v3_split with both pools backed by the same kv_cache (single-codec)."""
    from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
        triton_turboquant_unified_attention_split,
    )

    device = query.device
    Q = query.shape[0]
    blocks_a = split_token // block_size
    blocks_b = (seq_len - split_token + block_size - 1) // block_size
    full_bt = (
        torch.arange(blocks_a + blocks_b, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .contiguous()
    )
    pool_a_bt = full_bt[:, :blocks_a].contiguous() if blocks_a > 0 else \
        torch.empty((1, 0), device=device, dtype=torch.int32)
    pool_b_bt = full_bt[:, blocks_a:].contiguous() if blocks_b > 0 else \
        torch.empty((1, 0), device=device, dtype=torch.int32)
    pool_a_seq_lens = torch.tensor([split_token], device=device, dtype=torch.int32)
    pool_b_seq_lens = torch.tensor(
        [seq_len - split_token], device=device, dtype=torch.int32
    )
    query_start_loc = torch.tensor([0, Q], device=device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(query.shape[-1])
    return triton_turboquant_unified_attention_split(
        query=query,
        pool_a_kv_cache=kv_cache,
        pool_b_kv_cache=kv_cache,
        pool_a_block_table=pool_a_bt,
        pool_b_block_table=pool_b_bt,
        pool_a_seq_lens=pool_a_seq_lens,
        pool_b_seq_lens=pool_b_seq_lens,
        query_start_loc=query_start_loc,
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
        scale=scale,
        max_query_len=Q,
        max_pool_a_seq_len=split_token,
        max_pool_b_seq_len=seq_len - split_token,
    )


# ---------------------------------------------------------------------------
# Equivalence tests against the single-pool reference. Both fused pools use
# the same TQ44 codec backing the same kv_cache; only the split point and
# the block tables differ — so the fused result MUST equal the single-pool
# v3 result over the full sequence, modulo accumulation noise.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestV4FusedEquivalence:
    @pytest.mark.parametrize("preset", ["turboquant_4bit_nc"])
    @pytest.mark.parametrize(
        "Hq,Hk,D,Q,seq_len,split_token",
        [
            (16, 2, 128, 64, 256, 128),
            (32, 4, 128, 32, 512, 256),
            (64, 8, 64, 64, 1024, 512),
        ],
    )
    def test_fused_equiv_to_single_pool(
        self, preset, Hq, Hk, D, Q, seq_len, split_token
    ):
        block_size = 16
        cfg, Pi, PiT, cent, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            preset, Hk=Hk, D=D, seq_len=seq_len, block_size=block_size,
            seed=4242,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(7777)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)

        out_ref = _run_single_pool_v3(
            cfg, Pi, PiT, cent, kv_cache, query, seq_len, block_size,
        )
        out_fused = _run_fused(
            cfg, Pi, PiT, cent, kv_cache,
            cfg, Pi, PiT, cent, kv_cache,
            query, seq_len, split_token, block_size,
        )

        diff = (out_ref.float() - out_fused.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        tag = (
            f"[v4_fused {preset} Hq={Hq} Hk={Hk} D={D} Q={Q} "
            f"seq={seq_len} split={split_token}]"
        )
        assert max_err < 5e-2, (
            f"{tag} max abs err {max_err:.4e} exceeds 5e-2; "
            f"mean={mean_err:.4e}"
        )
        assert mean_err < 5e-3, (
            f"{tag} mean abs err {mean_err:.4e} exceeds 5e-3"
        )

    @pytest.mark.parametrize(
        "Hq,Hk,D,Q,seq_len,split_token",
        [
            (16, 2, 128, 64, 256, 128),
            (32, 4, 128, 32, 512, 256),
        ],
    )
    def test_fused_equiv_to_v3_split(
        self, Hq, Hk, D, Q, seq_len, split_token
    ):
        """Fused and v3_split must agree (within fp accumulation noise)
        when both run on the same input with the same codec."""
        block_size = 16
        cfg, Pi, PiT, cent, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=11,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(22)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)

        out_split = _run_v3_split_same_codec(
            cfg, Pi, PiT, cent, kv_cache, query, seq_len, split_token,
            block_size,
        )
        out_fused = _run_fused(
            cfg, Pi, PiT, cent, kv_cache,
            cfg, Pi, PiT, cent, kv_cache,
            query, seq_len, split_token, block_size,
        )
        diff = (out_split.float() - out_fused.float()).abs()
        assert diff.max().item() < 5e-2, (
            f"v4_fused vs v3_split: max_err={diff.max().item():.4e}"
        )
        assert diff.mean().item() < 5e-3, (
            f"v4_fused vs v3_split: mean_err={diff.mean().item():.4e}"
        )

    @pytest.mark.parametrize(
        "split_token,seq_len",
        [
            (16, 256),
            (240, 256),
        ],
    )
    def test_fused_uneven(self, split_token, seq_len):
        Q = seq_len - split_token
        Hq, Hk, D = 16, 2, 128
        block_size = 16
        cfg, Pi, PiT, cent, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=1234,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(2024)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)
        out_ref = _run_single_pool_v3(
            cfg, Pi, PiT, cent, kv_cache, query, seq_len, block_size,
        )
        out_fused = _run_fused(
            cfg, Pi, PiT, cent, kv_cache,
            cfg, Pi, PiT, cent, kv_cache,
            query, seq_len, split_token, block_size,
        )
        diff = (out_ref.float() - out_fused.float()).abs()
        assert diff.max().item() < 5e-2
        assert diff.mean().item() < 5e-3

    def test_fused_pool_a_empty(self):
        """split_token=0 ⇒ all data in pool B ⇒ fused must equal reference."""
        Hq, Hk, D, Q, seq_len = 16, 2, 128, 32, 256
        block_size = 16
        cfg, Pi, PiT, cent, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=99,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(11)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)
        out_ref = _run_single_pool_v3(
            cfg, Pi, PiT, cent, kv_cache, query, seq_len, block_size,
        )
        out_fused = _run_fused(
            cfg, Pi, PiT, cent, kv_cache,
            cfg, Pi, PiT, cent, kv_cache,
            query, seq_len, 0, block_size,
        )
        diff = (out_ref.float() - out_fused.float()).abs()
        assert diff.max().item() < 1e-2

    def test_fused_pool_b_empty_decode(self):
        """All KV in pool A, Q=1 attention (decode). Common case for tier
        boundary: every dialogue token is in pool B initially, but during
        an early-turn decode the prefix-only pool A is the entirety."""
        Hq, Hk, D, seq_len = 16, 2, 128, 256
        block_size = 16
        cfg, Pi, PiT, cent, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=77,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(88)
        # decode-style: 1 query, attending to all 256 prefix tokens in pool A
        query = torch.randn(1, Hq, D, device=device, dtype=torch.float16)
        # split = seq_len means pool A holds everything; the query is a NEW
        # token (not in any pool yet) — so pool B has length 1 for the
        # current Q chunk. Set up that way:
        # actually, easier: use single-pool ref directly with seq_len=256+1,
        # but the v3 single-pool kernel needs the Q-token to be stored, so
        # we cheat: compare fused-(pool_a_seq=256, pool_b_seq=0) Q=1 to
        # single-pool seq_len=256 Q=1 (current Q attends only to prefix).
        out_ref = _run_single_pool_v3(
            cfg, Pi, PiT, cent, kv_cache, query, seq_len, block_size,
        )
        out_fused = _run_fused(
            cfg, Pi, PiT, cent, kv_cache,
            cfg, Pi, PiT, cent, kv_cache,
            query, seq_len, seq_len, block_size,
        )
        diff = (out_ref.float() - out_fused.float()).abs()
        assert diff.max().item() < 5e-2

    @pytest.mark.parametrize("split_token", [128, 256])
    def test_fused_decode_q1(self, split_token):
        Hq, Hk, D, seq_len = 16, 2, 128, 512
        block_size = 16
        cfg, Pi, PiT, cent, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=555,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(303)
        query = torch.randn(1, Hq, D, device=device, dtype=torch.float16)
        out_ref = _run_single_pool_v3(
            cfg, Pi, PiT, cent, kv_cache, query, seq_len, block_size,
        )
        out_fused = _run_fused(
            cfg, Pi, PiT, cent, kv_cache,
            cfg, Pi, PiT, cent, kv_cache,
            query, seq_len, split_token, block_size,
        )
        diff = (out_ref.float() - out_fused.float()).abs()
        assert diff.max().item() < 5e-2
        assert diff.mean().item() < 5e-3


# ---------------------------------------------------------------------------
# head_dim=256: the specific case v3_split couldn't handle (SMEM OOM).
# This test ALONE justifies the fused kernel's existence in the codebase.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestV4FusedHeadDim256:
    """Qwen3-35B-A3B has head_dim=256. v3_split OOMs SMEM at prefill at this
    size (256 KB requested vs 64 KB MI300X cap) and falls back to the
    single-pool kernel — which means the two-pool dispatch path was never
    actually exercised on this model. The fused kernel's SMEM-aware
    BLOCK_M heuristic fits the same compute into < 50 KB SMEM and unblocks
    the experiment.
    """

    def test_fused_head_dim_256_prefill_smoke(self):
        """Prefill at head_dim=256: no surviving Triton reference (the
        single-pool kernel OOMs at BLOCK_M=128 × head_dim=256). Assert
        the fused kernel runs, output is finite, and has the right shape.
        Numerical equivalence is covered by the head_dim=128 tests above
        (the kernel has no head_dim-specific algebra).
        """
        Hq, Hk, D, Q, seq_len, split_token = 16, 2, 256, 64, 256, 128
        block_size = 16
        cfg, Pi, PiT, cent, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=42,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(909)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)
        out_fused = _run_fused(
            cfg, Pi, PiT, cent, kv_cache,
            cfg, Pi, PiT, cent, kv_cache,
            query, seq_len, split_token, block_size,
        )
        assert out_fused.shape == (Q, Hq, D)
        assert torch.isfinite(out_fused).all(), (
            "head_dim=256 prefill output has non-finite values"
        )
        m = out_fused.abs().mean().item()
        assert 1e-4 < m < 10.0, (
            f"head_dim=256 output magnitude {m:.4e} looks pathological"
        )

    def test_fused_head_dim_256_decode(self):
        """Decode at head_dim=256: BLOCK_M=16 fits SMEM for both kernels
        once FUSE_Q_ROT is disabled (the 256×256 PiT tile would otherwise
        cost 128 KB SMEM). Compare to single-pool reference."""
        Hq, Hk, D, seq_len, split_token = 16, 2, 256, 512, 256
        block_size = 16
        cfg, Pi, PiT, cent, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=505,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(606)
        query = torch.randn(1, Hq, D, device=device, dtype=torch.float16)
        # Reference: launcher-side Q rotation (host rocBLAS) to keep PiT
        # off the kernel's SMEM. Fused launcher auto-disables FUSE_Q_ROT
        # at head_dim>=192 via its SMEM guard.
        out_ref = _run_single_pool_v3(
            cfg, Pi, PiT, cent, kv_cache, query, seq_len, block_size,
            fuse_q_rot=False,
        )
        out_fused = _run_fused(
            cfg, Pi, PiT, cent, kv_cache,
            cfg, Pi, PiT, cent, kv_cache,
            query, seq_len, split_token, block_size,
        )
        diff = (out_ref.float() - out_fused.float()).abs()
        assert diff.max().item() < 5e-2, (
            f"head_dim=256 decode max_err={diff.max().item():.4e}"
        )
        assert diff.mean().item() < 5e-3


# ---------------------------------------------------------------------------
# Mixed-precision sanity (the actual production case). No analytical
# reference exists, so we only assert the output is finite and shaped
# correctly. The real check is the LCB accuracy delta (system-level).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestV4FusedMixedPrecision:
    def test_fused_tq84_a_tq44_b(self):
        """Pool A = TQ84 (FP8 keys, 4-bit values), Pool B = TQ44 (4-bit
        MSE keys, 4-bit values). Same raw KV data stored in two different
        codec presets — algebra of the fused kernel should accept the
        codec mismatch and produce a finite, correctly-shaped output."""
        Hq, Hk, D, Q, seq_len, split_token = 16, 2, 128, 32, 512, 256
        block_size = 16
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(111)
        raw_k = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        raw_v = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)

        # Pool A: TQ84 holding tokens [0, split_token).
        cfg_a, Pi_a, PiT_a, cent_a, _, kv_a, _, _, _ = (
            _build_and_store_tq_cache(
                "turboquant_k8v4_nc", Hk=Hk, D=D, seq_len=split_token,
                block_size=block_size, seed=0,
                raw_k=raw_k[:split_token].contiguous(),
                raw_v=raw_v[:split_token].contiguous(),
            )
        )
        # Pool B: TQ44 holding tokens [split_token, seq_len).
        cfg_b, Pi_b, PiT_b, cent_b, _, kv_b, _, _, _ = (
            _build_and_store_tq_cache(
                "turboquant_4bit_nc", Hk=Hk, D=D,
                seq_len=seq_len - split_token,
                block_size=block_size, seed=0,
                raw_k=raw_k[split_token:].contiguous(),
                raw_v=raw_v[split_token:].contiguous(),
            )
        )

        torch.manual_seed(222)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)
        out = _run_fused(
            cfg_a, Pi_a, PiT_a, cent_a, kv_a,
            cfg_b, Pi_b, PiT_b, cent_b, kv_b,
            query, seq_len, split_token, block_size,
        )
        assert out.shape == (Q, Hq, D), f"got shape {out.shape}"
        assert torch.isfinite(out).all(), (
            "mixed-precision fused output has non-finite values"
        )
        # Sanity: output magnitude should be roughly in attention-output
        # range (not zero, not blown up). Loose bounds.
        m = out.abs().mean().item()
        assert 1e-4 < m < 10.0, (
            f"mixed-precision output magnitude {m:.4e} looks pathological"
        )
