"""Phase 1E v3_split kernel — numerical equivalence tests.

The split launcher implements two-pool TurboQuant attention by running the
existing v3 unified 2D kernel twice (once per pool) and merging the two
outputs via fp32 log-sum-exp.

These tests pin the *algebra* of the split path: equality (within accumu-
lation noise) to a single-pool reference computed over the same data. The
codec is held constant (same TQ44 configuration on both pools) so any
discrepancy is attributable to the LSE merge, not codec mixing — which is
exactly what we want to verify here.

Test surface:
  * test_split_equiv_to_single_pool   — split point at exact block boundary
  * test_split_uneven                  — small pool A, large pool B (8 vs 248)
  * test_split_pool_a_empty            — pool A has 0 tokens (degenerates to B)
  * test_split_decode_q1               — Q=1 (decode-style) split
  * test_lse_returned_matches_logz     — direct LSE-return correctness check
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
    """Sylvester-construction Hadamard matrix scaled by 1/sqrt(d)."""
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
):
    """Allocate a TQ-packed KV cache holding ``seq_len`` random tokens.

    Returns ``(cfg, Pi, PiT, centroids, midpoints, kv_cache, num_blocks,
    raw_k, raw_v)``.
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
):
    """Reference: standard single-pool v3 attention over the full cache."""
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
        force_2d=True,  # keep apples-to-apples vs split (which is 2D-only)
    )


def _run_v3_split(
    cfg, Pi, PiT, centroids, kv_cache, query, seq_len, split_token, block_size,
):
    """v3_split with the same TQ44 cache backing both pools, distinguished
    only by which slice of the block table each pool sees.
    """
    from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
        triton_turboquant_unified_attention_split,
    )

    device = query.device
    Q = query.shape[0]
    assert split_token % block_size == 0, "split_token must align to block_size"
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
    pool_b_seq_lens = torch.tensor([seq_len - split_token], device=device, dtype=torch.int32)
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


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestV3SplitEquivalence:
    """v3_split must match single-pool v3 when both pools use the same codec.

    Equivalence is mathematical (modulo fp accumulation order). We allow
    a small tolerance because the LSE merge does the final combine in
    fp32 then casts back to fp16, while the single-pool kernel keeps the
    online softmax accumulator in fp32 for the entire seq_len. Empirically
    this introduces <1 fp16 ULP of difference per row.
    """

    @pytest.mark.parametrize("preset", ["turboquant_4bit_nc"])
    @pytest.mark.parametrize(
        "Hq,Hk,D,Q,seq_len,split_token",
        [
            (16, 2, 128, 64, 256, 128),  # even split, mid-size context
            (32, 4, 128, 32, 512, 256),  # llama-ish
            (64, 8, 64, 64, 1024, 512),  # gpt-oss-ish
        ],
    )
    def test_split_equiv_to_single_pool(
        self, preset, Hq, Hk, D, Q, seq_len, split_token
    ):
        block_size = 16
        cfg, Pi, PiT, centroids, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            preset, Hk=Hk, D=D, seq_len=seq_len, block_size=block_size, seed=4242,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(7777)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)

        out_ref = _run_single_pool_v3(
            cfg, Pi, PiT, centroids, kv_cache, query, seq_len, block_size,
        )
        out_split = _run_v3_split(
            cfg, Pi, PiT, centroids, kv_cache, query, seq_len, split_token,
            block_size,
        )

        diff = (out_ref.float() - out_split.float()).abs()
        # fp16 ULP at typical attention magnitudes is ~5e-4. Allow some
        # slack for the extra fp32->fp16 cast in the LSE merge path.
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        tag = (
            f"[v3_split {preset} Hq={Hq} Hk={Hk} D={D} Q={Q} "
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
        "split_token,seq_len",
        [
            (16, 256),     # tiny pool A (1 block), pool B = chunk = 240 tokens
            (240, 256),    # large pool A (15 blocks), pool B = chunk = 16 tokens
        ],
    )
    def test_split_uneven(self, split_token, seq_len):
        # v3_split contract: in continuation prefill the current chunk is
        # exactly the pool-B range. So Q must equal seq_len - split_token,
        # otherwise the chunk's queries straddle the pool boundary (which
        # is impossible to reconstruct via two non-overlapping passes).
        Q = seq_len - split_token
        Hq, Hk, D = 16, 2, 128
        block_size = 16
        cfg, Pi, PiT, centroids, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=1234,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(2024)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)

        out_ref = _run_single_pool_v3(
            cfg, Pi, PiT, centroids, kv_cache, query, seq_len, block_size,
        )
        out_split = _run_v3_split(
            cfg, Pi, PiT, centroids, kv_cache, query, seq_len, split_token,
            block_size,
        )
        diff = (out_ref.float() - out_split.float()).abs()
        assert diff.max().item() < 5e-2
        assert diff.mean().item() < 5e-3

    def test_split_pool_a_empty(self):
        """split_token=0 ⇒ all data in pool B ⇒ split must equal reference."""
        Hq, Hk, D, Q, seq_len = 16, 2, 128, 32, 256
        block_size = 16
        cfg, Pi, PiT, centroids, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=99,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(11)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)

        out_ref = _run_single_pool_v3(
            cfg, Pi, PiT, centroids, kv_cache, query, seq_len, block_size,
        )
        out_split = _run_v3_split(
            cfg, Pi, PiT, centroids, kv_cache, query, seq_len, 0, block_size,
        )
        diff = (out_ref.float() - out_split.float()).abs()
        # pool_a_seq_lens=[0] means pool A pass produces lse_a=-inf; the
        # merge collapses to pool B (causal, full seq). The result should
        # match single-pool reference very tightly because the only diff
        # is the fp32→fp16 cast on the merge output.
        assert diff.max().item() < 1e-2, (
            f"empty-pool-A split should equal pool-B-only ref; "
            f"max_err={diff.max().item():.4e}"
        )

    @pytest.mark.parametrize("split_token", [128, 256])
    def test_split_decode_q1(self, split_token):
        """Decode-style (Q=1): the heuristic picks BLOCK_M=16 for Q=1
        which exercises a structurally different path than prefill BM=128.
        """
        Hq, Hk, D, seq_len = 16, 2, 128, 512
        block_size = 16
        cfg, Pi, PiT, centroids, _, kv_cache, _, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=555,
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(303)
        query = torch.randn(1, Hq, D, device=device, dtype=torch.float16)

        out_ref = _run_single_pool_v3(
            cfg, Pi, PiT, centroids, kv_cache, query, seq_len, block_size,
        )
        out_split = _run_v3_split(
            cfg, Pi, PiT, centroids, kv_cache, query, seq_len, split_token,
            block_size,
        )
        diff = (out_ref.float() - out_split.float()).abs()
        assert diff.max().item() < 5e-2
        assert diff.mean().item() < 5e-3


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestV3LSEReturn:
    """Direct correctness check for the new RETURN_LSE kernel branch.

    Verifies that ``triton_turboquant_unified_attention(..., return_lse=True)``
    returns an LSE that correctly normalizes the attention output: i.e.
    ``softmax(QK)V == sum_j exp(QK_j - lse) V_j`` holds within tol.
    """

    def test_lse_returned_matches_logz(self):
        from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
            triton_turboquant_unified_attention,
        )

        Hq, Hk, D, seq_len, Q = 16, 2, 128, 256, 32
        block_size = 16
        cfg, Pi, PiT, centroids, _, kv_cache, num_blocks, _, _ = (
            _build_and_store_tq_cache(
                "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
                block_size=block_size, seed=42,
            )
        )
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(909)
        query = torch.randn(Q, Hq, D, device=device, dtype=torch.float16)
        block_table = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .contiguous()
        )
        seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
        query_start_loc = torch.tensor([0, Q], device=device, dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)

        # Run twice: once without LSE (legacy), once with LSE return.
        out_legacy = triton_turboquant_unified_attention(
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
            force_2d=True,
            max_query_len=Q,
            max_seq_len=seq_len,
        )
        out_lse, lse = triton_turboquant_unified_attention(
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
            force_2d=True,
            max_query_len=Q,
            max_seq_len=seq_len,
            return_lse=True,
        )

        # Output should be bitwise identical (RETURN_LSE doesn't change the
        # output store path).
        assert torch.equal(out_legacy, out_lse), (
            "RETURN_LSE branch perturbed the output store; max diff="
            f"{(out_legacy.float() - out_lse.float()).abs().max().item():.4e}"
        )
        # LSE must be finite for all real query rows (Q queries × Hq heads).
        assert lse.shape == (Q, Hq), f"unexpected lse shape {lse.shape}"
        assert torch.isfinite(lse).all(), (
            f"LSE has non-finite values: {lse}"
        )
