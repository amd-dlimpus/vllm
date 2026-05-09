"""Phase 1E: numerical equivalence tests for `_continuation_prefill_split`.

The split variant dequants TWO TQ pools, inverse-rotates per pool, and
concatenates [pool_a; pool_b; chunk] before running flash_attn. The
single-pool baseline (`_continuation_prefill`) does the same with a
single TQ pool covering all cached tokens.

When both pools use the SAME codec (same TQ44 config), the split output
must match the single-pool output up to fp16 accumulation noise. This
isolates the *plumbing correctness* (workspace allocation, dequant grid
sizing, concatenation order) from codec-specific concerns.

Codec-mixing (pool A = TQ84, pool B = TQ44) requires per-pool
``layer._tq_Pi_half`` / ``layer._tq_VRot_half`` and is exercised in the
end-to-end smoke test instead.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    solve_lloyd_max,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    reset_workspace_manager,
)


GPGPU_AVAILABLE = torch.cuda.is_available()
DEVICE_TYPE = "cuda"


def _build_hadamard(d: int, device_type: str) -> torch.Tensor:
    assert (d & (d - 1)) == 0
    h = torch.tensor([[1.0]], device=device_type)
    while h.shape[0] < d:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return (h / math.sqrt(d)).to(torch.float32)


def _build_and_store_tq_cache(preset, Hk, D, seq_len, block_size, seed):
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
        num_blocks, block_size, Hk, cfg.slot_size_aligned,
        device=device, dtype=torch.uint8,
    )
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
    triton_turboquant_store(
        raw_k, raw_v, kv_cache, slot_mapping, PiT, midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
        centroids=c_sorted,
        norm_correction=cfg.norm_correction,
    )
    return cfg, Pi, PiT, c_sorted, kv_cache, num_blocks, raw_k, raw_v


def _make_impl_instance(num_heads, num_kv_heads, head_size, scale, tq_config):
    """Construct a `TurboQuantAttentionImpl` without going through __init__.

    The full __init__ pulls `get_current_vllm_config()` which requires a
    fully-configured vLLM runtime. For unit tests we set the few
    attributes the methods under test actually read.
    """
    from vllm.v1.attention.backends.turboquant_attn import (
        TurboQuantAttentionImpl,
    )

    impl = TurboQuantAttentionImpl.__new__(TurboQuantAttentionImpl)
    impl.num_heads = num_heads
    impl.num_kv_heads = num_kv_heads
    impl.num_kv_groups = num_heads // num_kv_heads
    impl.head_size = head_size
    impl.scale = scale
    impl.kv_cache_dtype = "turboquant_4bit_nc"
    impl.tq_config = tq_config
    impl._mse_bytes = (
        math.ceil(head_size * tq_config.key_mse_bits / 8)
        if not tq_config.key_fp8
        else head_size
    )
    impl._val_data_bytes = math.ceil(
        head_size * tq_config.effective_value_quant_bits / 8
    )
    impl._n_centroids = tq_config.n_centroids if not tq_config.key_fp8 else 1
    impl.max_num_kv_splits = 16
    impl.sinks = None
    return impl


def _make_layer_stub(D, device, key_fp8: bool):
    """Mimic the layer attributes that `_continuation_prefill[_split]`
    reads. For MSE-key codecs, ``_tq_Pi_half`` is the inverse rotation;
    for FP8 codecs it's unused but we set it anyway for safety.
    """
    H = _build_hadamard(D, str(device))
    layer = SimpleNamespace()
    layer._tq_Pi_half = H.to(torch.float16) if not key_fp8 else None
    layer._tq_VRot_half = None  # only set for value_quant_bits == 2 paths
    return layer


@pytest.fixture
def _ws_manager_for_test():
    """Install a real WorkspaceManager for the test session."""
    reset_workspace_manager()  # ensure clean slate
    init_workspace_manager(torch.device(DEVICE_TYPE), num_ubatches=1)
    try:
        yield None
    finally:
        reset_workspace_manager()


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestContinuationPrefillSplit:
    """`_continuation_prefill_split` ≡ `_continuation_prefill` when both
    pools share the same codec.
    """

    @pytest.mark.parametrize(
        "Hq,Hk,D,q_len,seq_len,split_token",
        [
            (16, 2, 128, 64, 256, 128),
            (32, 4, 128, 32, 512, 256),
            (16, 2, 128, 16, 256, 240),  # tiny pool B
            (16, 2, 128, 240, 256, 16),  # tiny pool A
        ],
    )
    def test_split_matches_single_pool_continuation(
        self, _ws_manager_for_test,
        Hq, Hk, D, q_len, seq_len, split_token,
    ):
        # Invariant: q_len = seq_len - cached_total. cached_total is
        # everything in pool A + pool B = split_token + (seq_len - q_len -
        # split_token). For continuation chunk = q_len, cached_total =
        # seq_len - q_len. We split cached_total into pool_a_cached_len =
        # split_token, pool_b_cached_len = seq_len - q_len - split_token.
        cached_total = seq_len - q_len
        if split_token > cached_total:
            pytest.skip(f"split_token={split_token} > cached_total={cached_total}")
        pool_a_cached_len = split_token
        pool_b_cached_len = cached_total - split_token
        # Block-align the split for the test setup (so block tables are
        # straightforward — production aligns split to first-chunk-prefill
        # boundary which is also block-aligned).
        block_size = 16
        if pool_a_cached_len % block_size != 0:
            pytest.skip("test requires block-aligned split")

        cfg, Pi, PiT, centroids, kv_cache, num_blocks, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=4242,
        )

        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(7777)
        # Query corresponds to the LAST q_len tokens of the seq.
        # Build the chunk K/V from the same raw tokens used for the cache,
        # so concat [cached_pool_a; cached_pool_b; chunk] reconstructs the
        # full sequence verbatim.
        query = torch.randn(q_len, Hq, D, device=device, dtype=torch.float16)

        scale = 1.0 / math.sqrt(D)
        impl = _make_impl_instance(Hq, Hk, D, scale, cfg)
        layer = _make_layer_stub(D, device, key_fp8=cfg.key_fp8)

        # Reconstruct the chunk K/V by re-encoding/decoding through TQ
        # is hard; instead, we use the SAME random seed to generate the
        # same tensors that went into the cache. The cached tokens are
        # the FIRST cached_total = seq_len - q_len tokens. The chunk is
        # the LAST q_len tokens.
        torch.manual_seed(4242)
        full_raw_k = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        full_raw_v = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        chunk_key = full_raw_k[cached_total:].contiguous()
        chunk_val = full_raw_v[cached_total:].contiguous()

        # ----- Reference: single-pool continuation prefill. -----
        full_block_table = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0).contiguous()
        )
        out_ref = impl._continuation_prefill(
            layer=layer,
            query=query,
            key_chunk=chunk_key,
            val_chunk=chunk_val,
            kv_cache=kv_cache,
            block_table=full_block_table,
            cached_len=cached_total,
            seq_len=seq_len,
            Pi=Pi,
            centroids=centroids,
        )

        # ----- v3_split path: same codec on both pools, distinct block tables. -----
        blocks_a = pool_a_cached_len // block_size
        blocks_b_used = (pool_b_cached_len + block_size - 1) // block_size
        if pool_a_cached_len > 0:
            pool_a_bt = full_block_table[:, :blocks_a].contiguous()
        else:
            pool_a_bt = torch.empty((1, 0), device=device, dtype=torch.int32)
        if pool_b_cached_len > 0:
            pool_b_bt = full_block_table[:, blocks_a:blocks_a + blocks_b_used].contiguous()
        else:
            pool_b_bt = torch.empty((1, 0), device=device, dtype=torch.int32)

        out_split = impl._continuation_prefill_split(
            layer=layer,
            query=query,
            key_chunk=chunk_key,
            val_chunk=chunk_val,
            pool_a_kv_cache=kv_cache,
            pool_b_kv_cache=kv_cache,
            pool_a_block_table=pool_a_bt,
            pool_b_block_table=pool_b_bt,
            pool_a_cached_len=pool_a_cached_len,
            pool_b_cached_len=pool_b_cached_len,
            seq_len=seq_len,
            pool_a_centroids=centroids,
            pool_a_mse_bits=cfg.key_mse_bits,
            pool_a_mse_bytes=impl._mse_bytes,
            pool_a_val_data_bytes=impl._val_data_bytes,
            pool_a_value_quant_bits=cfg.effective_value_quant_bits,
            pool_a_key_fp8=cfg.key_fp8,
            pool_a_norm_correction=cfg.norm_correction,
            pool_b_centroids=centroids,
            pool_b_mse_bits=cfg.key_mse_bits,
            pool_b_mse_bytes=impl._mse_bytes,
            pool_b_val_data_bytes=impl._val_data_bytes,
            pool_b_value_quant_bits=cfg.effective_value_quant_bits,
            pool_b_key_fp8=cfg.key_fp8,
            pool_b_norm_correction=cfg.norm_correction,
        )

        diff = (out_ref.float() - out_split.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        # Same dequant + same concat order + same flash_attn → bitwise
        # identical, modulo Triton kernel non-determinism across launches.
        # Empirically max diff is well under fp16 ULP.
        assert max_err < 1e-2, (
            f"split-vs-single-pool max_err={max_err:.4e} too large; "
            f"mean_err={mean_err:.4e}"
        )


@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestTwoPoolMetadataDispatch:
    """Verify the metadata-level two-pool dispatch contract.

    These tests don't exercise full forward() (which requires a layer
    object with all `_tq_*` attributes set up via `_ensure_on_device`);
    they verify (1) `is_two_pool()` correctly reflects field presence,
    and (2) `_decode_attention_split` end-to-end on a same-codec setup.
    """

    def test_is_two_pool_predicate(self):
        from vllm.v1.attention.backends.turboquant_attn import TurboQuantMetadata

        device = torch.device(DEVICE_TYPE)
        bt = torch.zeros((1, 1), device=device, dtype=torch.int32)
        sl = torch.tensor([0], device=device, dtype=torch.int32)
        # All None → single-pool
        m = TurboQuantMetadata(
            seq_lens=sl, slot_mapping=sl, block_table=bt,
            query_start_loc=sl, num_actual_tokens=0,
        )
        assert m.is_two_pool() is False
        # Partial population → still False (defensive)
        m.pool_a_block_table = bt
        assert m.is_two_pool() is False
        m.pool_b_block_table = bt
        m.pool_a_seq_lens = sl
        m.pool_b_seq_lens = sl
        assert m.is_two_pool() is True

    def test_decode_attention_split_smoke(self, _ws_manager_for_test):
        from vllm.v1.attention.backends.turboquant_attn import TurboQuantMetadata

        Hq, Hk, D, seq_len, split_token = 16, 2, 128, 256, 128
        block_size = 16
        cfg, Pi, PiT, centroids, kv_cache, num_blocks, _, _ = _build_and_store_tq_cache(
            "turboquant_4bit_nc", Hk=Hk, D=D, seq_len=seq_len,
            block_size=block_size, seed=4242,
        )
        device = torch.device(DEVICE_TYPE)

        # Single-token decode query.
        torch.manual_seed(11)
        query = torch.randn(1, Hq, D, device=device, dtype=torch.float16)
        scale = 1.0 / math.sqrt(D)
        impl = _make_impl_instance(Hq, Hk, D, scale, cfg)
        # Layer needs the cached _tq_centroids attribute (used by
        # _decode_attention but not by _decode_attention_split). Set both
        # for safety so any future plumbing changes don't break this.
        layer = _make_layer_stub(D, device, key_fp8=cfg.key_fp8)
        layer._tq_centroids = centroids
        layer._tq_Pi = Pi
        layer._tq_PiT = PiT

        # Build two-pool metadata with same cache backing both pools.
        full_bt = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0).contiguous()
        )
        blocks_a = split_token // block_size
        m = TurboQuantMetadata(
            seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
            slot_mapping=torch.empty(0, device=device, dtype=torch.int32),
            block_table=full_bt,
            query_start_loc=torch.tensor([0, 1], device=device, dtype=torch.int32),
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=seq_len,
            is_prefill=False,
            num_decodes=1,
            num_decode_tokens=1,
            pool_a_block_table=full_bt[:, :blocks_a].contiguous(),
            pool_b_block_table=full_bt[:, blocks_a:].contiguous(),
            pool_a_seq_lens=torch.tensor([split_token], device=device, dtype=torch.int32),
            pool_b_seq_lens=torch.tensor([seq_len - split_token], device=device, dtype=torch.int32),
            pool_a_kv_cache=kv_cache,
            pool_b_kv_cache=kv_cache,
            max_pool_a_seq_len=split_token,
            max_pool_b_seq_len=seq_len - split_token,
        )
        assert m.is_two_pool()

        out = impl._decode_attention_split(query, m, Pi, centroids, PiT, layer)
        assert out.shape == (1, Hq, D)
        assert torch.isfinite(out).all(), "output has non-finite entries"

        # Reference: standard single-pool decode kernel (force_2d for
        # apples-to-apples since v3_split is 2D-only). Both should match.
        from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
            triton_turboquant_unified_attention,
        )
        out_ref = triton_turboquant_unified_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=full_bt,
            seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], device=device, dtype=torch.int32),
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
            max_query_len=1,
            max_seq_len=seq_len,
        )
        diff = (out.float() - out_ref.float()).abs()
        assert diff.max().item() < 5e-2, (
            f"_decode_attention_split diverged from single-pool ref: "
            f"max_err={diff.max().item():.4e}"
        )
