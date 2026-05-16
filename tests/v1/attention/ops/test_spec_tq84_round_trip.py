"""Spec-TQ84 (MSE_BITS=8) Triton kernel round-trip test.

Validates the read-path fix added to:
  - _tq_load_k_tile in triton_turboquant_unified_attention.py
  - _tq_full_dequant_kv in triton_turboquant_decode.py

For MSE_BITS=8 the store kernel writes 1 byte per element (no bit packing
and no cross-byte read). Prior to the fix the read kernels fell through to
the generic bit-extraction branch and performed an out-of-bounds +1 byte
load, crashing once the OOB address landed on an unmapped page.

This test stores a random K through the production launcher with the
spec-TQ84 preset, dequants via the patched _tq_full_dequant_kv, and asserts:

  (a) max-abs vs offline Python tq84_spec_round_trip <= TOL_VS_OFFLINE
      (Triton path equals the reference codec up to bf16/fp16 rounding)

  (b) max-abs vs raw input < 1.0 (sanity � quantization is lossy but
      bounded; the offline reference itself has a similar gap)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Make the offline reference importable.
REF_DIR = Path(
    "/scratch/dlimpus/kv-cache-research/directions/D_prefix_tier/"
    "phase2/analysis/e08_redistribution"
)
if str(REF_DIR) not in sys.path:
    sys.path.insert(0, str(REF_DIR))

from variants_tq84_spec import tq84_spec_round_trip  # noqa: E402

from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    kv_cache_flat_u16,
)
from vllm.v1.attention.ops.triton_turboquant_store import (
    triton_turboquant_store,
)


CENTROIDS_PATH = REF_DIR / "centroids_gauss_d256_b8.pt"
TOL_VS_OFFLINE = 1e-2  # bf16 store path + fp16 norm rounding


def _build_hadamard(d: int, device: torch.device) -> torch.Tensor:
    assert (d & (d - 1)) == 0
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < d:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return (h / math.sqrt(d)).to(torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_spec_tq84_round_trip():
    device = torch.device("cuda")
    torch.manual_seed(0)

    seq_len = 32  # small; not testing perf, just correctness
    Hk = 2
    D = 128
    block_size = 16

    # Spec-TQ84 preset: 8-bit MSE keys via Lloyd-Max table + Hadamard.
    cfg = TurboQuantConfig.from_cache_dtype(
        "turboquant_k8v4_spec_nc", head_dim=D
    )
    assert cfg.key_mse_bits == 8 and not cfg.key_fp8 and cfg.norm_correction

    PiT = _build_hadamard(D, device)  # Hadamard is its own transpose
    Pi = PiT  # symmetric

    # Load the cached Beta-fit centroids and derive midpoints.
    centroids = torch.load(CENTROIDS_PATH, map_location=device)["centroids"]
    centroids = centroids.float().contiguous()
    c_sorted, _ = centroids.sort()
    midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).contiguous()

    # Random input K (bf16 � production dtype).
    raw_k = torch.randn(seq_len, Hk, D, device=device, dtype=torch.bfloat16)
    raw_v = torch.randn(seq_len, Hk, D, device=device, dtype=torch.bfloat16)

    num_blocks = (seq_len + block_size - 1) // block_size + 1
    kv_cache = torch.zeros(
        num_blocks, block_size, Hk, cfg.slot_size_aligned,
        device=device, dtype=torch.uint8,
    )
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)

    triton_turboquant_store(
        raw_k.to(torch.float16), raw_v.to(torch.float16),
        kv_cache, slot_mapping,
        PiT, midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
        centroids=c_sorted,
        norm_correction=cfg.norm_correction,
    )

    # Dequant via patched _tq_full_dequant_kv.
    alloc_len = math.ceil(seq_len / block_size) * block_size
    k_buf = torch.empty(1, Hk, alloc_len, D, device=device, dtype=torch.float16)
    v_buf = torch.empty(1, Hk, alloc_len, D, device=device, dtype=torch.float16)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .unsqueeze(0).contiguous()
    )
    mse_bytes = math.ceil(D * cfg.key_mse_bits / 8)
    val_data_bytes = math.ceil(D * cfg.effective_value_quant_bits / 8)
    key_data_bytes = D if cfg.key_fp8 else mse_bytes
    data_bytes_per_slot = key_data_bytes + val_data_bytes
    meta_region_offset = block_size * Hk * data_bytes_per_slot
    num_soa_fields = 2 if cfg.key_fp8 else 3
    soa_v_scale = 0 if cfg.key_fp8 else 1
    soa_v_zero = 1 if cfg.key_fp8 else 2

    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    grid = (alloc_len, 1 * Hk)
    _tq_full_dequant_kv[grid](
        kv_cache, kv_cache_flat_u16(kv_cache),
        block_table, c_sorted,
        k_buf, v_buf,
        k_buf.stride(0), k_buf.stride(1), k_buf.stride(2),
        v_buf.stride(0), v_buf.stride(1), v_buf.stride(2),
        kv_cache.stride(0), block_table.stride(0),
        HEAD_DIM=D, BLOCK_SIZE=block_size, NUM_KV_HEADS=Hk,
        MSE_BYTES=mse_bytes,
        VQB=cfg.effective_value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        MSE_BITS=cfg.key_mse_bits,
        KEY_FP8=1 if cfg.key_fp8 else 0,
        KEY_DATA_BYTES=key_data_bytes,
        META_REGION_OFFSET=meta_region_offset,
        NUM_SOA_FIELDS=num_soa_fields,
        SOA_K_NORM=0,
        SOA_V_SCALE=soa_v_scale,
        SOA_V_ZERO=soa_v_zero,
        BLOCK_D=BLOCK_D,
        NORM_CORRECTION=1 if cfg.norm_correction else 0,
        FP8_E4B15=0,
        num_warps=4,
    )
    torch.cuda.synchronize()

    # _tq_full_dequant_kv returns K in rotated space (no inverse Hadamard
    # applied in-kernel). Apply Pi.T externally to match the offline reference,
    # which produces unrotated K via @ PiT.T at the end of tq84_spec_round_trip.
    k_recon_triton_rot = k_buf[0, :, :seq_len, :].permute(1, 0, 2).float()  # [seq, Hk, D] rotated
    k_recon_triton = k_recon_triton_rot @ PiT.T  # unrotate

    # Offline reference round-trip.
    k_recon_ref = tq84_spec_round_trip(raw_k, PiT, c_sorted)
    k_recon_ref = k_recon_ref.float()

    # (a) Triton vs offline reference.
    diff_off = (k_recon_triton - k_recon_ref).abs()
    max_off = float(diff_off.max())
    mean_off = float(diff_off.mean())
    print(
        f"\nTriton vs offline: max={max_off:.6e}, mean={mean_off:.6e}",
        flush=True,
    )
    assert max_off <= TOL_VS_OFFLINE, (
        f"Triton spec-TQ84 disagrees with offline reference: "
        f"max-abs-diff={max_off:.6e} (tol={TOL_VS_OFFLINE})"
    )

    # (b) Sanity vs raw input.
    diff_raw = (k_recon_triton - raw_k.float()).abs()
    max_raw = float(diff_raw.max())
    mean_raw = float(diff_raw.mean())
    print(
        f"Triton vs raw input: max={max_raw:.6e}, mean={mean_raw:.6e}",
        flush=True,
    )
    assert max_raw < 1.0, (
        f"Triton reconstruction wildly off raw input: max={max_raw:.6e}"
    )
    # And the offline reference itself should have similar magnitudes.
    diff_ref_raw = (k_recon_ref - raw_k.float()).abs()
    print(
        f"Offline vs raw input:  max={float(diff_ref_raw.max()):.6e}, "
        f"mean={float(diff_ref_raw.mean()):.6e}",
        flush=True,
    )


if __name__ == "__main__":
    test_spec_tq84_round_trip()
    print("PASS")
