"""Spec-TQ84 round-trip test at PRODUCTION SCALE.

Extends test_spec_tq84_round_trip.py (which validates seq_len=32 only) to
{32, 1K, 8K, 32K, 128K} so we can ask:

  - Does max-abs diff (Triton vs offline reference) grow with seq_len?
  - Do NaN or Inf appear in the reconstructed K at large N?
  - Does per-token fp16 norm overflow rate (> 65504) grow with seq_len?

The kernel-correctness-at-32 test passing while the 128K-sweep crashes is the
trigger for this diagnostic. If max-abs explodes or NaN appears past some N,
that points to a scale-dependent kernel bug as the LCB-sweep crash cause.

Each case is gated and printed even on PASS so we capture the trend.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

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


def _run_one(seq_len: int) -> dict:
    """Store random K via triton store, dequant via _tq_full_dequant_kv,
    compare against offline reference. Returns metrics dict.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)

    Hk = 2
    D = 128
    block_size = 16

    cfg = TurboQuantConfig.from_cache_dtype(
        "turboquant_k8v4_spec_nc", head_dim=D
    )
    assert cfg.key_mse_bits == 8 and not cfg.key_fp8 and cfg.norm_correction

    PiT = _build_hadamard(D, device)
    centroids = torch.load(CENTROIDS_PATH, map_location=device)["centroids"]
    c_sorted, _ = centroids.float().contiguous().sort()
    midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).contiguous()

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

    k_recon_triton_rot = k_buf[0, :, :seq_len, :].permute(1, 0, 2).float()
    k_recon_triton = k_recon_triton_rot @ PiT.T

    k_recon_ref = tq84_spec_round_trip(raw_k, PiT, c_sorted).float()

    # Sanity / NaN / Inf scans
    n_nan_triton = int(torch.isnan(k_recon_triton).sum())
    n_inf_triton = int(torch.isinf(k_recon_triton).sum())
    n_nan_ref = int(torch.isnan(k_recon_ref).sum())
    n_inf_ref = int(torch.isinf(k_recon_ref).sum())

    # fp16 per-token norm overflow (the codec stores norm as fp16 internally).
    per_token_norms = raw_k.float().norm(dim=-1)  # [seq, Hk]
    fp16_overflow_count = int((per_token_norms > 65504.0).sum())
    fp16_overflow_rate = fp16_overflow_count / per_token_norms.numel()
    max_per_token_norm = float(per_token_norms.max())

    # Triton vs offline
    diff_off = (k_recon_triton - k_recon_ref).abs()
    max_off = float(diff_off.max())
    mean_off = float(diff_off.mean())

    # Triton vs raw input (sanity bound)
    diff_raw = (k_recon_triton - raw_k.float()).abs()
    max_raw = float(diff_raw.max())
    mean_raw = float(diff_raw.mean())

    diff_ref_raw = (k_recon_ref - raw_k.float()).abs()
    max_ref_raw = float(diff_ref_raw.max())

    return {
        "seq_len": seq_len,
        "max_off": max_off,
        "mean_off": mean_off,
        "max_raw": max_raw,
        "mean_raw": mean_raw,
        "max_ref_raw": max_ref_raw,
        "n_nan_triton": n_nan_triton,
        "n_inf_triton": n_inf_triton,
        "n_nan_ref": n_nan_ref,
        "n_inf_ref": n_inf_ref,
        "max_per_token_norm": max_per_token_norm,
        "fp16_overflow_count": fp16_overflow_count,
        "fp16_overflow_rate": fp16_overflow_rate,
    }


SCALES = [32, 1024, 8192, 32768, 131072]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("seq_len", SCALES)
def test_spec_tq84_round_trip_scale(seq_len: int) -> None:
    m = _run_one(seq_len)
    print(
        f"\n[seq_len={seq_len}] max_off={m['max_off']:.6e} mean_off={m['mean_off']:.6e} "
        f"max_raw={m['max_raw']:.6e} mean_raw={m['mean_raw']:.6e} "
        f"nan_t={m['n_nan_triton']} inf_t={m['n_inf_triton']} "
        f"nan_r={m['n_nan_ref']} inf_r={m['n_inf_ref']} "
        f"fp16_ovf={m['fp16_overflow_count']} max_norm={m['max_per_token_norm']:.4f}",
        flush=True,
    )
    assert m["n_nan_triton"] == 0, f"NaN in Triton reconstruction at seq_len={seq_len}"
    assert m["n_inf_triton"] == 0, f"Inf in Triton reconstruction at seq_len={seq_len}"
    assert m["max_off"] <= TOL_VS_OFFLINE, (
        f"Triton spec-TQ84 disagrees with offline reference at seq_len={seq_len}: "
        f"max-abs-diff={m['max_off']:.6e} (tol={TOL_VS_OFFLINE})"
    )
    assert m["max_raw"] < 1.5, (
        f"Triton reconstruction wildly off raw input at seq_len={seq_len}: "
        f"max={m['max_raw']:.6e}"
    )


if __name__ == "__main__":
    rows = []
    for n in SCALES:
        try:
            rows.append(_run_one(n))
            print(f"[seq_len={n}] {rows[-1]}", flush=True)
        except Exception as e:  # pragma: no cover - diagnostic surface
            print(f"[seq_len={n}] EXCEPTION: {type(e).__name__}: {e}", flush=True)
            rows.append({"seq_len": n, "exception": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY ===")
    print(f"{'seq_len':>10} {'max_off':>12} {'mean_off':>12} {'nan_t':>6} {'inf_t':>6} {'fp16_ovf':>10} {'max_norm':>10}")
    for r in rows:
        if "exception" in r:
            print(f"{r['seq_len']:>10} EXCEPTION: {r['exception']}")
            continue
        print(
            f"{r['seq_len']:>10} {r['max_off']:>12.4e} {r['mean_off']:>12.4e} "
            f"{r['n_nan_triton']:>6d} {r['n_inf_triton']:>6d} "
            f"{r['fp16_overflow_count']:>10d} {r['max_per_token_norm']:>10.4f}"
        )
