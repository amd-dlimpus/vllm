# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Phase 1D per-layer kv_cache_dtype tiering.

Validates that ``CacheConfig.kv_cache_dtype_per_layer`` is correctly
plumbed through the page-size calculator and the per-layer dtype
override path in ``vllm.model_executor.layers.attention.attention``.
The tests do not boot a model (kept cheap so they run on any host).

Run:
  .venv/bin/python -m pytest tests/quantization/test_turboquant_per_layer.py -v
"""

import pytest

from vllm.config.cache import CacheConfig
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)


# Pure-config tests ----------------------------------------------------------


def test_per_layer_dtype_field_default_is_empty():
    """Default config does NOT set per-layer dtype overrides."""
    c = CacheConfig()
    assert c.kv_cache_dtype_per_layer == {}


def test_per_layer_dtype_field_accepts_dict():
    """Accepts {layer_idx_str: cache_dtype_str} mapping."""
    c = CacheConfig(
        cache_dtype="turboquant_4bit_nc",
        kv_cache_dtype_per_layer={
            "0": "turboquant_k8v4",
            "1": "turboquant_k8v4",
            "60": "turboquant_k8v4",
            "61": "turboquant_k8v4",
        },
    )
    assert c.cache_dtype == "turboquant_4bit_nc"
    assert c.kv_cache_dtype_per_layer == {
        "0": "turboquant_k8v4",
        "1": "turboquant_k8v4",
        "60": "turboquant_k8v4",
        "61": "turboquant_k8v4",
    }


# Slot-size invariants used by the platform page-size calc ------------------


@pytest.mark.parametrize("head_dim", [128])
def test_tq84_slot_size_strictly_larger_than_tq44(head_dim):
    """Phase 1D's premise: TQ84 layers need a larger physical block than TQ44.

    The platform's page-size calc must take max() over all per-layer
    presets so allocations are big enough for the largest preset. This
    test pins the absolute relationship so a regression in one of the
    preset definitions fails loudly.
    """
    tq44 = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", head_dim)
    tq84 = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", head_dim)
    assert tq84.slot_size_aligned > tq44.slot_size_aligned
    # At head_dim=128 the absolute sizes are pinned.
    if head_dim == 128:
        assert tq44.slot_size_aligned == 134
        assert tq84.slot_size_aligned == 196


# Page-size calc end-to-end (synthetic ModelConfig) -------------------------


def test_per_layer_override_grows_attn_page_size():
    """When kv_cache_dtype_per_layer puts some layers in a larger preset,
    the platform-level page-size calculation must take max() across all
    presets. We exercise the same logic snippet without booting a real
    platform: TQ84 slot_size > TQ44 slot_size, so a configuration with
    global=TQ44 and per_layer={"0": TQ84} must end up with TQ84-sized
    pages.

    This is a thin wrapper around the same arithmetic as the platform
    code in vllm/platforms/interface.py — we replicate it to assert the
    invariant without spinning up a worker.
    """
    head_dim = 128
    block_size = 16
    num_kv_heads = 8

    tq44_cfg = TurboQuantConfig.from_cache_dtype(
        "turboquant_4bit_nc", head_dim
    )
    tq84_cfg = TurboQuantConfig.from_cache_dtype(
        "turboquant_k8v4", head_dim
    )

    tq_page = block_size * num_kv_heads * tq44_cfg.slot_size_aligned
    per_layer_max_page = max(
        block_size * num_kv_heads * cfg.slot_size_aligned
        for cfg in [tq44_cfg, tq84_cfg]
    )

    # Pure-TQ44 (baseline) page size
    pure_tq44_page = tq_page

    # Mixed (global TQ44, override TQ84 on some layers) — alignment must
    # use max(TQ44, TQ84) = TQ84.
    mixed_page = max(tq_page, per_layer_max_page)

    assert mixed_page > pure_tq44_page
    assert mixed_page == block_size * num_kv_heads * tq84_cfg.slot_size_aligned


# Hash factor coverage (so per_layer override busts compile cache) ----------


def test_per_layer_dtype_changes_compute_hash():
    """Two CacheConfigs that differ only in kv_cache_dtype_per_layer must
    produce different compute_hash values, otherwise the compile cache
    will silently reuse a graph compiled for the wrong layout.
    """
    base = CacheConfig(cache_dtype="turboquant_4bit_nc")
    mixed = CacheConfig(
        cache_dtype="turboquant_4bit_nc",
        kv_cache_dtype_per_layer={"0": "turboquant_k8v4"},
    )
    assert base.compute_hash() != mixed.compute_hash()


# Layer-name resolution helper ----------------------------------------------


def test_resolve_per_layer_helper_with_synthetic_layers():
    """`_resolve_per_layer_cache_dtypes` translates {layer_idx: dtype}
    into {layer_name: dtype} by scanning attention layers in the vllm_config.

    We can't easily build a real VllmConfig in a unit test, so we monkey-
    patch the helpers it depends on and assert the translation logic.
    """
    from vllm.v1.worker.gpu import model_runner as mr

    class _FakeAttnLayer:
        pass

    fake_layers = {
        "model.layers.0.self_attn.attn": _FakeAttnLayer(),
        "model.layers.1.self_attn.attn": _FakeAttnLayer(),
        "model.layers.61.self_attn.attn": _FakeAttnLayer(),
    }

    class _FakeCacheCfg:
        kv_cache_dtype_per_layer = {
            "0": "turboquant_k8v4",
            "61": "turboquant_k8v4",
        }

    class _FakeVllmCfg:
        cache_config = _FakeCacheCfg()

    import vllm.config as vc

    original = vc.get_layers_from_vllm_config

    def _stub(_cfg, _typ):
        return fake_layers

    vc.get_layers_from_vllm_config = _stub  # type: ignore[assignment]
    try:
        result = mr._resolve_per_layer_cache_dtypes(_FakeVllmCfg())  # type: ignore[arg-type]
    finally:
        vc.get_layers_from_vllm_config = original  # type: ignore[assignment]

    assert result == {
        "model.layers.0.self_attn.attn": "turboquant_k8v4",
        "model.layers.61.self_attn.attn": "turboquant_k8v4",
    }
    # Layer 1 is NOT in per-layer override — must not appear in result.
    assert "model.layers.1.self_attn.attn" not in result
