"""188 strengthening: final-store cast capability and mask-before-centering bite.

Kept separate so packet193's reviewed files remain byte-frozen. These tests pin
both lowering and execution; source variance includes all BLOCK padded lanes.
"""
import importlib
from pathlib import Path
import sys

import pytest
import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fa_bwd_routing import _build_lowerer


@triton.jit
def _soft_final_half(X, O, N, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + r * N + c, c < N, other=-float("inf"))
    e = tl.exp(x - tl.max(x, 0))
    y = (e / tl.sum(e, 0)).to(tl.float16)
    tl.store(O + r * N + c, y, c < N)


@triton.jit
def _norm_mask_before_center(X, O, N, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + r * N + c, c < N, other=0.0)
    mean = tl.sum(x, 0) / N
    centered = tl.where(c < N, x, 0.0) - mean
    var = tl.sum(centered * centered, 0) / N
    y = centered * tl.rsqrt(var + 1.0e-5)
    tl.store(O + r * N + c, y, c < N)


@pytest.fixture
def routes(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    for fn in (_soft_final_half, _norm_mask_before_center):
        fn.device_caches.clear()
    cls = importlib.import_module("triton_msl.codegen.generic_lowerer").GenericLowerer
    hits = []
    for name in ("_lower_softmax_template", "_lower_layer_norm_template"):
        real = getattr(cls, name)

        def spy(self, *args, _name=name, _real=real, **kwargs):
            hits.append(_name)
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(cls, name, spy)
    return hits


@pytest.mark.parametrize("boundary", ["lowering", "gpu"])
def test_softmax_final_half_store_must_route(routes, boundary):
    if boundary == "lowering":
        text = _build_lowerer(_soft_final_half, {"X": "*fp32", "O": "*fp16", "N": "i32"}, {"BLOCK": 64}).lower()
        assert "kernel void" in text
    else:
        if not torch.backends.mps.is_available():
            pytest.skip("Metal GPU required")
        torch.manual_seed(188)
        cpu = torch.randn(8, 47)
        x = cpu.to("mps")
        out = torch.full((8, 47), float("nan"), dtype=torch.float16, device="mps")
        _soft_final_half[(8,)](x, out, 47, 64)
        torch.mps.synchronize()
        torch.testing.assert_close(out.cpu(), torch.softmax(cpu, dim=-1).half(), rtol=0, atol=0)
    assert routes == ["_lower_softmax_template"]


@pytest.mark.parametrize("boundary", ["lowering", "gpu"])
def test_layernorm_mask_before_center_source_math_must_compute(routes, boundary):
    if boundary == "lowering":
        text = _build_lowerer(_norm_mask_before_center, {"X": "*fp32", "O": "*fp32", "N": "i32"}, {"BLOCK": 64}).lower()
        assert "kernel void" in text
        # A safe generic fallback may compute this source, but the canonical
        # template must never discard the nonzero centered padding.
        assert not routes
        return
    if not torch.backends.mps.is_available():
        pytest.skip("Metal GPU required")
    torch.manual_seed(188)
    cpu = torch.randn(8, 47) * 3 + 2
    x = cpu.to("mps")
    out = torch.full_like(x, float("nan"))
    _norm_mask_before_center[(8,)](x, out, 47, 64)
    torch.mps.synchronize()
    mean = cpu.sum(-1, keepdim=True) / 47
    centered = cpu - mean
    variance = (centered.square().sum(-1, keepdim=True) + (64 - 47) * mean.square()) / 47
    reference = centered * torch.rsqrt(variance + 1e-5)
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-6, atol=1e-6)
    assert not routes
