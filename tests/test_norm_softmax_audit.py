"""Regression (audit #294): three silent-wrongs on the softmax/layer-norm fast-template
detectors, found by the reduce/norm audit sweep and fixed in _detect_softmax /
_detect_layer_norm (+ the LN template eps):

  F1  LayerNorm template hardcoded eps=1e-6, ignoring the kernel's rsqrt(var+eps).
  F2  softmax/LN picked input/output ptr by DECLARATION ORDER -> an output-first
      signature (out, x, n) swapped the buffers.
  F3  detectors matched op-PRESENCE, not a validated cone -> a constexpr *scale
      folded into the input (softmax(x*scale)) was silently dropped.

Each kernel below computes correctly now (via the fixed fast template OR, for the
scaled case, by falling through to the generic lowering).
"""

import math
import pytest
import torch
import triton
import triton.language as tl

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _ln_eps(x_ptr, o_ptr, N, BLOCK: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(m, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    tl.store(o_ptr + row * N + cols, xc * tl.rsqrt(var + EPS), mask=m)


@triton.jit
def _softmax_outfirst(o_ptr, x_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(o_ptr + row * N + cols, e / tl.sum(e, axis=0), mask=m)


@triton.jit
def _softmax_scaled(x_ptr, o_ptr, N, BLOCK: tl.constexpr, SCALE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=-float("inf")) * SCALE
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(o_ptr + row * N + cols, e / tl.sum(e, axis=0), mask=m)


@requires_mps
@pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-3])
def test_layernorm_uses_kernel_eps(eps):
    """F1: the LN fast template must use the kernel's eps, not a hardcoded 1e-6."""
    dev = "mps"
    torch.manual_seed(0)
    R, N = 8, 64
    x = torch.randn(R, N, device=dev)
    o = torch.zeros(R, N, device=dev)
    _ln_eps[(R,)](x, o, N, 64, eps)
    torch.mps.synchronize()
    ref = torch.nn.functional.layer_norm(x, (N,), eps=eps)
    assert (o - ref).abs().max().item() < 1e-4


@requires_mps
def test_softmax_output_first_signature():
    """F2: a (out_ptr, x_ptr, n) output-first softmax must not swap the buffers."""
    dev = "mps"
    torch.manual_seed(1)
    R, N = 8, 64
    x = torch.randn(R, N, device=dev)
    o = torch.zeros(R, N, device=dev)
    _softmax_outfirst[(R,)](o, x, N, 64)
    torch.mps.synchronize()
    assert (o - torch.softmax(x, dim=-1)).abs().max().item() < 1e-4


@requires_mps
@pytest.mark.parametrize("scale", [3.0, 0.5])
def test_softmax_constexpr_scale_not_dropped(scale):
    """F3: softmax(x * SCALE) with a constexpr scale must not drop the scale."""
    dev = "mps"
    torch.manual_seed(2)
    R, N = 8, 64
    x = torch.randn(R, N, device=dev)
    o = torch.zeros(R, N, device=dev)
    _softmax_scaled[(R,)](x, o, N, 64, scale)
    torch.mps.synchronize()
    assert (o - torch.softmax(x * scale, dim=-1)).abs().max().item() < 1e-4
