"""Regression: a multi-value ``tl.associative_scan`` must stage EACH slot in its own
dtype.

Background (2026-08-26). ``_lower_scan`` derived one ``(msl_type, shared_dtype)`` from
operand 0 and used it for every slot — threadgroup declarations, staging writes,
accumulator init, rhs loads, the combine body's block-arg types, write-back and result
registration. A mixed-dtype scan therefore truncated every slot but the first (re-audit
#14: an fp32 sum slot staged as i32 came back all zeros).

That was closed on 2026-06-23 (``e3b11bf``) by REFUSING mixed-dtype multi-value scans.
The refusal was correct about the defect but expensive: upstream's ``cummax`` is
``tl.associative_scan((value, index.to(tl.int64)), ...)`` — a legitimate mixed-dtype
scan — so the refusal turned ~316 upstream ``test_scan2d`` conformance tests into
failures, which went unnoticed because the recorded ratchet baseline predated it.

The fix stages per slot, so these now COMPUTE. These tests pin both directions: the
truncation the refusal guarded must stay fixed, and the mixed-dtype scans it refused must
keep working.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _sum_pair(a_val, a_cnt, b_val, b_cnt):
    return a_val + b_val, a_cnt + b_cnt


@triton.jit
def _mixed_scan_kernel(x_ptr, c_ptr, o_val, o_cnt, N: tl.constexpr):
    """fp32 value slot beside an int32 count slot — the exact re-audit #14 shape."""
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    c = tl.load(c_ptr + i)
    s, k = tl.associative_scan((x, c), axis=0, combine_fn=_sum_pair)
    tl.store(o_val + i, s)
    tl.store(o_cnt + i, k)


@requires_mps
@pytest.mark.parametrize("N", [32, 128, 256])
def test_mixed_fp32_i32_scan_truncates_neither_slot(N):
    # 2**24 + 1 is the smallest positive int that float32 cannot represent: staged as
    # fp32 it becomes 2**24. Putting it in the int slot makes a wrong staging dtype a
    # visible off-by-one rather than a rounding argument.
    dev = "mps"
    big = 2**24 + 1
    x = torch.arange(N, device=dev, dtype=torch.float32) * 0.5
    c = torch.zeros(N, device=dev, dtype=torch.int32)
    c[0] = big
    o_val = torch.zeros(N, device=dev, dtype=torch.float32)
    o_cnt = torch.zeros(N, device=dev, dtype=torch.int32)

    _mixed_scan_kernel[(1,)](x, c, o_val, o_cnt, N=N)
    torch.mps.synchronize()

    assert torch.allclose(o_val, torch.cumsum(x, 0), atol=1e-4), (
        f"fp32 slot wrong at N={N} (staged with the int slot's dtype?)")
    got = o_cnt.cpu().numpy()
    assert (got == big).all(), (
        f"int32 slot truncated at N={N}: got {got[0]} want {big} "
        "(staged with the fp32 slot's dtype loses 2**24+1)")


@triton.jit
def _cummax(v0, i0, v1, i1):
    gt = v0 > v1
    return tl.where(gt, v0, v1), tl.where(gt, i0, i1)


@triton.jit
def _cummax_kernel(x_ptr, o_ptr, N: tl.constexpr):
    """The upstream cummax shape: value slot beside an INT64 index slot."""
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    idx = tl.arange(0, N).to(tl.int64)
    _, z = tl.associative_scan((x, idx), axis=0, combine_fn=_cummax)
    tl.store(o_ptr + i, z)


@requires_mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("N", [32, 256])
def test_cummax_int64_index_slot(dtype, N):
    # Previously refused outright ("mixed operand dtypes"), which is what cost the
    # upstream test_scan2d family. Must compute, and the index slot must stay 64-bit.
    dev = "mps"
    rs = np.random.RandomState(17)
    host = rs.randint(-1000, 1000, size=N)
    x = torch.tensor(host, device=dev, dtype=dtype)
    o = torch.zeros(N, device=dev, dtype=torch.int64)

    _cummax_kernel[(1,)](x, o, N=N)
    torch.mps.synchronize()

    ref = torch.cummax(torch.tensor(host), dim=0).indices.numpy()
    assert (o.cpu().numpy() == ref).all(), f"cummax index wrong at N={N} dtype={dtype}"


@requires_mps
def test_same_dtype_multivalue_scan_unchanged():
    # The same-dtype path always worked and must not regress from the per-slot change.
    dev = "mps"
    N = 64
    x = torch.arange(N, device=dev, dtype=torch.float32)
    c = torch.ones(N, device=dev, dtype=torch.float32)
    o_val = torch.zeros(N, device=dev, dtype=torch.float32)
    o_cnt = torch.zeros(N, device=dev, dtype=torch.float32)

    _mixed_scan_kernel[(1,)](x, c, o_val, o_cnt, N=N)
    torch.mps.synchronize()

    assert torch.allclose(o_val, torch.cumsum(x, 0), atol=1e-4)
    assert torch.allclose(o_cnt, torch.cumsum(c, 0), atol=1e-4)
