"""Regression tests for the 2026-08-25 loop-carry/reduce re-review findings.

F1 (was ACTIVE silent-wrong): the first scf.for result's element type was applied to every
   iter-arg, so an i64 counter carried FIRST "upgraded" a co-carried fp32 accumulator to
   `long`, truncating the float every iteration (order-dependent). Both orders must be exact.
F2 (was pre-existing silent-wrong): the multipass reduce phase loop strides _loop_e over the
   kernel's TOTAL elements; a reduce input smaller than that wraps and is accumulated
   repeatedly (a 256-wide sum in a 512-element kernel counted every element twice). Now
   correct-or-refuse.
"""

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _cnt_first(inp, out, cnt_out, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    cnt = tl.zeros((), dtype=tl.int64)  # i64 counter FIRST
    acc = tl.zeros((BLOCK,), dtype=tl.float32)  # fp32 accumulator second
    for _ in range(0, STEPS):
        cnt += 2
        acc += tl.load(inp + offs)
    tl.store(out + offs, acc)
    tl.store(cnt_out, cnt)


@triton.jit
def _acc_first(inp, out, cnt_out, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)  # fp32 accumulator FIRST
    cnt = tl.zeros((), dtype=tl.int64)  # i64 counter second
    for _ in range(0, STEPS):
        cnt += 2
        acc += tl.load(inp + offs)
    tl.store(out + offs, acc)
    tl.store(cnt_out, cnt)


@requires_mps
@pytest.mark.parametrize("kern", [_cnt_first, _acc_first], ids=["i64-first", "fp32-first"])
def test_mixed_dtype_iter_args_order_independent(kern):
    dev = "mps"
    torch.manual_seed(0)
    x = torch.randn(64, device=dev)
    out = torch.empty(64, device=dev)
    cnt = torch.zeros(1, device=dev, dtype=torch.int64)
    kern[(1,)](x, out, cnt, BLOCK=64, STEPS=3, num_warps=2)
    torch.mps.synchronize()
    a_err = (out - x * 3).abs().max().item()
    assert a_err < 1e-4, f"fp32 accumulator corrupted (typed long?): err {a_err:.2e}"
    assert int(cnt[0]) == 6, f"i64 counter wrong: {int(cnt[0])}"


@triton.jit
def _small_reduce_in_big_kernel(x_ptr, z_ptr, out_ptr, s_ptr, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])  # M*N elements (the kernel's total)
    rz = tl.arange(0, R)
    z = tl.load(z_ptr + rz)  # R < M*N elements
    s = tl.sum(z)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], x * 2.0)
    tl.store(s_ptr, s)


@requires_mps
def test_small_reduce_in_multipass_correct_or_refuse():
    dev = "mps"
    torch.manual_seed(0)
    M, N, R = 16, 32, 256  # total 512, reduce input 256
    x = torch.randn(M, N, device=dev)
    z = torch.randn(R, device=dev)
    out = torch.zeros(M, N, device=dev)
    sv = torch.zeros(1, device=dev)
    try:
        _small_reduce_in_big_kernel[(1,)](x, z, out, sv, M, N, R, num_warps=2)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # refused loudly — safe (the old behavior silently doubled the sum)
    assert abs(float(sv[0]) - float(z.sum())) < 1e-2, "wrapped reduce over-counted"
    assert (out - x * 2.0).abs().max().item() < 1e-4
