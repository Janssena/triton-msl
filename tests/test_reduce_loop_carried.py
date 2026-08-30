"""In-loop 2-D axis reduce (GEMV-via-tl.sum): correct-or-refuse.

A 2-D→1-D axis reduce (`tl.sum(x, axis=1)`) produces its result in the
row-broadcast layout (thread ``lid`` holds row ``lid/N``). When that result is
accumulated across a K-loop, the loop-carried accumulator + 1-D store used to
assume one-row-per-thread and silently collapse every output row to the first;
this was refused until loop-carry layout propagation landed.

2026-08-29 (trifast #6b): loop-carry layout propagation LANDED — the layout
resolver follows scf.for carries (including the single-result form) and
convert_layout stages blocked values with the (lid % div) copy — so these
kernels now COMPUTE. The tests assert numeric correctness and explicitly
assert the original collapse-to-row-0 silent-wrong is absent.

The single-tile form (no K-loop) was always correct and must NOT be refused.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal

    from triton_msl.errors import MetalNonRecoverableError

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + triton needed")

if HAS:

    @triton.jit
    def _gemv_loop(x_ptr, w_ptr, o_ptr, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for _ in range(0, K, BK):
            x = tl.load(x_ptr + offs_k)
            w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk)
            acc += tl.sum(x[None, :] * w, axis=1)
            offs_k += BK
        tl.store(o_ptr + offs_n, acc)

    @triton.jit
    def _gemv_single(x_ptr, w_ptr, o_ptr, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        x = tl.load(x_ptr + offs_k)
        w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk)
        tl.store(o_ptr + offs_n, tl.sum(x[None, :] * w, axis=1))

    @triton.jit
    def _gemv_single_iterarg(x_ptr, w_ptr, o_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        # A plain fp32 GEMV with a SINGLE loop iter_arg (`acc`) and a
        # `range(0, K, BK)` induction var (no manual offs_k += BK). A single-result
        # scf.for has result_ids == None, which an earlier version of the guard failed
        # to cross — the same silent-wrong slipped through. It must refuse. (No dequant
        # / no sitofp, so it is NOT routed to the int8 GEMV kernel — it hits the guard.)
        pid = tl.program_id(0)
        offs_n = pid * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k in range(0, K, BK):
            x = tl.load(x_ptr + offs_k + k)
            w = tl.load(w_ptr + offs_n[:, None] * swn + (offs_k[None, :] + k) * swk)
            acc += tl.sum(x[None, :] * w, axis=1)
        tl.store(o_ptr + offs_n, acc)


@requires
def test_inloop_2d_axis_reduce_computes():
    # K > BK forces the K-loop that carries the reduce result. CONVERTED from a
    # refusal pin (2026-08-29, trifast #6b): the loop-carried "blocked" reduce
    # result is now staged with the blocked (lid % div) copy in convert_layout,
    # so the K-loop GEMV computes instead of refusing. The original failure mode
    # this pinned — every output row silently collapsing to row 0 — is asserted
    # against explicitly below.
    torch.manual_seed(0)
    N, K, BK = 32, 64, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    _gemv_loop[(1,)](x, w, o, K, w.stride(0), w.stride(1), BN=N, BK=BK)
    torch.mps.synchronize()
    assert not bool((o[1:] == o[0]).all()), "output collapsed to row 0 — the original silent-wrong"
    torch.testing.assert_close(o, w @ x, rtol=1e-4, atol=1e-4)


@requires
def test_single_iterarg_loop_carried_reduce_computes():
    # Single loop iter_arg (single-result scf.for, result_ids == None) crossing
    # the loop-carry. CONVERTED from a refusal pin (2026-08-29, trifast #6b):
    # the layout resolver and convert-layout carry-walk both special-case the
    # single-result scf.for (whose result id IS the op's own id), so the decode
    # GEMV computes instead of refusing — and must NOT silently collapse.
    torch.manual_seed(0)
    N, K, BN, BK = 128, 256, 32, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    _gemv_single_iterarg[(triton.cdiv(N, BN),)](x, w, o, N, K, w.stride(0), w.stride(1), BN=BN, BK=BK)
    torch.mps.synchronize()
    assert not bool((o[1:] == o[0]).all()), "output collapsed to row 0 — the original silent-wrong"
    torch.testing.assert_close(o, w.float() @ x.float(), rtol=1e-4, atol=1e-4)


@requires
def test_single_tile_2d_axis_reduce_still_correct():
    # No K-loop (BK == K): the reduce result is consumed directly, correctly.
    torch.manual_seed(0)
    N, K = 32, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    _gemv_single[(1,)](x, w, o, w.stride(0), w.stride(1), BN=N, BK=K)
    torch.mps.synchronize()
    torch.testing.assert_close(o, w @ x, rtol=1e-3, atol=1e-3)
