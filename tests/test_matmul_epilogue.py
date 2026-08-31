"""Fused matmul + pointwise/broadcast epilogue (#158).

A matmul whose result feeds a pointwise/broadcast epilogue (scale, bias,
activation, chains) must COMPUTE the epilogue — not drop it (pre-#157) and not
refuse (post-#157). Softmax keeps its own path. Unsupported epilogues still
refuse loudly (integrity).
"""

import numpy as np
import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires_metal = pytest.mark.skipif(not HAS, reason="Metal/torch/triton needed")

M = N = K = 32


def _ab():
    a = torch.randn(M, K) * 0.3
    b = torch.randn(K, N) * 0.3
    return a, b


if HAS:

    @triton.jit
    def _mm_scale(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        tl.store(C + om[:, None] * N + on[None, :], acc * 3.0 + 1.0)

    @triton.jit
    def _mm_relu(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        tl.store(C + om[:, None] * N + on[None, :], tl.maximum(acc, 0.0))

    @triton.jit
    def _mm_bias_relu(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        bias = tl.load(Bias + on)  # (N,)
        acc = acc + bias[None, :]
        tl.store(C + om[:, None] * N + on[None, :], tl.maximum(acc, 0.0))


@requires_metal
def test_matmul_scale_bias_const():
    a, b = _ab()
    c = torch.zeros(M, N)
    _mm_scale[(1,)](a, b, c, M=M, N=N, K=K)
    np.testing.assert_allclose(c.numpy(), (a @ b).numpy() * 3.0 + 1.0, atol=2e-2, rtol=2e-2)


@requires_metal
def test_matmul_relu():
    a, b = _ab()
    c = torch.zeros(M, N)
    _mm_relu[(1,)](a, b, c, M=M, N=N, K=K)
    np.testing.assert_allclose(c.numpy(), np.maximum((a @ b).numpy(), 0.0), atol=2e-2, rtol=2e-2)


@requires_metal
def test_matmul_bias_relu_linear_layer():
    a, b = _ab()
    bias = torch.randn(N) * 0.3
    c = torch.zeros(M, N)
    _mm_bias_relu[(1,)](a, b, bias, c, M=M, N=N, K=K)
    ref = np.maximum((a @ b).numpy() + bias.numpy()[None, :], 0.0)
    np.testing.assert_allclose(c.numpy(), ref, atol=2e-2, rtol=2e-2)


if HAS:

    @triton.jit
    def _mm_chain(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        # chained: scale, exp-ish (bounded), clamp
        acc = tl.maximum(acc * 2.0 - 0.5, 0.0)
        acc = tl.minimum(acc, 5.0)
        tl.store(C + om[:, None] * N + on[None, :], acc)

    @triton.jit
    def _mm_rowreduce(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        acc = tl.dot(a, b)
        acc = acc - tl.sum(acc, axis=1)[:, None]  # reduce, NOT softmax
        tl.store(C + om[:, None] * N + on[None, :], acc)


@requires_metal
def test_matmul_chained_pointwise():
    a, b = _ab()
    c = torch.zeros(M, N)
    _mm_chain[(1,)](a, b, c, M=M, N=N, K=K)
    mm = (a @ b).numpy()
    ref = np.minimum(np.maximum(mm * 2.0 - 0.5, 0.0), 5.0)
    np.testing.assert_allclose(c.numpy(), ref, atol=2e-2, rtol=2e-2)


@requires_metal
def test_matmul_rowreduce_epilogue_computes():
    # matmul + row-reduce-subtract epilogue (acc - sum(acc, axis=1)). The fused
    # epilogue template can't represent the reduce, so this used to REFUSE loudly:
    # routing matmul-with-epilogue kernels to the generic lowerer had been tried and
    # reverted because the combination fuzzer showed it correct at some shapes and
    # grossly wrong at others (re-audit #6).
    #
    # CONVERTED 2026-08-30 (dot-recovery stage 1a): re-audit #6's failures were
    # re-measured and are NON-UNIFORM shapes (M32xN64 etc.). This kernel is uniform
    # 32x32x32 and non-looped — inside the generic dot path's proven envelope — so it
    # now computes, reduce epilogue included. The old boundary still holds outside the
    # envelope (test_dot_epilogue_generic.py::test_nonuniform_fused_accumulator_still_refuses).
    torch.manual_seed(5)
    a = torch.randn(M, K, device="mps") * 0.3
    b = torch.randn(K, N, device="mps") * 0.3
    c = torch.zeros(M, N, device="mps")
    _mm_rowreduce[(1,)](a, b, c, M=M, N=N, K=K)
    torch.mps.synchronize()
    base = a @ b
    torch.testing.assert_close(c, base - base.sum(dim=1)[:, None], rtol=1e-4, atol=1e-4)
    assert (c - base).abs().max().item() > 1e-2, "reduce epilogue dropped — silent-wrong"


if HAS:

    @triton.jit
    def _mm_scale_runtime_arg(A, B, C, alpha, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        a = tl.load(A + om[:, None] * K + ok[None, :])
        b = tl.load(B + ok[:, None] * N + on[None, :])
        tl.store(C + om[:, None] * N + on[None, :], tl.dot(a, b) * alpha)


@requires_metal
def test_matmul_runtime_scalar_epilogue_computes():
    # An epilogue that scales by a RUNTIME scalar arg (a kernel-arg splat leaf) the
    # fused template can't lower: it used to REFUSE loudly rather than silently resolve
    # the scalar to 0 (-> wrong output). CONVERTED 2026-08-30 (dot-recovery stage 1a):
    # inside the generic dot path's proven envelope the generic lowerer resolves the
    # splat scalar natively, so the kernel computes. The alpha-as-zero silent-wrong the
    # refusal guarded is asserted against explicitly.
    torch.manual_seed(5)
    a = torch.randn(M, K, device="mps") * 0.3
    b = torch.randn(K, N, device="mps") * 0.3
    c = torch.zeros(M, N, device="mps")
    _mm_scale_runtime_arg[(1,)](a, b, c, 2.5, M=M, N=N, K=K)
    torch.mps.synchronize()
    torch.testing.assert_close(c, (a @ b) * 2.5, rtol=1e-4, atol=1e-4)
    assert c.abs().max().item() > 1e-2, "alpha resolved to 0 — the silent-wrong is back"


if HAS:

    @triton.jit
    def _mm_kloop_fma(
        A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
    ):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            acc += tl.dot(
                tl.load(A + rm[:, None] * sam + kk[None, :] * sak), tl.load(B + kk[:, None] * sbk + rn[None, :] * sbn)
            )
        tl.store(C + rm[:, None] * scm + rn[None, :] * scn, tl.math.fma(acc, 2.0, 1.0))


@requires_metal
def test_matmul_kloop_epilogue_refuses_not_dropped():
    # re-audit #6: a LOOPED matmul (dot inside scf.for) with a trailing epilogue was
    # claimed by the inline-dot template, which stored the raw accumulator and SILENTLY
    # DROPPED the fma (returned the bare dot, maxerr ~16-33). The fused-epilogue
    # template only follows a DIRECT dot, so the looped case can't be emitted — it must
    # REFUSE loudly rather than drop the epilogue (generic routing proved unreliable).
    from triton_msl.errors import MetalNonRecoverableError

    a, b = _ab()
    c = torch.zeros(M, N)
    with pytest.raises(MetalNonRecoverableError):
        _mm_kloop_fma[(1, 1)](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BM=M,
            BN=N,
            BK=K,
        )
