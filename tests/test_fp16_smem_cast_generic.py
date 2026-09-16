"""Regression: an fp16 dtype cast applied to a SHARED-MEMORY-backed value must lower to
valid MSL, compute correctly, and still quantize.

Background (2026-08-26). ``_lower_truncf``'s scalar branch round-trips a value through the
narrow type (``static_cast<float>(static_cast<half>(x))``) so that a mid-computation
``.to(tl.float16)`` actually loses precision. But an oversized tile / loop-carried
accumulator lives in a threadgroup array, and ``_lookup`` resolves it to the bare ARRAY
NAME — so that branch emitted ``static_cast<half>(smem_iter_N)``, casting a
``threadgroup float*`` to a scalar, which Metal rejects:

    error: static_cast from 'threadgroup float *' to 'half' is not allowed

That reached any fp16 varlen-shaped FA kernel that fell to the generic path (originally
reported only as an H=1 side finding; reproduced here at H=4 through the GQA guard, and
independently at H=2 through a shifted-cu_seqlens kernel). fp32 was unaffected (no cast)
and bf16 is refused earlier by the dense-FA dtype gate.

The fix quantizes the threadgroup array cooperatively in place and re-registers the result
as shared-memory-backed, mirroring the binary elementwise path. These tests pin all three
properties: valid MSL, correct numerics, and preserved quantization.
"""

import math

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
def _gqa_varlen_generic(
    Q,
    K,
    V,
    Out,
    OutFP32,
    cu_q,
    cu_k,
    sqt,
    sqh,
    sqd,
    skt,
    skh,
    skd,
    svt,
    svh,
    svd,
    sot,
    soh,
    sod,
    H,
    GROUP,
    max_seqlen,
    SCALE: tl.constexpr,
    STORE_FP32: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    """Varlen FA whose K/V use a GQA head (``h // GROUP``), so the varlen detector's
    head-parity guard refuses the specialized route and the kernel lowers generically —
    the path that emitted invalid MSL for an fp16 output."""
    sm = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    hkv = h // GROUP
    qs = tl.load(cu_q + b)
    slq = tl.load(cu_q + b + 1) - qs
    ks = tl.load(cu_k + b)
    slk = tl.load(cu_k + b + 1) - ks
    om = sm * BM + tl.arange(0, BM)
    on = tl.arange(0, BN)
    od = tl.arange(0, D)
    q = tl.load(Q + (qs + om)[:, None] * sqt + h * sqh + od[None, :] * sqd, mask=om[:, None] < slq, other=0.0) * SCALE
    mi = tl.full([BM], float("-inf"), tl.float32)
    li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for sn in range(0, max_seqlen, BN):
        kn = sn + on
        k = tl.load(K + (ks + kn)[:, None] * skt + hkv * skh + od[None, :] * skd, mask=kn[:, None] < slk, other=0.0)
        qk = tl.where(kn[None, :] < slk, tl.dot(q, tl.trans(k).to(q.dtype)), float("-inf"))
        m2 = tl.maximum(mi, tl.max(qk, 1))
        a = tl.exp(mi - m2)
        p = tl.exp(qk - m2[:, None])
        li = li * a + tl.sum(p, 1)
        acc = acc * a[:, None]
        vv = tl.load(V + (ks + kn)[:, None] * svt + hkv * svh + od[None, :] * svd, mask=kn[:, None] < slk, other=0.0)
        acc += tl.dot(p.to(tl.float32), vv.to(tl.float32))
        mi = m2
    result = acc / li[:, None]
    tl.store(
        Out + (qs + om)[:, None] * sot + h * soh + od[None, :] * sod,
        result.to(Out.dtype.element_ty),
        mask=om[:, None] < slq,
    )
    if STORE_FP32:
        # A second consumer of `result` pins SSA semantics for the shared-memory cast:
        # producing the fp16 output must not quantize this original fp32 value in place.
        tl.store(OutFP32 + (qs + om)[:, None] * sot + h * soh + od[None, :] * sod, result, mask=om[:, None] < slq)


def _run_generic_varlen(dtype, lens, D=64, nw=4, H=4, Hkv=2):
    dev = "mps"
    torch.manual_seed(0)
    group = H // Hkv
    cu = torch.tensor([0] + list(torch.tensor(list(lens)).cumsum(0)), device=dev, dtype=torch.int32)
    total = int(cu[-1])
    q = torch.randn(total, H, D, device=dev, dtype=dtype)
    k = torch.randn(total, Hkv, D, device=dev, dtype=dtype)
    v = torch.randn(total, Hkv, D, device=dev, dtype=dtype)
    o = torch.zeros(total, H, D, device=dev, dtype=dtype)
    o_fp32 = torch.zeros(total, H, D, device=dev, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    mx = max(lens)
    _gqa_varlen_generic[(triton.cdiv(mx, 32), len(lens) * H)](
        q,
        k,
        v,
        o,
        o_fp32,
        cu,
        cu,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        H,
        group,
        mx,
        scale,
        False,
        32,
        32,
        D,
        num_warps=nw,
    )
    torch.mps.synchronize()
    ref = torch.zeros_like(o)
    for b in range(len(lens)):
        s, e = int(cu[b]), int(cu[b + 1])
        for h in range(H):
            hk = h // group
            sc = (q[s:e, h].float() * scale) @ k[s:e, hk].float().T
            ref[s:e, h] = (torch.softmax(sc, -1) @ v[s:e, hk].float()).to(ref.dtype)
    return (o.float() - ref.float()).abs().max().item()


@requires_mps
@pytest.mark.parametrize("nw", [4, 8])
@pytest.mark.parametrize("lens", [(48, 32), (64, 16, 33)])
def test_fp16_smem_cast_lowers_and_is_correct(nw, lens):
    # Previously: MetalCompilationError (invalid MSL). Now: valid MSL AND correct.
    err = _run_generic_varlen(torch.float16, lens, nw=nw)
    assert err < 2e-2, f"fp16 generic varlen wrong after the smem-cast fix: err {err:.2e}"


@requires_mps
def test_fp32_generic_varlen_unchanged():
    # fp32 has no narrowing cast and must be untouched by the fix.
    err = _run_generic_varlen(torch.float32, (48, 32))
    assert err < 1e-3, f"fp32 generic varlen regressed: err {err:.2e}"


@requires_mps
def test_bf16_generic_varlen_stays_correct_or_refuses():
    # bf16 is refused earlier by the dense-FA dtype gate; if that ever changes, it must
    # still be correct rather than silently wrong.
    try:
        err = _run_generic_varlen(torch.bfloat16, (48, 32))
    except MetalNonRecoverableError:
        return
    assert err < 6e-2, f"bf16 generic varlen wrong: err {err:.2e}"


@triton.jit
def _smem_acc_quantize(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, STEPS: tl.constexpr):
    """Loop-carried (shared-memory-backed) accumulator, cast to fp16 MID-COMPUTATION and
    then used in further arithmetic — so a passthrough would silently skip quantization."""
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for _ in range(0, STEPS):
        acc += tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    q = acc.to(tl.float16).to(tl.float32) + 0.0
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], q)


@requires_mps
def test_smem_cast_still_quantizes():
    # 2049.0 is not representable in fp16 and must quantize to 2048.0. A passthrough
    # would leave 2049.0 — the 2026-06-22 silent-wrong class this round-trip exists to
    # prevent — so the cooperative in-place fix must preserve it.
    dev = "mps"
    M, N = 16, 32
    x = torch.full((M, N), 2049.0, device=dev, dtype=torch.float32)
    o = torch.zeros(M, N, device=dev, dtype=torch.float32)
    _smem_acc_quantize[(1,)](x, o, M, N, 1, num_warps=1)
    torch.mps.synchronize()
    assert float(o[0, 0]) == 2048.0, f"fp16 cast did not quantize: got {float(o[0, 0])}"


@requires_mps
def test_smem_cast_with_second_source_consumer_refuses():
    # The generic varlen result is shared-memory-backed. Quantizing that array in place
    # makes a later fp32 store observe the rounded fp16 values, violating SSA semantics.
    # Until the lowerer can provide separate shared storage, this shape must refuse.
    dev = "mps"
    H, Hkv, D = 4, 2, 64
    group = H // Hkv
    cu = torch.tensor([0, 48, 80], device=dev, dtype=torch.int32)
    torch.manual_seed(7)
    q = torch.randn(80, H, D, device=dev, dtype=torch.float16)
    k = torch.randn(80, Hkv, D, device=dev, dtype=torch.float16)
    v = torch.randn(80, Hkv, D, device=dev, dtype=torch.float16)
    o = torch.zeros(80, H, D, device=dev, dtype=torch.float16)
    o_fp32 = torch.zeros(80, H, D, device=dev, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    with pytest.raises(MetalNonRecoverableError, match="another consumer"):
        _gqa_varlen_generic[(2, 8)](
            q,
            k,
            v,
            o,
            o_fp32,
            cu,
            cu,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *o.stride(),
            H,
            group,
            48,
            scale,
            True,
            32,
            32,
            D,
            num_warps=4,
        )
