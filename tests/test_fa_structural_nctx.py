"""Structural N_CTX resolution for standard FlashAttention (the trifast-#1
name-heuristic class): a kernel that names its sequence-length arg anything but
``N_CTX`` (``seqlen``, ``L``, ...) still routes. The seq-len is resolved from the
bounds-mask cmpi (``offs < seqlen``), which references N_CTX in BOTH causal and
non-causal kernels — unlike the KV-loop upper bound, which is ``(start_m+1)*BLOCK_M``
under causal and would resolve the WRONG arg (a silent-wrong).

Safety: the resolver must NOT loosen the anti-silent-wrong gates —
  * a TIGHTER store mask than the tile boundary still refuses (the hd128 template
    drops the store mask, so it self-refuses a mask it can't honor), and
  * an AMBIGUOUS seq-len (two distinct args compared to a tile index) refuses
    rather than guessing which is N_CTX.
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
def _fa_renamed(
    Q, K, V, Out,
    sqz, sqh, sqm, sqk, skz, skh, skn, skk, svz, svh, svn, svk, soz, soh, som, sok,
    Z, H, seqlen,                       # N_CTX renamed
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0); off_hz = tl.program_id(1)
    off_z = off_hz // H; off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N); offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + off_z * sqz + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqk,
                mask=offs_m[:, None] < seqlen, other=0.0)
    q = q * (1.0 / tl.sqrt(float(HEAD_DIM)))
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    hi = seqlen
    if IS_CAUSAL:
        hi = min((start_m + 1) * BLOCK_M, seqlen)
    for start_n in range(0, hi, BLOCK_N):
        k = tl.load(K + off_z * skz + off_h * skh + (start_n + offs_n)[:, None] * skn + offs_d[None, :] * skk,
                    mask=(start_n + offs_n)[:, None] < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        m_ij = tl.max(qk, 1); m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new); p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None]
        v = tl.load(V + off_z * svz + off_h * svh + (start_n + offs_n)[:, None] * svn + offs_d[None, :] * svk,
                    mask=(start_n + offs_n)[:, None] < seqlen, other=0.0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32)); m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + off_z * soz + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sok,
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)


def _ref(q, k, v, causal):
    d = q.shape[-1]
    s = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(d)
    if causal:
        n = q.shape[-2]
        m = torch.tril(torch.ones(n, n, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~m, float("-inf"))
    return (torch.softmax(s, -1) @ v.float()).to(q.dtype)


@requires_mps
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("Z,H,N,D", [(1, 2, 64, 128), (2, 2, 96, 128), (1, 4, 128, 64)])
def test_structural_nctx_routes_and_computes(Z, H, N, D, causal):
    dev = "mps"
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, D, device=dev); k = torch.randn(Z, H, N, D, device=dev)
    v = torch.randn(Z, H, N, D, device=dev); o = torch.empty_like(q)
    BM = min(32, N); BN = min(32, N)
    _fa_renamed[(triton.cdiv(N, BM), Z * H)](
        q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        Z, H, N, BM, BN, D, causal)
    torch.mps.synchronize()
    assert (o - _ref(q, k, v, causal)).abs().max().item() < 2e-3


@triton.jit
def _fa_tight_store(
    Q, K, V, Out,
    sqz, sqh, sqm, sqk, skz, skh, skn, skk, svz, svh, svn, svk, soz, soh, som, sok,
    Z, H, seqlen,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0); off_hz = tl.program_id(1)
    off_z = off_hz // H; off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N); offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + off_z * sqz + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqk,
                mask=offs_m[:, None] < seqlen, other=0.0)
    q = q * (1.0 / tl.sqrt(float(HEAD_DIM)))
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, seqlen, BLOCK_N):
        k = tl.load(K + off_z * skz + off_h * skh + (start_n + offs_n)[:, None] * skn + offs_d[None, :] * skk,
                    mask=(start_n + offs_n)[:, None] < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        m_ij = tl.max(qk, 1); m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new); p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None]
        v = tl.load(V + off_z * svz + off_h * svh + (start_n + offs_n)[:, None] * svn + offs_d[None, :] * svk,
                    mask=(start_n + offs_n)[:, None] < seqlen, other=0.0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32)); m_i = m_new
    acc = acc / l_i[:, None]
    om = offs_m[:, None]
    # (om < seqlen) & (om < 16): the const-16 leaf on a multi-block index is not
    # provably trivial -> the mask-dropping hd128 template must refuse.
    tl.store(Out + off_z * soz + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sok,
             acc.to(Out.dtype.element_ty), mask=(om < seqlen) & (om < 16))


@triton.jit
def _fa_ambiguous(
    Q, K, V, Out,
    sqz, sqh, sqm, sqk, skz, skh, skn, skk, svz, svh, svn, svk, soz, soh, som, sok,
    Z, H, seqlen, other_bound,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0); off_hz = tl.program_id(1)
    off_z = off_hz // H; off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N); offs_d = tl.arange(0, HEAD_DIM)
    # two DISTINCT args compared to a tile index -> ambiguous which is N_CTX
    q = tl.load(Q + off_z * sqz + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqk,
                mask=(offs_m[:, None] < seqlen) & (offs_m[:, None] < other_bound), other=0.0)
    q = q * (1.0 / tl.sqrt(float(HEAD_DIM)))
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, seqlen, BLOCK_N):
        k = tl.load(K + off_z * skz + off_h * skh + (start_n + offs_n)[:, None] * skn + offs_d[None, :] * skk,
                    mask=(start_n + offs_n)[:, None] < other_bound, other=0.0)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        m_ij = tl.max(qk, 1); m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new); p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None]
        v = tl.load(V + off_z * svz + off_h * svh + (start_n + offs_n)[:, None] * svn + offs_d[None, :] * svk,
                    mask=(start_n + offs_n)[:, None] < seqlen, other=0.0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32)); m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + off_z * soz + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sok,
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)


def _launch(kernel, extra_scalars):
    Z, H, N, D = 1, 1, 64, 128
    dev = "mps"
    q = torch.randn(Z, H, N, D, device=dev); k = torch.randn(Z, H, N, D, device=dev)
    v = torch.randn(Z, H, N, D, device=dev); o = torch.empty_like(q)
    args = [q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(), Z, H, N] + extra_scalars
    kernel[(N // 32, Z * H)](*args, 32, 32, D)
    torch.mps.synchronize()


@requires_mps
def test_structural_nctx_tighter_store_mask_refuses():
    # The structural resolver must NOT loosen the store-mask gate.
    with pytest.raises(MetalNonRecoverableError):
        _launch(_fa_tight_store, [])


@requires_mps
def test_structural_nctx_ambiguous_refuses():
    # Two distinct seq-len candidates -> refuse rather than guess.
    with pytest.raises(MetalNonRecoverableError):
        _launch(_fa_ambiguous, [64])
