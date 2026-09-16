"""VARLEN (packed cu_seqlens) FlashAttention must lower via the GENERIC per-element
path and compute CORRECTLY — it must NOT be routed to the dense [Z,H,N,D] FA template
(which bakes fixed strides and ignores cu_seqlens -> would silent-wrong).

This is a GUARD test: the dense FA detector declines varlen because its packed 3-index +
dynamic cu_seqlens-offset addressing doesn't match the dense 4-index pattern, and the
structural N_CTX resolver finds no scalar-arg bound (the per-batch seqlen is
subi(load(cu[b+1]), load(cu[b]))). If a future change loosens FA detection to fire on
varlen, this test catches the resulting silent-wrong (correctness diverges from the
per-sequence reference) instead of it slipping through.
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
def _varlen_fwd(
    Q,
    K,
    V,
    Out,
    cu_q,
    cu_k,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    H,
    max_seqlen,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    q_start = tl.load(cu_q + off_b)
    q_end = tl.load(cu_q + off_b + 1)
    seqlen_q = q_end - q_start
    k_start = tl.load(cu_k + off_b)
    k_end = tl.load(cu_k + off_b + 1)
    seqlen_k = k_end - k_start
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + (q_start + offs_m)[:, None] * stride_qt + off_h * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    q = q * (1.0 / tl.sqrt(float(HEAD_DIM)))
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, max_seqlen, BLOCK_N):
        kn = start_n + offs_n
        k_ptrs = K + (k_start + kn)[:, None] * stride_kt + off_h * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        valid = kn[None, :] < seqlen_k
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= kn[None, :])
        qk = tl.where(valid, qk, float("-inf"))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = V + (k_start + kn)[:, None] * stride_vt + off_h * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new
    acc = acc / l_i[:, None]
    o_ptrs = Out + (q_start + offs_m)[:, None] * stride_ot + off_h * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


def _ref_varlen(q, k, v, cu, H, D, causal):
    out = torch.zeros_like(q)
    for b in range(len(cu) - 1):
        s, e = int(cu[b]), int(cu[b + 1])
        n = e - s
        for h in range(H):
            sc = (q[s:e, h].float() @ k[s:e, h].float().transpose(-2, -1)) / math.sqrt(D)
            if causal:
                m = torch.tril(torch.ones(n, n, device=q.device, dtype=torch.bool))
                sc = sc.masked_fill(~m, float("-inf"))
            out[s:e, h] = (torch.softmax(sc, -1) @ v[s:e, h].float()).to(out.dtype)
    return out


@requires_mps
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lens", [[48, 32], [64, 16, 48]])
def test_varlen_lowers_generic_and_is_correct(lens, causal):
    dev = "mps"
    torch.manual_seed(0)
    H, D = 2, 64
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
    T = int(cu[-1])
    q = torch.randn(T, H, D, device=dev)
    k = torch.randn(T, H, D, device=dev)
    v = torch.randn(T, H, D, device=dev)
    o = torch.zeros(T, H, D, device=dev)
    BM = BN = 32
    max_seqlen = max(lens)
    grid = (triton.cdiv(max_seqlen, BM), len(lens) * H)
    _varlen_fwd[grid](
        q, k, v, o, cu, cu, *q.stride(), *k.stride(), *v.stride(), *o.stride(), H, max_seqlen, causal, BM, BN, D
    )
    torch.mps.synchronize()
    ref = _ref_varlen(q, k, v, cu, H, D, causal)
    assert (o - ref).abs().max().item() < 1e-2
