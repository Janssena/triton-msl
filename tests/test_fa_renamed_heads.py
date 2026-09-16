"""Regression (audit #282, Finding 1): a standard (non-biased) FlashAttention kernel
whose HEADS scalar is NOT named literally "H" must still compute correctly for
multi-head. _detect_flash_attention used to resolve H by literal arg NAME and fall
back to H=1 when absent, so a renamed heads arg silently collapsed every head to
head 0 (the trifast-#1 name-heuristic class). H is now resolved STRUCTURALLY from
the batch/head decomposition (off_hz // H), independent of the arg name.
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
def _fa_named_heads_nheads(  # heads scalar is deliberately named NHEADS, not H
    Q,
    K,
    V,
    Out,
    sqz,
    sqh,
    sqm,
    sqk,
    skz,
    skh,
    skn,
    skk,
    svz,
    svh,
    svn,
    svk,
    soz,
    soh,
    som,
    sok,
    Z,
    NHEADS,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // NHEADS
    off_h = off_hz % NHEADS
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        Q + off_z * sqz + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqk,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )
    q = q * (1.0 / tl.sqrt(float(HEAD_DIM)))
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(
            K + off_z * skz + off_h * skh + (start_n + offs_n)[:, None] * skn + offs_d[None, :] * skk,
            mask=(start_n + offs_n)[:, None] < N_CTX,
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            V + off_z * svz + off_h * svh + (start_n + offs_n)[:, None] * svn + offs_d[None, :] * svk,
            mask=(start_n + offs_n)[:, None] < N_CTX,
            other=0.0,
        )
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(
        Out + off_z * soz + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sok,
        acc.to(Out.dtype.element_ty),
        mask=offs_m[:, None] < N_CTX,
    )


@requires_mps
@pytest.mark.parametrize("Z,H,N,D", [(1, 4, 64, 128), (2, 2, 64, 128), (1, 1, 64, 128)])
def test_fa_renamed_heads_multihead_correct(Z, H, N, D):
    dev = "mps"
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, D, device=dev)
    k = torch.randn(Z, H, N, D, device=dev)
    v = torch.randn(Z, H, N, D, device=dev)
    scale = 1.0 / math.sqrt(D)
    ref = torch.softmax((q * scale) @ k.transpose(-2, -1), dim=-1) @ v
    out = torch.zeros_like(q)
    st = lambda t: t.stride()
    _fa_named_heads_nheads[(triton.cdiv(N, 32), Z * H)](
        q, k, v, out, *st(q), *st(k), *st(v), *st(out), Z, H, N, 32, 32, D
    )
    torch.mps.synchronize()
    # Before the structural-H fix this was ~1.17 (heads collapsed to head 0) for H>1.
    assert (out - ref).abs().max().item() < 1e-2
