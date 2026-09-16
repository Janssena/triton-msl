"""Regression tests for the 2026-08 dual-lens audit fixes (correct-or-refuse, never
silent-wrong). Covers the two CONFIRMED silent-wrongs + the hardening residuals:

  1. Biased/triangle FA drops a within-tile output store mask  -> now REFUSES.
  2. int4 GEMV non-standard nibble packing (high-first / double-swap) routed + garbage
     -> now REFUSES (full loop-k pinning: byte//2, nibble%2, activation x[k] share k).
  3. Structural N_CTX picked a unique range-compared scalar that isn't the seq-len
     -> now REFUSES (cross-check against the scf.for KV-loop bound).
  4. backward dK/dV head_dim too large to subtile -> clean MetalNonRecoverableError
     (parity with the dQ/dbias branches), not a bare ValueError.
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


# ------------------------- 1. biased-FA store mask --------------------------
@triton.jit
def _biased_tighter_store(
    o_ptr,
    o_sz,
    o_sh,
    o_sm,
    o_sk,
    lse_ptr,
    lse_sz,
    lse_sh,
    lse_sm,
    q_ptr,
    q_sz,
    q_sh,
    q_sm,
    q_sk,
    k_ptr,
    k_sz,
    k_sh,
    k_sn,
    k_sk,
    v_ptr,
    v_sz,
    v_sh,
    v_sn,
    v_sk,
    b_ptr,
    b_sz,
    b_sh,
    b_sm,
    b_sn,
    sm_scale,
    Z,
    H,
    N,
    DIM: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    inv_ln2: tl.constexpr = 1.4426950408889634
    ln2: tl.constexpr = 0.6931471824645996
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    mi = pid_m * BM + tl.arange(0, BM)
    ni = tl.arange(0, BN)
    di = tl.arange(0, DIM)
    qp = q_ptr + z * q_sz + h * q_sh + mi[:, None] * q_sm + di[None, :] * q_sk
    kp = k_ptr + z * k_sz + h * k_sh + di[:, None] * k_sk + ni[None, :] * k_sn
    vp = v_ptr + z * v_sz + h * v_sh + ni[:, None] * v_sn + di[None, :] * v_sk
    bp = b_ptr + z * b_sz + h * b_sh + mi[:, None] * b_sm + ni[None, :] * b_sn
    op = o_ptr + z * o_sz + h * o_sh + mi[:, None] * o_sm + di[None, :] * o_sk
    lp = lse_ptr + z * lse_sz + h * lse_sh + mi * lse_sm
    smax = tl.full([BM], value=-float("inf"), dtype=tl.float32)
    den = tl.full([BM], value=0, dtype=tl.float32)
    acc = tl.full([BM, DIM], value=0, dtype=tl.float32)
    mm = mi < N
    q = tl.load(qp, mm[:, None])
    q = q * tl.full([1], value=sm_scale, dtype=q.type.element_ty)
    for sn in tl.range(0, N, BN):
        mn = (ni + sn) < N
        kt = tl.load(kp, mn[None, :])
        bb = tl.load(bp, mm[:, None] & mn[None, :])
        s = bb.to(tl.float32)
        s = tl.dot(q, kt, s)
        s *= inv_ln2
        s = tl.where(mm[:, None] & mn[None, :], s, -float("inf"))
        bmax = tl.maximum(smax, tl.max(s, 1))
        s = s - bmax[:, None]
        es = tl.math.exp2(s)
        summ = tl.sum(es, 1)
        esc = tl.math.exp2(smax - bmax)
        den = den * esc + summ
        acc = acc * esc[:, None]
        vb = tl.load(vp, mn[:, None])
        acc = tl.dot(es.to(q.type.element_ty), vb, acc)
        smax = bmax
        kp += BN * k_sn
        vp += BN * v_sn
        bp += BN * b_sn
    tl.store(op, (acc / den[:, None]).to(q.type.element_ty), mask=mm[:, None] & (mi < 16)[:, None])
    tl.store(lp, (smax * ln2) + tl.log(den), mask=mm)


@requires_mps
def test_biased_fa_tighter_output_store_mask_refuses():
    dev = "mps"
    Z, H, N, DIM = 1, 2, 32, 64
    torch.manual_seed(0)
    sm = 1.0 / math.sqrt(DIM)
    q = torch.randn(Z, H, N, DIM, device=dev)
    k = torch.randn(Z, H, N, DIM, device=dev)
    v = torch.randn(Z, H, N, DIM, device=dev)
    b = torch.randn(Z, H, N, N, device=dev)
    o = torch.zeros(Z, H, N, DIM, device=dev)
    lse = torch.zeros(Z, H, N, device=dev)
    st = lambda t: t.stride()
    with pytest.raises(MetalNonRecoverableError):
        _biased_tighter_store[(triton.cdiv(N, 32), Z * H)](
            o, *st(o), lse, *st(lse), q, *st(q), k, *st(k), v, *st(v), b, *st(b), sm, Z, H, N, DIM, 32, 32
        )


# ------------------------- 2. int4 GEMV nibble pinning -----------------------
@triton.jit
def _int4_gemv_highfirst(
    x_ptr, w_ptr, o_ptr, s_ptr, z_ptr, N, K, ng, swn, ssn, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr
):
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + ok
        packed = tl.load(w_ptr + on[:, None] * swn + (kk // 2)[None, :])
        w4 = (packed >> (((kk + 1) % 2) * 4)[None, :]) & 0xF  # HIGH nibble for even k
        g = kk // G
        s = tl.load(s_ptr + on[:, None] * ssn + g[None, :])
        z = tl.load(z_ptr + on[:, None] * ssn + g[None, :])
        acc += tl.sum(tl.load(x_ptr + kk)[None, :] * ((w4.to(tl.float32) - z) * s), axis=1)
    tl.store(o_ptr + on, acc)


@triton.jit
def _int4_gemv_doubleswap(
    x_ptr, w_ptr, o_ptr, s_ptr, z_ptr, N, K, ng, swn, ssn, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr
):
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + ok
        kk2 = kk + 1
        packed = tl.load(w_ptr + on[:, None] * swn + (kk2 // 2)[None, :])  # byte (k+1)//2
        w4 = (packed >> ((kk2 % 2) * 4)[None, :]) & 0xF  # nibble (k+1)%2
        g = kk // G
        s = tl.load(s_ptr + on[:, None] * ssn + g[None, :])
        z = tl.load(z_ptr + on[:, None] * ssn + g[None, :])
        acc += tl.sum(tl.load(x_ptr + kk)[None, :] * ((w4.to(tl.float32) - z) * s), axis=1)
    tl.store(o_ptr + on, acc)


def _run_int4_gemv(kernel):
    dev = "mps"
    N, K, G = 128, 256, 128
    ng = K // G
    torch.manual_seed(0)
    w4 = torch.randint(0, 16, (N, K), device=dev, dtype=torch.int32)
    packed = ((w4[:, 0::2] << 4) | w4[:, 1::2]).to(torch.uint8).contiguous()
    x = torch.randn(K, device=dev)
    s = (torch.rand(N, ng, device=dev) * 0.02 + 0.005).contiguous()
    z = torch.randint(0, 16, (N, ng), device=dev).float().contiguous()
    o = torch.zeros(N, device=dev)
    kernel[(triton.cdiv(N, 32),)](x, packed, o, s, z, N, K, ng, packed.stride(0), s.stride(0), BN=32, BK=64, G=G)
    torch.mps.synchronize()


@requires_mps
@pytest.mark.parametrize("kernel", [_int4_gemv_highfirst, _int4_gemv_doubleswap])
def test_int4_gemv_nonstandard_packing_refuses(kernel):
    # Non-standard nibble/byte packing must NOT route to make_int4_gemv (low-nibble-even)
    # and silently dequantize the wrong nibble; the loop-k pinning refuses it.
    with pytest.raises(MetalNonRecoverableError):
        _run_int4_gemv(kernel)


# ------------------------- 3. N_CTX cross-check -----------------------------
@triton.jit
def _fa_nonlength_scalar(
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
    H,
    seqlen,
    valid_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    # ONLY valid_len is compared to a tile index; seqlen drives the loop (divisibility).
    q = tl.load(
        Q + off_z * sqz + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqk,
        mask=offs_m[:, None] < valid_len,
        other=0.0,
    )
    q = q * (1.0 / tl.sqrt(float(HEAD_DIM)))
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, seqlen, BLOCK_N):
        k = tl.load(K + off_z * skz + off_h * skh + (start_n + offs_n)[:, None] * skn + offs_d[None, :] * skk)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + off_z * svz + off_h * svh + (start_n + offs_n)[:, None] * svn + offs_d[None, :] * svk)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(
        Out + off_z * soz + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sok,
        acc.to(Out.dtype.element_ty),
        mask=offs_m[:, None] < valid_len,
    )


@requires_mps
def test_structural_nctx_non_length_scalar_refuses():
    # valid_len (not the loop bound) is the unique range-compared scalar; the scf.for
    # cross-check rejects it rather than mis-using it as N_CTX. Must not silent-wrong.
    dev = "mps"
    Z, H, N, D = 1, 2, 64, 128
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, D, device=dev)
    k = torch.randn(Z, H, N, D, device=dev)
    v = torch.randn(Z, H, N, D, device=dev)
    o = torch.empty_like(q)
    st = lambda t: t.stride()
    with pytest.raises(MetalNonRecoverableError):
        _fa_nonlength_scalar[(triton.cdiv(N, 32), Z * H)](
            q, k, v, o, *st(q), *st(k), *st(v), *st(o), Z, H, N, N, 32, 32, D
        )


# ------------------------- 4. bwd_kv error-type parity ----------------------
@requires_mps
def test_bwd_kv_head_dim256_refuses_cleanly():
    # head_dim=256 can't be k-subtiled within Metal's limits -> the kv branch must raise
    # MetalNonRecoverableError (CPU fallback), NOT a bare ValueError, matching q/b.
    from .test_fa_bwd_routing import _bwd_kv, _reference

    dev = "mps"
    Hc, Hh, I, N, DIM = 1, 1, 1, 32, 256
    BJ = BK = 32
    NEG = -1e9
    r = _reference(Hc, Hh, I, N, DIM)
    st = lambda t: tuple(t.stride())
    dk = torch.zeros(Hc, I, N, DIM, device=dev)
    dv = torch.zeros(Hc, I, N, DIM, device=dev)
    q, k, v, bias, mask, do, lse, delta, sm = (
        r["q"],
        r["k"],
        r["v"],
        r["bias"],
        r["mask"],
        r["do"],
        r["lse"],
        r["delta"],
        r["sm"],
    )
    with pytest.raises(MetalNonRecoverableError):
        _bwd_kv[(triton.cdiv(N, BK), I, Hc)](
            delta,
            *st(delta),
            q,
            *st(q),
            k,
            *st(k),
            v,
            *st(v),
            bias,
            *st(bias),
            lse,
            *st(lse),
            mask,
            *st(mask),
            do,
            *st(do),
            dk,
            *st(dk),
            dv,
            *st(dv),
            sm,
            NEG,
            N,
            Hh,
            DIM,
            N,
            BJ,
            BK,
        )
