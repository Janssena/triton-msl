"""Maskless biased attention (ALiBi / T5-style additive bias, NO boolean mask):
scores = sm_scale*Q@Kᵀ + bias, softmax, P@V. The biased-FA detector previously
required exactly one boolean mask load (refused no-mask — correctly, since falling
through to standard FA would DROP the bias). The mask is now OPTIONAL: 0 masks ->
route to the biased template with its mask code gated off; 1 mask -> trifast (as
before); 2+ -> refuse (never fall through). This routes ALiBi and computes correctly
on both the tiled (head_dim 32) and simd (head_dim 64) paths.
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
def _alibi_fa(
    o_ptr, o_sz, o_sh, o_sm, o_sk, lse_ptr, lse_sz, lse_sh, lse_sm,
    q_ptr, q_sz, q_sh, q_sm, q_sk, k_ptr, k_sz, k_sh, k_sn, k_sk,
    v_ptr, v_sz, v_sh, v_sn, v_sk, b_ptr, b_sz, b_sh, b_sm, b_sn,
    sm_scale, Z, H, N, DIM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    inv_ln2: tl.constexpr = 1.4426950408889634
    ln2: tl.constexpr = 0.6931471824645996
    pid_m = tl.program_id(0); pid_zh = tl.program_id(1); z = pid_zh // H; h = pid_zh % H
    mi = pid_m * BM + tl.arange(0, BM); ni = tl.arange(0, BN); di = tl.arange(0, DIM)
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
    q = tl.load(qp, mm[:, None]); q = q * tl.full([1], value=sm_scale, dtype=q.type.element_ty)
    for sn in tl.range(0, N, BN):
        mn = (ni + sn) < N
        kt = tl.load(kp, mn[None, :]); bb = tl.load(bp, mm[:, None] & mn[None, :])
        s = bb.to(tl.float32); s = tl.dot(q, kt, s); s *= inv_ln2
        s = tl.where(mm[:, None] & mn[None, :], s, -float("inf"))
        bmax = tl.maximum(smax, tl.max(s, 1)); s = s - bmax[:, None]; es = tl.math.exp2(s)
        summ = tl.sum(es, 1); esc = tl.math.exp2(smax - bmax); den = den * esc + summ; acc = acc * esc[:, None]
        vb = tl.load(vp, mn[:, None]); acc = tl.dot(es.to(q.type.element_ty), vb, acc)
        smax = bmax; kp += BN * k_sn; vp += BN * v_sn; bp += BN * b_sn
    tl.store(op, (acc / den[:, None]).to(q.type.element_ty), mask=mm[:, None])
    tl.store(lp, (smax * ln2) + tl.log(den), mask=mm)


@requires_mps
@pytest.mark.parametrize("Z,H,N,DIM", [(1, 2, 64, 32), (2, 2, 64, 64), (1, 4, 96, 64)])
def test_alibi_maskless_biased_routes_and_computes(Z, H, N, DIM):
    dev = "mps"
    torch.manual_seed(0)
    sm = 1.0 / math.sqrt(DIM)
    q = torch.randn(Z, H, N, DIM, device=dev)
    k = torch.randn(Z, H, N, DIM, device=dev)
    v = torch.randn(Z, H, N, DIM, device=dev)
    b = torch.randn(Z, H, N, N, device=dev)
    o = torch.zeros(Z, H, N, DIM, device=dev)
    lse = torch.zeros(Z, H, N, device=dev)
    st = lambda t: t.stride()
    _alibi_fa[(triton.cdiv(N, 32), Z * H)](
        o, *st(o), lse, *st(lse), q, *st(q), k, *st(k), v, *st(v), b, *st(b), sm, Z, H, N, DIM, 32, 32)
    torch.mps.synchronize()
    raw = sm * (q @ k.transpose(-2, -1)) + b
    p = torch.softmax(raw, dim=-1)
    assert (o - p @ v).abs().max().item() < 1e-3
    assert (lse - torch.logsumexp(raw, dim=-1)).abs().max().item() < 1e-3
