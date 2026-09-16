# _bwd_kv_vshift is adapted from TriFast's _bwd_kv.
# Copyright (c) 2025 Liam Atkinson; MIT License (third_party/trifast/LICENSE).
# Source revision and local adaptations: docs/TRIFAST_PROVENANCE.md.
"""Regression tests for the 2026-08-25 adversarial re-review of the FA routing guards.
Confirmed bypass classes, all now correct-or-refuse:

  A. SPLIT SCALE: the detectors baked only the Q-side constant; a kernel that also folds a
     constant into K (the sqrt-split ``q *= a; k *= b``) routed with K's factor silently
     DROPPED (varlen + dense, err ~2.0). Both now account for every constant factor.
  B. STORE HEAD: the guards compared only the INPUT loads' head offsets; a kernel writing
     the output at ``off_h + 1`` routed and the template wrote the WRONG head (varlen +
     dense + backward). The store's offsets are now part of the parity check.
  C. BACKWARD V: the backward GQA guard compared only Q vs K; a V read at ``pid_h + 1``
     routed and mis-addressed dK/dV. V (and dO/O + every gradient store) now checked.

Each test asserts correct-or-refuse: refuse loudly OR match the true reference — never the
template's mis-routed result.
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


# ---------- A. split scale (dense hd128 — the highest-impact confirmed case) ----------


@triton.jit
def _dense_fa_split_scale(
    Q,
    K,
    V,
    O,
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
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    sm = tl.program_id(0)
    hz = tl.program_id(1)
    z = hz // H
    h = hz % H
    om = sm * BM + tl.arange(0, BM)
    on = tl.arange(0, BN)
    od = tl.arange(0, D)
    q = tl.load(Q + z * sqz + h * sqh + om[:, None] * sqm + od[None, :] * sqk, mask=om[:, None] < N, other=0.0)
    q = q * 0.3  # HALF the scale on Q...
    mi = tl.full([BM], -float("inf"), tl.float32)
    li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for kn in range(0, N, BN):
        kk = kn + on
        k = tl.load(K + z * skz + h * skh + kk[:, None] * skn + od[None, :] * skk, mask=kk[:, None] < N, other=0.0)
        k = k * 0.5  # ...the OTHER HALF on K (sqrt-split)
        qk = tl.dot(q, tl.trans(k))
        m2 = tl.maximum(mi, tl.max(qk, 1))
        a = tl.exp(mi - m2)
        p = tl.exp(qk - m2[:, None])
        li = li * a + tl.sum(p, 1)
        acc = acc * a[:, None]
        v = tl.load(V + z * svz + h * svh + kk[:, None] * svn + od[None, :] * svk, mask=kk[:, None] < N, other=0.0)
        acc += tl.dot(p, v)
        mi = m2
    tl.store(O + z * soz + h * soh + om[:, None] * som + od[None, :] * sok, acc / li[:, None], mask=om[:, None] < N)


@requires_mps
def test_dense_split_scale_correct_or_refuse():
    dev = "mps"
    torch.manual_seed(0)
    Z, H, N, D = 1, 2, 64, 128
    q = torch.randn(Z, H, N, D, device=dev)
    k = torch.randn(Z, H, N, D, device=dev)
    v = torch.randn(Z, H, N, D, device=dev)
    o = torch.zeros(Z, H, N, D, device=dev)
    st = lambda t: t.stride()
    try:
        _dense_fa_split_scale[(triton.cdiv(N, 32), Z * H)](
            q, k, v, o, *st(q), *st(k), *st(v), *st(o), Z, H, N, 32, 32, D
        )
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # refused loudly — safe
    ref = torch.zeros_like(o)
    for h in range(H):
        sc = (q[0, h].float() * 0.3) @ (k[0, h].float() * 0.5).T  # effective 0.15
        ref[0, h] = torch.softmax(sc, -1) @ v[0, h].float()
    err = (o - ref).abs().max().item()
    assert err < 1e-2, f"split scale mis-computed (K factor dropped?): err {err:.2e}"


# ---------- B. store head (varlen — confirmed P2) ----------


@triton.jit
def _varlen_store_shift(
    Q,
    K,
    V,
    Out,
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
    max_seqlen,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    sm = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    h_out = (h + 1) % H  # writes the NEXT head
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
        k = tl.load(K + (ks + kn)[:, None] * skt + h * skh + od[None, :] * skd, mask=kn[:, None] < slk, other=0.0)
        qk = tl.where(kn[None, :] < slk, tl.dot(q, tl.trans(k).to(q.dtype)), float("-inf"))
        m2 = tl.maximum(mi, tl.max(qk, 1))
        a = tl.exp(mi - m2)
        p = tl.exp(qk - m2[:, None])
        li = li * a + tl.sum(p, 1)
        acc = acc * a[:, None]
        vv = tl.load(V + (ks + kn)[:, None] * svt + h * svh + od[None, :] * svd, mask=kn[:, None] < slk, other=0.0)
        acc += tl.dot(p.to(tl.float32), vv.to(tl.float32))
        mi = m2
    tl.store(
        Out + (qs + om)[:, None] * sot + h_out * soh + od[None, :] * sod,
        (acc / li[:, None]).to(Out.dtype.element_ty),
        mask=om[:, None] < slq,
    )


@requires_mps
def test_varlen_store_head_shift_not_misrouted():
    dev = "mps"
    torch.manual_seed(0)
    H, D = 2, 64
    lens = [48, 32]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
    T = int(cu[-1])
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(T, H, D, device=dev)
    k = torch.randn(T, H, D, device=dev)
    v = torch.randn(T, H, D, device=dev)
    o = torch.zeros(T, H, D, device=dev)
    try:
        _varlen_store_shift[(triton.cdiv(max(lens), 32), len(lens) * H)](
            q, k, v, o, cu, cu, *q.stride(), *k.stride(), *v.stride(), *o.stride(), H, max(lens), scale, 32, 32, D
        )
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # refused loudly — safe
    ref = torch.zeros_like(o)
    for b in range(len(lens)):
        s, e = int(cu[b]), int(cu[b + 1])
        for h in range(H):
            sc = (q[s:e, h].float() @ k[s:e, h].float().T) * scale
            ref[s:e, (h + 1) % H] = (torch.softmax(sc, -1) @ v[s:e, h].float()).to(ref.dtype)
    err = (o - ref).abs().max().item()
    assert err < 1e-2, f"shifted-store varlen mis-computed (routed writing off_h?): err {err:.2e}"


# ---------- C. backward V head (confirmed B1: V read at pid_h+1 routed) ----------
# Reuses the GQA-backward kernel family from test_bwd_gqa_refuse; here the SHIFT variant.


@triton.jit
def _bwd_kv_vshift(
    d_ptr,
    stride_dh,
    stride_dm,
    stride_dn,
    q_ptr,
    stride_qh,
    stride_qm,
    stride_qn,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_km,
    stride_kn,
    stride_kd,
    v_ptr,
    stride_vh,
    stride_vm,
    stride_vn,
    stride_vd,
    b_ptr,
    stride_bh,
    stride_bm,
    stride_bn,
    l_ptr,
    stride_lh,
    stride_lm,
    stride_ln,
    m_ptr,
    stride_mh,
    stride_mm,
    stride_mn,
    do_ptr,
    stride_doh,
    stride_dom,
    stride_don,
    stride_dod,
    dk_ptr,
    stride_dkh,
    stride_dkm,
    stride_dkn,
    stride_dkd,
    dv_ptr,
    stride_dvh,
    stride_dvm,
    stride_dvn,
    stride_dvd,
    sm_scale,
    neg_inf,
    N,
    H,
    NH,
    DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty
    pid_k = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)
    inv_ln2: tl.constexpr = 1.4426950408889634
    mask_start_h = pid_h // H
    start_h = pid_h
    start_h_v = (pid_h + 1) % NH  # V read at the NEXT head
    start_i = pid_i
    start_k = pid_k * BLOCK_K
    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J)
    d_idxs = tl.arange(0, DIM)
    q_ptrs = (
        q_ptr
        + (start_h * stride_qh)
        + (start_i * stride_qm)
        + (j_idxs[:, None] * stride_qn)
        + (d_idxs[None, :] * stride_qd)
    )
    kt_ptrs = (
        k_ptr
        + (start_h * stride_kh)
        + (start_i * stride_km)
        + (d_idxs[:, None]) * stride_kd
        + (k_idxs[None, :] * stride_kn)
    )
    b_ptrs = b_ptr + (start_h * stride_bh) + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn)
    vt_ptrs = (
        v_ptr
        + (start_h_v * stride_vh)
        + (start_i * stride_vm)
        + (d_idxs[:, None] * stride_vd)
        + (k_idxs[None, :] * stride_vn)
    )
    l_ptrs = l_ptr + (start_h * stride_lh) + (start_i * stride_lm) + (j_idxs * stride_ln)
    mask_ptrs = m_ptr + (mask_start_h * stride_mh) + (start_i * stride_mm) + (k_idxs * stride_mn)
    do_ptrs = (
        do_ptr
        + (start_h * stride_doh)
        + (start_i * stride_dom)
        + (j_idxs[:, None] * stride_don)
        + (d_idxs[None, :] * stride_dod)
    )
    dk_ptrs = (
        dk_ptr
        + (start_h * stride_dkh)
        + (start_i * stride_dkm)
        + (k_idxs[:, None] * stride_dkn)
        + (d_idxs[None, :] * stride_dkd)
    )
    dv_ptrs = (
        dv_ptr
        + (start_h * stride_dvh)
        + (start_i * stride_dvm)
        + (k_idxs[:, None] * stride_dvn)
        + (d_idxs[None, :] * stride_dvd)
    )
    d_ptrs = d_ptr + (start_h * stride_dh) + (start_i * stride_dm) + (j_idxs * stride_dn)
    mask_k = k_idxs < N
    vt_block = tl.load(vt_ptrs, mask_k[None, :])
    kt_block = tl.load(kt_ptrs, mask_k[None, :]) * tl.full([1], value=sm_scale, dtype=input_dtype)
    m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg")
    dk_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    dv_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    for start_j in range(0, N, BLOCK_J):
        start_j = tl.multiple_of(start_j, BLOCK_J)
        mask_j = (j_idxs + start_j) < N
        q_block = tl.load(q_ptrs, mask_j[:, None])
        b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :]).to(tl.float32)
        scores = tl.dot(q_block, kt_block, b_block, input_precision="ieee")
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
        scores = tl.where(m_block[None, :], neg_inf, scores)
        row_max = tl.load(l_ptrs, mask=mask_j)
        sm_value = tl.math.exp2((scores - row_max[:, None]) * inv_ln2)
        do = tl.load(do_ptrs, mask_j[:, None])
        dv_block += tl.dot(tl.trans(sm_value).to(input_dtype), do, input_precision="ieee")
        delta = tl.load(d_ptrs, mask_j)
        dsm = tl.dot(do, vt_block, tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32), input_precision="ieee")
        dscores = (sm_value * (dsm - delta[:, None])).to(input_dtype)
        dk_block += tl.dot(tl.trans(dscores), q_block, input_precision="ieee")
        q_ptrs += BLOCK_J * stride_qn
        d_ptrs += BLOCK_J * stride_dn
        b_ptrs += BLOCK_J * stride_bm
        l_ptrs += BLOCK_J * stride_ln
        do_ptrs += BLOCK_J * stride_don
        mask_ptrs += BLOCK_J * stride_mn
    dk_block *= sm_scale
    tl.store(dk_ptrs, dk_block.to(input_dtype), mask_k[:, None])
    tl.store(dv_ptrs, dv_block.to(input_dtype), mask_k[:, None])


@requires_mps
def test_bwd_v_head_shift_refuses():
    # V read at pid_h+1: must NOT route to the template (which forces V to Q's head).
    # The detector refuses -> the generic/fallback path runs the kernel as-written OR the
    # launch refuses loudly. Assert only not-silently-templated: outputs must NOT match
    # the template's V@pid_h behavior when they differ from the kernel's own semantics.
    dev = "mps"
    torch.manual_seed(0)
    Hc, Hh, I, N, DIM = 4, 2, 2, 64, 32
    sm = 1.0 / math.sqrt(DIM)
    BJ = BK = 32
    NEG = -1e9
    q = torch.randn(Hc, I, N, DIM, device=dev)
    k = torch.randn(Hc, I, N, DIM, device=dev)
    v = torch.randn(Hc, I, N, DIM, device=dev)
    bias = torch.randn(Hc, N, N, device=dev)
    mask = (torch.rand(Hc // Hh, I, N, device=dev) < 0.15).to(torch.uint8)
    do = torch.randn(Hc, I, N, DIM, device=dev)
    lse = torch.randn(Hc, I, N, device=dev)
    delta = torch.randn(Hc, I, N, device=dev)
    dk = torch.zeros(Hc, I, N, DIM, device=dev)
    dv = torch.zeros(Hc, I, N, DIM, device=dev)
    st = lambda t: tuple(t.stride())
    try:
        _bwd_kv_vshift[(triton.cdiv(N, BK), I, Hc)](
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
            Hc,
            DIM,
            N,
            BJ,
            BK,
        )
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # refused loudly — safe
    # If it ran, dP must have used V at (h+1): recompute the kernel's own semantics for
    # h=0 (V from head 1) and require a match — the template (V@h) would diverge.
    h, hv = 0, 1
    i = 0
    m_h = mask[h // Hh, i].bool()
    sc = (q[h, i].float() * sm) @ k[h, i].float().T + bias[h]
    sc = sc.masked_fill(m_h[None, :], NEG)
    P = torch.exp2((sc - lse[h, i][:, None]) * 1.4426950408889634)
    dP = do[h, i].float() @ v[hv, i].float().T
    dS = P * (dP - delta[h, i][:, None])
    dk_ref = (dS.T @ q[h, i].float()) * sm
    err = (dk[h, i] - dk_ref).abs().max().item()
    assert err < 1e-2, f"V-shift backward mis-computed (templated with V@h?): err {err:.2e}"
