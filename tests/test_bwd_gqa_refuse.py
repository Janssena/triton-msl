# _bwd_kv_gqa and _bwd_kv_2d are adapted from TriFast's _bwd_kv.
# Copyright (c) 2025 Liam Atkinson; MIT License (third_party/trifast/LICENSE).
# Source revision and local adaptations: docs/TRIFAST_PROVENANCE.md.
"""Backward GQA/MQA guard (parity with the forward paths): a biased-attention backward
_bwd_kv where K/V have FEWER heads than Q (grouped-query: off_h_kv = pid_h // GROUP) must
NOT be routed to the dK/dV template — that template applies Q's head offset to K/V and would
silently mis-compute dK/dV (verified: a biased-GQA _bwd_kv otherwise routes kind=kv). The
detector refuses (Q/K batch-head offset mismatch) so it lowers on the generic/fallback path,
which computes the kernel as-written. Correct-or-refuse: the OUTPUT must match a GQA autograd
reference (never the wrong MHA-template result).
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
def _bwd_kv_gqa(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    l_ptr, stride_lh, stride_lm, stride_ln,
    m_ptr, stride_mh, stride_mm, stride_mn,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    dk_ptr, stride_dkh, stride_dkm, stride_dkn, stride_dkd,
    dv_ptr, stride_dvh, stride_dvm, stride_dvn, stride_dvd,
    sm_scale, neg_inf, N, H, GROUP, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty
    pid_k = tl.program_id(0); pid_i = tl.program_id(1); pid_h = tl.program_id(2)
    inv_ln2: tl.constexpr = 1.4426950408889634
    mask_start_h = pid_h // H
    start_h = pid_h
    start_h_kv = pid_h // GROUP                # GQA: K/V/dK/dV use fewer heads
    start_i = pid_i; start_k = pid_k * BLOCK_K
    k_idxs = tl.arange(0, BLOCK_K) + start_k; j_idxs = tl.arange(0, BLOCK_J); d_idxs = tl.arange(0, DIM)
    q_ptrs = q_ptr + (start_h * stride_qh) + (start_i * stride_qm) + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)
    kt_ptrs = k_ptr + (start_h_kv * stride_kh) + (start_i * stride_km) + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn)
    b_ptrs = b_ptr + (start_h * stride_bh) + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn)
    vt_ptrs = v_ptr + (start_h_kv * stride_vh) + (start_i * stride_vm) + (d_idxs[:, None] * stride_vd) + (k_idxs[None, :] * stride_vn)
    l_ptrs = l_ptr + (start_h * stride_lh) + (start_i * stride_lm) + (j_idxs * stride_ln)
    mask_ptrs = m_ptr + (mask_start_h * stride_mh) + (start_i * stride_mm) + (k_idxs * stride_mn)
    do_ptrs = do_ptr + (start_h * stride_doh) + (start_i * stride_dom) + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod)
    dk_ptrs = dk_ptr + (start_h_kv * stride_dkh) + (start_i * stride_dkm) + (k_idxs[:, None] * stride_dkn) + (d_idxs[None, :] * stride_dkd)
    dv_ptrs = dv_ptr + (start_h_kv * stride_dvh) + (start_i * stride_dvm) + (k_idxs[:, None] * stride_dvn) + (d_idxs[None, :] * stride_dvd)
    d_ptrs = d_ptr + (start_h * stride_dh) + (start_i * stride_dm) + (j_idxs * stride_dn)
    mask_k = k_idxs < N
    vt_block = tl.load(vt_ptrs, mask_k[None, :]); kt_block = tl.load(kt_ptrs, mask_k[None, :])
    kt_block = kt_block * tl.full([1], value=sm_scale, dtype=input_dtype)
    m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg")
    dk_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32); dv_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    for start_j in range(0, N, BLOCK_J):
        start_j = tl.multiple_of(start_j, BLOCK_J); mask_j = (j_idxs + start_j) < N
        q_block = tl.load(q_ptrs, mask_j[:, None]); b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :]).to(tl.float32)
        scores = tl.dot(q_block, kt_block, b_block, input_precision="ieee")
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
        scores = tl.where(m_block[None, :], neg_inf, scores)
        row_max = tl.load(l_ptrs, mask=mask_j); sm_value = tl.math.exp2((scores - row_max[:, None]) * inv_ln2)
        do = tl.load(do_ptrs, mask_j[:, None]); dv_block += tl.dot(tl.trans(sm_value).to(input_dtype), do, input_precision="ieee")
        delta = tl.load(d_ptrs, mask_j)
        dsm_value = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32); dsm_value = tl.dot(do, vt_block, dsm_value, input_precision="ieee")
        dscores = (sm_value * (dsm_value - delta[:, None])).to(input_dtype)
        dk_block += tl.dot(tl.trans(dscores), q_block, input_precision="ieee")
        q_ptrs += BLOCK_J * stride_qn; d_ptrs += BLOCK_J * stride_dn; b_ptrs += BLOCK_J * stride_bm
        l_ptrs += BLOCK_J * stride_ln; do_ptrs += BLOCK_J * stride_don; mask_ptrs += BLOCK_J * stride_mn
    dk_block *= sm_scale
    tl.store(dk_ptrs, dk_block.to(input_dtype), mask_k[:, None]); tl.store(dv_ptrs, dv_block.to(input_dtype), mask_k[:, None])


def _gqa_reference(Hc, GROUP, Hh, I, N, DIM, seed=0):
    """Autograd GQA backward: K/V have Hc//GROUP heads, expanded to Hc in the forward, so
    k.grad/v.grad sum each group's contributions -> the correct grouped dK/dV."""
    torch.manual_seed(seed); dev = "mps"; sm = 1.0 / math.sqrt(DIM)
    Hkv = Hc // GROUP
    q = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    k = torch.randn(Hkv, I, N, DIM, device=dev, requires_grad=True)
    v = torch.randn(Hkv, I, N, DIM, device=dev, requires_grad=True)
    bias = torch.randn(Hc, N, N, device=dev)
    mask = (torch.rand(Hc // Hh, I, N, device=dev) < 0.15).to(torch.uint8)
    do = torch.randn(Hc, I, N, DIM, device=dev)
    k_e = k.repeat_interleave(GROUP, dim=0); v_e = v.repeat_interleave(GROUP, dim=0)
    qk = torch.einsum("hijd,hikd->hijk", q, k_e) * sm
    mh = mask[torch.arange(Hc, device=dev) // Hh]
    raw = (qk + bias[:, None, :, :]).masked_fill(mh[:, :, None, :].bool(), float("-inf"))
    p = torch.softmax(raw, dim=-1)
    o = torch.einsum("hijk,hikd->hijd", p, v_e); o.retain_grad(); o.backward(do)
    lse = torch.logsumexp(raw, dim=-1).detach()
    delta = (o.detach() * do).sum(-1).contiguous()
    return dict(q=q.detach(), k=k.detach(), v=v.detach(), bias=bias, mask=mask, do=do,
                lse=lse, delta=delta, sm=sm, dk_ref=k.grad.detach(), dv_ref=v.grad.detach())


@triton.jit
def _bwd_kv_2d(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    l_ptr, stride_lh, stride_lm, stride_ln,
    m_ptr, stride_mh, stride_mm, stride_mn,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    dk_ptr, stride_dkh, stride_dkm, stride_dkn, stride_dkd,
    dv_ptr, stride_dvh, stride_dvm, stride_dvn, stride_dvd,
    sm_scale, neg_inf, N, H, I, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty
    pid_k = tl.program_id(0); pid_ih = tl.program_id(1)      # 2-D grid: i + head combined
    pid_i = pid_ih % I; pid_h = pid_ih // I
    inv_ln2: tl.constexpr = 1.4426950408889634
    mask_start_h = pid_h // H; start_h = pid_h; start_i = pid_i; start_k = pid_k * BLOCK_K
    k_idxs = tl.arange(0, BLOCK_K) + start_k; j_idxs = tl.arange(0, BLOCK_J); d_idxs = tl.arange(0, DIM)
    q_ptrs = q_ptr + (start_h * stride_qh) + (start_i * stride_qm) + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)
    kt_ptrs = k_ptr + (start_h * stride_kh) + (start_i * stride_km) + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn)
    b_ptrs = b_ptr + (start_h * stride_bh) + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn)
    vt_ptrs = v_ptr + (start_h * stride_vh) + (start_i * stride_vm) + (d_idxs[:, None] * stride_vd) + (k_idxs[None, :] * stride_vn)
    l_ptrs = l_ptr + (start_h * stride_lh) + (start_i * stride_lm) + (j_idxs * stride_ln)
    mask_ptrs = m_ptr + (mask_start_h * stride_mh) + (start_i * stride_mm) + (k_idxs * stride_mn)
    do_ptrs = do_ptr + (start_h * stride_doh) + (start_i * stride_dom) + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod)
    dk_ptrs = dk_ptr + (start_h * stride_dkh) + (start_i * stride_dkm) + (k_idxs[:, None] * stride_dkn) + (d_idxs[None, :] * stride_dkd)
    dv_ptrs = dv_ptr + (start_h * stride_dvh) + (start_i * stride_dvm) + (k_idxs[:, None] * stride_dvn) + (d_idxs[None, :] * stride_dvd)
    d_ptrs = d_ptr + (start_h * stride_dh) + (start_i * stride_dm) + (j_idxs * stride_dn)
    mask_k = k_idxs < N
    vt_block = tl.load(vt_ptrs, mask_k[None, :]); kt_block = tl.load(kt_ptrs, mask_k[None, :]) * tl.full([1], value=sm_scale, dtype=input_dtype)
    m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg")
    dk_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32); dv_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    for start_j in range(0, N, BLOCK_J):
        start_j = tl.multiple_of(start_j, BLOCK_J); mask_j = (j_idxs + start_j) < N
        q_block = tl.load(q_ptrs, mask_j[:, None]); b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :]).to(tl.float32)
        scores = tl.dot(q_block, kt_block, b_block, input_precision="ieee")
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf); scores = tl.where(m_block[None, :], neg_inf, scores)
        row_max = tl.load(l_ptrs, mask=mask_j); sm_value = tl.math.exp2((scores - row_max[:, None]) * inv_ln2)
        do = tl.load(do_ptrs, mask_j[:, None]); dv_block += tl.dot(tl.trans(sm_value).to(input_dtype), do, input_precision="ieee")
        delta = tl.load(d_ptrs, mask_j); dsm = tl.dot(do, vt_block, tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32), input_precision="ieee")
        dscores = (sm_value * (dsm - delta[:, None])).to(input_dtype); dk_block += tl.dot(tl.trans(dscores), q_block, input_precision="ieee")
        q_ptrs += BLOCK_J * stride_qn; d_ptrs += BLOCK_J * stride_dn; b_ptrs += BLOCK_J * stride_bm
        l_ptrs += BLOCK_J * stride_ln; do_ptrs += BLOCK_J * stride_don; mask_ptrs += BLOCK_J * stride_mn
    dk_block *= sm_scale
    tl.store(dk_ptrs, dk_block.to(input_dtype), mask_k[:, None]); tl.store(dv_ptrs, dv_block.to(input_dtype), mask_k[:, None])


@requires_mps
def test_bwd_kv_2d_grid_correct_or_refuse():
    # A biased backward on a 2-D grid (pid_i and pid_h combined -> n_pid=2 -> grid_3d=False)
    # exercises the q/kv templates' z=zh//H,h=zh%H head/batch decode, which no kernel
    # validates and could silently swap batch<->head. The detector refuses the 2-D path ->
    # generic/fallback runs the kernel as-written (correct). Output must match the reference.
    dev = "mps"
    Hc, Hh, I, N, DIM = 4, 2, 2, 64, 32
    BJ = BK = 32
    NEG = -1e9
    r = _gqa_reference(Hc, 1, Hh, I, N, DIM)   # GROUP=1 -> MHA reference
    st = lambda t: tuple(t.stride())
    dk = torch.zeros(Hc, I, N, DIM, device=dev)
    dv = torch.zeros(Hc, I, N, DIM, device=dev)
    q, k, v, bias, mask, do, lse, delta, sm = (
        r["q"], r["k"], r["v"], r["bias"], r["mask"], r["do"], r["lse"], r["delta"], r["sm"])
    from triton_msl.errors import MetalNonRecoverableError
    try:
        _bwd_kv_2d[(triton.cdiv(N, BK), I * Hc)](
            delta, *st(delta), q, *st(q), k, *st(k), v, *st(v), bias, *st(bias),
            lse, *st(lse), mask, *st(mask), do, *st(do), dk, *st(dk), dv, *st(dv),
            sm, NEG, N, Hh, I, DIM, N, BJ, BK)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # refused loudly — safe
    dk_err = (dk - r["dk_ref"]).abs().max().item()
    dv_err = (dv - r["dv_ref"]).abs().max().item()
    assert dk_err < 3e-3 and dv_err < 3e-3, (
        f"2-D backward mis-computed (routed through the unvalidated decode?): dK {dk_err:.2e} dV {dv_err:.2e}")


@requires_mps
def test_bwd_kv_gqa_not_misrouted():
    dev = "mps"
    Hc, GROUP, Hh, I, N, DIM = 4, 2, 2, 2, 64, 32
    Hkv = Hc // GROUP
    BJ = BK = 32
    NEG = -1e9
    r = _gqa_reference(Hc, GROUP, Hh, I, N, DIM)
    st = lambda t: tuple(t.stride())
    dk = torch.zeros(Hkv, I, N, DIM, device=dev)
    dv = torch.zeros(Hkv, I, N, DIM, device=dev)
    q, k, v, bias, mask, do, lse, delta, sm = (
        r["q"], r["k"], r["v"], r["bias"], r["mask"], r["do"], r["lse"], r["delta"], r["sm"])
    from triton_msl.errors import MetalNonRecoverableError
    try:
        _bwd_kv_gqa[(triton.cdiv(N, BK), I, Hc)](
            delta, *st(delta), q, *st(q), k, *st(k), v, *st(v), bias, *st(bias),
            lse, *st(lse), mask, *st(mask), do, *st(do), dk, *st(dk), dv, *st(dv),
            sm, NEG, N, Hh, GROUP, DIM, N, BJ, BK)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # refused loudly — safe
    dk_err = (dk - r["dk_ref"]).abs().max().item()
    dv_err = (dv - r["dv_ref"]).abs().max().item()
    assert dk_err < 3e-3 and dv_err < 3e-3, (
        f"GQA backward mis-computed (routed as MHA?): dK {dk_err:.2e} dV {dv_err:.2e}")
