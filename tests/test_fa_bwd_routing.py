"""End-to-end: trifast's UNMODIFIED backward `_bwd_kv` @triton.jit is detected,
routed to the biased-FA backward dK/dV template, and computes correct gradients on
Metal — where the generic path used to fail-closed (no `max` op -> the forward FA
block never fires -> #7 host-path refuse).

The backward pattern = FA-2 recompute P = exp2((sm_scale*QK + bias - lse)*inv_ln2)
with lse LOADED (row_max, not computed via a max reduction), then dV += P^T @ dO,
dP = dO @ V^T, dS = P*(dP - delta), dK += dS^T @ Q. See
generic_lowerer._detect_biased_fa_backward (bwd_kind="kv").

This kernel is copied verbatim from trifast/src/trifast/triton.py (`_bwd_kv`), the
same source the external user (Janssena) is porting. The reference lse/delta inputs
are produced directly in torch so the test is self-contained (no forward kernel).
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


# fmt: off
@triton.jit
def _bwd_kv(
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
    sm_scale,
    neg_inf,
    N, H, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty
    pid_k = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)
    inv_ln2: tl.constexpr = 1.4426950408889634

    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = pid_i
    start_j = 0
    start_k = pid_k * BLOCK_K

    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J)
    d_idxs = tl.arange(0, DIM)

    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)

    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    kt_ptrs = base_k_ptr + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn)

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn)

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    vt_ptrs = base_v_ptr + (d_idxs[:, None] * stride_vd) + (k_idxs[None, :] * stride_vn)

    base_l_ptr = l_ptr + (start_h * stride_lh) + (start_i * stride_lm)
    l_ptrs = base_l_ptr + (j_idxs * stride_ln)

    base_mask_ptr = m_ptr + (mask_start_h * stride_mh)
    mask_ptrs = base_mask_ptr + (start_i * stride_mm) + (k_idxs * stride_mn)

    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod)

    base_dk_ptr = dk_ptr + (start_h * stride_dkh) + (start_i * stride_dkm)
    dk_ptrs = base_dk_ptr + (k_idxs[:, None] * stride_dkn) + (d_idxs[None, :] * stride_dkd)

    base_dv_ptr = dv_ptr + (start_h * stride_dvh) + (start_i * stride_dvm)
    dv_ptrs = base_dv_ptr + (k_idxs[:, None] * stride_dvn) + (d_idxs[None, :] * stride_dvd)

    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)

    mask_k = k_idxs < N

    vt_block = tl.load(vt_ptrs, mask_k[None, :])
    kt_block = tl.load(kt_ptrs, mask_k[None, :])
    kt_block = kt_block * tl.full([1], value=sm_scale, dtype=input_dtype)
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

        dsm_value = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)
        dsm_value = tl.dot(do, vt_block, dsm_value, input_precision="ieee")

        dscores = sm_value * (dsm_value - delta[:, None])
        dscores = dscores.to(input_dtype)

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
# fmt: on


def _reference(Hc, Hh, I, N, DIM, seed=0):
    """torch autograd reference for the trifast biased/triangle attention, plus the
    (lse, delta) the backward consumes as inputs."""
    torch.manual_seed(seed)
    dev = "mps"
    sm = 1.0 / math.sqrt(DIM)
    q = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    k = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    v = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    bias = torch.randn(Hc, N, N, device=dev, requires_grad=True)
    mask = (torch.rand(Hc // Hh, I, N, device=dev) < 0.15).to(torch.uint8)
    do = torch.randn(Hc, I, N, DIM, device=dev)

    qk = torch.einsum("hijd,hikd->hijk", q, k) * sm
    mh = mask[torch.arange(Hc, device=dev) // Hh]
    raw = qk + bias[:, None, :, :]
    raw = raw.masked_fill(mh[:, :, None, :].bool(), float("-inf"))
    p = torch.softmax(raw, dim=-1)
    o = torch.einsum("hijk,hikd->hijd", p, v)
    o.retain_grad()
    o.backward(do)

    lse = torch.logsumexp(raw, dim=-1).detach()          # [Hc,I,N] (row_max the bwd loads)
    delta = (o.detach() * do).sum(-1).contiguous()       # [Hc,I,N]
    return dict(
        q=q.detach(), k=k.detach(), v=v.detach(), bias=bias.detach(), mask=mask,
        do=do, lse=lse, delta=delta, sm=sm,
        dk_ref=k.grad.detach(), dv_ref=v.grad.detach(),
    )


@requires_mps
@pytest.mark.parametrize("Hc,Hh,I,N,DIM", [
    (4, 2, 3, 64, 32),   # cross-head mask (batch=2), 2 k-blocks
    (2, 1, 2, 96, 32),   # single batch, N=96 -> 3 k-blocks + longer j-loop
    (1, 1, 1, 48, 32),   # minimal, N not a block multiple (mask_k tail)
])
def test_bwd_kv_routes_and_computes(Hc, Hh, I, N, DIM):
    # NOTE: BK*head_dim threads/threadgroup caps head_dim at 32 for BK=32 (DIM=64
    # would need 2048 threads > Metal's 1024 max and ~49KB > 32KB tg memory). Larger
    # head dims need a BK=16 tiling of this template -- tracked as follow-up.
    dev = "mps"
    BJ = BK = 32
    NEG = -1e9
    r = _reference(Hc, Hh, I, N, DIM)
    st = lambda t: tuple(t.stride())
    dk = torch.zeros(Hc, I, N, DIM, device=dev)
    dv = torch.zeros(Hc, I, N, DIM, device=dev)
    q, k, v, bias, mask, do, lse, delta, sm = (
        r["q"], r["k"], r["v"], r["bias"], r["mask"], r["do"], r["lse"], r["delta"], r["sm"])
    _bwd_kv[(triton.cdiv(N, BK), I, Hc)](
        delta, *st(delta), q, *st(q), k, *st(k), v, *st(v), bias, *st(bias),
        lse, *st(lse), mask, *st(mask), do, *st(do), dk, *st(dk), dv, *st(dv),
        sm, NEG, N, Hh, DIM, N, BJ, BK)
    torch.mps.synchronize()
    dk_err = (dk - r["dk_ref"]).abs().max().item()
    dv_err = (dv - r["dv_ref"]).abs().max().item()
    assert dk_err < 3e-3, f"dK wrong: {dk_err:.3e} (ref scale {r['dk_ref'].abs().max():.2f})"
    assert dv_err < 3e-3, f"dV wrong: {dv_err:.3e} (ref scale {r['dv_ref'].abs().max():.2f})"
