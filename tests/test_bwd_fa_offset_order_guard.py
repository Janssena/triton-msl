# _bwd_kv_v_offsets_swapped is adapted from TriFast's _bwd_kv.
# Copyright (c) 2025 Liam Atkinson; MIT License (third_party/trifast/LICENSE).
# Source revision and local adaptations: docs/TRIFAST_PROVENANCE.md.
"""Regression: biased-FA backward must preserve ordered head/instance offsets."""

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
def _bwd_kv_v_offsets_swapped(
    d_ptr,
    stride_dh,
    stride_di,
    stride_dn,
    q_ptr,
    stride_qh,
    stride_qi,
    stride_qn,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_ki,
    stride_kn,
    stride_kd,
    v_ptr,
    stride_vh,
    stride_vi,
    stride_vn,
    stride_vd,
    b_ptr,
    stride_bh,
    stride_bm,
    stride_bn,
    l_ptr,
    stride_lh,
    stride_li,
    stride_ln,
    m_ptr,
    stride_mh,
    stride_mi,
    stride_mn,
    do_ptr,
    stride_doh,
    stride_doi,
    stride_don,
    stride_dod,
    dk_ptr,
    stride_dkh,
    stride_dki,
    stride_dkn,
    stride_dkd,
    dv_ptr,
    stride_dvh,
    stride_dvi,
    stride_dvn,
    stride_dvd,
    sm_scale,
    neg_inf,
    N,
    H_MASK,
    N_HEADS,
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
    mask_start_h = pid_h // H_MASK
    start_h = pid_h
    start_i = pid_i
    start_k = pid_k * BLOCK_K
    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J)
    d_idxs = tl.arange(0, DIM)

    q_ptrs = (
        q_ptr + start_h * stride_qh + start_i * stride_qi + j_idxs[:, None] * stride_qn + d_idxs[None, :] * stride_qd
    )
    kt_ptrs = (
        k_ptr + start_h * stride_kh + start_i * stride_ki + d_idxs[:, None] * stride_kd + k_idxs[None, :] * stride_kn
    )
    b_ptrs = b_ptr + start_h * stride_bh + j_idxs[:, None] * stride_bm + k_idxs[None, :] * stride_bn

    # Same two offset SSAs as Q/K, but paired with the opposite scalar stride legs.
    vt_ptrs = (
        v_ptr + start_i * stride_vh + start_h * stride_vi + d_idxs[:, None] * stride_vd + k_idxs[None, :] * stride_vn
    )
    l_ptrs = l_ptr + start_h * stride_lh + start_i * stride_li + j_idxs * stride_ln
    mask_ptrs = m_ptr + mask_start_h * stride_mh + start_i * stride_mi + k_idxs * stride_mn
    do_ptrs = (
        do_ptr
        + start_h * stride_doh
        + start_i * stride_doi
        + j_idxs[:, None] * stride_don
        + d_idxs[None, :] * stride_dod
    )
    dk_ptrs = (
        dk_ptr
        + start_h * stride_dkh
        + start_i * stride_dki
        + k_idxs[:, None] * stride_dkn
        + d_idxs[None, :] * stride_dkd
    )
    dv_ptrs = (
        dv_ptr
        + start_h * stride_dvh
        + start_i * stride_dvi
        + k_idxs[:, None] * stride_dvn
        + d_idxs[None, :] * stride_dvd
    )
    d_ptrs = d_ptr + start_h * stride_dh + start_i * stride_di + j_idxs * stride_dn

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
        dsm = tl.dot(
            do,
            vt_block,
            tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32),
            input_precision="ieee",
        )
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
def test_backward_swapped_offset_pairing_correct_or_refuse(monkeypatch):
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    route = {"calls": 0, "tag": None}
    real = fa_dispatch.dispatch_flash_attention

    def spy(*args, **kwargs):
        route["calls"] += 1
        if len(args) > 1 and isinstance(args[1], (tuple, list)) and args[1]:
            route["tag"] = args[1][0]
        return real(*args, **kwargs)

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)

    device = "mps"
    torch.manual_seed(8401)
    heads, instances, N, D = 4, 2, 32, 32
    mask_heads = 2
    scale = 1.0 / math.sqrt(D)
    neg_inf = -1e9
    q = torch.randn(heads, instances, N, D, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.randn(heads, N, N, device=device)
    mask = (torch.rand(heads // mask_heads, instances, N, device=device) < 0.15).to(torch.uint8)
    do = torch.randn_like(q)
    lse = torch.randn(heads, instances, N, device=device)
    delta = torch.randn(heads, instances, N, device=device)
    dk = torch.zeros_like(q)
    dv = torch.zeros_like(q)
    strides = lambda tensor: tuple(tensor.stride())

    try:
        _bwd_kv_v_offsets_swapped[(triton.cdiv(N, 32), instances, heads)](
            delta,
            *strides(delta),
            q,
            *strides(q),
            k,
            *strides(k),
            v,
            *strides(v),
            bias,
            *strides(bias),
            lse,
            *strides(lse),
            mask,
            *strides(mask),
            do,
            *strides(do),
            dk,
            *strides(dk),
            dv,
            *strides(dv),
            scale,
            neg_inf,
            N,
            mask_heads,
            heads,
            DIM=D,
            CLOSEST_N=N,
            BLOCK_J=32,
            BLOCK_K=32,
        )
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        print(f"BWD_FA_OFFSET_ORDER_PROBE refused route={route}")
        return

    h, instance = 1, 0
    mask_row = mask[h // mask_heads, instance].bool()
    scores = (q[h, instance].float() * scale) @ k[h, instance].float().T + bias[h]
    scores = scores.masked_fill(mask_row[None, :], neg_inf)
    probabilities = torch.exp2((scores - lse[h, instance][:, None]) * 1.4426950408889634)
    swapped_flat_index = instance * instances + h
    v_swapped = v.reshape(heads * instances, N, D)[swapped_flat_index]
    dp = do[h, instance].float() @ v_swapped.float().T
    ds = probabilities * (dp - delta[h, instance][:, None])
    dk_ref = (ds.T @ q[h, instance].float()) * scale
    err = (dk[h, instance] - dk_ref).abs().max().item()
    print(f"BWD_FA_OFFSET_ORDER_PROBE route={route} err={err}")
    assert err < 1e-2, (
        f"biased-FA backward violated correct-or-refuse for swapped offset/stride pairing: route={route}, max_err={err}"
    )
