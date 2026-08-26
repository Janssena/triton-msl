"""Regression: varlen FA must prove cu end/start indices are exactly b+1 and b."""

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
def _varlen_cu_probe(
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
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CU_SHIFT: tl.constexpr,
    UNSIGNED_BH: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    if UNSIGNED_BH:
        off_bh_u = off_bh.to(tl.uint32)
        h_u = H.to(tl.uint32)
        off_b = off_bh_u // h_u
        off_h = off_bh_u % h_u
    else:
        off_b = off_bh // H
        off_h = off_bh % H

    q_start_index = off_b + CU_SHIFT
    q_end_index = off_b + CU_SHIFT + 1
    k_start_index = off_b + CU_SHIFT
    k_end_index = off_b + CU_SHIFT + 1
    q_start = tl.load(cu_q + q_start_index)
    seqlen_q = tl.load(cu_q + q_end_index) - q_start
    k_start = tl.load(cu_k + k_start_index)
    seqlen_k = tl.load(cu_k + k_end_index) - k_start

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = (
        Q
        + (q_start + offs_m)[:, None] * stride_qt
        + off_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0) * SCALE
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, max_seqlen, BLOCK_N):
        kn = start_n + offs_n
        k_ptrs = (
            K
            + (k_start + kn)[:, None] * stride_kt
            + off_h * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        qk = tl.where(kn[None, :] < seqlen_k, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = (
            V
            + (k_start + kn)[:, None] * stride_vt
            + off_h * stride_vh
            + offs_d[None, :] * stride_vd
        )
        value = tl.load(v_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        acc += tl.dot(p.to(tl.float32), value.to(tl.float32))
        m_i = m_new

    o_ptrs = (
        Out
        + (q_start + offs_m)[:, None] * stride_ot
        + off_h * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(
        o_ptrs,
        (acc / l_i[:, None]).to(Out.dtype.element_ty),
        mask=offs_m[:, None] < seqlen_q,
    )


@requires_mps
def test_varlen_shifted_cu_pair_correct_or_refuse(monkeypatch):
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    route = {"calls": 0, "tag": None, "hardcodes_unshifted_cu": False}
    real = fa_dispatch.dispatch_flash_attention

    def spy(*args, **kwargs):
        route["calls"] += 1
        if len(args) > 1 and isinstance(args[1], (tuple, list)) and args[1]:
            descriptor = args[1]
            route["tag"] = descriptor[0]
            if len(descriptor) > 1 and isinstance(descriptor[1], str):
                source = descriptor[1]
                route["hardcodes_unshifted_cu"] = (
                    "CUQ[bb]" in source
                    and ("CUQ[bb + 1u]" in source or "CUQ[bb+1u]" in source)
                    and "CUK[bb]" in source
                )
        return real(*args, **kwargs)

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)

    device = "mps"
    torch.manual_seed(8501)
    lengths = [16, 48, 32]
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), device=device, dtype=torch.int32)
    total = int(cu[-1])
    batches_processed = len(lengths) - 1
    H, D = 2, 64
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(total, H, D, device=device, dtype=torch.float16)
    k = torch.randn(total, H, D, device=device, dtype=torch.float16)
    v = torch.randn(total, H, D, device=device, dtype=torch.float16)
    out = torch.zeros_like(q)

    try:
        _varlen_cu_probe[(triton.cdiv(max(lengths[1:]), 32), batches_processed * H)](
            q,
            k,
            v,
            out,
            cu,
            cu,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            H,
            max(lengths[1:]),
            SCALE=scale,
            BLOCK_M=32,
            BLOCK_N=32,
            HEAD_DIM=D,
            CU_SHIFT=1,
            UNSIGNED_BH=False,
        )
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        print(f"VARLEN_CU_PROBE refused route={route}")
        return

    ref = torch.zeros_like(out)
    for sequence in (1, 2):
        start, end = int(cu[sequence]), int(cu[sequence + 1])
        for h in range(H):
            scores = (q[start:end, h].float() * scale) @ k[start:end, h].float().T
            ref[start:end, h] = (torch.softmax(scores, -1) @ v[start:end, h].float()).half()

    err = (out - ref).abs().max().item()
    print(f"VARLEN_CU_PROBE route={route} err={err}")
    assert err < 2e-2, (
        "varlen FA violated correct-or-refuse for shifted cu[b+1]/cu[b+2] semantics: "
        f"route={route}, max_err={err}"
    )


@requires_mps
def test_varlen_unsigned_pid_division_routes_and_is_correct(monkeypatch):
    """A casted program id with divui/remui is still the canonical CU convention."""
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
    torch.manual_seed(8502)
    lengths = [16, 48]
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), device=device, dtype=torch.int32)
    total = int(cu[-1])
    H, D = 2, 64
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(total, H, D, device=device, dtype=torch.float16)
    k = torch.randn(total, H, D, device=device, dtype=torch.float16)
    v = torch.randn(total, H, D, device=device, dtype=torch.float16)
    out = torch.zeros_like(q)

    _varlen_cu_probe[(triton.cdiv(max(lengths), 32), len(lengths) * H)](
        q,
        k,
        v,
        out,
        cu,
        cu,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        H,
        max(lengths),
        SCALE=scale,
        BLOCK_M=32,
        BLOCK_N=32,
        HEAD_DIM=D,
        CU_SHIFT=0,
        UNSIGNED_BH=True,
    )
    torch.mps.synchronize()

    ref = torch.zeros_like(out)
    for sequence in range(len(lengths)):
        start, end = int(cu[sequence]), int(cu[sequence + 1])
        for h in range(H):
            scores = (q[start:end, h].float() * scale) @ k[start:end, h].float().T
            ref[start:end, h] = (torch.softmax(scores, -1) @ v[start:end, h].float()).half()

    err = (out - ref).abs().max().item()
    print(f"VARLEN_CU_UNSIGNED route={route} err={err}")
    assert route == {"calls": 1, "tag": "flash_attention"}
    assert err < 2e-2
