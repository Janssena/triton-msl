"""FlashAttention templates must reproduce every value-path transformation they accept.

The templates reload raw Q/K/V and emit a canonical output normalization.  A detector
may therefore route a kernel only when the values reaching both dots and the output
store have exactly the semantics the template replays.  These pins cover three
GPU-confirmed silent-wrong gaps and a canonical positive:

* transpose Q before or after the canonical scale: correct or loudly refuse;
* transpose K before the dot's canonical transpose: correct or loudly refuse;
* apply arithmetic after softmax or normalized output: correct or loudly refuse.

Each mode is a separate constexpr specialization of a private kernel, so randomized
test order cannot warm a compile cache entry that bypasses the lowering spy.
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

_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _varlen_fa_value_path(
    Q, K, V, Out, cu_q, cu_k,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    H, max_seqlen, RUNTIME_SCALE, SCALE: tl.constexpr, MODE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    q_start = tl.load(cu_q + off_b)
    seqlen_q = tl.load(cu_q + off_b + 1) - q_start
    k_start = tl.load(cu_k + off_b)
    seqlen_k = tl.load(cu_k + off_b + 1) - k_start

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = (
        Q
        + (q_start + offs_m)[:, None] * stride_qt
        + off_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    if MODE == 0:
        q = tl.trans(q) * SCALE
    elif MODE == 7:
        q = q * (SCALE * _LOG2E)
    elif MODE == 10 or MODE == 11:
        pass  # #6a spellings put the complete scale after the QK dot.
    elif MODE == 13:
        q = q * RUNTIME_SCALE
    else:
        q = q * SCALE
    if MODE == 1:
        # Detection has already recognized load*scale; this must not be ignored.
        q = tl.trans(q)

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
        if MODE == 2:
            # This cancels the canonical transpose below in square test tiles.
            k = tl.trans(k)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        if MODE == 10:
            qk = qk * SCALE
        elif MODE == 11:
            qk = qk * RUNTIME_SCALE
        elif MODE == 12 or MODE == 13:
            qk = qk * _LOG2E
        qk = tl.where(kn[None, :] < seqlen_k, qk, float("-inf"))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        if MODE == 7 or MODE == 12 or MODE == 13:
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
        elif MODE == 8:
            alpha = tl.exp2((m_i - m_new) * _LOG2E)
            p = tl.exp2((qk - m_new[:, None]) * _LOG2E)
        elif MODE == 9:
            alpha = tl.math.exp2((m_i - m_new) * _LOG2E)
            p = tl.math.exp2((qk - m_new[:, None]) * _LOG2E)
        else:
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
        if MODE == 6:
            p = tl.exp(p)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = (
            V
            + (k_start + kn)[:, None] * stride_vt
            + off_h * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new
    acc = acc / l_i[:, None]
    if MODE == 3:
        acc = acc * 2.0 + 0.25
    elif MODE == 5:
        acc = acc / 2.0
    o_ptrs = (
        Out
        + (q_start + offs_m)[:, None] * stride_ot
        + off_h * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(
        o_ptrs,
        acc.to(Out.dtype.element_ty),
        mask=offs_m[:, None] < seqlen_q,
    )


def _reference(q, k, v, mode, scale):
    out = torch.zeros_like(q)
    for h in range(q.shape[1]):
        qb = q[:, h].float()
        kb = k[:, h].float()
        if mode == 0:
            qb = qb.transpose(0, 1)
        qb = qb * scale
        if mode == 1:
            qb = qb.transpose(0, 1)
        if mode == 2:
            kb = kb.transpose(0, 1)
        scores = qb @ kb.transpose(0, 1)
        if mode == 6:
            weights = torch.exp(torch.exp(scores - scores.max(dim=-1, keepdim=True).values))
            weights = weights / weights.sum(dim=-1, keepdim=True)
        else:
            weights = torch.softmax(scores, dim=-1)
        got = weights @ v[:, h].float()
        if mode == 3:
            got = got * 2.0 + 0.25
        elif mode == 5:
            got = got / 2.0
        out[:, h] = got.to(out.dtype)
    return out


def _run(mode, monkeypatch):
    import triton_msl.codegen.generic_lowerer as gl

    hits = []
    real = gl.GenericLowerer._lower_varlen_flash_attention

    def spy(self, info):
        hits.append(info["head_dim"])
        return real(self, info)

    monkeypatch.setattr(gl.GenericLowerer, "_lower_varlen_flash_attention", spy)

    torch.manual_seed(20260828)
    H = 2
    D = 32
    length = 32
    scale = 1.0 / math.sqrt(D)
    cu = torch.tensor([0, length], device="mps", dtype=torch.int32)
    q = torch.randn(length, H, D, device="mps", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.zeros_like(q)
    _varlen_fa_value_path[(1, H)](
        q, k, v, out, cu, cu,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        H, length, scale, scale, mode, 32, 32, D,
    )
    torch.mps.synchronize()
    return q, k, v, out, hits, scale


@requires_mps
def test_canonical_fa_value_paths_still_route_and_compute(monkeypatch):
    q, k, v, out, hits, scale = _run(4, monkeypatch)
    assert hits == [32], "canonical varlen FA did not reach its specialized template"
    err = (out - _reference(q, k, v, 4, scale)).abs().max().item()
    assert err < 1e-3, f"canonical varlen FA is wrong: err {err}"


@pytest.mark.parametrize(
    ("mode", "label"),
    [
        (7, "log2e in Q scale plus exp2"),
        (8, "exp2(x * log2e)"),
        (9, "tl.math.exp2(x * log2e)"),
    ],
)
@requires_mps
def test_varlen_equivalent_exp_spelling_routes_and_computes(mode, label, monkeypatch):
    q, k, v, out, hits, scale = _run(mode, monkeypatch)
    assert hits == [32], f"{label} did not reach the specialized varlen FA template"
    err = (out - _reference(q, k, v, mode, scale)).abs().max().item()
    assert err < 1e-3, f"varlen FA miscompiled {label}: err {err}"


@pytest.mark.parametrize(
    ("mode", "label"),
    [
        (10, "post-dot constexpr scale"),
        (12, "Q scale plus post-dot log2e with exp2"),
    ],
)
@requires_mps
def test_varlen_dot_result_scale_routes_and_computes(mode, label, monkeypatch):
    q, k, v, out, hits, scale = _run(mode, monkeypatch)
    assert hits == [32], f"{label} did not reach the specialized varlen FA template"
    err = (out - _reference(q, k, v, mode, scale)).abs().max().item()
    assert err < 1e-3, f"varlen FA miscompiled {label}: err {err}"


@pytest.mark.parametrize(
    ("mode", "label"),
    [
        (0, "Q transpose before scale"),
        (1, "Q transpose after scale"),
        (2, "K transpose"),
        (3, "output epilogue"),
        (5, "post-normalization division"),
        (6, "post-softmax exponential"),
        (11, "post-dot runtime scale"),
        (13, "runtime Q scale plus post-dot log2e with exp2"),
    ],
)
@requires_mps
def test_fa_unreplayed_value_path_is_correct_or_refuses(mode, label, monkeypatch):
    try:
        q, k, v, out, _hits, scale = _run(mode, monkeypatch)
    except MetalNonRecoverableError as exc:
        assert "FlashAttention" in str(exc)
        return

    err = (out - _reference(q, k, v, mode, scale)).abs().max().item()
    assert err < 1e-3, f"FA silently dropped {label}: err {err}"
