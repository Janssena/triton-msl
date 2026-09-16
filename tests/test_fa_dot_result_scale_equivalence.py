"""Post-dot FA score scaling must be composed exactly or refused.

Issue #4/#6a is the canonical ``qk = tl.dot(...); qk *= scale`` spelling.  The
specialized templates re-emit scores from raw Q/K, so accepting that spelling requires
folding every proven score-side factor together with the exp/exp2 base conversion.
This surface deliberately separates score scaling from P scaling and scalar bias.
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
def _fa_dot_result_scale(
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
    N_CTX,
    SCALE,
    BIAS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VARIANT: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + off_z * sqz + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    base_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    if VARIANT == 0 or VARIANT == 4 or VARIANT == 6 or VARIANT == 7:
        q = q * base_scale
    elif VARIANT == 5:
        q = q * SCALE

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        kn = start_n + offs_n
        k_ptrs = K + off_z * skz + off_h * skh + kn[:, None] * skn + offs_d[None, :] * skk
        k = tl.load(k_ptrs, mask=kn[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        if VARIANT == 1:
            qk = qk * base_scale
        elif VARIANT == 2:
            qk = qk * SCALE
        elif VARIANT == 3:
            qk = qk * (base_scale * _LOG2E)
        elif VARIANT == 4 or VARIANT == 5:
            qk = qk * _LOG2E
        elif VARIANT == 7:
            qk = qk + BIAS
        elif VARIANT == 8:
            qk = qk / tl.sqrt(float(HEAD_DIM))
        elif VARIANT == 9:
            qk = qk * 0.5
            qk = qk * (2.0 * base_scale)
        elif VARIANT == 10:
            qk = qk * SCALE
            qk = qk * _LOG2E
        elif VARIANT == 11:
            qk = qk * SCALE
            qk = qk + BIAS

        if VARIANT == 12:
            # Intentionally non-equivalent: the row max sees raw scores while only
            # the probability/recurrence leg below sees the scale.
            m_ij = tl.max(qk, 1)
            qk = qk * base_scale
        else:
            m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        if VARIANT == 3 or VARIANT == 4 or VARIANT == 5 or VARIANT == 10:
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
        else:
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
        if VARIANT == 6:
            p = p * 0.5
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = V + off_z * svz + off_h * svh + kn[:, None] * svn + offs_d[None, :] * svk
        v = tl.load(v_ptrs, mask=kn[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    o_ptrs = Out + off_z * soz + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sok
    tl.store(o_ptrs, acc / l_i[:, None], mask=offs_m[:, None] < N_CTX)


def _run(variant, monkeypatch):
    import triton_msl.codegen.generic_lowerer as gl

    hits = []
    real = gl.GenericLowerer._lower_flash_attention_template

    def spy(self, info):
        hits.append(info["head_dim"])
        return real(self, info)

    monkeypatch.setattr(gl.GenericLowerer, "_lower_flash_attention_template", spy)
    torch.manual_seed(20260829)
    z, h, n_ctx, head_dim = 1, 2, 64, 64
    q = torch.randn(z, h, n_ctx, head_dim, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_dim)
    bias = 0.25
    _fa_dot_result_scale[(n_ctx // 32, z * h)](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        z,
        h,
        n_ctx,
        scale,
        bias,
        32,
        32,
        head_dim,
        variant,
    )
    torch.mps.synchronize()
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)
    return out, ref, hits


@pytest.mark.parametrize(
    ("variant", "label"),
    [
        (0, "canonical Q-side constant scale"),
        (1, "post-dot constant scale"),
        (3, "post-dot constant scale times log2e with exp2"),
        (4, "Q scale plus post-dot log2e with exp2"),
        (8, "post-dot constant division"),
        (9, "chained post-dot constant factors"),
    ],
)
@requires_mps
def test_equivalent_dot_result_scale_routes_and_computes(variant, label, monkeypatch):
    out, ref, hits = _run(variant, monkeypatch)
    assert hits == [64], f"{label} did not reach the specialized dense FA template"
    err = (out.float() - ref).abs().max().item()
    assert err < 3e-3, f"dense FA miscompiled {label}: err {err}"


@pytest.mark.parametrize(
    ("variant", "label"),
    [
        (2, "post-dot runtime scale"),
        (5, "runtime Q scale plus post-dot log2e with exp2"),
        (6, "scale applied after probability exp"),
        (7, "post-dot scalar bias"),
        (10, "post-dot runtime scale then log2e with exp2"),
        (11, "post-dot runtime scale plus scalar bias"),
    ],
)
@requires_mps
def test_non_scale_score_epilogue_is_correct_or_refuses(variant, label, monkeypatch):
    try:
        out, ref, _hits = _run(variant, monkeypatch)
    except MetalNonRecoverableError as exc:
        assert "FlashAttention" in str(exc)
        return
    err = (out.float() - ref).abs().max().item()
    assert err < 3e-3, f"dense FA silently dropped {label}: err {err}"


@requires_mps
def test_score_scale_after_row_max_refuses(monkeypatch):
    with pytest.raises(MetalNonRecoverableError, match="FlashAttention"):
        _run(12, monkeypatch)


@requires_mps
def test_biased_result_scale_is_not_double_folded(monkeypatch):
    """The biased detector already proves inv_ln2*ln(2)==1; keep it single-folded."""
    import tests.test_fa_biased_routing as biased
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    hits = []
    real = fa_dispatch.dispatch_flash_attention

    def spy(rt, descriptor, *args, **kwargs):
        result = real(rt, descriptor, *args, **kwargs)
        hits.append((descriptor[0], result))
        return result

    monkeypatch.setattr(
        fa_dispatch,
        "dispatch_flash_attention",
        spy,
    )
    out, lse, out_ref, lse_ref = biased._run(1, 2, 64, 32)
    assert hits == [("flash_attention", True)], "trifast-form biased FA did not take its specialized native-grid route"
    assert (out - out_ref).abs().max().item() < 1e-3
    finite = torch.isfinite(lse_ref)
    assert (lse[finite] - lse_ref[finite]).abs().max().item() < 1e-3
