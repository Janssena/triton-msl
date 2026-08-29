"""Dense FA templates must preserve the composed score-to-probability function.

The template always emits a natural-exponential online softmax.  Equivalent Triton
spellings may put the base conversion either after ``score - max`` or into Q's scale;
the lowering must reason about that composition, not admit/reject the slots separately.
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
def _fa_score_probability(
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
    q_ptrs = (
        Q
        + off_z * sqz
        + off_h * sqh
        + offs_m[:, None] * sqm
        + offs_d[None, :] * sqk
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    qk_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    if VARIANT == 3 or VARIANT == 4 or VARIANT == 8 or VARIANT == 9:
        q = q * (qk_scale * _LOG2E)
    else:
        q = q * qk_scale

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        kn = start_n + offs_n
        k_ptrs = (
            K
            + off_z * skz
            + off_h * skh
            + kn[:, None] * skn
            + offs_d[None, :] * skk
        )
        k = tl.load(k_ptrs, mask=kn[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        if VARIANT == 7 or VARIANT == 8:
            qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))
        elif VARIANT == 9:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        elif VARIANT == 10:
            qk = tl.where((kn[None, :] % 2) == 0, qk, float("-inf"))
        elif VARIANT == 12:
            qk = tl.where((kn + 1)[None, :] < N_CTX, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        if VARIANT == 1:
            alpha = tl.exp2((m_i - m_new) * _LOG2E)
            p = tl.exp2((qk - m_new[:, None]) * _LOG2E)
        elif VARIANT == 2:
            alpha = tl.math.exp2((m_i - m_new) * _LOG2E)
            p = tl.math.exp2((qk - m_new[:, None]) * _LOG2E)
        elif VARIANT == 3 or VARIANT == 8 or VARIANT == 9:
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
        elif VARIANT == 4:
            alpha = tl.exp2(m_i - m_new)
            centered = qk - m_new[:, None]
            p = tl.exp2(centered)
        elif VARIANT == 11:
            alpha = tl.exp(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
        else:
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = (
            V
            + off_z * svz
            + off_h * svh
            + kn[:, None] * svn
            + offs_d[None, :] * svk
        )
        v = tl.load(v_ptrs, mask=kn[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = (
        Out
        + off_z * soz
        + off_h * soh
        + offs_m[:, None] * som
        + offs_d[None, :] * sok
    )
    if VARIANT == 5:
        tl.store(
            o_ptrs,
            acc.to(Out.dtype.element_ty),
            mask=offs_m[:, None] < N_CTX,
        )
    else:
        tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def _reference(q, k, v, variant):
    scale = 1.0 / math.sqrt(q.shape[-1])
    if variant == 9:
        return torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), scale=scale, is_causal=True
        )
    if variant == 10:
        score = q.float() @ k.float().transpose(-1, -2) * scale
        keep = torch.arange(q.shape[-2], device=q.device) % 2 == 0
        score = score.masked_fill(~keep[None, None, None, :], float("-inf"))
        return torch.softmax(score, dim=-1) @ v.float()
    if variant == 12:
        score = q.float() @ k.float().transpose(-1, -2) * scale
        keep = torch.arange(q.shape[-2], device=q.device) + 1 < q.shape[-2]
        score = score.masked_fill(~keep[None, None, None, :], float("-inf"))
        return torch.softmax(score, dim=-1) @ v.float()
    if variant == 11:
        # Replay the deliberately mismatched online recurrence exactly.  It is a valid
        # kernel but not a natural-softmax identity, so the canonical template must
        # refuse unless it can reproduce this composition.
        qf = q.float() * scale
        kf = k.float()
        vf = v.float()
        m_i = torch.full(qf.shape[:-1], float("-inf"), device=q.device)
        l_i = torch.zeros_like(m_i)
        acc = torch.zeros_like(qf)
        for start_n in range(0, q.shape[-2], 32):
            score = qf @ kf[..., start_n : start_n + 32, :].transpose(-1, -2)
            m_new = torch.maximum(m_i, score.max(dim=-1).values)
            alpha = torch.exp(m_i - m_new)
            p = torch.exp2(score - m_new[..., None])
            l_i = l_i * alpha + p.sum(dim=-1)
            acc = acc * alpha[..., None] + p @ vf[..., start_n : start_n + 32, :]
            m_i = m_new
        return acc / l_i[..., None]
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=scale
    )


def _run(variant, monkeypatch):
    import triton_msl.codegen.generic_lowerer as gl

    hits = []
    real = gl.GenericLowerer._lower_flash_attention_template

    def spy(self, info):
        hits.append(info["head_dim"])
        return real(self, info)

    monkeypatch.setattr(gl.GenericLowerer, "_lower_flash_attention_template", spy)
    torch.manual_seed(20260829)
    z, h, n_ctx, head_dim = 1, 4, 64, 64
    dtype = torch.float16 if variant == 6 else torch.float32
    q = torch.randn(z, h, n_ctx, head_dim, device="mps", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)
    _fa_score_probability[(n_ctx // 32, z * h)](
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
        32,
        32,
        head_dim,
        variant,
    )
    torch.mps.synchronize()
    return q, k, v, out, hits


@pytest.mark.parametrize(
    ("variant", "label"),
    [
        (0, "canonical exp"),
        (1, "exp2(x * log2e)"),
        (2, "tl.math.exp2(x * log2e)"),
        (3, "log2e in Q scale plus exp2"),
        (4, "split centered score with log2e in Q scale"),
        (5, "explicit output cast"),
        (6, "fp16 input and fp32 accumulation"),
        (7, "bounds mask before softmax"),
        (8, "tutorial exp2 plus bounds mask"),
        (9, "tutorial exp2 plus causal mask"),
    ],
)
@requires_mps
def test_equivalent_score_probability_spelling_routes_and_computes(
    variant, label, monkeypatch
):
    q, k, v, out, hits = _run(variant, monkeypatch)
    assert hits == [64], f"{label} did not reach the specialized dense FA template"
    err = (out.float() - _reference(q, k, v, variant)).abs().max().item()
    tolerance = 3e-2 if variant == 6 else 3e-3
    assert err < tolerance, f"dense FA miscompiled {label}: err {err}"


@pytest.mark.parametrize(
    ("variant", "label"),
    [
        (10, "noncanonical even-key mask"),
        (11, "mismatched alpha/P exponent bases"),
        (12, "offset N_CTX bounds mask"),
    ],
)
@requires_mps
def test_non_equivalent_score_probability_path_is_correct_or_refuses(
    variant, label, monkeypatch
):
    try:
        q, k, v, out, _hits = _run(variant, monkeypatch)
    except MetalNonRecoverableError as exc:
        assert "FlashAttention" in str(exc)
        return
    err = (out.float() - _reference(q, k, v, variant)).abs().max().item()
    assert err < 3e-3, f"dense FA silently dropped {label}: err {err}"
