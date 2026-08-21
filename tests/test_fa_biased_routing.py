"""End-to-end: a trifast-style BIASED / triangle-attention @triton.jit kernel is
detected, routed to the biased tiled template, and computes correctly on Metal —
where the generic path used to refuse (dot-result-scale guard). Plus correct-or-
refuse coverage: standard FA is NOT misrouted, and a mismatched softmax
temperature is refused.

The biased pattern = standard FA (2 dots + exp + max) PLUS a bias tile as the QK
dot's C operand, a loaded mask -> -inf, a per-row lse store, a runtime scale into
Q, and a const result-scale (inv_ln2) with exp2 (== natural softmax over
sm_scale*QK + bias). See generic_lowerer._detect_biased_flash_attention.
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
def _biased_fa(
    o_ptr, o_sz, o_sh, o_sm, o_sk,
    lse_ptr, lse_sz, lse_sh, lse_sm,
    q_ptr, q_sz, q_sh, q_sm, q_sk,
    k_ptr, k_sz, k_sh, k_sn, k_sk,
    v_ptr, v_sz, v_sh, v_sn, v_sk,
    b_ptr, b_sz, b_sh, b_sm, b_sn,
    mask_ptr, m_sz, m_sh, m_sn,
    sm_scale, neg_inf, Z, H, N,
    DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BAD_TEMP: tl.constexpr = False,
):
    inv_ln2: tl.constexpr = 1.4426950408889634
    ln2: tl.constexpr = 0.6931471824645996
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    start_m = pid_m * BLOCK_M
    m_idxs = start_m + tl.arange(0, BLOCK_M)
    n_idxs = tl.arange(0, BLOCK_N)
    d_idxs = tl.arange(0, DIM)
    q_ptrs = q_ptr + z * q_sz + h * q_sh + m_idxs[:, None] * q_sm + d_idxs[None, :] * q_sk
    kt_ptrs = k_ptr + z * k_sz + h * k_sh + d_idxs[:, None] * k_sk + n_idxs[None, :] * k_sn
    v_ptrs = v_ptr + z * v_sz + h * v_sh + n_idxs[:, None] * v_sn + d_idxs[None, :] * v_sk
    b_ptrs = b_ptr + z * b_sz + h * b_sh + m_idxs[:, None] * b_sm + n_idxs[None, :] * b_sn
    mask_ptrs = mask_ptr + z * m_sz + h * m_sh + n_idxs * m_sn
    o_ptrs = o_ptr + z * o_sz + h * o_sh + m_idxs[:, None] * o_sm + d_idxs[None, :] * o_sk
    lse_ptrs = lse_ptr + z * lse_sz + h * lse_sh + m_idxs * lse_sm

    scores_max = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    sm_denom = tl.full([BLOCK_M], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_M, DIM], value=0, dtype=tl.float32)
    mask_m = m_idxs < N
    q_block = tl.load(q_ptrs, mask_m[:, None])
    q_block = q_block * tl.full([1], value=sm_scale, dtype=q_block.type.element_ty)

    for start_n in tl.range(0, N, BLOCK_N):
        mask_n = (n_idxs + start_n) < N
        kt_block = tl.load(kt_ptrs, mask_n[None, :])
        b_block = tl.load(b_ptrs, mask_m[:, None] & mask_n[None, :])
        m_block = tl.load(mask_ptrs, mask_n)
        scores = b_block.to(tl.float32)
        scores = tl.dot(q_block, kt_block, scores)
        if not BAD_TEMP:
            scores *= inv_ln2  # inv_ln2 + exp2 == natural softmax
        scores = tl.where(m_block[None, :], neg_inf, scores)
        scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, neg_inf)
        block_max = tl.maximum(scores_max, tl.max(scores, 1))
        scores = scores - block_max[:, None]
        exp_scores = tl.math.exp2(scores)
        summed = tl.sum(exp_scores, 1)
        exp_scale = tl.math.exp2(scores_max - block_max)
        sm_denom = sm_denom * exp_scale + summed
        acc = acc * exp_scale[:, None]
        v_block = tl.load(v_ptrs, mask_n[:, None])
        exp_scores = exp_scores.to(q_block.type.element_ty)
        acc = tl.dot(exp_scores, v_block, acc)
        scores_max = block_max
        kt_ptrs += BLOCK_N * k_sn
        v_ptrs += BLOCK_N * v_sn
        b_ptrs += BLOCK_N * b_sn
        mask_ptrs += BLOCK_N * m_sn

    normalize = acc / sm_denom[:, None]
    tl.store(o_ptrs, normalize.to(q_block.type.element_ty), mask=mask_m[:, None])
    lse = (scores_max * ln2) + tl.log(sm_denom)
    tl.store(lse_ptrs, lse, mask=mask_m)


def _run(Z, H, N, DIM, bad_temp=False):
    torch.manual_seed(0)
    dev = "mps"
    sm = 1.0 / math.sqrt(DIM)
    q = torch.randn(Z, H, N, DIM, device=dev)
    k = torch.randn(Z, H, N, DIM, device=dev)
    v = torch.randn(Z, H, N, DIM, device=dev)
    b = torch.randn(Z, H, N, N, device=dev)
    mask = (torch.rand(Z, H, N, device=dev) < 0.25).to(torch.uint8)
    o = torch.zeros(Z, H, N, DIM, device=dev)
    lse = torch.zeros(Z, H, N, device=dev)
    st = lambda t: tuple(t.stride())
    grid = (triton.cdiv(N, 32), Z * H)
    _biased_fa[grid](
        o, *st(o), lse, *st(lse), q, *st(q), k, *st(k), v, *st(v),
        b, *st(b), mask, *st(mask), sm, -1e9, Z, H, N, DIM, 32, 32, bad_temp)
    torch.mps.synchronize()
    raw = sm * (q @ k.transpose(-2, -1)) + b
    raw = raw.masked_fill(mask[:, :, None, :].bool(), float("-inf"))
    p = torch.softmax(raw, dim=-1)
    o_ref = torch.nan_to_num(p, nan=0.0) @ v
    lse_ref = torch.logsumexp(raw, dim=-1)
    return o, lse, o_ref, lse_ref


@requires_mps
@pytest.mark.parametrize("Z,H,N,DIM", [(1, 2, 64, 32), (1, 1, 96, 32), (2, 2, 64, 32)])
def test_biased_fa_routes_and_computes(Z, H, N, DIM):
    o, lse, o_ref, lse_ref = _run(Z, H, N, DIM)
    assert (o - o_ref).abs().max().item() < 1e-3
    fin = torch.isfinite(lse_ref)
    assert (lse[fin] - lse_ref[fin]).abs().max().item() < 1e-3


@requires_mps
def test_biased_fa_bad_temperature_refuses():
    """exp2 WITHOUT the inv_ln2 result-scale => softmax temperature = ln2 != 1 =>
    the detector must refuse (routing it to the natural-exp template would be wrong).
    The refuse surfaces as either a MetalNonRecoverableError or a codegen warning;
    either proves it was NOT silently mis-routed to the template."""
    import warnings
    from triton_msl.errors import MetalNonRecoverableError

    refused = False
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            _run(1, 2, 64, 32, bad_temp=True)
        except MetalNonRecoverableError as e:
            refused = "temperature" in str(e) or "Refusing" in str(e)
    refused = refused or any(
        "temperature" in str(w.message) or "Refusing" in str(w.message) for w in wl
    )
    assert refused, "bad-temperature biased FA must be refused, not silently mis-routed"


def test_detector_returns_none_for_standard_fa():
    """The biased detector must NOT fire on a standard FA kernel (whose QK dot C
    operand is a zero/loop-acc, not a load) — else it would misroute it."""
    from tests.test_fa_detect import _build_fa_lowerer  # reuse the standard FA fixture

    low = _build_fa_lowerer(causal=False, head_dim=128, block=32)
    assert low._detect_biased_flash_attention() is None
