"""Regression: the FlashAttention template must VERIFY the V pointer role, not trust it.

Background (issue #4 item 3, residual class — 2026-08-27). Q, K and Out were
cross-checked against the generic dot-pointer resolver, but V was the ONE unverified
role: it is the second operand of dot 1 (P@V), not dot 0, so the Q/K/Out mechanism
could not see it and the detector's stride-chain trace stood unchecked. The reporter's
silent-wrong lived exactly there — a kernel where V resolved to a distinct-but-WRONG
arg index sails past the four-distinct-roles gate, and the template reads attention
values from the wrong tensor.

V IS structurally verifiable: find the earliest dot whose first operand depends on
dot 0's result (that is P@V by construction), trace its second operand to a kernel arg
with the same tracer Q/K/Out use, and require agreement with the detector. Every leg
fails safe (no dependent dot / untraceable operand -> today's behavior); only a
POSITIVE structural trace that DISAGREES refuses.

Two pins, both on the canonical template-routed kernel
(``tests/test_flash_attention._flash_attn_fwd``, the same kernel+shape
``test_real_fa_routes_through_dispatch`` proves runs on the simd template):

- agreement: with the verification live, the kernel still computes and matches SDPA;
- the catch: forcing the detector's V index to a wrong arg (injected by wrapping the
  template entry point — AFTER the detector's four-distinct-roles gate, which is
  exactly where the real evasion class lives) must REFUSE with the V-role message.
  This also proves the dot-1 trace resolves POSITIVELY on a real kernel: had it fallen
  back, no refusal would fire and the test would fail.

(Two earlier drafts failed for instructive reasons: a spare 5th pointer arg knocks the
kernel off the FA template path entirely, and a hand-rolled FA kernel shape missed the
template's routing conditions — so the test now reuses the proven-routed kernel and
injects the disagreement post-detection.)
"""

import math
import sys

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)

sys.path.insert(0, "tests")


@triton.jit
def _varlen_fwd_private(
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
):
    """Private clone of test_varlen_fa_routing._varlen_fwd. Deliberately a COPY, not an
    import: the catch test below monkeypatches the varlen LOWERING, which only runs on
    a fresh compile — if this file shared the routing tests' kernel, whichever ran
    first under randomized order would warm the cache and the mutated lowering would
    silently never execute (observed: passed in file order, failed under
    --randomly-seed=90210). A distinct function has a distinct cache key, always."""
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

    q_ptrs = Q + (q_start + offs_m)[:, None] * stride_qt + off_h * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    q = q * SCALE
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, max_seqlen, BLOCK_N):
        kn = start_n + offs_n
        k_ptrs = K + (k_start + kn)[:, None] * stride_kt + off_h * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        qk = tl.where(kn[None, :] < seqlen_k, qk, float("-inf"))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = V + (k_start + kn)[:, None] * stride_vt + off_h * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new
    acc = acc / l_i[:, None]
    o_ptrs = Out + (q_start + offs_m)[:, None] * stride_ot + off_h * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


@triton.jit
def _varlen_v_transform(
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
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Adversarial varlen FA: P@V consumes transformed V, not raw V."""
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
    q_ptrs = Q + (q_start + offs_m)[:, None] * stride_qt + off_h * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0) * SCALE
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, max_seqlen, BLOCK_N):
        kn = start_n + offs_n
        k_ptrs = K + (k_start + kn)[:, None] * stride_kt + off_h * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        qk = tl.where(kn[None, :] < seqlen_k, qk, float("-inf"))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = V + (k_start + kn)[:, None] * stride_vt + off_h * stride_vh + offs_d[None, :] * stride_vd
        if MODE == 2:
            # A pointer-valued select is legal for ordinary loads.  When the
            # selected load feeds a cooperatively staged dot operand, codegen
            # must either reconstruct every staged address exactly or refuse.
            k_as_v_ptrs = K + (k_start + kn)[:, None] * stride_kt + off_h * stride_kh + offs_d[None, :] * stride_kd
            v_ptrs = tl.where(off_h == 0, v_ptrs, k_as_v_ptrs)
        v = tl.load(v_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        if MODE == 0:
            v_operand = v + k
        elif MODE == 1:
            v_operand = tl.trans(v)
        else:
            v_operand = v
        acc += tl.dot(p.to(tl.float32), v_operand.to(tl.float32))
        m_i = m_new
    acc = acc / l_i[:, None]
    o_ptrs = Out + (q_start + offs_m)[:, None] * stride_ot + off_h * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


def _run_v_transform(mode, lens, H, D, scale):
    """Launch the private kernel with its additional transformation selector."""
    dev = "mps"
    torch.manual_seed(0)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
    total = int(cu[-1])
    q = torch.randn(total, H, D, device=dev, dtype=torch.float32)
    k = torch.randn(total, H, D, device=dev, dtype=torch.float32)
    v = torch.randn(total, H, D, device=dev, dtype=torch.float32)
    out = torch.zeros_like(q)
    block_m = block_n = 32
    max_seqlen = max(lens)
    grid = (triton.cdiv(max_seqlen, block_m), len(lens) * H)
    _varlen_v_transform[grid](
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
        max_seqlen,
        scale,
        mode,
        block_m,
        block_n,
        D,
    )
    torch.mps.synchronize()
    return q, k, v, out, cu


def _launch(D, causal=False):
    from test_flash_attention import _flash_attn_fwd

    Z, H, N = 1, 4, 512
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, D, device="mps", dtype=torch.float32)
    k = torch.randn(Z, H, N, D, device="mps", dtype=torch.float32)
    v = torch.randn(Z, H, N, D, device="mps", dtype=torch.float32)
    out = torch.empty_like(q)
    _flash_attn_fwd[(N // 32, Z * H)](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        Z,
        H,
        N,
        BLOCK_M=32,
        BLOCK_N=32,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
    )
    return q, k, v, out


@requires_mps
def test_fa_computes_with_v_verification_live():
    # Agreement path: the structural dot-1 trace and the detector agree on V, and the
    # verification does not over-refuse the canonical kernel.
    q, k, v, out = _launch(128)
    torch.mps.synchronize()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(128))
    err = (out - ref).abs().max().item()
    assert err == err and err < 2e-3, f"FA wrong with V verification live: err {err}"


@requires_mps
def test_fa_v_role_disagreement_refuses(monkeypatch):
    # The catch, on the VARLEN path — which (like the reporter's biased/triangle path)
    # has NO first-four ordering gate, so the structural dot-1 check is its only V
    # cross-check. Inject the evasion: the detector's roles claim V is a wrong (but
    # real, distinct) pointer arg while dot 1's actual second operand traces to the
    # true V. The mutation lands after detection — the position the real evasion
    # occupies. Must refuse via the V-role message specifically.
    #
    # (The dense path can't host this test: it pins Q/K/V/Out to args 0-3 IN ORDER
    # before the V check, so any wrong V index trips that gate first.)
    import triton_msl.codegen.generic_lowerer as gl

    from test_varlen_fa_routing import _run_varlen

    real = gl.GenericLowerer._lower_varlen_flash_attention

    def evil(self, info):
        bad = dict(info)
        bad["roles"] = dict(info["roles"])
        wrong = bad["roles"]["cuq"]  # a real ptr arg that is NOT V
        assert bad["roles"]["v"] != wrong, "test setup: V already resolved to the decoy"
        bad["roles"]["v"] = wrong
        return real(self, bad)

    monkeypatch.setattr(gl.GenericLowerer, "_lower_varlen_flash_attention", evil)
    with pytest.raises(MetalNonRecoverableError, match="V pointer role disagrees"):
        _run_varlen(_varlen_fwd_private, [64, 32], [64, 32], H=2, D=64, dtype=torch.float32, scale=1.0 / math.sqrt(64))


@requires_mps
def test_fa_v_arithmetic_is_correct_or_refuses():
    """A role trace must not mistake raw V for the semantics of ``V + K``.

    Pre-fix, both the varlen detector and _fa_verify_v_role followed operand 0 through
    the add and agreed on V. The specialized route ran exactly once, matched plain-V
    attention to 4.8e-7, and missed the kernel's V+K semantics by 1.392.
    """
    from test_varlen_fa_routing import _ref_varlen

    H, D = 2, 64
    scale = 1.0 / math.sqrt(D)
    try:
        q, k, v, out, cu_q = _run_v_transform(0, [64, 32], H, D, scale)
    except MetalNonRecoverableError as exc:
        assert "not a direct V load" in str(exc)
        return

    ref = _ref_varlen(q, k, v + k, cu_q, cu_q, H, D, scale)
    err = (out - ref).abs().max().item()
    assert err < 1e-3, f"FA template dropped arithmetic on V: err {err}"


@requires_mps
def test_fa_v_transpose_is_correct_or_refuses():
    """A matrix transpose is not a representation-only wrapper around V."""
    from test_varlen_fa_routing import _ref_varlen

    H, D = 2, 32
    scale = 1.0 / math.sqrt(D)
    try:
        q, k, v, out, cu = _run_v_transform(1, [32], H, D, scale)
    except MetalNonRecoverableError as exc:
        assert "not a direct V load" in str(exc)
        return

    # With T == D == 32, this is the kernel's per-head transpose expressed in the
    # packed [token, head, dim] host layout.
    transposed_v = v.permute(2, 1, 0).contiguous()
    ref = _ref_varlen(q, k, transposed_v, cu, cu, H, D, scale)
    err = (out - ref).abs().max().item()
    assert err < 1e-3, f"FA template dropped transpose on V: err {err}"


@requires_mps
def test_fa_v_selected_pointer_cooperative_staging_computes():
    """Cooperative staging rebuilds both arms rather than collapsing to offset 0."""
    from test_varlen_fa_routing import _ref_varlen

    H, D = 2, 64
    scale = 1.0 / math.sqrt(D)
    q, k, v, out, cu = _run_v_transform(2, [64], H, D, scale)

    # Head 0 selects V; head 1 selects K.  The source oracle is independent
    # of the selected pointer expression and therefore catches any staged
    # address collapse even when the resulting shader remains well formed.
    source_v = torch.stack((v[:, 0], k[:, 1]), dim=1)
    ref = _ref_varlen(q, k, source_v, cu, cu, H, D, scale)
    err = (out - ref).abs().max().item()
    assert err < 1e-3, f"cooperative staging changed selected-pointer semantics: err {err}"
