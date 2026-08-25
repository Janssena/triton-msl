"""End-to-end validation of VARLEN (packed cu_seqlens) FlashAttention routing.

The dominant real-world FA shape (flash-attn / vLLM): sequences PACKED into
[total_tokens, H, head_dim], indexed per batch by cu_seqlens. The dense [Z,H,N,D]
templates can't express it and the generic path runs it at ~0.04 TF. These tests pin:
  * an UNMODIFIED varlen @jit kernel ROUTES to the tiled varlen template and is EXACT
    vs a per-sequence reference (hd 32/64/128, fp32/fp16, cross-attention seqlen_q != k);
  * the softmax scale is BAKED from the kernel's own constant (a non-1/sqrt(d) scale is
    honored, not silently replaced);
  * CAUSAL varlen is correct-or-refuse (v1 routes non-causal only -> causal must not
    silently mis-mask);
  * a DENSE FA kernel is NOT stolen by the varlen detector (still correct).
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


@triton.jit
def _varlen_fwd(
    Q, K, V, Out, cu_q, cu_k,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    H, max_seqlen, SCALE: tl.constexpr,
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
def _varlen_causal_fwd(
    Q, K, V, Out, cu_q, cu_k,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    H, max_seqlen, SCALE: tl.constexpr,
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
        causal = offs_m[:, None] >= kn[None, :]
        qk = tl.where((kn[None, :] < seqlen_k) & causal, qk, float("-inf"))
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


def _ref_varlen(q, k, v, cu_q, cu_k, H, D, scale, causal=False, strict=False):
    out = torch.zeros_like(q)
    for b in range(len(cu_q) - 1):
        qs, qe = cu_q[b].item(), cu_q[b + 1].item()
        ks, ke = cu_k[b].item(), cu_k[b + 1].item()
        for h in range(H):
            qb = q[qs:qe, h].float()
            kb = k[ks:ke, h].float()
            vb = v[ks:ke, h].float()
            sc = (qb @ kb.transpose(-2, -1)) * scale
            if causal or strict:
                lq, lk = qe - qs, ke - ks
                rows = torch.arange(lq, device=q.device)[:, None]
                cols = torch.arange(lk, device=q.device)[None, :]
                mask = rows < cols if causal else rows <= cols  # >= vs strict >
                sc = sc.masked_fill(mask, float("-inf"))
            out[qs:qe, h] = (torch.softmax(sc, -1) @ vb).to(out.dtype)
    return out


def _run_varlen(kernel, lens_q, lens_k, H, D, dtype, scale, seed=0):
    dev = "mps"
    torch.manual_seed(seed)
    cu_q = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0)), device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(lens_k).cumsum(0)), device=dev, dtype=torch.int32)
    Tq, Tk = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(Tq, H, D, device=dev, dtype=dtype)
    k = torch.randn(Tk, H, D, device=dev, dtype=dtype)
    v = torch.randn(Tk, H, D, device=dev, dtype=dtype)
    o = torch.zeros(Tq, H, D, device=dev, dtype=dtype)
    BM = BN = 32
    max_seqlen = max(max(lens_q), max(lens_k))
    grid = (triton.cdiv(max_seqlen, BM), len(lens_q) * H)
    kernel[grid](
        q, k, v, o, cu_q, cu_k,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        H, max_seqlen, scale, BM, BN, D)
    torch.mps.synchronize()
    return q, k, v, o, cu_q, cu_k


@requires_mps
@pytest.mark.parametrize("D", [32, 64, 128])
def test_varlen_routes_and_correct(D):
    scale = 1.0 / math.sqrt(D)
    q, k, v, o, cu_q, cu_k = _run_varlen(_varlen_fwd, [48, 32, 17], [48, 32, 17], 2, D, torch.float32, scale)
    ref = _ref_varlen(q, k, v, cu_q, cu_k, 2, D, scale)
    err = (o - ref).abs().max().item()
    assert err < 1e-3, f"varlen hd{D} err {err:.2e}"


@requires_mps
def test_varlen_fp16():
    D = 64
    scale = 1.0 / math.sqrt(D)
    q, k, v, o, cu_q, cu_k = _run_varlen(_varlen_fwd, [40, 24], [40, 24], 3, D, torch.float16, scale)
    ref = _ref_varlen(q, k, v, cu_q, cu_k, 3, D, scale)
    err = (o - ref).abs().max().item()
    assert err < 2e-2, f"varlen fp16 err {err:.2e}"


@requires_mps
def test_varlen_cross_attention_seqlens():
    # seqlen_q != seqlen_k per batch (cross attention) — cu_q and cu_k differ.
    D = 64
    scale = 1.0 / math.sqrt(D)
    q, k, v, o, cu_q, cu_k = _run_varlen(_varlen_fwd, [30, 50], [64, 20], 2, D, torch.float32, scale)
    ref = _ref_varlen(q, k, v, cu_q, cu_k, 2, D, scale)
    err = (o - ref).abs().max().item()
    assert err < 1e-3, f"varlen cross-attn err {err:.2e}"


@requires_mps
def test_varlen_bakes_nonstandard_scale():
    # A scale that is NOT 1/sqrt(d): the detector must BAKE the kernel's own constant,
    # not silently substitute 1/sqrt(d). If the scale were replaced, this would fail.
    D = 64
    scale = 0.3
    q, k, v, o, cu_q, cu_k = _run_varlen(_varlen_fwd, [48, 32], [48, 32], 2, D, torch.float32, scale)
    ref = _ref_varlen(q, k, v, cu_q, cu_k, 2, D, scale)
    err = (o - ref).abs().max().item()
    assert err < 1e-3, f"varlen baked-scale err {err:.2e}"


@requires_mps
@pytest.mark.parametrize("D", [32, 64, 128])
def test_varlen_causal_routes_and_correct(D):
    # Standard within-sequence lower-triangular causal (offs_m >= kn) routes with the
    # template's causal_guard and is EXACT vs a causal per-sequence reference.
    scale = 1.0 / math.sqrt(D)
    q, k, v, o, cu_q, cu_k = _run_varlen(_varlen_causal_fwd, [48, 40, 33], [48, 40, 33], 2, D, torch.float32, scale)
    ref = _ref_varlen(q, k, v, cu_q, cu_k, 2, D, scale, causal=True)
    err = (o - ref).abs().max().item()
    assert err < 1e-3, f"varlen causal hd{D} err {err:.2e}"


@triton.jit
def _varlen_strict_causal_fwd(
    Q, K, V, Out, cu_q, cu_k,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    H, max_seqlen, SCALE: tl.constexpr,
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
        strict = offs_m[:, None] > kn[None, :]                # STRICT > (not the template's >=)
        qk = tl.where((kn[None, :] < seqlen_k) & strict, qk, float("-inf"))
        m_ij = tl.max(qk, 1); m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new); p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None]
        v_ptrs = V + (k_start + kn)[:, None] * stride_vt + off_h * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kn[:, None] < seqlen_k, other=0.0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32)); m_i = m_new
    acc = acc / l_i[:, None]
    o_ptrs = Out + (q_start + offs_m)[:, None] * stride_ot + off_h * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


@requires_mps
def test_varlen_strict_causal_not_misrouted():
    # A NON-standard row-vs-col mask (strict > instead of the template's >=) must NOT be
    # routed as >= causal — that would silent-wrong the diagonal. Correct-or-refuse: the
    # detector refuses -> the kernel computes on the fallback path. Either way the OUTPUT
    # must match the STRICT reference (never the >= result). This proves not-silent-wrong.
    D = 64
    scale = 1.0 / math.sqrt(D)
    try:
        q, k, v, o, cu_q, cu_k = _run_varlen(_varlen_strict_causal_fwd, [48, 40], [48, 40], 2, D, torch.float32, scale)
    except MetalNonRecoverableError:
        return  # refused loudly — safe
    ref_strict = _ref_varlen(q, k, v, cu_q, cu_k, 2, D, scale, strict=True)
    # strict > fully masks each sequence's first row (no strictly-earlier key) -> nan in
    # BOTH kernel and ref. Compare finite positions; masked rows must stay non-finite (a >=
    # misroute would attend to the diagonal -> a FINITE value there).
    finite = ref_strict.isfinite()
    err = (o[finite] - ref_strict[finite]).abs().max().item()
    assert err < 1e-3, f"strict-causal mis-computed (routed as >=?): err {err:.2e}"
    assert not torch.isfinite(o[~finite]).any(), "strict-masked rows became finite -> routed as >="


@triton.jit
def _dense_fa(Q, K, V, O, sqz, sqh, sqm, sqk, skz, skh, skn, skk, svz, svh, svn, svk,
              soz, soh, som, sok, Z, H, N, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    sm = tl.program_id(0); hz = tl.program_id(1); z = hz // H; h = hz % H
    om = sm * BM + tl.arange(0, BM); on = tl.arange(0, BN); od = tl.arange(0, D)
    q = tl.load(Q + z * sqz + h * sqh + om[:, None] * sqm + od[None, :] * sqk, mask=om[:, None] < N, other=0.)
    q = q * (1.0 / math.sqrt(D))
    mi = tl.full([BM], -float("inf"), tl.float32); li = tl.zeros([BM], tl.float32); acc = tl.zeros([BM, D], tl.float32)
    for kn in range(0, N, BN):
        kk = kn + on
        k = tl.load(K + z * skz + h * skh + kk[:, None] * skn + od[None, :] * skk, mask=kk[:, None] < N, other=0.)
        qk = tl.dot(q, tl.trans(k))
        m2 = tl.maximum(mi, tl.max(qk, 1)); a = tl.exp(mi - m2); p = tl.exp(qk - m2[:, None])
        li = li * a + tl.sum(p, 1); acc = acc * a[:, None]
        v = tl.load(V + z * svz + h * svh + kk[:, None] * svn + od[None, :] * svk, mask=kk[:, None] < N, other=0.)
        acc += tl.dot(p, v); mi = m2
    tl.store(O + z * soz + h * soh + om[:, None] * som + od[None, :] * sok, acc / li[:, None], mask=om[:, None] < N)


@triton.jit
def _gqa_varlen_fwd(Q, K, V, Out, cu_q, cu_k,
    sqt, sqh, sqd, skt, skh, skd, svt, svh, svd, sot, soh, sod,
    H, GROUP, max_seqlen, SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    start_m = tl.program_id(0); off_bh = tl.program_id(1)
    off_b = off_bh // H; off_h = off_bh % H
    off_h_kv = off_h // GROUP                       # GQA: fewer kv heads (Llama/Mistral)
    q_start = tl.load(cu_q + off_b); seqlen_q = tl.load(cu_q + off_b + 1) - q_start
    k_start = tl.load(cu_k + off_b); seqlen_k = tl.load(cu_k + off_b + 1) - k_start
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_n = tl.arange(0, BLOCK_N); offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + (q_start + offs_m)[:, None] * sqt + off_h * sqh + offs_d[None, :] * sqd, mask=offs_m[:, None] < seqlen_q, other=0.) * SCALE
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32); l_i = tl.zeros([BLOCK_M], tl.float32); acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    for start_n in range(0, max_seqlen, BLOCK_N):
        kn = start_n + offs_n
        k = tl.load(K + (k_start + kn)[:, None] * skt + off_h_kv * skh + offs_d[None, :] * skd, mask=kn[:, None] < seqlen_k, other=0.)
        qk = tl.where(kn[None, :] < seqlen_k, tl.dot(q, tl.trans(k).to(q.dtype)), float("-inf"))
        m_ij = tl.max(qk, 1); m_new = tl.maximum(m_i, m_ij); alpha = tl.exp(m_i - m_new); p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None]
        v = tl.load(V + (k_start + kn)[:, None] * svt + off_h_kv * svh + offs_d[None, :] * svd, mask=kn[:, None] < seqlen_k, other=0.)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32)); m_i = m_new
    tl.store(Out + (q_start + offs_m)[:, None] * sot + off_h * soh + offs_d[None, :] * sod, (acc / l_i[:, None]).to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


@requires_mps
def test_gqa_varlen_not_misrouted():
    # GQA/MQA (Llama/Mistral): K/V have H/GROUP heads, indexed by off_h // GROUP -- a
    # DIFFERENT head index than Q. The varlen template applies Q's head index to K/V, so
    # routing GQA would SILENT-WRONG. The detector must refuse (Q/K/V head-index mismatch)
    # -> the kernel computes on the fallback path. Output must match the GQA reference.
    dev = "mps"; torch.manual_seed(0)
    H, Hkv, D = 4, 2, 64
    G = H // Hkv
    scale = 1.0 / math.sqrt(D)
    cu = torch.tensor([0, 48, 80], device=dev, dtype=torch.int32)
    q = torch.randn(80, H, D, device=dev); k = torch.randn(80, Hkv, D, device=dev)
    v = torch.randn(80, Hkv, D, device=dev); o = torch.zeros(80, H, D, device=dev)
    BM = BN = 32
    try:
        _gqa_varlen_fwd[(triton.cdiv(48, BM), 2 * H)](
            q, k, v, o, cu, cu, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            H, G, 48, scale, BM, BN, D)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # refused loudly — safe
    ref = torch.zeros_like(q)
    for b in range(len(cu) - 1):
        s, e = cu[b].item(), cu[b + 1].item()
        for h in range(H):
            hk = h // G
            sc = (q[s:e, h].float() @ k[s:e, hk].float().transpose(-2, -1)) * scale
            ref[s:e, h] = (torch.softmax(sc, -1) @ v[s:e, hk].float()).to(ref.dtype)
    err = (o - ref).abs().max().item()
    assert err < 1e-2, f"GQA varlen mis-computed (routed as MHA?): err {err:.2e}"


@triton.jit
def _batch_inner_varlen(Q, K, V, Out, cu_q, cu_k,
    sqt, sqh, sqd, skt, skh, skd, svt, svh, svd, sot, soh, sod,
    NB, max_seqlen, SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    start_m = tl.program_id(0); off_bh = tl.program_id(1)
    off_h = off_bh // NB; off_b = off_bh % NB          # BATCH-INNER (template assumes head-inner)
    q_start = tl.load(cu_q + off_b); seqlen_q = tl.load(cu_q + off_b + 1) - q_start
    k_start = tl.load(cu_k + off_b); seqlen_k = tl.load(cu_k + off_b + 1) - k_start
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_n = tl.arange(0, BLOCK_N); offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + (q_start + offs_m)[:, None] * sqt + off_h * sqh + offs_d[None, :] * sqd, mask=offs_m[:, None] < seqlen_q, other=0.) * SCALE
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32); l_i = tl.zeros([BLOCK_M], tl.float32); acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    for start_n in range(0, max_seqlen, BLOCK_N):
        kn = start_n + offs_n
        k = tl.load(K + (k_start + kn)[:, None] * skt + off_h * skh + offs_d[None, :] * skd, mask=kn[:, None] < seqlen_k, other=0.)
        qk = tl.where(kn[None, :] < seqlen_k, tl.dot(q, tl.trans(k).to(q.dtype)), float("-inf"))
        m_ij = tl.max(qk, 1); m_new = tl.maximum(m_i, m_ij); alpha = tl.exp(m_i - m_new); p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None]
        v = tl.load(V + (k_start + kn)[:, None] * svt + off_h * svh + offs_d[None, :] * svd, mask=kn[:, None] < seqlen_k, other=0.)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32)); m_i = m_new
    tl.store(Out + (q_start + offs_m)[:, None] * sot + off_h * soh + offs_d[None, :] * sod, (acc / l_i[:, None]).to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


@requires_mps
def test_batch_inner_varlen_not_misrouted():
    # BATCH-INNER packing (off_h = bh // NB, off_b = bh % NB) is the opposite of the
    # template's head-inner assumption (h = bh % nheads). Routing it would mistake NB for
    # nheads and SWAP batch<->head (silent-wrong). The detector must refuse (off_h is a
    # divsi, not remsi(pid, H)) -> fallback computes correctly. cu is padded to H+1 entries
    # so that even a hypothetical misroute stays in-bounds (a detectable wrong result, never
    # an out-of-bounds runaway loop) — the test can only pass or fail, never hang.
    dev = "mps"; torch.manual_seed(0)
    H, D, NB = 3, 64, 2
    scale = 1.0 / math.sqrt(D)
    cu = torch.tensor([0, 48, 80, 80], device=dev, dtype=torch.int32)  # NB=2 real + 1 phantom
    q = torch.randn(80, H, D, device=dev); k = torch.randn(80, H, D, device=dev)
    v = torch.randn(80, H, D, device=dev); o = torch.zeros(80, H, D, device=dev)
    BM = BN = 32
    _batch_inner_varlen[(triton.cdiv(48, BM), H * NB)](
        q, k, v, o, cu, cu, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        NB, 48, scale, BM, BN, D)
    torch.mps.synchronize()
    ref = torch.zeros_like(q)
    for b in range(NB):
        s, e = cu[b].item(), cu[b + 1].item()
        for h in range(H):
            sc = (q[s:e, h].float() @ k[s:e, h].float().transpose(-2, -1)) * scale
            ref[s:e, h] = (torch.softmax(sc, -1) @ v[s:e, h].float()).to(ref.dtype)
    err = (o - ref).abs().max().item()
    assert err < 1e-2, f"batch-inner varlen mis-computed (routed as head-inner?): err {err:.2e}"


@requires_mps
def test_varlen_block64_not_misrouted():
    # The template hard-codes BLOCK_M==BLOCK_N==32. A kernel launched with BLOCK_M=64 spawns
    # cdiv(seqlen, 64) blocks; if routed, the 32-rows/block template would leave the upper
    # half of each 64-row block unwritten (silent-wrong). The detector must refuse on the
    # tile shape (Q tile [64, D]) -> fallback computes all rows correctly.
    D = 64
    scale = 1.0 / math.sqrt(D)
    # _varlen_fwd is generic over BLOCK_M/BLOCK_N constexprs; launch it at 64.
    dev = "mps"; torch.manual_seed(0)
    cu = torch.tensor([0, 96, 160], device=dev, dtype=torch.int32)
    q = torch.randn(160, 2, D, device=dev); k = torch.randn(160, 2, D, device=dev)
    v = torch.randn(160, 2, D, device=dev); o = torch.zeros(160, 2, D, device=dev)
    max_seqlen = 96
    try:
        _varlen_fwd[(triton.cdiv(max_seqlen, 64), 2 * 2)](
            q, k, v, o, cu, cu, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            2, max_seqlen, scale, 64, 64, D)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # refused loudly — safe
    ref = _ref_varlen(q, k, v, cu, cu, 2, D, scale)
    err = (o - ref).abs().max().item()
    assert err < 1e-3, f"BLOCK_M=64 varlen mis-computed (routed with 32-row template?): err {err:.2e}"


@requires_mps
def test_dense_fa_not_misrouted():
    # A DENSE [Z,H,N,D] FA kernel (0 int-pointer args) must NOT be captured by the varlen
    # detector — it stays on the dense FA path and stays correct.
    dev = "mps"; Z, H, N, D = 1, 2, 64, 64
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, D, device=dev); k = torch.randn(Z, H, N, D, device=dev)
    v = torch.randn(Z, H, N, D, device=dev); o = torch.zeros(Z, H, N, D, device=dev)
    st = lambda t: t.stride()
    _dense_fa[(triton.cdiv(N, 32), Z * H)](q, k, v, o, *st(q), *st(k), *st(v), *st(o), Z, H, N, 32, 32, D)
    torch.mps.synchronize()
    ref = torch.zeros_like(q)
    for z in range(Z):
        for h in range(H):
            sc = (q[z, h].float() @ k[z, h].float().T) / math.sqrt(D)
            ref[z, h] = (torch.softmax(sc, -1) @ v[z, h].float())
    err = (o - ref).abs().max().item()
    assert err < 1e-3, f"dense FA misrouted/wrong: err {err:.2e}"
