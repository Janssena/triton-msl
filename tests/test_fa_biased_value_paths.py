"""Packet 113/114: the biased (trifast) replacement must prove the VALUES it reconstructs,
not only the masks on the memory operations that feed them.

On ``e2330ec`` the biased route identified a Bias pointer through arithmetic, never tied the
QK dot's accumulator to the Bias load, counted a boolean Mask load without proving it drives
the score select, treated every ``arith.select`` as transparent, hard-coded ``-INFINITY``
where the source carries a runtime ``neg_inf`` scalar, and never traced the Lse value. GPT's
113 rows (all one specialized dispatch, finite wrong output): Lse ``+1`` (lse err 1.0), Bias
``*2`` (0.94 / 3.15), inverted Mask (3.38), reversed select polarity (2.84), zero fill in the
Mask select (0.27), zero fill in the tile-tail select at N = 48 (0.27), runtime ``neg_inf =
0`` (0.15), and the usual ``-1e9`` sentinel with every key masked (template non-finite,
source finite).

Now ``_fa_verify_value_paths(biased=...)`` proves: the accumulator IS the unique Bias load;
the softmax input is exactly ``select(row & col boundary, [select(Mask != 0, S, score)],
S)`` on ONE sentinel SSA value ``S``; the constant scale sits between the dot and the
selects; only those selects are transparent to the score-scale walker; the row max reduces
the same select; the Lse store is ``max * (1/c) + log(l)`` on the loop's verified results.
The sentinel is FORWARDED into both templates (runtime fp32 arg, or a literal) in
natural units, and every masked / tail cell holds it and counts in the sum — so a fully
masked row reproduces the source's finite uniform recurrence exactly. A literal ``-inf`` is
also exact: a nonfinite row is contained internally and restored as NaN at its stores.
"""

import importlib.util
import math
import pathlib

import pytest

try:
    import torch
    import triton
    import triton.language as tl

    import triton_msl.autotuning._fa_dispatch as fa_dispatch
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))


@pytest.fixture
def fa_spy(monkeypatch):
    hits = []
    real = fa_dispatch.dispatch_flash_attention

    def spy(*a, **k):
        r = real(*a, **k)
        hits.append(bool(r))
        return r

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)
    return hits


if HAS:

    @triton.jit
    def _biased_value(
        o_ptr, o_sz, o_sh, o_sm, o_sk,
        lse_ptr, lse_sz, lse_sh, lse_sm,
        q_ptr, q_sz, q_sh, q_sm, q_sk,
        k_ptr, k_sz, k_sh, k_sn, k_sk,
        v_ptr, v_sz, v_sh, v_sn, v_sk,
        b_ptr, b_sz, b_sh, b_sm, b_sn,
        mask_ptr, m_sz, m_sh, m_sn,
        sm_scale, neg_inf, Z, H, N,
        DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        MODE: tl.constexpr,
    ):
        """GPT's 113 harness. 0 canonical; 1 Lse + 1; 2 Bias * 2; 3 inverted loaded Mask;
        4 reversed Mask-select polarity; 5 zero fill in the Mask select; 6 zero fill in the
        tile-tail select; 7/8/9/10 canonical spelling (runtime sentinel / N varied by the
        caller); 11 a literal -inf in BOTH selects (with a Mask load)."""
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
            if MODE == 2:
                b_block = b_block * 2.0
            m_block = tl.load(mask_ptrs, mask_n)
            if MODE == 3:
                m_block = m_block == 0
            scores = tl.dot(q_block, kt_block, b_block.to(tl.float32))
            scores *= inv_ln2
            if MODE == 4:
                scores = tl.where(m_block[None, :], scores, neg_inf)
            elif MODE == 5:
                scores = tl.where(m_block[None, :], 0.0, scores)
            elif MODE == 11:
                scores = tl.where(m_block[None, :], -float("inf"), scores)
            else:
                scores = tl.where(m_block[None, :], neg_inf, scores)
            if MODE == 6:
                scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, 0.0)
            elif MODE == 11:
                scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, -float("inf"))
            else:
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
        if MODE == 1:
            lse = lse + 1.0
        tl.store(lse_ptrs, lse, mask=mask_m)


def _reference(q, k, v, bias, mask, scale, mode, neg_inf, n_tiles_cols):
    """GPT's 113 oracle: the SOURCE program modelled directly in its base-2 domain, with the
    final tile's invalid columns (N % 32 != 0) carrying the sentinel (or the mode-6 zero)
    and zero V rows — they count in the recurrence exactly as the source computes them."""
    bias = bias.float() * (2.0 if mode == 2 else 1.0)
    scores = (scale * (q.float() @ k.float().transpose(-2, -1)) + bias) * (1.0 / math.log(2.0))
    masked = mask.bool()
    if mode == 3:
        masked = ~masked
    fill = float("-inf") if mode == 11 else neg_inf
    if mode == 4:
        scores = scores.masked_fill(~masked[:, :, None, :], fill)
    elif mode == 5:
        scores = scores.masked_fill(masked[:, :, None, :], 0.0)
    else:
        scores = scores.masked_fill(masked[:, :, None, :], fill)
    pad = n_tiles_cols - scores.shape[-1]
    v = v.float()
    if pad:
        tail = 0.0 if mode == 6 else fill
        scores = torch.cat([scores, torch.full((*scores.shape[:-1], pad), tail, device=scores.device)], dim=-1)
        v = torch.cat([v, torch.zeros(*v.shape[:-2], pad, v.shape[-1], device=v.device)], dim=-2)
    # Model the SOURCE loop, not a mathematically reassociated one-shot softmax.  The
    # distinction is observable for nonfinite data: a first all--inf tile makes the
    # carried state NaN even if a later tail tile contains a finite runtime sentinel.
    maximum = torch.full(scores.shape[:-1], float("-inf"), device=scores.device)
    denom = torch.zeros_like(maximum)
    acc = torch.zeros(*scores.shape[:-1], v.shape[-1], device=scores.device)
    for start in range(0, n_tiles_cols, 32):
        score_block = scores[..., start : start + 32]
        block_max = torch.maximum(maximum, score_block.max(dim=-1).values)
        numer = torch.exp2(score_block - block_max[..., None])
        alpha = torch.exp2(maximum - block_max)
        denom = denom * alpha + numer.sum(dim=-1)
        acc = acc * alpha[..., None] + numer @ v[..., start : start + 32, :]
        maximum = block_max
    out = acc / denom[..., None]
    lse = maximum * math.log(2.0) + torch.log(denom) + (1.0 if mode == 1 else 0.0)
    return out, lse


def _run(
    mode,
    n=64,
    neg_inf=-1e9,
    all_masked=False,
    seed=None,
    d=64,
    bias_neg_inf_rows=(),
    mask_all_false=False,
    force_tiled=False,
):
    if hasattr(_biased_value, "device_caches"):
        _biased_value.device_caches.clear()
    torch.manual_seed(13100 + mode if seed is None else seed)
    z, h = 1, 2
    scale = 1.0 / math.sqrt(d)
    q, k, v = (torch.randn(z, h, n, d, device=D) for _ in range(3))
    bias = torch.randn(z, h, n, n, device=D)
    mask = (torch.rand(z, h, n, device=D) < 0.25).to(torch.uint8)
    for row in bias_neg_inf_rows:
        bias[..., row, :] = float("-inf")
    if all_masked:
        mask.fill_(1)
    if mask_all_false:
        mask.zero_()
    if force_tiled:
        out_storage = torch.full((z, h, n, d * 2), float("nan"), device=D)
        out = out_storage[..., ::2]
    else:
        out = torch.full_like(q, float("nan"))
    lse = torch.full((z, h, n), float("nan"), device=D)
    st = lambda t: tuple(t.stride())
    try:
        _biased_value[((n + 31) // 32, z * h)](
            out, *st(out), lse, *st(lse), q, *st(q), k, *st(k), v, *st(v), bias, *st(bias), mask, *st(mask),
            scale, neg_inf, z, h, n, d, 32, 32, mode,
        )
        torch.mps.synchronize()
        state = "computed"
    except MetalNonRecoverableError:
        state = "refused"
    ref = _reference(q, k, v, bias, mask, scale, mode, neg_inf, ((n + 31) // 32) * 32)
    return state, out, lse, ref


def _assert_matches(out, lse, ref, label):
    out_ref, lse_ref = ref
    assert torch.isfinite(out).all() == torch.isfinite(out_ref).all(), f"{label}: finiteness differs from the source"
    assert (out.float() - out_ref).abs().max().item() < 2e-3, label
    fin = torch.isfinite(lse_ref)
    assert (lse[fin] - lse_ref[fin]).abs().max().item() < 2e-3, label


def _assert_nonfinite_matches(out, lse, ref, label):
    """Compare nonfinite placement exactly, then compare every mutually finite value."""
    out_ref, lse_ref = ref
    for got, want, name in ((out, out_ref, "Out"), (lse, lse_ref, "Lse")):
        assert torch.equal(torch.isnan(got), torch.isnan(want)), f"{label}: {name} NaN placement differs"
        assert torch.equal(torch.isposinf(got), torch.isposinf(want)), f"{label}: {name} +inf placement differs"
        assert torch.equal(torch.isneginf(got), torch.isneginf(want)), f"{label}: {name} -inf placement differs"
        finite = torch.isfinite(got) & torch.isfinite(want)
        if finite.any():
            assert (got[finite].float() - want[finite].float()).abs().max().item() < 2e-3, label


@requires_gpu
@pytest.mark.parametrize(
    "mode,n,neg_inf,all_masked,label",
    [
        (0, 64, -1e9, False, "canonical control"),
        (7, 64, 0.0, False, "canonical spelling, runtime neg_inf = 0 (forwarded, not approximated)"),
        (8, 64, -1e9, True, "every key masked, runtime -1e9: finite uniform recurrence"),
        (9, 48, -1e9, False, "canonical at N = 48: the tile tail holds the sentinel"),
        (10, 48, -1e9, True, "every key masked at N = 48: 16 tail cells count in the denominator"),
    ],
)
def test_biased_canonical_forms_route_and_match(cold_gpu_caches, fa_spy, mode, n, neg_inf, all_masked, label):
    """Bites on e2330ec for modes 8 and 10 (the template hard-coded -INFINITY: non-finite
    where the source is finite) and 7 (err 0.15); 0 and 9 are controls."""
    state, out, lse, ref = _run(mode, n=n, neg_inf=neg_inf, all_masked=all_masked)
    assert state == "computed", label
    assert True in fa_spy, f"{label}: the biased replacement must be the path taken"
    _assert_matches(out, lse, ref, label)


@requires_gpu
@pytest.mark.parametrize(
    "mode,n,label",
    [
        (1, 64, "Lse epilogue (+1)"),
        (2, 64, "Bias transformed (*2) before the dot"),
        (3, 64, "loaded Mask inverted before its select"),
        (4, 64, "Mask select polarity reversed"),
        (5, 64, "Mask select fills zero, not the sentinel"),
        (6, 48, "tile-tail select fills zero, not the sentinel"),
    ],
)
def test_biased_unreplayed_values_refuse_or_match(cold_gpu_caches, fa_spy, mode, n, label):
    """Bites on e2330ec for modes 1-6 (113 §3: one dispatch each, finite wrong output).
    Now: refused before any dispatch, or the exact source semantics."""
    state, out, lse, ref = _run(mode, n=n)
    if state == "computed":
        _assert_matches(out, lse, ref, label)
    else:
        assert True not in fa_spy, f"{label}: refusal must happen before any dispatch"


@requires_gpu
@pytest.mark.parametrize("force_tiled,n", [(False, 64), (True, 64), (True, 48)])
def test_literal_neg_inf_with_loaded_mask_is_exact(
    cold_gpu_caches, fa_spy, force_tiled, n
):
    """The old conservative guard can be relaxed only after the fully-masked row
    is exact.  This pin uses the literal in both source selects, forces every Mask
    entry true, and covers simd, tiled, and the partial final tile.
    """
    state, out, lse, ref = _run(
        11,
        n=n,
        all_masked=True,
        seed=9073,
        force_tiled=force_tiled,
    )
    assert state == "computed"
    assert True in fa_spy
    _assert_nonfinite_matches(
        out, lse, ref, f"literal -inf + loaded Mask, tiled={force_tiled}, n={n}"
    )


@requires_gpu
@pytest.mark.parametrize("force_tiled,n", [(False, 64), (True, 64), (True, 48)])
@pytest.mark.parametrize("neg_inf", [float("-inf"), float("inf"), float("nan")])
def test_runtime_nonfinite_sentinel_is_exact_or_refuses_before_dispatch(
    cold_gpu_caches, fa_spy, force_tiled, n, neg_inf
):
    """Packet 115 P0-A: the literal-only check did not constrain a runtime scalar.

    On packet 114 all nine rows route. Out happens to be NaN like the source, but both
    simd and tiled templates collapse the source's NaN Lse to -inf via ``l > 0``.
    """
    state, out, lse, ref = _run(
        0,
        n=n,
        neg_inf=neg_inf,
        all_masked=True,
        seed=9071,
        force_tiled=force_tiled,
    )
    if state == "computed":
        _assert_nonfinite_matches(
            out, lse, ref, f"runtime sentinel {neg_inf}, tiled={force_tiled}, n={n}"
        )
    else:
        assert True not in fa_spy


@requires_gpu
@pytest.mark.parametrize(
    "force_tiled,n,rows",
    [
        (False, 64, (0,)),      # simd: one bad row must not poison its 8-row MMA fragment
        (False, 64, tuple(range(64))),
        (True, 64, (0,)),       # tiled fallback via a proven non-unit output stride
        (True, 48, (0,)),       # tiled fallback + partial final key tile
    ],
)
def test_loaded_neg_inf_bias_is_exact_or_refuses_before_dispatch(
    cold_gpu_caches, fa_spy, force_tiled, n, rows
):
    """Packet 115 P0-B: runtime data can make a source row nonfinite after detection."""
    state, out, lse, ref = _run(
        0,
        n=n,
        neg_inf=-1e9,
        mask_all_false=True,
        bias_neg_inf_rows=rows,
        seed=9071,
        force_tiled=force_tiled,
    )
    if state == "computed":
        _assert_nonfinite_matches(
            out, lse, ref, f"loaded -inf Bias, tiled={force_tiled}, n={n}, rows={rows}"
        )
    else:
        assert True not in fa_spy


# ------------------------------------------------------------------ maskless, literal -inf


def _alibi_kernel():
    p = pathlib.Path(__file__).with_name("test_fa_alibi.py")
    spec = importlib.util.spec_from_file_location("alibi_value_src", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod._alibi_fa
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    return fn


@requires_gpu
@pytest.mark.parametrize("N", [96, 48])
def test_maskless_literal_neg_inf_sentinel_routes_and_matches(cold_gpu_caches, fa_spy, N):
    """The ALiBi-style maskless biased kernel spells its boundary select with a literal
    -inf: admitted (no Mask load -> no stored row can be fully masked), decoded from the
    walker's raw bit word (0xFF800000 — the packet-114 candidate's first cut baked that word
    as a finite +4.3e9 sentinel and zeroed every output; §7 of the packet)."""
    fn = _alibi_kernel()
    Z, H, DIM = 1, 4, 64
    torch.manual_seed(0)
    sm = 1.0 / math.sqrt(DIM)
    q, k, v = (torch.randn(Z, H, N, DIM, device=D) for _ in range(3))
    b = torch.randn(Z, H, N, N, device=D)
    o = torch.full((Z, H, N, DIM), float("nan"), device=D)
    lse = torch.full((Z, H, N), float("nan"), device=D)
    st = lambda t: tuple(t.stride())
    fn[(triton.cdiv(N, 32), Z * H)](o, *st(o), lse, *st(lse), q, *st(q), k, *st(k), v, *st(v), b, *st(b), sm, Z, H, N, DIM, 32, 32)
    torch.mps.synchronize()
    assert True in fa_spy, "the biased replacement must be the path taken"
    raw = sm * (q @ k.transpose(-2, -1)) + b
    p = torch.softmax(raw, dim=-1)
    assert (o - p @ v).abs().max().item() < 1e-3
    assert (lse - torch.logsumexp(raw, dim=-1)).abs().max().item() < 1e-3


@requires_gpu
@pytest.mark.parametrize("force_tiled", [False, True])
def test_maskless_loaded_neg_inf_bias_row_is_exact(cold_gpu_caches, fa_spy, force_tiled):
    """The maskless ALiBi spelling has the same data-dependent Bias boundary.

    A literal boundary sentinel is structurally safe, but a loaded Bias row may still be
    all ``-inf``.  Simd must contain the bad row; tiled must not map its NaN state to zero.
    """
    fn = _alibi_kernel()
    z, h, n, dim = 1, 2, 64, 64
    torch.manual_seed(9072)
    sm = 1.0 / math.sqrt(dim)
    q, k, v = (torch.randn(z, h, n, dim, device=D) for _ in range(3))
    bias = torch.randn(z, h, n, n, device=D)
    bias[..., 0, :] = float("-inf")
    if force_tiled:
        out_storage = torch.full((z, h, n, dim * 2), float("nan"), device=D)
        out = out_storage[..., ::2]
    else:
        out = torch.full_like(q, float("nan"))
    lse = torch.full((z, h, n), float("nan"), device=D)
    st = lambda t: tuple(t.stride())
    fn[(triton.cdiv(n, 32), z * h)](
        out, *st(out), lse, *st(lse), q, *st(q), k, *st(k), v, *st(v),
        bias, *st(bias), sm, z, h, n, dim, 32, 32,
    )
    torch.mps.synchronize()
    assert True in fa_spy
    mask = torch.zeros(z, h, n, device=D, dtype=torch.uint8)
    ref = _reference(q, k, v, bias, mask, sm, 0, float("-inf"), n)
    _assert_nonfinite_matches(out, lse, ref, f"maskless loaded -inf Bias, tiled={force_tiled}")
