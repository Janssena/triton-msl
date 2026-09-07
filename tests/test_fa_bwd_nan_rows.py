"""Packet 134: the biased-FA BACKWARD templates must reproduce the source's NaN semantics.

Packet 117 made the forward store a faithful NaN lse. The backward makers recompute
``P = exp2((s - lse) * inv_ln2)`` but hard-coded ``P = 0`` on every masked or out-of-range
cell. The trifast source applies ONE sentinel through two ``tl.where`` selects (bounds and
loaded Mask, either order) and THEN subtracts the row's lse, so a NaN lse poisons masked
cells too: ``exp2((sentinel - NaN) * inv_ln2)`` is NaN. Census on the committed tree
(cbff49c): dK / dV with a NaN lse row left exactly the masked key columns finite (56 of 64
rows NaN where the source has 64) on both the simd and scalar kv makers; dbias likewise
left the masked cells of that row at 0 where the source has NaN.

Per-query-row outputs (dQ, delta) were already row-local: a plain fragment multiply keeps
a NaN row in its own output row (the packet-117 mechanism needed the diagonal rescale,
which the backward does not use). Those rows are pinned as controls.

Now the backward detector proves the two selects and their ONE shared sentinel (an fp32
argument or a literal), the lowering forwards it, and every maker computes
``exp2((neg_inf - lse) * inv_ln2)`` on in-bounds masked cells — which underflows to exactly
0 for finite data (bit-identical to before) and propagates NaN otherwise.
"""

import math
import sys

import pytest

try:
    import torch
    import triton

    import triton_msl.codegen._msl_templates as M
    from triton_msl.errors import MetalNonRecoverableError

    sys.path.insert(0, "tests")
    from test_fa_bwd_routing import _bwd_b, _bwd_kv, _bwd_q

    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")


@pytest.fixture
def maker_spy(monkeypatch):
    hits = []
    for name in ("make_flash_attention_bwd_kv_kernel", "make_flash_attention_bwd_kv_kernel_simd",
                 "make_flash_attention_bwd_q_kernel", "make_flash_attention_bwd_q_kernel_simd",
                 "make_flash_attention_bwd_b_kernel"):
        real = getattr(M, name)

        def spy(*a, _r=real, _n=name, **k):
            hits.append(_n.replace("make_flash_attention_bwd_", ""))
            return _r(*a, **k)

        monkeypatch.setattr(M, name, spy)
    return hits


def _problem(DIM, poison, I=1, N=64, seed=11):
    """Inputs plus the SOURCE-algorithm reference (trifast math, masked_fill -inf, exp of
    (raw - lse)) with one row poisoned."""
    Hc = Hh = 1
    torch.manual_seed(seed)
    sm = 1.0 / math.sqrt(DIM)
    q, k, v, do = (torch.randn(Hc, I, N, DIM, device=D) for _ in range(4))
    bias = torch.randn(Hc, N, N, device=D)
    mask = (torch.rand(Hc // Hh, I, N, device=D) < 0.15).to(torch.uint8)
    if poison == "Q":
        q[0, 0, 5, :] = float("nan")
    if poison == "dO":
        do[0, 0, 5, :] = float("nan")
    raw = (torch.einsum("hijd,hikd->hijk", q, k) * sm + bias[:, None]).masked_fill(mask[:, :, None, :].bool(), float("-inf"))
    lse = torch.logsumexp(raw, -1).clone()
    o = (torch.softmax(raw, -1) @ v).contiguous()
    delta = (o * do).sum(-1).contiguous()
    if poison == "lse":
        lse[0, 0, 5] = float("nan")
    P = torch.exp(raw - lse[..., None])
    dP = torch.einsum("hijd,hikd->hijk", do, v)
    dS = P * (dP - delta[..., None])
    ref = dict(
        dQ=sm * torch.einsum("hijk,hikd->hijd", dS, k),
        dK=sm * torch.einsum("hijk,hijd->hikd", dS, q),
        dV=torch.einsum("hijk,hijd->hikd", P, do),
        db=dS.sum(1),
        delta=delta,
    )
    return dict(q=q, k=k, v=v, do=do, bias=bias, mask=mask, lse=lse, o=o, delta=delta, sm=sm, Hh=Hh, N=N, DIM=DIM), ref


def _assert_nan_exact(got, ref, label):
    assert torch.equal(torch.isnan(got), torch.isnan(ref)), f"{label}: NaN placement differs from the source"
    fin = ~torch.isnan(ref)
    if fin.any():
        assert (got[fin] - ref[fin]).abs().max().item() < 1e-2, label


_st = lambda t: tuple(t.stride())


def _run_kv(p, DIM):
    N = p["N"]; BJ = BK = 32
    dk = torch.zeros_like(p["k"]); dv = torch.zeros_like(p["v"])
    if hasattr(_bwd_kv, "device_caches"):
        _bwd_kv.device_caches.clear()
    _bwd_kv[(triton.cdiv(N, BK), 1, 1)](
        p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
        p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
        dk, *_st(dk), dv, *_st(dv), p["sm"], -1e9, N, p["Hh"], DIM, N, BJ, BK)
    torch.mps.synchronize()
    return dk, dv


def _run_q(p, DIM):
    N = p["N"]; BJ = BK = 32
    dq = torch.zeros_like(p["q"]); dlt = torch.zeros(1, 1, N, device=D)
    if hasattr(_bwd_q, "device_caches"):
        _bwd_q.device_caches.clear()
    _bwd_q[(triton.cdiv(N, BJ), 1, 1)](
        dlt, *_st(dlt), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]), p["bias"], *_st(p["bias"]),
        p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["o"], *_st(p["o"]), p["do"], *_st(p["do"]),
        dq, *_st(dq), p["sm"], -1e9, N, p["Hh"], DIM, N, BJ, BK)
    torch.mps.synchronize()
    return dq, dlt


@requires_gpu
@pytest.mark.parametrize("DIM,maker", [(32, "kv_kernel_simd"), (64, "kv_kernel")])
@pytest.mark.parametrize("poison", [None, "lse", "dO", "Q"])
def test_bwd_kv_nan_rows_match_the_source(cold_gpu_caches, maker_spy, DIM, maker, poison):
    """Bites on cbff49c (cell-level NaN placement on dK and dV separately): poison='lse'
    on both makers and poison='Q' on both makers (the template never computed P on masked
    cells, so the masked key columns stayed finite where the source has NaN), and
    poison='dO' on the SCALAR maker (its `P != 0` fast-skip dropped the NaN that
    `0 * (NaN - delta)` produces). The dO row on the simd maker and the controls agreed.
    A row-level census had called the Q / dO rows "agreeing"; the cell-level pin is the
    truth and is what is retained."""
    p, ref = _problem(DIM, poison)
    dk, dv = _run_kv(p, DIM)
    assert maker_spy == [maker]
    _assert_nan_exact(dk, ref["dK"], f"dK {maker} {poison}")
    _assert_nan_exact(dv, ref["dV"], f"dV {maker} {poison}")


@requires_gpu
@pytest.mark.parametrize("DIM,maker", [(32, "q_kernel_simd"), (64, "q_kernel")])
@pytest.mark.parametrize("poison", [None, "lse", "dO", "Q"])
def test_bwd_q_nan_rows_stay_row_local(cold_gpu_caches, maker_spy, DIM, maker, poison):
    """dQ and delta are per-query-row: exactly row 5 is NaN, every sibling row finite and
    exact (controls; the backward has no diagonal rescale, so no fragment contamination)."""
    p, ref = _problem(DIM, poison)
    dq, dlt = _run_q(p, DIM)
    assert maker_spy == [maker]
    _assert_nan_exact(dq, ref["dQ"], f"dQ {maker} {poison}")
    if poison in ("dO", "Q"):
        assert torch.isnan(dlt[0, 0, 5]) and torch.isfinite(dlt[0, 0, :5]).all() and torch.isfinite(dlt[0, 0, 6:]).all()
    else:
        assert torch.isfinite(dlt).all()


@requires_gpu
def test_bwd_b_nan_lse_row_matches_the_source(cold_gpu_caches, maker_spy):
    """Bites on cbff49c: dbias stores dS per cell; the source has NaN across the whole
    poisoned row (masked cells included), the old template left the masked cells at 0."""
    p, ref = _problem(32, "lse", I=64)
    N = p["N"]; BJ = BK = 32
    db = torch.zeros(1, N, N, device=D)
    if hasattr(_bwd_b, "device_caches"):
        _bwd_b.device_caches.clear()
    _bwd_b[(triton.cdiv(N, BJ), triton.cdiv(N, BK), 1)](
        p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
        p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
        db, *_st(db), p["sm"], -1e9, p["Hh"], N, 32, N, BJ, BK)
    torch.mps.synchronize()
    assert maker_spy == ["b_kernel"]
    _assert_nan_exact(db, ref["db"], "dbias NaN lse")


@requires_gpu
def test_bwd_sentinel_must_be_one_shared_value(cold_gpu_caches, maker_spy, tmp_path):
    """The suite's own trifast kv kernel, mutated in ONE place: the loaded-Mask select fills
    with a second scalar (`neg_inf2`) instead of the shared sentinel. Everything else is the
    recognized backward, so the detector's sentinel proof is what must refuse — before any
    maker is built."""
    import importlib.util
    import pathlib

    src = pathlib.Path(__file__).with_name("test_fa_bwd_routing.py").read_text()
    start = src.index("@triton.jit\ndef _bwd_kv(")
    end = src.index("\ndef ", start + 20)
    kernel_src = src[start:end]
    assert kernel_src.count("    neg_inf,\n") == 1 and kernel_src.count("tl.where(m_block[None, :], neg_inf, scores)") == 1
    kernel_src = kernel_src.replace("    neg_inf,\n", "    neg_inf,\n    neg_inf2,\n", 1)
    kernel_src = kernel_src.replace("tl.where(m_block[None, :], neg_inf, scores)", "tl.where(m_block[None, :], neg_inf2, scores)", 1)
    mod_path = tmp_path / "bwd_kv_two_sentinels.py"
    mod_path.write_text("import triton\nimport triton.language as tl\n\n" + kernel_src + "\n")
    spec = importlib.util.spec_from_file_location("bwd_kv_two_sentinels", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod._bwd_kv

    p, _ = _problem(32, None)
    N = p["N"]; dk = torch.zeros_like(p["k"]); dv = torch.zeros_like(p["v"])
    with pytest.raises(MetalNonRecoverableError, match="sentinel"):
        fn[(2, 1, 1)](
            p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
            p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
            dk, *_st(dk), dv, *_st(dv), p["sm"], -1e9, -2e9, N, p["Hh"], 32, N, 32, 32)
        torch.mps.synchronize()
    assert maker_spy == []


# ----------------------------------------------------------------------- packet 136 (135 HOLD)


def _mutated_bwd_kv(tmp_path, replacements, name):
    """The suite's own trifast kv kernel with exactly the given source replacements."""
    import importlib.util
    import pathlib

    src = pathlib.Path(__file__).with_name("test_fa_bwd_routing.py").read_text()
    import re

    start = src.index("@triton.jit\ndef _bwd_kv(")
    end = start + 20 + re.search(r"\n(?=@|def )", src[start + 20:]).start()
    kernel_src = src[start:end]
    for old, new in replacements:
        assert kernel_src.count(old) == 1, old
        kernel_src = kernel_src.replace(old, new, 1)
    mod_path = tmp_path / f"{name}.py"
    mod_path.write_text("import triton\nimport triton.language as tl\n\n" + kernel_src + "\n")
    spec = importlib.util.spec_from_file_location(name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._bwd_kv


_BOUNDS = "scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)"
_EXP = "sm_value = tl.math.exp2((scores - row_max[:, None]) * inv_ln2)"


@requires_gpu
@pytest.mark.parametrize(
    "label,replacements,field",
    [
        ("third select before the two", [(_BOUNDS, "scores = tl.where(mask_k[None, :], scores, neg_inf)\n        " + _BOUNDS)], "third select"),
        ("bounds select uses N // 2", [(_BOUNDS, "scores = tl.where(mask_j[:, None] & (k_idxs < N // 2)[None, :], scores, neg_inf)")], "N_CTX"),
        ("subtraction is lse - scores", [(_EXP, "sm_value = tl.math.exp2((row_max[:, None] - scores) * inv_ln2)")], "right-hand side"),
        ("exp2 coefficient is 1", [(_EXP, "sm_value = tl.math.exp2((scores - row_max[:, None]))")], "log2"),
    ],
)
def test_bwd_score_path_mutations_refuse_before_any_maker(cold_gpu_caches, maker_spy, tmp_path, label, replacements, field):
    """Packet 135 F2: four source mutations that the two-shell proof ADMITTED (each a GPU
    silent-wrong on cbff49c: dK err 4.1 / 3.3e9 / 2.9e9 / 1.26). Now each refuses by name
    at the lowering boundary, before any backward maker is built."""
    fn = _mutated_bwd_kv(tmp_path, replacements, "bwd_kv_" + "".join(ch if ch.isalnum() else "_" for ch in label))
    p, _ = _problem(32, None)
    N = p["N"]; dk = torch.zeros_like(p["k"]); dv = torch.zeros_like(p["v"])
    with pytest.raises(MetalNonRecoverableError, match=field):
        fn[(2, 1, 1)](
            p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
            p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
            dk, *_st(dk), dv, *_st(dv), p["sm"], -1e9, N, p["Hh"], 32, N, 32, 32)
        torch.mps.synchronize()
    assert maker_spy == []


def _tail_problem(DIM, N=48, phys=64):
    """N = 48 rows of real data inside PHYSICAL 64-row tensors whose padding rows 48..63 are
    NaN. The source's masked loads (other = 0) never observe the padding."""
    p, ref = _problem(DIM, None, N=phys)
    q, k, v, do = p["q"][..., :N, :], p["k"][..., :N, :], p["v"][..., :N, :], p["do"][..., :N, :]
    for t in (p["q"], p["k"], p["v"], p["do"]):
        t[..., N:, :] = float("nan")
    bias = p["bias"][..., :N, :N]; mask = p["mask"][..., :N]
    raw = (torch.einsum("hijd,hikd->hijk", q, k) * p["sm"] + bias[:, None]).masked_fill(mask[:, :, None, :].bool(), float("-inf"))
    lse = torch.logsumexp(raw, -1).contiguous(); o = (torch.softmax(raw, -1) @ v).contiguous(); delta = (o * do).sum(-1).contiguous()
    P = torch.exp(raw - lse[..., None]); dP = torch.einsum("hijd,hikd->hijk", do, v); dS = P * (dP - delta[..., None])
    ref = dict(dK=p["sm"] * torch.einsum("hijk,hijd->hikd", dS, q), dV=torch.einsum("hijk,hijd->hikd", P, do),
               dQ=p["sm"] * torch.einsum("hijk,hikd->hijd", dS, k), db=dS.sum(1))
    # pass the PHYSICAL tensors (padding rows present in memory) with N = 48
    return dict(q=p["q"], k=p["k"], v=p["v"], do=p["do"], bias=p["bias"], mask=p["mask"], lse=lse, o=o, delta=delta,
                sm=p["sm"], Hh=1, N=N, DIM=DIM), ref


@requires_gpu
@pytest.mark.parametrize("kind,DIM,maker", [("kv", 64, "kv_kernel"), ("q", 64, "q_kernel"), ("kv", 32, "kv_kernel_simd"), ("q", 32, "q_kernel_simd")])
def test_bwd_tail_never_reads_masked_out_padding(cold_gpu_caches, maker_spy, kind, DIM, maker):
    """Packet 135 F1: with N = 48 inside 64-row physical tensors whose padding rows are NaN,
    the outputs must be finite and exact — the source's masked loads see zeros there. The
    packet-134 candidate's scalar makers read dO past N_CTX (0 * NaN = NaN): dK 3072/3072
    NaN on the scalar kv maker. The simd makers stage dO with a range guard (controls)."""
    p, ref = _tail_problem(DIM)
    N = p["N"]; BJ = BK = 32
    if kind == "kv":
        dk = torch.zeros(1, 1, N, DIM, device=D); dv = torch.zeros(1, 1, N, DIM, device=D)
        if hasattr(_bwd_kv, "device_caches"):
            _bwd_kv.device_caches.clear()
        _bwd_kv[(triton.cdiv(N, BK), 1, 1)](
            p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
            p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
            dk, *_st(dk), dv, *_st(dv), p["sm"], -1e9, N, p["Hh"], DIM, N, BJ, BK)
        torch.mps.synchronize()
        assert maker_spy == [maker]
        assert torch.isfinite(dk).all() and torch.isfinite(dv).all(), "padding NaN leaked into dK/dV"
        assert (dk - ref["dK"]).abs().max().item() < 1e-2 and (dv - ref["dV"]).abs().max().item() < 1e-2
    else:
        dq = torch.zeros(1, 1, N, DIM, device=D); dlt = torch.zeros(1, 1, N, device=D)
        if hasattr(_bwd_q, "device_caches"):
            _bwd_q.device_caches.clear()
        _bwd_q[(triton.cdiv(N, BJ), 1, 1)](
            dlt, *_st(dlt), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]), p["bias"], *_st(p["bias"]),
            p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["o"], *_st(p["o"]), p["do"], *_st(p["do"]),
            dq, *_st(dq), p["sm"], -1e9, N, p["Hh"], DIM, N, BJ, BK)
        torch.mps.synchronize()
        assert maker_spy == [maker]
        assert torch.isfinite(dq).all() and torch.isfinite(dlt).all(), "padding NaN leaked into dQ/delta"
        assert (dq - ref["dQ"]).abs().max().item() < 1e-2


# ----------------------------------------------------------------------- packet 138 (137 HOLD)


@requires_gpu
def test_bwd_bounds_leaf_must_be_the_key_load_mask_not_a_lookalike(cold_gpu_caches, maker_spy, tmp_path):
    """Packet 137 P0: a key-bound leaf built from the HEAD program id (`pid(2) * BLOCK_K +
    range`) has the shape of a tile index and the right extent, but it is not the key
    coordinate the K/V loads use. The 136 proof admitted it (GPU: dK err 5.84, dV 2.92 on
    N = 48 / Hc = 2). The bounds select's leaves must BE the Q-load and K-load mask values."""
    fn = _mutated_bwd_kv(
        tmp_path,
        [(_BOUNDS, "fake_k = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)\n        scores = tl.where(mask_j[:, None] & (fake_k < N)[None, :], scores, neg_inf)")],
        "bwd_kv_wrong_pid_bound",
    )
    # GPT's row: N = 48, two heads (Hc = 2 sharing one Mask group), DIM = 32, BLOCK = 32. For
    # head 1 the mutated source's key bound is (32 + range) < 48, i.e. columns 16..31 of EVERY
    # key tile are sentinel-filled; head 0 coincides with the canonical bound. Correct-or-refuse:
    # either the lowering refuses before any maker, or the GPU result equals THAT semantics.
    Hc, Hh, N, DIM, B = 2, 2, 48, 32, 32
    torch.manual_seed(11)
    sm = 1.0 / math.sqrt(DIM)
    q, k, v, do = (torch.randn(Hc, 1, N, DIM, device=D) for _ in range(4))
    bias = torch.randn(Hc, N, N, device=D)
    mask = (torch.rand(Hc // Hh, 1, N, device=D) < 0.15).to(torch.uint8)
    raw = (torch.einsum("hijd,hikd->hijk", q, k) * sm + bias[:, None]).masked_fill(mask[:, :, None, :].bool(), float("-inf"))
    lse = torch.logsumexp(raw, -1).contiguous()
    o = (torch.softmax(raw, -1) @ v).contiguous()
    delta = (o * do).sum(-1).contiguous()
    cols = torch.arange(N, device=D)
    fake_masked = torch.stack([(h * B + cols % B) >= N for h in range(Hc)])  # [Hc, N]
    raw_mut = raw.masked_fill(fake_masked[:, None, None, :], float("-inf"))
    P = torch.exp(raw_mut - lse[..., None])
    dS = P * (torch.einsum("hijd,hikd->hijk", do, v) - delta[..., None])
    dK_mut = sm * torch.einsum("hijk,hijd->hikd", dS, q)
    dV_mut = torch.einsum("hijk,hijd->hikd", P, do)
    assert fake_masked[1].any() and not fake_masked[0].any()
    dk = torch.zeros_like(k); dv = torch.zeros_like(v)
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    try:
        fn[(triton.cdiv(N, B), 1, Hc)](
            delta, *_st(delta), q, *_st(q), k, *_st(k), v, *_st(v), bias, *_st(bias), lse, *_st(lse),
            mask, *_st(mask), do, *_st(do), dk, *_st(dk), dv, *_st(dv), sm, -1e9, N, Hh, DIM, N, B, B)
        torch.mps.synchronize()
    except MetalNonRecoverableError as e:
        assert "load mask" in str(e), str(e)
        assert maker_spy == [], "refusal must precede every maker"
        return
    # routed: it is only correct if it computed the MUTATED source's semantics
    err_k = (dk - dK_mut).abs().max().item(); err_v = (dv - dV_mut).abs().max().item()
    assert err_k < 1e-2 and err_v < 1e-2, f"routed the lookalike bound and computed the canonical one: dK err {err_k:.3g}, dV err {err_v:.3g}"


@requires_gpu
def test_bwd_bound_rhs_must_be_the_unnarrowed_n_ctx(cold_gpu_caches, maker_spy, tmp_path):
    """Packet 139 (GPT repo review): `k_idxs < N.to(tl.int8).to(tl.int32)` was ADMITTED by
    the 138 candidate because the N_CTX lookup peeled the narrowing cast. At N = 160 the
    narrowed bound is -96, so the mutated source sentinel-fills EVERY cell (P = 0 for a
    finite lse: dK = dV = 0) while the template replays the real bound. Correct-or-refuse:
    refuse before any maker, or equal the mutated source's semantics on the GPU."""
    fn = _mutated_bwd_kv(
        tmp_path,
        [(_BOUNDS, "scores = tl.where(mask_j[:, None] & (k_idxs < N.to(tl.int8).to(tl.int32))[None, :], scores, neg_inf)")],
        "bwd_kv_narrowed_n_ctx",
    )
    p, _ = _problem(32, None, N=160)
    N = p["N"]; B = 32
    assert ((N + 128) % 256) - 128 == -96
    dk = torch.zeros_like(p["k"]); dv = torch.zeros_like(p["v"])
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    try:
        fn[(triton.cdiv(N, B), 1, 1)](
            p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
            p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
            dk, *_st(dk), dv, *_st(dv), p["sm"], -1e9, N, p["Hh"], 32, N, B, B)
        torch.mps.synchronize()
    except MetalNonRecoverableError as e:
        assert "N_CTX" in str(e), str(e)
        assert maker_spy == [], "refusal must precede every maker"
        return
    err_k = dk.abs().max().item(); err_v = dv.abs().max().item()
    assert err_k < 1e-6 and err_v < 1e-6, f"routed the narrowed bound and computed the real one: |dK| {err_k:.3g}, |dV| {err_v:.3g} (mutated source: all zero)"


@requires_gpu
def test_bwd_bound_lhs_narrowing_cast_refuses(cold_gpu_caches, maker_spy, tmp_path):
    """Packet 139, index side: `(k_idxs.to(tl.int8).to(tl.int32)) < N` decomposed to the
    same signature as the K-load mask because the shared decomposer peeled `arith.trunci`.
    Lowering-boundary pin only: with masked loads (other = 0) every cell the narrowing
    flips lies past N_CTX, where this kernel family's loads and stores are masked, so no
    GPU witness exists here — the proof hole is still a proof hole and must refuse."""
    fn = _mutated_bwd_kv(
        tmp_path,
        [(_BOUNDS, "scores = tl.where(mask_j[:, None] & ((k_idxs.to(tl.int8).to(tl.int32)) < N)[None, :], scores, neg_inf)")],
        "bwd_kv_narrowed_k_index",
    )
    p, _ = _problem(32, None, N=160)
    N = p["N"]; B = 32
    dk = torch.zeros_like(p["k"]); dv = torch.zeros_like(p["v"])
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    with pytest.raises(MetalNonRecoverableError, match="load mask|one tile"):
        fn[(triton.cdiv(N, B), 1, 1)](
            p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
            p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
            dk, *_st(dk), dv, *_st(dv), p["sm"], -1e9, N, p["Hh"], 32, N, B, B)
        torch.mps.synchronize()
    assert maker_spy == []


# --- packet 141 / 144: the score→select and lse→subtraction VALUE paths ----------------------

def _bwd_source(kernel_name):
    import pathlib
    import re

    src = pathlib.Path(__file__).with_name("test_fa_bwd_routing.py").read_text()
    start = src.index(f"@triton.jit\ndef {kernel_name}(")
    end = start + 20 + re.search(r"\n(?=@|def )", src[start + 20:]).start()
    return src[start:end]


def _load_kernel_src(tmp_path, kernel_name, kernel_src, module_name):
    import importlib.util

    mod_path = tmp_path / f"{module_name}.py"
    mod_path.write_text("import triton\nimport triton.language as tl\n\n" + kernel_src + "\n")
    spec = importlib.util.spec_from_file_location(module_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, kernel_name)


_COL = {"_bwd_kv": "k_idxs", "_bwd_q": "(k_idxs + start_k)", "_bwd_b": "k_idxs"}

# spelling -> (transform factory, expected disposition, refusal substring)
_SPELLINGS = {
    "canonical": ("id", "admit", None),
    "bound_both_extsi": ("bound", "admit", None),          # sign-extending BOTH operands preserves slt/ult
    "bound_rhs_narrow": ("bound", "refuse", "N_CTX"),
    "bound_lhs_narrow": ("bound", "refuse", "one tile"),
    "bound_both_zext": ("bound", "refuse", "N_CTX"),
    "score_fp16_round": ("after_dot", "refuse", "dot value"),
    "score_bf16_round": ("after_dot", "refuse", "dot value"),
    "lse_fp16_round": ("lse", "refuse", "loaded lse VALUE"),
    "lse_bf16_round": ("lse", "refuse", "loaded lse VALUE"),
    "lse_times_1p0001": ("lse", "refuse", "loaded lse VALUE"),
    # `+ 0.0` is not folded by the compiler; the lse pointer discovery (which walks through a
    # multiply but not an add) already fails to resolve it, so it refuses one link earlier.
    "lse_plus_0": ("lse", "refuse", r"loaded lse VALUE|lse \(row_max\) pointer"),
    # packet 145 / 146: the right vector along the WRONG score axis (one slice change, no
    # arithmetic, no conversion) — admitted on `cbff49c` and on the 144 tree, dV wrong.
    "lse_axis": ("sub", "refuse", "score ROWS"),
    "bounds_axes": ("bounds_line", "refuse", "swapped broadcast axis"),
    "mask_axis": ("sub", "refuse", "score COLUMNS"),
    "and_commuted": ("bounds_line", "admit", None),        # harmless operand order of the AND
}
_SUBS = {
    "lse_axis": (r"\(scores - (\w+)\[:, None\]\)", r"(scores - \1[None, :])"),
    "mask_axis": (r"m_block\[None, :\]", "m_block[:, None]"),
}
# edits confined to the bounds `tl.where` line (the same conjunction also masks the bias load)
_BOUNDS_LINE = {
    "bounds_axes": "mask_j[None, :] & mask_k[:, None]",
    "and_commuted": "mask_k[None, :] & mask_j[:, None]",
}
_SPELL_TEXT = {
    "bound_both_extsi": "{col}.to(tl.int64) < N.to(tl.int64)",
    "bound_rhs_narrow": "{col} < N.to(tl.int8).to(tl.int32)",
    "bound_lhs_narrow": "{col}.to(tl.int8).to(tl.int32) < N",
    "bound_both_zext": "{col}.to(tl.uint32).to(tl.uint64) < N.to(tl.uint32).to(tl.uint64)",
    "score_fp16_round": "scores = scores.to(tl.float16).to(tl.float32)",
    "score_bf16_round": "scores = scores.to(tl.bfloat16).to(tl.float32)",
    "lse_fp16_round": ".to(tl.float16).to(tl.float32)",
    "lse_bf16_round": ".to(tl.bfloat16).to(tl.float32)",
    "lse_times_1p0001": " * 1.0001",
    "lse_plus_0": " + 0.0",
}


def _spell(kernel_name, mode):
    """One source-level spelling of the backward score / lse path, applied to KERNEL_NAME."""
    import re

    kind, _, _ = _SPELLINGS[mode]
    if kind == "id":
        return lambda s: s
    if kind == "sub":
        pat, repl = _SUBS[mode]

        def t(s):
            out, n = re.subn(pat, repl, s)
            assert n == 1, (kernel_name, mode, n)
            return out
        return t
    if kind == "bounds_line":
        def t(s):
            assert s.count(_BOUNDS) == 1, kernel_name
            return s.replace(_BOUNDS, _BOUNDS.replace("mask_j[:, None] & mask_k[None, :]", _BOUNDS_LINE[mode]))
        return t
    text = _SPELL_TEXT[mode].format(col=_COL[kernel_name])
    if kind == "bound":
        def t(s):
            assert s.count(_BOUNDS) == 1, kernel_name
            return s.replace(_BOUNDS, _BOUNDS.replace("mask_k[None, :]", f"({text})[None, :]"))
        return t
    if kind == "after_dot":
        def t(s):
            m = re.search(r"\n(\s*)scores = tl\.dot\([^\n]*\n", s)
            assert m, kernel_name
            return s[:m.end()] + m.group(1) + text + "\n" + s[m.end():]
        return t
    if kind == "lse":
        def t(s):
            m = re.search(r"\w+ = tl\.load\(l_ptrs, (?:mask=)?mask_j(?:, cache_modifier=\"\.cg\")?\)", s)
            assert m, kernel_name
            return s[:m.end()] + text + s[m.end():]
        return t
    raise AssertionError(kind)


def _lower_direct(fn):
    """Fresh TTGIR → GenericLowerer.lower() with no launch and no cache (protocol rule 9)."""
    from test_fa_bwd_routing import _build_lowerer

    cex = {}
    for i in fn.constexprs:
        n = fn.arg_names[i]
        cex[n] = 32 if (n == "DIM" or n.startswith("BLOCK")) else 256 if n == "CLOSEST_N" else None
        assert cex[n] is not None, n
    sig = {n: ("*u8" if ("mask" in n or n == "m_ptr") else "*fp32") if n.endswith("_ptr")
           else "fp32" if n in ("sm_scale", "neg_inf") else "i32"
           for n in fn.arg_names if n not in cex}
    return _build_lowerer(fn, sig, cex).lower()


@pytest.mark.skipif(not HAS_GPU, reason="Triton frontend + Metal backend import needed")
@pytest.mark.parametrize("mode", list(_SPELLINGS))
@pytest.mark.parametrize("kernel_name", ["_bwd_kv", "_bwd_q", "_bwd_b"])
def test_bwd_value_path_spelling_matrix_direct_lowering(maker_spy, tmp_path, kernel_name, mode):
    """Packet 141 §5.4: the spelling matrix over the three backward callers, at the lowering
    boundary via a direct `.lower()` call. Bound spellings: both-operand sign extension is the
    same comparison and ADMITS; narrowing either operand or zero-extending both REFUSES. Value
    spellings: any conversion or arithmetic between the score dot and the selects, or between
    the lse load and the subtraction, REFUSES — the template replays neither (141 F1 / F2: the
    fp16 round trips were admitted on `cbff49c` and computed the canonical values)."""
    fn = _load_kernel_src(tmp_path, kernel_name, _spell(kernel_name, mode)(_bwd_source(kernel_name)), f"m_{kernel_name}_{mode}")
    _, expect, needle = _SPELLINGS[mode]
    if expect == "admit":
        msl = _lower_direct(fn)
        assert isinstance(msl, str) and "kernel void" in msl
        assert len(maker_spy) == 1, maker_spy
    else:
        with pytest.raises(MetalNonRecoverableError, match=needle):
            _lower_direct(fn)
        assert maker_spy == [], "refusal must precede every maker"


def _value_witness_problem(DIM, bias_value):
    """GPT's 141 construction: N = 64 full tiles, Q = K = 0, V = dO = 1, delta = 0, Mask = 0,
    constant bias, supplied lse = bias + log(64) in fp32. Scores are exactly `bias`."""
    N = 64
    q = torch.zeros(1, 1, N, DIM, device=D); k = torch.zeros(1, 1, N, DIM, device=D)
    v = torch.ones(1, 1, N, DIM, device=D); do = torch.ones(1, 1, N, DIM, device=D)
    bias = torch.full((1, N, N), bias_value, device=D)
    mask = torch.zeros(1, 1, N, device=D, dtype=torch.uint8)
    lse = torch.full((1, 1, N), bias_value + math.log(N), device=D)
    delta = torch.zeros(1, 1, N, device=D)
    return dict(q=q, k=k, v=v, do=do, bias=bias, mask=mask, lse=lse, delta=delta, sm=1.0 / math.sqrt(DIM), N=N, DIM=DIM)


def _value_witness_reference(p, mode):
    """The MUTATED source's own math on the SAME supplied lse / delta: the specified fp16 round
    trip applied to the score or to the loaded lse, then exp2 / dS / dK / dV as the source spells them."""
    N = p["N"]; inv_ln2 = 1.4426950408889634
    s = p["bias"][:, None].expand(1, 1, N, N).clone()  # Q@Kᵀ = 0, so scores == bias exactly
    lse = p["lse"].clone()
    if mode == "score_fp16_round":
        s = s.to(torch.float16).to(torch.float32)
    elif mode == "lse_fp16_round":
        lse = lse.to(torch.float16).to(torch.float32)
    else:
        raise AssertionError(mode)
    P = torch.exp2((s - lse[..., None]) * inv_ln2)
    dP = torch.einsum("hijd,hikd->hijk", p["do"], p["v"])
    dS = P * (dP - p["delta"][..., None])
    dK = p["sm"] * torch.einsum("hijk,hijd->hikd", dS, p["q"])
    dV = torch.einsum("hijk,hijd->hikd", P, p["do"])
    return dK, dV


def _assert_same_values(got, ref, label):
    assert torch.equal(torch.isnan(got), torch.isnan(ref)), f"{label}: NaN placement differs from the mutated source"
    assert torch.equal(torch.isposinf(got), torch.isposinf(ref)) and torch.equal(torch.isneginf(got), torch.isneginf(ref)), f"{label}: Inf placement differs"
    fin = torch.isfinite(ref)
    if fin.any():
        err = (got[fin] - ref[fin]).abs().max().item()
        assert err < 1e-3, f"{label}: max |got - mutated source| = {err:.4g}"


@requires_gpu
@pytest.mark.parametrize("bias_value", [1000.24, 80000.0], ids=["finite", "overflow"])
@pytest.mark.parametrize("DIM,maker", [(32, "kv_kernel_simd"), (64, "kv_kernel")])
@pytest.mark.parametrize("mode", ["score_fp16_round", "lse_fp16_round"])
def test_bwd_value_path_round_trip_correct_or_refuse(cold_gpu_caches, maker_spy, tmp_path, mode, DIM, maker, bias_value):
    """Packet 141 F1 / F2 on the GPU, two-arm: the kv kernel with the score (F1) or the loaded lse
    (F2) rounded through fp16 must refuse before any maker, or compute the MUTATED source's own
    values. On `cbff49c` both route and compute the canonical values: finite case dV 1.0 where
    the mutated source has 0.7866 (F1) / 0.9038 (F2); overflow case zeros where the mutated
    source has 2048 NaNs in dK and +Inf in dV (F1)."""
    fn = _load_kernel_src(tmp_path, "_bwd_kv", _spell("_bwd_kv", mode)(_bwd_source("_bwd_kv")), f"g_bwd_kv_{mode}")
    p = _value_witness_problem(DIM, bias_value)
    dK_ref, dV_ref = _value_witness_reference(p, mode)
    if mode == "score_fp16_round" and bias_value > 60000:
        assert torch.isnan(dK_ref).sum().item() == 64 * DIM and torch.isposinf(dV_ref).all()
    N = p["N"]; B = 32
    dk = torch.zeros_like(p["k"]); dv = torch.zeros_like(p["v"])
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    try:
        fn[(triton.cdiv(N, B), 1, 1)](
            p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
            p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
            dk, *_st(dk), dv, *_st(dv), p["sm"], -1e9, N, 1, DIM, N, B, B)
        torch.mps.synchronize()
    except MetalNonRecoverableError as e:
        assert ("dot value" if mode.startswith("score") else "loaded lse VALUE") in str(e), str(e)
        assert maker_spy == [], "refusal must precede every maker"
        return
    assert maker_spy == [maker], maker_spy
    _assert_same_values(dk, dK_ref, f"{mode} dK")
    _assert_same_values(dv, dV_ref, f"{mode} dV")


def _axis_witness_problem(DIM):
    """GPT's 145 construction: N = 48 (one full and one partial 32-tile), Q = K = delta = 0,
    V = dO = 1, bias = 0, lse[j] = log 48 + 0.025 j (non-constant), loaded Mask true where
    k % 7 == 2 (non-uniform), so an axis error cannot hide behind uniform data."""
    N = 48
    q = torch.zeros(1, 1, N, DIM, device=D); k = torch.zeros(1, 1, N, DIM, device=D)
    v = torch.ones(1, 1, N, DIM, device=D); do = torch.ones(1, 1, N, DIM, device=D)
    bias = torch.zeros(1, N, N, device=D)
    mask = ((torch.arange(N, device=D) % 7) == 2).to(torch.uint8).reshape(1, 1, N)
    lse = (math.log(N) + 0.025 * torch.arange(N, device=D, dtype=torch.float32)).reshape(1, 1, N)
    delta = torch.zeros(1, 1, N, device=D)
    return dict(q=q, k=k, v=v, do=do, bias=bias, mask=mask, lse=lse, delta=delta, sm=1.0 / math.sqrt(DIM), N=N, DIM=DIM)


def _tiled_kv_reference(mode, p, B=32):
    """An explicit tiled CPU-side replay of the kv SOURCE (`_bwd_kv` with every load `other=0`)
    under one axis mutation: the same tiles, the same masked-load zero fills (including the
    zero-filled padded lse lanes the wrong-axis lse exposes), the same selects, the same
    masked stores. `mode` in canonical / lse_axis / bounds_axes / mask_axis."""
    N, DIM = p["N"], p["DIM"]; NEG = -1e9; inv_ln2 = 1.4426950408889634
    q, k, v, do = p["q"][0, 0], p["k"][0, 0], p["v"][0, 0], p["do"][0, 0]
    lse, delta, mask, bias = p["lse"][0, 0], p["delta"][0, 0], p["mask"][0, 0], p["bias"][0]

    def rows(t, start):  # masked row load, other = 0
        out = torch.zeros((B,) + tuple(t.shape[1:]), device=D, dtype=t.dtype)
        n = min(B, N - start)
        if n > 0:
            out[:n] = t[start:start + n]
        return out

    dK = torch.zeros(N, DIM, device=D); dV = torch.zeros(N, DIM, device=D)
    ar = torch.arange(B, device=D)
    for k0 in range(0, N, B):
        mask_k = (ar + k0) < N
        kt = rows(k, k0) * p["sm"]; vt = rows(v, k0)
        m_block = rows(mask, k0).bool()
        dk_acc = torch.zeros(B, DIM, device=D); dv_acc = torch.zeros(B, DIM, device=D)
        for j0 in range(0, N, B):
            mask_j = (ar + j0) < N
            q_block = rows(q, j0)
            b_block = torch.zeros(B, B, device=D)
            nj, nk = min(B, N - j0), min(B, N - k0)
            b_block[:nj, :nk] = bias[j0:j0 + nj, k0:k0 + nk]
            scores = q_block @ kt.T + b_block
            bounds = (mask_j[None, :] & mask_k[:, None]) if mode == "bounds_axes" else (mask_j[:, None] & mask_k[None, :])
            scores = torch.where(bounds, scores, torch.full_like(scores, NEG))
            mv = m_block[:, None] if mode == "mask_axis" else m_block[None, :]
            scores = torch.where(mv.expand(B, B), torch.full_like(scores, NEG), scores)
            row_max = rows(lse, j0)
            lb = row_max[None, :] if mode == "lse_axis" else row_max[:, None]
            P = torch.exp2((scores - lb) * inv_ln2)
            do_b = rows(do, j0)
            dv_acc += P.T @ do_b
            dl = rows(delta, j0)
            dP = do_b @ vt.T
            dS = P * (dP - dl[:, None])
            dk_acc += dS.T @ q_block
        n = min(B, N - k0)
        dK[k0:k0 + n] = dk_acc[:n]; dV[k0:k0 + n] = dv_acc[:n]
    return dK.reshape(1, 1, N, DIM), dV.reshape(1, 1, N, DIM)


_AXIS_NEEDLE = {"lse_axis": "score ROWS", "bounds_axes": "swapped broadcast axis", "mask_axis": "score COLUMNS"}


@requires_gpu
@pytest.mark.parametrize("DIM,maker", [(32, "kv_kernel_simd"), (64, "kv_kernel")])
@pytest.mark.parametrize("mode", ["canonical", "lse_axis", "bounds_axes", "mask_axis"])
def test_bwd_axis_correct_or_refuse(cold_gpu_caches, maker_spy, tmp_path, mode, DIM, maker):
    """Packet 145 F1–F3 on the GPU, two-arm, on the explicit-`other=0` kv source: refuse before
    any maker naming the axis, or equal the mutated source's tiled replay. The canonical row must
    ROUTE and match the same replay (it validates the reference). On `cbff49c` and the 144 tree
    all three mutations route and compute the canonical values: dV max error 15.85 (lse axis),
    0.186 (swapped bounds), 0.541 (mask axis) — GPT's 145 table."""
    src = _explicit_other_zero(_bwd_source("_bwd_kv"))
    if mode != "canonical":
        src = _spell("_bwd_kv", mode)(src)
    fn = _load_kernel_src(tmp_path, "_bwd_kv", src, f"ax_bwd_kv_{mode}")
    p = _axis_witness_problem(DIM)
    dK_ref, dV_ref = _tiled_kv_reference(mode, p)
    N = p["N"]; B = 32
    dk = torch.zeros_like(p["k"]); dv = torch.zeros_like(p["v"])
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    try:
        fn[(triton.cdiv(N, B), 1, 1)](
            p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
            p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
            dk, *_st(dk), dv, *_st(dv), p["sm"], -1e9, N, 1, DIM, N, B, B)
        torch.mps.synchronize()
    except MetalNonRecoverableError as e:
        assert mode != "canonical", str(e)
        assert _AXIS_NEEDLE[mode] in str(e), str(e)
        assert maker_spy == [], "refusal must precede every maker"
        return
    assert maker_spy == [maker], maker_spy
    _assert_same_values(dk, dK_ref, f"{mode} dK")
    _assert_same_values(dv, dV_ref, f"{mode} dV")
    if mode == "canonical":
        assert abs(dv[0, 0, 0, 0].item() - 0.5896477) < 1e-5, dv[0, 0, 0, 0].item()  # GPT's measured control


# --- packet 147 / 148: the DOWNSTREAM graph — orientation, roles, accumulation, stores ---------

# (kernel, mode) -> (old text, new text, expect, needle). One-token changes that stay type-valid on
# square tiles; every one of them EMITTED MSL on `cbff49c` and on the 146 tree (GPT's 147 matrix).
_DOWNSTREAM = {
    ("_bwd_kv", "dv_p_no_transpose"): ("tl.dot(tl.trans(sm_value).to(input_dtype), do,", "tl.dot(sm_value.to(input_dtype), do,", "refuse", r"dV = P"),
    ("_bwd_kv", "dk_ds_no_transpose"): ("tl.dot(tl.trans(dscores), q_block,", "tl.dot(dscores, q_block,", "refuse", r"dK = dS"),
    ("_bwd_kv", "dp_v_wrong_transpose"): ("tl.dot(do, vt_block, dsm_value,", "tl.dot(do, tl.trans(vt_block), dsm_value,", "refuse", r"dP dot"),
    ("_bwd_kv", "delta_wrong_axis"): ("(dsm_value - delta[:, None])", "(dsm_value - delta[None, :])", "refuse", r"dS = P|delta"),
    ("_bwd_q", "dp_v_wrong_transpose"): ("tl.dot(do_block, vt_block,", "tl.dot(do_block, tl.trans(vt_block),", "refuse", r"dP dot"),
    ("_bwd_q", "dq_ds_wrong_transpose"): ("tl.dot(dscores, k_block,", "tl.dot(tl.trans(dscores), k_block,", "refuse", r"dQ = dS"),
    ("_bwd_q", "delta_wrong_axis"): ("(dsm_value - delta[:, None])", "(dsm_value - delta[None, :])", "refuse", r"dS = P|delta"),
    ("_bwd_q", "delta_reduce_axis0"): ("tl.sum(o_block * do_block, axis=1)", "tl.sum(o_block * do_block, axis=0)", "refuse", r"delta|dS = P"),
    ("_bwd_b", "dp_v_no_transpose"): ("tl.dot(do, tl.trans(v_block),", "tl.dot(do, v_block,", "refuse", r"dP dot"),
    ("_bwd_b", "delta_wrong_axis"): ("(dsm_value - delta[:, None])", "(dsm_value - delta[None, :])", "refuse", r"dS = P|delta"),
    ("_bwd_b", "db_store_transpose"): ("tl.store(db_ptrs, db_block.to(input_dtype),", "tl.store(db_ptrs, tl.trans(db_block).to(input_dtype),", "refuse", r"dbias"),
    # 147 §5: explicit compare spellings of the loaded-Mask select (positives + the wrong axis)
    ("_bwd_kv", "mask_compare_before_expand"): ("tl.where(m_block[None, :], neg_inf, scores)", "tl.where((m_block != 0)[None, :], neg_inf, scores)", "admit", None),
    ("_bwd_kv", "mask_compare_commuted"): ("tl.where(m_block[None, :], neg_inf, scores)", "tl.where((0 != m_block)[None, :], neg_inf, scores)", "admit", None),
    ("_bwd_kv", "mask_compare_wrong_axis"): ("tl.where(m_block[None, :], neg_inf, scores)", "tl.where((m_block != 0)[:, None], neg_inf, scores)", "refuse", r"score COLUMNS"),
    ("_bwd_q", "mask_compare_before_expand"): ("tl.where(m_block[None, :], neg_inf, scores)", "tl.where((m_block != 0)[None, :], neg_inf, scores)", "admit", None),
    ("_bwd_b", "mask_compare_before_expand"): ("tl.where(m_block[None, :], neg_inf, scores)", "tl.where((m_block != 0)[None, :], neg_inf, scores)", "admit", None),
}


def _downstream_src(kernel_name, mode):
    old, new, _, _ = _DOWNSTREAM[(kernel_name, mode)]
    src = _bwd_source(kernel_name)
    assert src.count(old) == 1, (kernel_name, mode, src.count(old))
    return src.replace(old, new)


@pytest.mark.skipif(not HAS_GPU, reason="Triton frontend + Metal backend import needed")
@pytest.mark.parametrize("kernel_name,mode", sorted(_DOWNSTREAM))
def test_bwd_downstream_matrix_direct_lowering(maker_spy, tmp_path, kernel_name, mode):
    """Packet 147 §3: eleven one-token downstream changes (P / dS transposes into dV / dK / dQ,
    V's orientation in dP, delta's broadcast axis, the delta reduce axis, the dbias store
    orientation) all EMITTED on `cbff49c` — the graph was classified by pointer role, never
    proved by orientation. Now `_prove_downstream` assigns every tile a (row, col) coordinate
    pair, contracts every dot on matching coordinates, checks each store's value against its
    mask's projection, and refuses any dtype conversion. Explicit `m != 0` spellings admit."""
    fn = _load_kernel_src(tmp_path, kernel_name, _downstream_src(kernel_name, mode), f"ds_{kernel_name}_{mode}")
    _, _, expect, needle = _DOWNSTREAM[(kernel_name, mode)]
    if expect == "admit":
        msl = _lower_direct(fn)
        assert isinstance(msl, str) and "kernel void" in msl
        assert len(maker_spy) == 1, maker_spy
    else:
        with pytest.raises(MetalNonRecoverableError, match=needle):
            _lower_direct(fn)
        assert maker_spy == [], "refusal must precede every maker"


def _tiled_kv_reference_dn(mode, p, BJ=32, BK=32):
    """Tiled replay of the kv source (all loads `other=0`) under a DOWNSTREAM mutation:
    `dv_p_no_transpose` accumulates P @ dO instead of Pᵀ @ dO; `dk_ds_no_transpose` accumulates
    dS @ Q instead of dSᵀ @ Q (both type-valid only because BJ == BK). Rectangular canonical
    tiles are replayed with their own extents."""
    N, DIM = p["N"], p["DIM"]; NEG = -1e9; inv_ln2 = 1.4426950408889634
    q, k, v, do = p["q"][0, 0], p["k"][0, 0], p["v"][0, 0], p["do"][0, 0]
    lse, delta, mask, bias = p["lse"][0, 0], p["delta"][0, 0], p["mask"][0, 0], p["bias"][0]

    def rows(t, start, B):
        out = torch.zeros((B,) + tuple(t.shape[1:]), device=D, dtype=t.dtype)
        n = min(B, N - start)
        if n > 0:
            out[:n] = t[start:start + n]
        return out

    dK = torch.zeros(N, DIM, device=D); dV = torch.zeros(N, DIM, device=D)
    for k0 in range(0, N, BK):
        mask_k = (torch.arange(BK, device=D) + k0) < N
        kt = rows(k, k0, BK) * p["sm"]; vt = rows(v, k0, BK); m_block = rows(mask, k0, BK).bool()
        dk_acc = torch.zeros(BK, DIM, device=D); dv_acc = torch.zeros(BK, DIM, device=D)
        for j0 in range(0, N, BJ):
            mask_j = (torch.arange(BJ, device=D) + j0) < N
            q_block = rows(q, j0, BJ)
            b_block = torch.zeros(BJ, BK, device=D)
            nj, nk = min(BJ, N - j0), min(BK, N - k0)
            b_block[:nj, :nk] = bias[j0:j0 + nj, k0:k0 + nk]
            scores = q_block @ kt.T + b_block
            scores = torch.where(mask_j[:, None] & mask_k[None, :], scores, torch.full_like(scores, NEG))
            scores = torch.where(m_block[None, :].expand(BJ, BK), torch.full_like(scores, NEG), scores)
            P = torch.exp2((scores - rows(lse, j0, BJ)[:, None]) * inv_ln2)
            do_b = rows(do, j0, BJ)
            dv_acc += (P @ do_b) if mode == "dv_p_no_transpose" else (P.T @ do_b)
            dP = do_b @ vt.T
            dS = P * (dP - rows(delta, j0, BJ)[:, None])
            dk_acc += (dS @ q_block) if mode == "dk_ds_no_transpose" else (dS.T @ q_block)
        n = min(BK, N - k0)
        dK[k0:k0 + n] = p["sm"] * dk_acc[:n]; dV[k0:k0 + n] = dv_acc[:n]
    return dK.reshape(1, 1, N, DIM), dV.reshape(1, 1, N, DIM)


def _launch_kv(fn, p, BJ, BK):
    N, DIM = p["N"], p["DIM"]
    dk = torch.zeros_like(p["k"]); dv = torch.zeros_like(p["v"])
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    fn[(triton.cdiv(N, BK), 1, 1)](
        p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
        p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
        dk, *_st(dk), dv, *_st(dv), p["sm"], -1e9, N, 1, DIM, N, BJ, BK)
    torch.mps.synchronize()
    return dk, dv


@requires_gpu
@pytest.mark.parametrize("DIM,maker", [(32, "kv_kernel_simd"), (64, "kv_kernel")])
@pytest.mark.parametrize("mode", ["canonical", "dv_p_no_transpose", "dk_ds_no_transpose"])
def test_bwd_orientation_correct_or_refuse(cold_gpu_caches, maker_spy, tmp_path, mode, DIM, maker):
    """Packet 147 F1 / F2 on the GPU, two-arm, asymmetric random data (N = 64, two full tiles):
    the kv source with the P transpose dropped from dV (F1) or the dS transpose dropped from dK
    (F2) must refuse before any maker, or equal the mutated source's tiled replay. On `cbff49c`
    and the 146 tree both route and compute the canonical values: dV err 2.17 (F1), dK err 12.3
    simd / 17.4 scalar (F2). The canonical row routes and matches the same replay."""
    src = _explicit_other_zero(_bwd_source("_bwd_kv"))
    if mode != "canonical":
        old, new, _, _ = _DOWNSTREAM[("_bwd_kv", mode)]
        assert src.count(old) == 1
        src = src.replace(old, new)
    fn = _load_kernel_src(tmp_path, "_bwd_kv", src, f"or_bwd_kv_{mode}")
    p, _ = _problem(DIM, None, N=64, seed=7)
    p = dict(p, bias=p["bias"], mask=p["mask"])
    dK_ref, dV_ref = _tiled_kv_reference_dn(mode, p)
    try:
        dk, dv = _launch_kv(fn, p, 32, 32)
    except MetalNonRecoverableError as e:
        assert mode != "canonical", str(e)
        assert ("dV = P" if mode == "dv_p_no_transpose" else "dK = dS") in str(e), str(e)
        assert maker_spy == [], "refusal must precede every maker"
        return
    assert maker_spy == [maker], maker_spy
    err_k = (dk - dK_ref).abs().max().item(); err_v = (dv - dV_ref).abs().max().item()
    assert err_k < 2e-3 and err_v < 2e-3, f"{mode}: routed and computed something else: dK err {err_k:.3g}, dV err {err_v:.3g}"


@requires_gpu
@pytest.mark.parametrize("DIM,BJ,BK,maker", [(32, 16, 32, "kv_kernel_simd"), (32, 32, 16, "kv_kernel_simd"), (64, 16, 32, "kv_kernel")])
def test_bwd_kv_rectangular_tiles_compute(cold_gpu_caches, maker_spy, tmp_path, DIM, BJ, BK, maker):
    """Packet 147 §4 P0: the canonical kv source at BJ = 16, BK = 32, D = 32 routed to the simd
    maker, whose `tg_P` / `tg_dS` scratch [BJ*BK] was reused as the dV / dK store scratch
    [BK*D] — 512 floats declared, 1024 indexed: dK err 3.06 on `cbff49c` and the 146 tree
    (dV happened to survive). The scratch is now sized for both interpretations. Both sides of
    D <= BJ, plus the scalar control."""
    fn = _load_kernel_src(tmp_path, "_bwd_kv", _explicit_other_zero(_bwd_source("_bwd_kv")), f"rect_kv_{BJ}_{BK}_{DIM}")
    p, _ = _problem(DIM, None, N=64, seed=3)
    dK_ref, dV_ref = _tiled_kv_reference_dn("canonical", p, BJ=BJ, BK=BK)
    dk, dv = _launch_kv(fn, p, BJ, BK)
    assert maker_spy == [maker], maker_spy
    err_k = (dk - dK_ref).abs().max().item(); err_v = (dv - dV_ref).abs().max().item()
    assert err_k < 2e-3 and err_v < 2e-3, f"BJ{BJ}/BK{BK}/D{DIM}: dK err {err_k:.3g}, dV err {err_v:.3g}"


@requires_gpu
@pytest.mark.parametrize("DIM,BJ,BK,maker", [(32, 32, 16, "q_kernel_simd"), (32, 16, 32, "q_kernel_simd")])
def test_bwd_q_rectangular_tiles_compute(cold_gpu_caches, maker_spy, tmp_path, DIM, BJ, BK, maker):
    """The q simd sibling: `tg_P[BJ*BK]` reused as the dQ store scratch [BJ*D]. At BJ 32 / BK 16
    the GPU result happened to match while indexing past the declared extent; both sides of
    D <= BK now compute against the full-formula reference (N = 64 is a multiple of both tiles)."""
    fn = _load_kernel_src(tmp_path, "_bwd_q", _explicit_other_zero(_bwd_source("_bwd_q")), f"rect_q_{BJ}_{BK}_{DIM}")
    p, ref = _problem(DIM, None, N=64, seed=5)
    N = p["N"]
    dq = torch.zeros_like(p["q"]); dlt = torch.zeros(1, 1, N, device=D)
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    fn[(triton.cdiv(N, BJ), 1, 1)](
        dlt, *_st(dlt), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]), p["bias"], *_st(p["bias"]),
        p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["o"], *_st(p["o"]), p["do"], *_st(p["do"]),
        dq, *_st(dq), p["sm"], -1e9, N, 1, DIM, N, BJ, BK)
    torch.mps.synchronize()
    assert maker_spy == [maker], maker_spy
    assert (dq - ref["dQ"]).abs().max().item() < 2e-3 and (dlt - ref["delta"]).abs().max().item() < 2e-3


@pytest.mark.skipif(not HAS_GPU, reason="templates import")
@pytest.mark.parametrize("maker,D,BJ,BK", [
    ("make_flash_attention_bwd_kv_kernel_simd", 32, 16, 32),
    ("make_flash_attention_bwd_kv_kernel_simd", 32, 32, 16),
    ("make_flash_attention_bwd_q_kernel_simd", 32, 32, 16),
    ("make_flash_attention_bwd_q_kernel_simd", 32, 16, 32),
])
def test_bwd_simd_scratch_declared_for_every_use(maker, D, BJ, BK):
    """Emission-boundary pin (147 §4): every reused threadgroup array is declared at least as large
    as its largest use — kv: tg_P / tg_dS hold [BJ,BK] and then [BK,D]; q: tg_P holds [BJ,BK] and
    then [BJ,D]. On `cbff49c` the declarations were BJ*BK only (512 here, indexed to 1023)."""
    import re

    from triton_msl.codegen import _msl_templates as T_

    names = ["Q", "K", "V", "Bias", "Mask", "Lse", "Delta", "dO", "DK", "DV", "DQ", "O", "DB"]
    decls = [f"    device float* {n} [[buffer({i})]]" for i, n in enumerate(names)] + ["    constant uint* _bpk [[buffer(20)]]"]
    class _AnyBindings(dict):
        """Every logical stride the maker asks for resolves to a packed slot; this pin only
        inspects the emitted threadgroup declarations."""

        def __contains__(self, k):
            return True

        def __missing__(self, k):
            return "_bpk[0]"

    bindings = _AnyBindings(scale="as_type<float>(_bpk[60])", neg_inf="as_type<float>(_bpk[61])")
    src = getattr(T_, maker)(head_dim=D, BLOCK_J=BJ, BLOCK_K=BK, arg_decls=decls, bindings=bindings, grid_3d=True, runtime_neg_inf=True)
    # largest use of each reused array: kv holds [BJ,BK] then [BK,D]; q holds [BJ,BK] then [BJ,D]
    arrays = {"tg_P": max(BJ * BK, BK * D), "tg_dS": max(BJ * BK, BK * D)} if "kv" in maker else {"tg_P": max(BJ * BK, BJ * D)}
    for arr, need in arrays.items():
        m = re.search(rf"threadgroup float {arr}\[(\d+)\]", src)
        assert m, arr
        assert int(m.group(1)) >= need, f"{maker} D{D} BJ{BJ} BK{BK}: {arr} declared {m.group(1)} < {need} used"


# --- packet 149 / 150: VALUE identity per role and the exact accumulator recurrence -------------

# (kernel, mode) -> (replacements, expect, needle). `replacements` is a list of (old, new) applied
# to the kernel source, each exactly once.
_VALUES = {
    # F1: pointer provenance is not value identity (`load_addr` walked through the multiply)
    ("_bwd_kv", "dv_do_times2"): ([("tl.trans(sm_value).to(input_dtype), do,", "tl.trans(sm_value).to(input_dtype), do * 2.0,")], "refuse", r"load VALUE itself"),
    ("_bwd_kv", "dk_q_times2"): ([("tl.trans(dscores), q_block,", "tl.trans(dscores), q_block * 2.0,")], "refuse", r"load VALUE itself"),
    ("_bwd_kv", "dp_do_times2"): ([("tl.dot(do, vt_block,", "tl.dot(do * 2.0, vt_block,")], "refuse", r"load VALUE itself"),
    ("_bwd_kv", "bias_times2"): ([("tl.dot(q_block, kt_block, b_block,", "tl.dot(q_block, kt_block, b_block * 2.0,")], "refuse", r"Bias load VALUE"),
    ("_bwd_q", "dp_do_times2"): ([("tl.dot(do_block, vt_block,", "tl.dot(do_block * 2.0, vt_block,")], "refuse", r"load VALUE itself"),
    ("_bwd_q", "dq_k_times2"): ([("tl.dot(dscores, k_block,", "tl.dot(dscores, k_block * 2.0,")], "refuse", r"scaled K value"),
    ("_bwd_b", "dp_do_times2"): ([("tl.dot(do, tl.trans(v_block),", "tl.dot(do * 2.0, tl.trans(v_block),")], "refuse", r"load VALUE itself"),
    # F2: a zero-initialised carry is not interchangeable with the store's own carry
    ("_bwd_kv", "dv_uses_dk_acc"): ([("dv_block += tl.dot(", "dv_block = dk_block + tl.dot(")], "refuse", r"ITS OWN loop carry"),
    ("_bwd_kv", "nonzero_dv_init"): ([("dv_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)", "dv_block = tl.full([BLOCK_K, DIM], 5.0, dtype=tl.float32)")], "refuse", r"literal zero|oriented as"),
    # (kv's dP dot with the dV carry as C refuses one link earlier, at the dP orientation)
    ("_bwd_kv", "dp_c_is_dv_acc"): ([("dsm_value = tl.dot(do, vt_block, dsm_value,", "dsm_value = tl.dot(do, vt_block, dv_block,")], "refuse", r"literal zero|the dP dot"),
    ("_bwd_q", "dp_c_is_dq_acc"): ([("dsm_value = tl.dot(do_block, vt_block,", "dsm_value = tl.dot(do_block, vt_block, dq_block,")], "refuse", r"literal zero"),
    ("_bwd_b", "dp_c_is_db_acc"): ([("dsm_value = tl.dot(do, tl.trans(v_block),", "dsm_value = tl.dot(do, tl.trans(v_block), db_block,")], "refuse", r"literal zero"),
    # 149 §4 positive: K loaded [k, d] under the row mask and transposed at the score dot — the
    # lowering's stride mapping now follows the PROVED orientation (on `cbff49c`: dK err 7.5e-3)
    ("_bwd_kv", "k_load_kd"): ([
        ("kt_ptrs = base_k_ptr + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn)", "kt_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :] * stride_kd)"),
        ("kt_block = tl.load(kt_ptrs, mask_k[None, :])", "kt_block = tl.load(kt_ptrs, mask_k[:, None])"),
        ("tl.dot(q_block, kt_block,", "tl.dot(q_block, tl.trans(kt_block),"),
    ], "admit", None),
}


def _values_src(kernel_name, mode, explicit_other=False):
    reps, _, _ = _VALUES[(kernel_name, mode)]
    src = _bwd_source(kernel_name)
    for old, new in reps:   # mutate first: the explicit-`other` rewrite changes the load spellings
        assert src.count(old) == 1, (kernel_name, mode, old, src.count(old))
        src = src.replace(old, new)
    if explicit_other:
        src = _explicit_other_zero(src)
    return src


@pytest.mark.skipif(not HAS_GPU, reason="Triton frontend + Metal backend import needed")
@pytest.mark.parametrize("kernel_name,mode", sorted(_VALUES))
def test_bwd_value_and_recurrence_direct_lowering(maker_spy, tmp_path, kernel_name, mode):
    """Packet 149: every dot operand must BE the recognised load value through layout / transpose
    ops only (the score dot's K (kv, q) or Q (b) operand exactly `load * splat(sm_scale)`, and dQ's
    K operand that same scaled value); the bias accumulator the Bias load (a widening replayed);
    dP accumulates from a literal zero; every output accumulates into ITS OWN loop carry from a
    literal-zero init. All of the refusals here EMITTED on `cbff49c` and the 148 tree; the K[k,d]
    positive admits (and computed wrongly on `cbff49c`)."""
    fn = _load_kernel_src(tmp_path, kernel_name, _values_src(kernel_name, mode), f"val_{kernel_name}_{mode}")
    _, expect, needle = _VALUES[(kernel_name, mode)]
    if expect == "admit":
        msl = _lower_direct(fn)
        assert isinstance(msl, str) and "kernel void" in msl
        assert len(maker_spy) == 1, maker_spy
    else:
        with pytest.raises(MetalNonRecoverableError, match=needle):
            _lower_direct(fn)
        assert maker_spy == [], "refusal must precede every maker"


def _tiled_kv_reference_val(mode, p, BJ=32, BK=32):
    """Tiled replay of the kv source under one 149 mutation: an operand doubled where the source
    doubles it, or dV computed from the PREVIOUS dK carry (`dv = dk_prev + Pᵀ @ dO`, then dK)."""
    N, DIM = p["N"], p["DIM"]; NEG = -1e9; inv_ln2 = 1.4426950408889634
    q, k, v, do = p["q"][0, 0], p["k"][0, 0], p["v"][0, 0], p["do"][0, 0]
    lse, delta, mask, bias = p["lse"][0, 0], p["delta"][0, 0], p["mask"][0, 0], p["bias"][0]

    def rows(t, start, B):
        out = torch.zeros((B,) + tuple(t.shape[1:]), device=D, dtype=t.dtype)
        n = min(B, N - start)
        if n > 0:
            out[:n] = t[start:start + n]
        return out

    dK = torch.zeros(N, DIM, device=D); dV = torch.zeros(N, DIM, device=D)
    for k0 in range(0, N, BK):
        mask_k = (torch.arange(BK, device=D) + k0) < N
        kt = rows(k, k0, BK) * p["sm"]; vt = rows(v, k0, BK); m_block = rows(mask, k0, BK).bool()
        dk_acc = torch.zeros(BK, DIM, device=D); dv_acc = torch.zeros(BK, DIM, device=D)
        for j0 in range(0, N, BJ):
            mask_j = (torch.arange(BJ, device=D) + j0) < N
            q_block = rows(q, j0, BJ)
            b_block = torch.zeros(BJ, BK, device=D)
            nj, nk = min(BJ, N - j0), min(BK, N - k0)
            b_block[:nj, :nk] = bias[j0:j0 + nj, k0:k0 + nk]
            scores = q_block @ kt.T + b_block
            scores = torch.where(mask_j[:, None] & mask_k[None, :], scores, torch.full_like(scores, NEG))
            scores = torch.where(m_block[None, :].expand(BJ, BK), torch.full_like(scores, NEG), scores)
            P = torch.exp2((scores - rows(lse, j0, BJ)[:, None]) * inv_ln2)
            do_b = rows(do, j0, BJ)
            prev_dk = dk_acc.clone()
            dv_acc = (prev_dk if mode == "dv_uses_dk_acc" else dv_acc) + P.T @ (do_b * (2.0 if mode == "dv_do_times2" else 1.0))
            dP = (do_b * (2.0 if mode == "dp_do_times2" else 1.0)) @ vt.T
            dS = P * (dP - rows(delta, j0, BJ)[:, None])
            dk_acc = dk_acc + dS.T @ (q_block * (2.0 if mode == "dk_q_times2" else 1.0))
        n = min(BK, N - k0)
        dK[k0:k0 + n] = p["sm"] * dk_acc[:n]; dV[k0:k0 + n] = dv_acc[:n]
    return dK.reshape(1, 1, N, DIM), dV.reshape(1, 1, N, DIM)


_VAL_NEEDLE = {"dv_do_times2": "load VALUE itself", "dk_q_times2": "load VALUE itself", "dp_do_times2": "load VALUE itself", "dv_uses_dk_acc": "ITS OWN loop carry"}


@requires_gpu
@pytest.mark.parametrize("DIM,maker", [(32, "kv_kernel_simd"), (64, "kv_kernel")])
@pytest.mark.parametrize("mode", ["canonical", "dv_do_times2", "dk_q_times2", "dp_do_times2", "dv_uses_dk_acc", "k_load_kd"])
def test_bwd_value_recurrence_correct_or_refuse(cold_gpu_caches, maker_spy, tmp_path, mode, DIM, maker):
    """Packet 149 F1 / F2 and the §4 positive on the GPU, two-arm, asymmetric random data (N = 64,
    two query tiles so a wrong carry is observable): refuse before any maker, or equal the mutated
    source's tiled replay. `k_load_kd` and `canonical` must ROUTE and match. On `cbff49c` and the
    148 tree the four mutations route and compute the canonical values (dV err 0.27 / 0.44, dK
    err 0.11 / 0.13, dV err 0.42 / 0.88); `k_load_kd` computed dK err 7.5e-3 on `cbff49c`."""
    src = _explicit_other_zero(_bwd_source("_bwd_kv")) if mode == "canonical" else _values_src("_bwd_kv", mode, explicit_other=True)
    fn = _load_kernel_src(tmp_path, "_bwd_kv", src, f"vr_bwd_kv_{mode}")
    p, _ = _problem(DIM, None, N=64, seed=149)
    dK_ref, dV_ref = _tiled_kv_reference_val("canonical" if mode == "k_load_kd" else mode, p)
    try:
        dk, dv = _launch_kv(fn, p, 32, 32)
    except MetalNonRecoverableError as e:
        assert mode in _VAL_NEEDLE, str(e)
        assert _VAL_NEEDLE[mode] in str(e), str(e)
        assert maker_spy == [], "refusal must precede every maker"
        return
    assert maker_spy == [maker], maker_spy
    err_k = (dk - dK_ref).abs().max().item(); err_v = (dv - dV_ref).abs().max().item()
    assert err_k < 2e-3 and err_v < 2e-3, f"{mode}: routed and computed something else: dK err {err_k:.3g}, dV err {err_v:.3g}"


# --- packet 151 / 152: the q route's COMPUTED delta — producer values + combiner ------------------

_COMBINER_PLUS1 = "@triton.jit\ndef _combine_plus1(a, b):\n    return a + b + 1.0\n\n"
_COMBINER_ADD = "@triton.jit\ndef _combine_add(a, b):\n    return a + b\n\n"
_DELTA_LINE = "delta = tl.sum(o_block * do_block, axis=1)"
# mode -> (replacement delta line, prelude, expect, needle)
_DELTA = {
    "delta_o_times2": ("delta = tl.sum((o_block * 2.0) * do_block, axis=1)", "", "refuse", r"O load VALUE"),
    "delta_do_times2": ("delta = tl.sum(o_block * (do_block * 2.0), axis=1)", "", "refuse", r"O load VALUE"),
    "delta_o_square": ("delta = tl.sum((o_block * o_block) * do_block, axis=1)", "", "refuse", r"O load VALUE"),
    "delta_combine_plus1": ("delta = tl.reduce(o_block * do_block, 1, _combine_plus1)", _COMBINER_PLUS1, "refuse", r"combiner|delta"),
    "delta_commuted": ("delta = tl.sum(do_block * o_block, axis=1)", "", "admit", None),
    "delta_reduce_plain_add": ("delta = tl.reduce(o_block * do_block, 1, _combine_add)", _COMBINER_ADD, "admit", None),
}


def _delta_src(mode, explicit_other=False):
    line, prelude, _, _ = _DELTA[mode]
    src = _bwd_source("_bwd_q")
    assert src.count(_DELTA_LINE) == 1
    src = src.replace(_DELTA_LINE, line)
    if explicit_other:
        src = _explicit_other_zero(src)
    return prelude + src


@pytest.mark.skipif(not HAS_GPU, reason="Triton frontend + Metal backend import needed")
@pytest.mark.parametrize("mode", sorted(_DELTA))
def test_bwd_q_delta_contract_direct_lowering(maker_spy, tmp_path, mode):
    """Packet 151: the reduce input must be exactly the O load VALUE times the dO load VALUE (either
    order) and the reduction region exactly `a + b` of its two block arguments. On `cbff49c` and the
    150 tree `(o*2)*do`, `o*(do*2)`, `(o*o)*do` and the `a + b + 1.0` combiner all EMITTED
    (delta err 14.0 / 14.0 / 33.7 / 31.0, dQ wrong with them). Commuted O/dO and an explicit
    plain-add combiner admit."""
    fn = _load_kernel_src(tmp_path, "_bwd_q", _delta_src(mode), f"dl_{mode}")
    _, _, expect, needle = _DELTA[mode]
    if expect == "admit":
        msl = _lower_direct(fn)
        assert isinstance(msl, str) and "kernel void" in msl
        assert len(maker_spy) == 1, maker_spy
    else:
        with pytest.raises(MetalNonRecoverableError, match=needle):
            _lower_direct(fn)
        assert maker_spy == [], "refusal must precede every maker"


def _delta_reference(mode, p):
    """The mutated source's math on the same inputs (N = 64 full tiles): the mutated delta, then
    dS = P * (dP - delta) and dQ = sm * dS @ K with THAT delta. A binary reduction of D elements
    performs D - 1 combines, so `a + b + 1.0` adds exactly D - 1."""
    o, do = p["o"][0, 0], p["do"][0, 0]
    if mode == "delta_o_times2" or mode == "delta_do_times2":
        delta = (2.0 * o * do).sum(-1)
    elif mode == "delta_o_square":
        delta = (o * o * do).sum(-1)
    elif mode == "delta_combine_plus1":
        delta = (o * do).sum(-1) + (p["DIM"] - 1)
    else:
        delta = (o * do).sum(-1)
    q, k, v, bias, mask, lse = p["q"][0, 0], p["k"][0, 0], p["v"][0, 0], p["bias"][0], p["mask"][0, 0], p["lse"][0, 0]
    raw = (q @ k.T * p["sm"] + bias).masked_fill(mask[None, :].bool(), float("-inf"))
    P = torch.exp(raw - lse[:, None])
    dP = do @ v.T
    dS = P * (dP - delta[:, None])
    dQ = p["sm"] * dS @ k
    N, DIM = p["N"], p["DIM"]
    return dQ.reshape(1, 1, N, DIM), delta.reshape(1, 1, N)


@requires_gpu
@pytest.mark.parametrize("DIM,maker", [(32, "q_kernel_simd"), (64, "q_kernel")])
@pytest.mark.parametrize("mode", ["canonical", "delta_o_times2", "delta_do_times2", "delta_o_square", "delta_combine_plus1"])
def test_bwd_q_delta_correct_or_refuse(cold_gpu_caches, maker_spy, tmp_path, mode, DIM, maker):
    """Packet 151 F1 / F2 on the GPU, two-arm, on both q makers with asymmetric random data (O is an
    independent supplied input): refuse before any maker, or equal the mutated source's own delta
    AND the dQ computed from it. The canonical row must route and match."""
    src = _explicit_other_zero(_bwd_source("_bwd_q")) if mode == "canonical" else _delta_src(mode, explicit_other=True)
    fn = _load_kernel_src(tmp_path, "_bwd_q", src, f"dq_{mode}")
    p, _ = _problem(DIM, None, N=64, seed=151)
    dQ_ref, dlt_ref = _delta_reference(mode, p)
    N = p["N"]; B = 32
    dq = torch.zeros_like(p["q"]); dlt = torch.zeros(1, 1, N, device=D)
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    try:
        fn[(triton.cdiv(N, B), 1, 1)](
            dlt, *_st(dlt), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]), p["bias"], *_st(p["bias"]),
            p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["o"], *_st(p["o"]), p["do"], *_st(p["do"]),
            dq, *_st(dq), p["sm"], -1e9, N, 1, DIM, N, B, B)
        torch.mps.synchronize()
    except MetalNonRecoverableError as e:
        assert mode != "canonical", str(e)
        assert ("combiner" in str(e) or "delta" in str(e) or "O load VALUE" in str(e)), str(e)
        assert maker_spy == [], "refusal must precede every maker"
        return
    assert maker_spy == [maker], maker_spy
    err_d = (dlt - dlt_ref).abs().max().item(); err_q = (dq - dQ_ref).abs().max().item()
    assert err_d < 1e-2 and err_q < 2e-3, f"{mode}: routed and computed something else: delta err {err_d:.3g}, dQ err {err_q:.3g}"


# --- packet 153 / 154: the reduction region's RETURNED value, at the direct-TTGIR boundary ------------

def _ttgir_text(fn):
    """The canonical q kernel's TTGIR text (the same pipeline `_build_lowerer` runs)."""
    from triton._C.libtriton import ir
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend.compiler import MetalBackend

    cex = {}
    for i in fn.constexprs:
        n = fn.arg_names[i]
        cex[n] = 32 if (n == "DIM" or n.startswith("BLOCK")) else 256 if n == "CLOSEST_N" else None
        assert cex[n] is not None, n
    sig = {n: ("*u8" if ("mask" in n or n == "m_ptr") else "*fp32") if n.endswith("_ptr")
           else "fp32" if n in ("sm_scale", "neg_inf") else "i32"
           for n in fn.arg_names if n not in cex}
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    src = ASTSource(fn=fn, signature=sig, constexprs=cex)
    ctx = ir.context(); ir.load_dialects(ctx)
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx)
    md = {}
    mod = backend.make_ttir(mod, md, options)
    mod = backend.make_ttgir(mod, md, options)
    return str(mod)


def _lower_ttgir_text(text, tmp_path, name):
    """Direct TTGIR input (`IRSource`, the input `triton.compile` accepts for a `.ttgir` path):
    verify the module, walk it, `GenericLowerer.lower()` — no launch, no cache."""
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import IRSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir

    path = tmp_path / f"{name}.ttgir"
    path.write_text(text)
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({})
    ctx = ir.context(); ir.load_dialects(ctx)
    source = IRSource(str(path), ctx, backend)
    if hasattr(source.module, "verify"):
        assert source.module.verify(), "the mutated TTGIR must still verify"
    return GenericLowerer(walk_ttgir(source.module, options), options).lower()


_RET_RE = r"(%[\w]+) = arith.addf (%[\w]+), (%[\w]+) : f32[^\n]*\n(\s*)tt.reduce.return \1 : f32"


def _reduce_return_variants(text):
    """From the canonical TTGIR: the region's `%s = addf %a, %b ; return %s` rewritten to return
    %a, return %b (the dead add retained — legal, verifying IR) and a commuted add (positive)."""
    import re

    m = re.search(_RET_RE, text)
    assert m and text.count("tt.reduce.return ") == 1, "one reduction region with a returned add"
    s_, a, b, ws = m.group(1), m.group(2), m.group(3), m.group(4)
    ret = f"tt.reduce.return {s_} : f32"
    add = f"{s_} = arith.addf {a}, {b} : f32"
    assert text.count(ret) == 1 and text.count(add) == 1
    return {
        "canonical": text,
        "return_left": text.replace(ret, f"tt.reduce.return {a} : f32"),
        "return_right": text.replace(ret, f"tt.reduce.return {b} : f32"),
        "commuted_add": text.replace(add, f"{s_} = arith.addf {b}, {a} : f32"),
    }


@pytest.mark.skipif(not HAS_GPU, reason="Triton frontend + Metal backend import needed")
@pytest.mark.parametrize("mode", ["canonical", "return_left", "return_right", "commuted_add"])
def test_bwd_q_reduce_return_direct_ttgir(maker_spy, tmp_path, mode):
    """Packet 153: the walker dropped `tt.reduce.return`, so a region `%s = addf %a, %b ;
    return %a` (a dead add with a projection returned — legal, verifying TTGIR reachable through
    `IRSource`) was indistinguishable from the sum: on the 152 tree both projections EMITTED and
    the emitted shader returned D where the IR returns 1 (delta err 31 / 63). Python source
    cannot spell it (DCE removes the dead add), so this pin feeds the mutated TTGIR directly. The
    reduce op now carries `return_ids`; the combiner proof requires them to name the addition."""
    variants = _reduce_return_variants(_ttgir_text(_bwd_q))
    if mode in ("canonical", "commuted_add"):
        msl = _lower_ttgir_text(variants[mode], tmp_path, mode)
        assert isinstance(msl, str) and "kernel void" in msl
        assert len(maker_spy) == 1, maker_spy
    else:
        with pytest.raises(MetalNonRecoverableError, match=r"combiner|delta"):
            _lower_ttgir_text(variants[mode], tmp_path, mode)
        assert maker_spy == [], "refusal must precede every maker"


def test_bwd_reduce_op_carries_return_ids():
    """The walker records which SSA value each reduction region returns (`attrs["return_ids"]`),
    on the entry-function path; the combiner proof refuses when the information is absent."""
    if not HAS_GPU:
        pytest.skip("Triton frontend needed")
    from test_fa_bwd_routing import _build_lowerer

    fn = _bwd_q
    cex = {n: (32 if (n == "DIM" or n.startswith("BLOCK")) else 256) for n in [fn.arg_names[i] for i in fn.constexprs]}
    sig = {n: ("*u8" if ("mask" in n or n == "m_ptr") else "*fp32") if n.endswith("_ptr") else "fp32" if n in ("sm_scale", "neg_inf") else "i32" for n in fn.arg_names if n not in cex}
    lw = _build_lowerer(fn, sig, cex)

    def walk(ops):
        for o in ops:
            yield o
            if getattr(o, "region_ops", None):
                yield from walk(o.region_ops)

    red = [o for o in walk(lw.graph.ops) if o.op == "tt.reduce"]
    assert len(red) == 1
    body = red[0].region_ops
    assert len(body) == 1 and body[0].op == "arith.addf"
    assert red[0].attrs.get("return_ids") == [body[0].id]


def _explicit_other_zero(kernel_src):
    """Every masked `tl.load(...)` in the source gets an explicit `other=0` (packet 137: the
    tail pin must not assume the backend's masked-load default)."""
    import re

    def fix(m):
        args = m.group(1)
        return m.group(0) if ("other" in args or "," not in args) else f"tl.load({args}, other=0)"

    out = re.sub(r"tl\.load\(([^()]*?)\)", fix, kernel_src)
    assert out.count("other=0") >= 5, out.count("other=0")
    return out


def _mutated_bwd_src(tmp_path, kernel_name, transform, module_name):
    import importlib.util
    import pathlib

    src = pathlib.Path(__file__).with_name("test_fa_bwd_routing.py").read_text()
    import re

    start = src.index(f"@triton.jit\ndef {kernel_name}(")
    end = start + 20 + re.search(r"\n(?=@|def )", src[start + 20:]).start()
    kernel_src = transform(src[start:end])
    mod_path = tmp_path / f"{module_name}.py"
    mod_path.write_text("import triton\nimport triton.language as tl\n\n" + kernel_src + "\n")
    spec = importlib.util.spec_from_file_location(module_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, kernel_name)


@requires_gpu
@pytest.mark.parametrize("kind,DIM,maker", [("kv", 64, "kv_kernel"), ("q", 64, "q_kernel"), ("kv", 32, "kv_kernel_simd"), ("q", 32, "q_kernel_simd")])
def test_bwd_tail_with_explicit_other_zero(cold_gpu_caches, maker_spy, tmp_path, kind, DIM, maker):
    """Packet 137: the same N = 48 tail with NaN physical padding, but every masked source load
    says `other=0` explicitly, so the reference is the language contract, not a backend
    default. (The implicit-`other` rows above stay as controls.)"""
    fn = _mutated_bwd_src(tmp_path, "_bwd_kv" if kind == "kv" else "_bwd_q", _explicit_other_zero, f"bwd_{kind}_other0")
    p, ref = _tail_problem(DIM)
    N = p["N"]; BJ = BK = 32
    if kind == "kv":
        dk = torch.zeros(1, 1, N, DIM, device=D); dv = torch.zeros(1, 1, N, DIM, device=D)
        fn[(triton.cdiv(N, BK), 1, 1)](
            p["delta"], *_st(p["delta"]), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]),
            p["bias"], *_st(p["bias"]), p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["do"], *_st(p["do"]),
            dk, *_st(dk), dv, *_st(dv), p["sm"], -1e9, N, p["Hh"], DIM, N, BJ, BK)
        torch.mps.synchronize()
        assert maker_spy == [maker]
        assert torch.isfinite(dk).all() and torch.isfinite(dv).all()
        assert (dk - ref["dK"]).abs().max().item() < 1e-2 and (dv - ref["dV"]).abs().max().item() < 1e-2
    else:
        dq = torch.zeros(1, 1, N, DIM, device=D); dlt = torch.zeros(1, 1, N, device=D)
        fn[(triton.cdiv(N, BJ), 1, 1)](
            dlt, *_st(dlt), p["q"], *_st(p["q"]), p["k"], *_st(p["k"]), p["v"], *_st(p["v"]), p["bias"], *_st(p["bias"]),
            p["lse"], *_st(p["lse"]), p["mask"], *_st(p["mask"]), p["o"], *_st(p["o"]), p["do"], *_st(p["do"]),
            dq, *_st(dq), p["sm"], -1e9, N, p["Hh"], DIM, N, BJ, BK)
        torch.mps.synchronize()
        assert maker_spy == [maker]
        assert torch.isfinite(dq).all() and torch.isfinite(dlt).all()
        assert (dq - ref["dQ"]).abs().max().item() < 1e-2
