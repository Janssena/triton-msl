"""Packet 160 / 162 — item F: conversion passthroughs on template value paths.

Census (packet 160): every list in codegen that treats a float / integer conversion as transparent
was classified and probed. Two template routes dropped casts the source spells:

* F-h — the softmax template's store chain: ``_detect_softmax`` (and ``_detect_layer_norm``)
  validated the INPUT cone but never tied the STORED value to the computation, so
  ``(y * 4).to(int32).to(f32)`` before the store emitted MSL byte-identical to the canonical
  kernel (GPU err 2.24) and ``y.to(f16).to(f32)`` likewise (err 1.4e-4). Now the stored value
  must root at the division (softmax) / the normalisation multiply (layer norm) through layout
  ops and at most ONE ``truncf`` whose type equals the store pointer's element type — the cast
  the template's own store replays.
* F-b — the matmul + epilogue emitter substituted a passthrough op with its operand's expression,
  so a conversion listed in ``_EPI_PASSTHROUGH`` / ``_EPILOGUE_ALLOWED`` would be dropped by
  construction (not reached with any probed shape: the K-chunk path claims them first and replays
  casts exactly). Both lists now hold layout ops only.

The remaining census sites already refused or replayed; they are pinned here so the census stays
executable: single-dot output casts and a narrowed row index (refuse), K-loop input casts (refuse),
K-chunk epilogue casts (replayed, err 0), the i64 → i32 reduce store (replayed), the layer-norm eps
through an fp16 round trip (Triton folds it; the template bakes the folded value).
"""

import difflib
import importlib.util
import pathlib
import sys

import pytest

try:
    import torch
    import triton

    import triton_msl
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    from triton_msl.errors import MetalNonRecoverableError

    sys.path.insert(0, "tests")
    from test_fa_bwd_routing import _build_lowerer

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
def route_spy(monkeypatch):
    """Records which GenericLowerer template route(s) claimed the kernel.

    The class is resolved through the MODULE at fixture time: `tests/test_mept_*.py` reload
    `triton_msl.codegen.generic_lowerer` at import, which creates a new class object, so a class
    captured at this file's import would be a dead spy whenever those modules are collected first
    (the packet-162 randomized gate found exactly that: 10 positive rows with an empty spy)."""
    import importlib

    cls = importlib.import_module("triton_msl.codegen.generic_lowerer").GenericLowerer
    hits = []
    for name in dir(cls):
        if name.startswith("_lower_") and any(k in name for k in ("softmax", "norm", "dot", "matmul", "reduce", "epilogue")):
            real = getattr(cls, name)

            def mk(_r, _n):
                def spy(self, *a, **k):
                    hits.append(_n)
                    return _r(self, *a, **k)
                return spy

            monkeypatch.setattr(cls, name, mk(real, name))
    return hits


def _load_src(tmp_path, src, modname, fn_name):
    p = tmp_path / f"{modname}.py"
    p.write_text("import triton\nimport triton.language as tl\n\n" + src + "\n")
    spec = importlib.util.spec_from_file_location(modname, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


def _lower(fn, sig, cex):
    return _build_lowerer(fn, sig, cex).lower()


# ----------------------------------------------------------------------------------------------
# F-h: softmax and layer-norm store chains
# ----------------------------------------------------------------------------------------------

_SM = '''
@triton.jit
def _sm(x_ptr, o_ptr, N, BLOCK: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=-float("inf"))
    x = x - tl.max(x, axis=0); e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    if MODE == 1:
        y = y.to(tl.float16).to(tl.float32)
    if MODE == 2:
        y = (y * 4.0).to(tl.int32).to(tl.float32)
    if MODE == 3:
        y = y.to(tl.float16)
    tl.store(o_ptr + row * N + cols, y, mask=m)
'''

_LN = '''
@triton.jit
def _ln(x_ptr, o_ptr, N, BLOCK: tl.constexpr, EPS: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(m, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    if MODE == 4:
        y = xc * tl.rsqrt(var + (tl.full([], EPS, tl.float16)).to(tl.float32))
    else:
        y = xc * tl.rsqrt(var + EPS)
    if MODE == 1:
        y = y.to(tl.float16).to(tl.float32)
    if MODE == 2:
        y = (y * 4.0).to(tl.int32).to(tl.float32)
    if MODE == 3:
        y = y.to(tl.float16)
    tl.store(o_ptr + row * N + cols, y, mask=m)
'''


def _sm_ref(x, mode):
    y = torch.softmax(x, -1)
    if mode == 1:
        return y.half().float()
    if mode == 2:
        return (y * 4.0).to(torch.int32).float()
    if mode == 3:
        return y.half().float()
    return y


def _ln_ref(x, mode, eps=1e-5):
    N = x.shape[-1]
    mean = x.mean(-1, keepdim=True); xc = x - mean; var = (xc * xc).sum(-1, keepdim=True) / N
    e = torch.tensor(eps, dtype=torch.float16).float().item() if mode == 4 else eps
    y = xc * torch.rsqrt(var + e)
    if mode == 1 or mode == 3:
        return y.half().float()
    if mode == 2:
        return (y * 4.0).to(torch.int32).float()
    return y


@pytest.mark.skipif(not HAS_GPU, reason="Triton frontend + Metal backend import needed")
@pytest.mark.parametrize("family,mode,expect", [
    ("sm", 0, "template"), ("sm", 1, "refuse-or-generic"), ("sm", 2, "refuse-or-generic"), ("sm", 3, "template"),
    ("ln", 0, "template"), ("ln", 1, "refuse-or-generic"), ("ln", 2, "refuse-or-generic"), ("ln", 3, "template"), ("ln", 4, "template"),
])
def test_store_chain_casts_direct_lowering(route_spy, tmp_path, family, mode, expect):
    """Direct lowering: a round trip or an integer cast on the store chain must NOT reach the
    softmax / layer-norm template (it refuses or the generic lowering, which replays every op,
    takes it); the canonical kernel and a single output cast to an fp16 pointer still route to
    the template; the fp16-rounded eps (folded by Triton) still routes."""
    src, fn_name = (_SM, "_sm") if family == "sm" else (_LN, "_ln")
    fn = _load_src(tmp_path, src, f"dl_{family}_{mode}", fn_name)
    out_ptr = "*fp16" if mode == 3 else "*fp32"
    sig = {"x_ptr": "*fp32", "o_ptr": out_ptr, "N": "i32"}
    cex = {"BLOCK": 64, "MODE": mode} if family == "sm" else {"BLOCK": 64, "EPS": 1e-5, "MODE": mode}
    tmpl = "_lower_softmax_template" if family == "sm" else "_lower_layer_norm_template"
    try:
        msl = _lower(fn, sig, cex)
    except MetalNonRecoverableError:
        assert expect == "refuse-or-generic"
        return
    assert isinstance(msl, str) and "kernel void" in msl
    if expect == "template":
        assert tmpl in route_spy, route_spy
    else:
        assert tmpl not in route_spy, f"the template claimed a store chain with an unreplayed cast: {route_spy}"


@requires_gpu
@pytest.mark.parametrize("family,mode", [("sm", 0), ("sm", 1), ("sm", 2), ("sm", 3), ("ln", 0), ("ln", 1), ("ln", 2), ("ln", 3), ("ln", 4)])
def test_store_chain_casts_correct_or_refuse(cold_gpu_caches, route_spy, tmp_path, family, mode):
    """GPU, two-arm: refuse before any template, or equal the mutated source's semantics. On
    `cbff49c` the softmax template computed the raw division for modes 1 and 2."""
    src, fn_name = (_SM, "_sm") if family == "sm" else (_LN, "_ln")
    fn = _load_src(tmp_path, src, f"gpu_{family}_{mode}", fn_name)
    torch.manual_seed(2); R, N = 8, 64
    x = torch.randn(R, N, device=D) * 3
    o = torch.zeros(R, N, device=D, dtype=torch.float16 if mode == 3 else torch.float32)
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    try:
        if family == "sm":
            fn[(R,)](x, o, N, 64, mode)
        else:
            fn[(R,)](x, o, N, 64, 1e-5, mode)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        assert mode in (1, 2)
        return
    ref = _sm_ref(x, mode) if family == "sm" else _ln_ref(x, mode)
    err = (o.float() - ref).abs().max().item()
    assert err < (2e-3 if mode == 3 else 1e-4), f"{family} mode {mode}: err {err:.3e} vs the mutated source (route {sorted(set(route_spy))})"
    if mode in (0, 3, 4):
        tmpl = "_lower_softmax_template" if family == "sm" else "_lower_layer_norm_template"
        assert tmpl in route_spy, route_spy


# ----------------------------------------------------------------------------------------------
# F-b: matmul + epilogue — contract + the probed positives (K-chunk replays)
# ----------------------------------------------------------------------------------------------

def test_epilogue_allow_lists_hold_no_conversions():
    """Contract pin (160 F-b): the substituting epilogue emitter may only pass layout ops through."""
    from triton_msl.codegen._lowerer_detection import _EPI_PASSTHROUGH, _DetectionMixin

    conv = {"arith.truncf", "arith.extf", "arith.sitofp", "arith.fptosi", "arith.uitofp", "arith.fptoui", "tt.fp_to_fp", "arith.trunci", "arith.extsi", "arith.extui"}
    assert not (_EPI_PASSTHROUGH & conv), _EPI_PASSTHROUGH & conv
    assert not (_DetectionMixin._EPILOGUE_ALLOWED & conv), _DetectionMixin._EPILOGUE_ALLOWED & conv


_EPI = '''
@triton.jit
def _dot_epi(X, Y, B, Z, S: tl.constexpr, MODE: tl.constexpr):
    om = tl.arange(0, S); on = tl.arange(0, S); ok = tl.arange(0, S)
    x = tl.load(X + om[:, None] * S + ok[None, :]); y = tl.load(Y + ok[:, None] * S + on[None, :])
    z = tl.dot(x, y)
    if MODE == 1:
        z = z.to(tl.int32).to(tl.float32)
    if MODE == 2:
        z = z.to(tl.float16).to(tl.float32)
    z = tl.maximum(z * 0.5 + tl.load(B + on)[None, :], 0.0)
    tl.store(Z + om[:, None] * S + on[None, :], z)
'''


@requires_gpu
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_epilogue_casts_replayed_or_refused(cold_gpu_caches, route_spy, tmp_path, mode):
    """A conversion inside a matmul epilogue: the K-chunk path replays it exactly (probed: err 0);
    the substituting epilogue emitter may not claim it. Two-arm: refuse, or equal the mutated source."""
    fn = _load_src(tmp_path, _EPI, f"epi_{mode}", "_dot_epi")
    torch.manual_seed(3); S = 32
    x = torch.randn(S, S, device=D) * 3; y = torch.randn(S, S, device=D) * 3; b = torch.randn(S, device=D)
    z = torch.zeros(S, S, device=D)
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    try:
        fn[(1,)](x, y, b, z, S=S, MODE=mode, num_warps=4)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        assert mode != 0
        return
    zz = x @ y
    if mode == 1:
        zz = zz.to(torch.int32).float()
    if mode == 2:
        zz = zz.half().float()
    ref = torch.clamp(zz * 0.5 + b[None, :], min=0.0)
    assert (z - ref).abs().max().item() < 1e-4
    assert "_lower_matmul_epilogue_template" not in route_spy or mode == 0


# ----------------------------------------------------------------------------------------------
# Census sites already closed — pinned so the census stays executable
# ----------------------------------------------------------------------------------------------

_DOT32 = '''
@triton.jit
def _dot32(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
    om = tl.arange(0, BM)
    if MODE == 3:
        om = om.to(tl.int8).to(tl.int32)
    on = tl.arange(0, BN); ok = tl.arange(0, BK)
    a = tl.load(a_ptr + om[:, None] * BK + ok[None, :]); b = tl.load(b_ptr + ok[:, None] * BN + on[None, :])
    c = tl.dot(a, b)
    if MODE == 1:
        c = c.to(tl.float16).to(tl.float32)
    if MODE == 2:
        c = c.to(tl.int32).to(tl.float32)
    tl.store(c_ptr + om[:, None] * BN + on[None, :], c)
'''

_KLOOP = '''
@triton.jit
def _kloop(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, BK)
    acc = tl.zeros([M, N], dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(A + rm[:, None] * K + (k0 + rk)[None, :]); b = tl.load(B + (k0 + rk)[:, None] * N + rn[None, :])
        if MODE == 1:
            a = a.to(tl.float16).to(tl.float32); b = b.to(tl.float16).to(tl.float32)
        if MODE == 2:
            a = a.to(tl.float16); b = b.to(tl.float16)
        acc = tl.dot(a, b, acc)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)
'''


@pytest.mark.skipif(not HAS_GPU, reason="Triton frontend + Metal backend import needed")
@pytest.mark.parametrize("family,mode,needle", [
    ("dot32", 1, "round-trip"), ("dot32", 2, "sitofp"), ("dot32", 3, "trunci"),
    ("kloop", 1, "extf"), ("kloop", 2, "truncf"),
])
def test_closed_sites_still_refuse(route_spy, tmp_path, family, mode, needle):
    """Single-dot output round trips / narrowed row index and K-loop input casts refuse by name
    (closed by packets 079–083 and 139–140; re-pinned by the 160 census)."""
    if family == "dot32":
        fn = _load_src(tmp_path, _DOT32, f"d32_{mode}", "_dot32")
        sig, cex = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"}, {"BM": 32, "BN": 32, "BK": 32, "MODE": mode}
    else:
        fn = _load_src(tmp_path, _KLOOP, f"kl_{mode}", "_kloop")
        sig, cex = {"A": "*fp32", "B": "*fp32", "C": "*fp32"}, {"M": 64, "N": 64, "K": 128, "BK": 32, "MODE": mode}
    with pytest.raises(MetalNonRecoverableError, match=needle):
        _lower(fn, sig, cex)


_RSUM = '''
@triton.jit
def _r1d_sum(a, o, N: tl.constexpr):
    tl.store(o, tl.sum(tl.load(a + tl.arange(0, N)), 0).to(tl.int32))
'''


@requires_gpu
def test_reduce_store_int_narrowing_replayed(cold_gpu_caches, tmp_path):
    """`sum(i64).to(int32)` stored to an int32 pointer: the 1-D i64 reduce path emits the
    narrowing (`static_cast<int>`); values beyond 2^31 wrap exactly as the source specifies."""
    fn = _load_src(tmp_path, _RSUM, "rsum", "_r1d_sum")
    a = torch.full((1024,), 3_000_000, device=D, dtype=torch.int64)  # sum = 3.072e9 > 2^31
    o = torch.zeros(1, device=D, dtype=torch.int32)
    fn[(1,)](a, o, 1024)
    torch.mps.synchronize()
    ref = int(a.sum().item())
    ref = ((ref + 2**31) % 2**32) - 2**31
    assert o.item() == ref, (o.item(), ref)
