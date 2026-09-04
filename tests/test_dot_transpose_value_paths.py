"""Packet 102: a transpose between a dot operand's load and the dot is a VALUE transform, and
only the templates that replay it may admit it.

On clean ``ba21e9d`` (and on the P0 tree ``e2330ec``) the K-loop matmul template stored the raw
product for a square in-tile ``tl.trans`` on A or on B (err 41 / 49): P0's shared value-path
vocabulary admitted ``ttg.memdesc_trans`` by opcode while the K-loop templates replay no
transpose at all (both canonical transposed layouts — A stored [K,M], B stored [N,K] — were
already refused there). The single-tile templates DO replay transposes (trans_a / trans_b,
probe-proven for the in-tile square forms and the canonical [N,K] weight), so the non-loop
branch keeps them and this file pins that as positives.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir

    from triton_msl.backend.compiler import MetalBackend
    import triton_msl.codegen.generic_lowerer as generic_lowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False

requires = pytest.mark.skipif(not HAS, reason="Triton compiler needed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")

if HAS:

    @triton.jit
    def _kloop(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        om = pid_m * BM + tl.arange(0, BM)
        on = pid_n * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
        bp = b_ptr + ok[:, None] * sbk + on[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            if MODE == 3:
                a = tl.load(a_ptr + om[:, None] * sam + (ok[None, :] + k) * sak)
            else:
                a = tl.load(ap)
            b = tl.load(bp)
            if MODE == 1:
                a = tl.trans(a)
            elif MODE == 2:
                b = tl.trans(b)
            acc += tl.dot(a, b)
            ap += BK * sak
            bp += BK * sbk
        tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)

    @triton.jit
    def _single(X, Y, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        if MODE == 4:
            y = tl.load(Y + on[:, None] * K + ok[None, :])  # Y stored [N,K] (canonical)
            y = tl.trans(y)
        else:
            y = tl.load(Y + ok[:, None] * N + on[None, :])
        if MODE == 1:
            x = tl.trans(x)
        elif MODE == 2:
            y = tl.trans(y)
        tl.store(C + om[:, None] * N + on[None, :], tl.dot(x, y))

    def _direct_lowerer_for(fn, signature, constexprs):
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        target = GPUTarget("metal", "apple-m4", 32)
        backend = MetalBackend(target)
        options = backend.parse_options({"num_warps": 4})
        src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
        context = ir.context()
        ir.load_dialects(context)
        mod = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context)
        metadata = {}
        mod = backend.make_ttir(mod, metadata, options)
        mod = backend.make_ttgir(mod, metadata, options)
        return generic_lowerer.GenericLowerer(walk_ttgir(mod, options), options)


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    for fn in (_kloop, _single):
        if hasattr(fn, "device_caches"):
            fn.device_caches.clear()


def _tile_trans(a, bm, bk):
    t = torch.empty_like(a)
    for m0 in range(0, a.shape[0], bm):
        for k0 in range(0, a.shape[1], bk):
            t[m0:m0 + bm, k0:k0 + bk] = a[m0:m0 + bm, k0:k0 + bk].T
    return t


def _witness(launch, c, raw, intended):
    sep = (raw - intended).abs().max().item()
    assert sep > 0
    try:
        launch()
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        assert bool(c.isnan().all()), "a refused kernel must touch nothing"
        return "refused"
    err = (c - intended).abs().max().item()
    assert err == err and err <= sep / 10, f"computed the raw product, not the transposed one (err {err}, separation {sep})"
    return "computed"


def _kloop_case(mode):
    M = N = K = 64
    BM = BN = BK = 32
    torch.manual_seed(10111)
    a = torch.randn(M, K, device="mps")
    b = torch.randn(K, N, device="mps")
    raw = a @ b
    intended = _tile_trans(a, BM, BK) @ b if mode == 1 else (a @ _tile_trans(b, BK, BN) if mode == 2 else raw)
    c = torch.full((M, N), float("nan"), device="mps")
    launch = lambda: _kloop[(M // BM, N // BN)](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BM, BN, BK, MODE=mode)
    return launch, c, raw, intended


@requires_gpu
@pytest.mark.parametrize("mode", [1, 2], ids=["transA", "transB"])
def test_kloop_in_tile_transpose_never_raw(cold_gpu_caches, mode):
    """Bites on ba21e9d AND e2330ec: the K-loop template stored the raw product."""
    launch, c, raw, intended = _kloop_case(mode)
    _witness(launch, c, raw, intended)


@requires_gpu
@pytest.mark.parametrize("mode", [0, 3], ids=["carried-pointer", "in-loop-(range+k)"])
def test_kloop_native_spellings_compute(cold_gpu_caches, mode):
    """Positives: the carried-pointer and in-loop ``(ok + k) * sak`` A spellings compute exactly
    (the latter through the stride tracer's induction-variable admission, packet 102)."""
    launch, c, raw, _ = _kloop_case(mode)
    launch()
    torch.mps.synchronize()
    assert (c - raw).abs().max().item() < 1e-3


@requires_gpu
@pytest.mark.parametrize("mode,M,N,K", [(1, 32, 64, 32), (2, 64, 32, 32), (4, 64, 32, 32)],
                         ids=["single-transA-square", "single-transB-square", "single-Y[N,K]-canonical"])
def test_single_tile_transposes_replayed(cold_gpu_caches, mode, M, N, K):
    """Positives (probe-proven): the single-tile template replays in-tile and canonical
    transposes exactly (outside the generic envelope: M != N)."""
    torch.manual_seed(10111)
    x = torch.randn(M, K, device="mps")
    y = torch.randn(N, K, device="mps") if mode == 4 else torch.randn(K, N, device="mps")
    c = torch.full((M, N), float("nan"), device="mps")
    _single[(1,)](x, y, c, M=M, N=N, K=K, MODE=mode, num_warps=4)
    torch.mps.synchronize()
    ref = (x.T @ y) if mode == 1 else (x @ y.T)
    torch.testing.assert_close(c, ref, rtol=1e-4, atol=1e-4)


_SIG_K = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "i32", "N": "i32", "K": "i32",
          "sam": "i32", "sak": "constexpr", "sbk": "i32", "sbn": "constexpr", "scm": "i32", "scn": "constexpr"}
_CEX_K = {"BM": 32, "BN": 32, "BK": 32, "sak": 1, "sbn": 1, "scn": 1}


@requires
@pytest.mark.parametrize("mode,role", [(1, "A"), (2, "B")])
def test_lowering_boundary_kloop_transpose_refuses(mode, role):
    """Rule 9 (bites on ba21e9d / e2330ec): the K-loop predicate names the transposed role."""
    lw = _direct_lowerer_for(_kloop, _SIG_K, {**_CEX_K, "MODE": mode})
    with pytest.raises(MetalNonRecoverableError, match=f"dot operand {role} .*transposed"):
        lw.lower()


@requires
def test_lowering_boundary_kloop_in_loop_spelling_lowers():
    """Positive control: the in-loop ``(range + k)`` spelling lowers on the K-loop template."""
    lw = _direct_lowerer_for(_kloop, _SIG_K, {**_CEX_K, "MODE": 3})
    msl = lw.lower()
    assert "kernel void" in msl and "UNSUPPORTED" not in msl
