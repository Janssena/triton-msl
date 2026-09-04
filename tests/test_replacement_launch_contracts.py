"""Packet 105/106: a replacement template (quantized GEMM/GEMV, the ordinary fast matmul and
its split-K, the MLA attention dispatch, the host path's direct matmul variant) may stand in
for the compiled kernel ONLY under the caller's invocation geometry, and only when every
template-level semantic is the kernel's.

On clean ``ba21e9d`` and on the P0 tree ``e2330ec``:
- the int4 GEMV template applied one scale/zero group to both nibbles of a byte, so an odd
  group size (G = 3) stored the template's result (err 8.3 vs the kernel's semantics);
- every replacement dispatcher rebuilt a FULL launch from M/N/K and never saw the caller's
  grid: a canonical kernel launched for a subset of its programs (``grid=(1,)`` for N = 64,
  BN = 32) ran the template over the whole output, writing memory the kernel never touches;
- the bare K-loop template's coordinate proof admitted a missing program_id, ``pid * 2BM``
  and shifted ``make_range`` bounds (the packet-103 class on the compiled templates);
- the host path's ``__mmdirect`` variant hard-coded a 2-D pid mapping, mis-mapping the
  tutorial's 1-D split and un-tiled axes whenever the extents were tile-aligned.

Now the descriptors carry the SOURCE program mapping (BM/BN, pid axes, the 1-D/2-D map), the
driver passes the launch grid, each dispatcher requires exactly ``cdiv(M, BM) x cdiv(N, BN)``
(``cdiv(N, BN)`` for a GEMV; ``cdiv(N_CTX, BLOCK_M) x Z*H`` for MLA; unused axes 1) — a
mismatch declines to the compiled kernel where that is correct (fast matmul) or refuses without
touching the output (quantized, MLA); the int4 GEMV requires an even group; the value-path
predicate requires the exact tile (``pid * BLOCK + make_range(0, BLOCK)``) and records
per-axis pid presence, which every emitted variant replays (an un-tiled axis is tile 0).
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir

    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.msl_emitter import emit_msl
    from triton_msl.errors import MetalNonRecoverableError
    import triton_msl.autotuning._quant_matmul_dispatch as quant_dispatch
    import triton_msl.autotuning._fast_matmul_dispatch as fast_dispatch
    from triton_msl.autotuning._fa_dispatch import _dispatch_mla

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False

requires = pytest.mark.skipif(not HAS, reason="Triton compiler needed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"
SENT = 98765.0

if HAS:

    @triton.jit
    def _g8(x_ptr, w_ptr, o_ptr, scale_ptr, zero_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        on = pid * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        scale = tl.load(scale_ptr + on)
        zero = tl.load(zero_ptr + on)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k in range(0, K, BK):
            x = tl.load(x_ptr + rk + k)
            w = tl.load(w_ptr + on[:, None] * swn + (rk[None, :] + k) * swk).to(tl.float32)
            acc += tl.sum(x[None, :] * (w - zero[:, None]), axis=1)
        tl.store(o_ptr + on, acc * scale)

    @triton.jit
    def _g4(x_ptr, w_ptr, o_ptr, s_ptr, z_ptr, N, K, ng, swn, ssn, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
        pid = tl.program_id(0)
        on = pid * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k in range(0, K, BK):
            kk = k + ok
            packed = tl.load(w_ptr + on[:, None] * swn + (kk // 2)[None, :])
            w4 = (packed >> ((kk % 2) * 4)[None, :]) & 0xF
            g = kk // G
            s = tl.load(s_ptr + on[:, None] * ssn + g[None, :])
            z = tl.load(z_ptr + on[:, None] * ssn + g[None, :])
            x = tl.load(x_ptr + kk)
            acc += tl.sum(x[None, :] * ((w4.to(tl.float32) - z) * s), axis=1)
        tl.store(o_ptr + on, acc)

    @triton.jit
    def _pg(a_ptr, w_ptr, c_ptr, scale_ptr, zero_ptr, M, N, K, sam, sak, swk, swn, ssg, ssn, zsg, zsn, scm, scn,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        om = pid_m * BM + tl.arange(0, BM)
        on = pid_n * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
        wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            g = k // G
            s = tl.load(scale_ptr + g * ssg + on * ssn)
            z = tl.load(zero_ptr + g * zsg + on * zsn)
            w = (tl.load(wp).to(tl.float32) - z[None, :]) * s[None, :]
            acc += tl.dot(tl.load(ap), w)
            ap += BK * sak
            wp += BK * swk
        tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)

    @triton.jit
    def _sym(a_ptr, w_ptr, c_ptr, s_ptr, M, N, K, sam, sak, swk, swn, scm, scn,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        om = pid_m * BM + tl.arange(0, BM)
        on = pid_n * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
        wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
        scale = tl.load(s_ptr + on)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(ap), tl.load(wp).to(tl.float32) * scale[None, :])
            ap += BK * sak
            wp += BK * swk
        tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)

    @triton.jit
    def _pn(a_ptr, w_ptr, c_ptr, scale_ptr, zero_ptr, M, N, K, sam, sak, swk, swn, scm, scn,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        om = pid_m * BM + tl.arange(0, BM)
        on = pid_n * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
        wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
        scale = tl.load(scale_ptr + on)
        zero = tl.load(zero_ptr + on)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            w = (tl.load(wp).to(tl.float32) - zero[None, :]) * scale[None, :]
            acc += tl.dot(tl.load(ap), w)
            ap += BK * sak
            wp += BK * swk
        tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)

    @triton.jit
    def _fd(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ONE_D: tl.constexpr, MODE: tl.constexpr):
        if ONE_D:
            pid = tl.program_id(0)
            npm = tl.cdiv(M, BM)
            pid_m = pid % npm
            pid_n = pid // npm
        else:
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        if MODE == 30:
            om = rm
            on = rn
        elif MODE == 31:
            om = pid_m * (2 * BM) + rm
            on = pid_n * BN + rn
        elif MODE == 37:
            om = pid_m * BM + rm
            on = pid_n * BN + tl.arange(BN, 2 * BN)
        else:
            om = pid_m * BM + rm
            on = pid_n * BN + rn
        if MODE == 38:
            ok = tl.arange(BK, 2 * BK)
        else:
            ok = tl.arange(0, BK)
        ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
        bp = b_ptr + ok[:, None] * sbk + on[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(ap), tl.load(bp))
            ap += BK * sak
            bp += BK * sbk
        tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)

    def _emit_for(fn, signature, constexprs):
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
        out = {}
        return emit_msl(mod, out, options), out


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    for fn in (_g8, _g4, _pg, _sym, _pn, _fd):
        if hasattr(fn, "device_caches"):
            fn.device_caches.clear()


@pytest.fixture
def spies(monkeypatch):
    """quant / fast replacement dispatch outcomes, recorded per launch."""
    q, f = [], []
    oq, of = quant_dispatch.dispatch_quant_matmul, fast_dispatch.dispatch_fast_matmul
    monkeypatch.setattr(quant_dispatch, "dispatch_quant_matmul", lambda *a, **k: (lambda r: (q.append(bool(r)), r)[1])(oq(*a, **k)))
    monkeypatch.setattr(fast_dispatch, "dispatch_fast_matmul", lambda *a, **k: (lambda r: (f.append(bool(r)), r)[1])(of(*a, **k)))
    return q, f


def _cmp(a, b):
    return (torch.nan_to_num(a, nan=SENT) - torch.nan_to_num(b, nan=SENT)).abs().max().item()


def _launch(fn_launch, out):
    """Returns 'refused' (output untouched) or 'computed'."""
    try:
        fn_launch()
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        assert bool(out.isnan().all()), "a refused launch must touch nothing"
        return "refused"
    return "computed"


# ---------------------------------------------------------------- A. odd int4 group size
def _g4_case(G, N=64, K=192, BN=32, BK=64):
    torch.manual_seed(5)
    ng = K // G
    x = torch.randn(K, device=D)
    w4 = torch.randint(0, 16, (N, K), device=D, dtype=torch.int32)
    packed = (w4[:, 0::2] | (w4[:, 1::2] << 4)).to(torch.uint8).contiguous()
    s = (torch.rand(N, ng, device=D) * 0.05 + 0.01).contiguous()
    z = torch.randint(0, 16, (N, ng), device=D).float().contiguous()
    gi = torch.arange(K, device=D) // G
    ref = ((w4.float() - z[:, gi]) * s[:, gi]) @ x
    o = torch.full((N,), float("nan"), device=D)
    launch = lambda grid=(N // BN,): _g4[grid](x, packed, o, s, z, N, K, ng, packed.stride(0), s.stride(0), BN=BN, BK=BK, G=G)
    return launch, o, ref


@requires_gpu
def test_int4_gemv_odd_group_refuses_or_matches_kernel(cold_gpu_caches, spies):
    """G = 3 (bites on ba21e9d / e2330ec: template stored err 8.3 vs the kernel)."""
    launch, o, ref = _g4_case(3)
    if _launch(launch, o) == "computed":
        assert _cmp(o, ref) < 1e-2, "the specialized template mis-grouped the two nibbles"
        assert spies[0] != [True]


@requires_gpu
@pytest.mark.parametrize("G", [2, 128])
def test_int4_gemv_even_group_routes_and_computes(cold_gpu_caches, spies, G):
    launch, o, ref = _g4_case(G, K=256 if G == 128 else 192)
    launch()
    torch.mps.synchronize()
    assert spies[0] == [True]
    assert _cmp(o, ref) < 1e-2


_SIG_G4 = {"x_ptr": "*fp32", "w_ptr": "*u8", "o_ptr": "*fp32", "s_ptr": "*fp32", "z_ptr": "*fp32", "N": "i32", "K": "i32", "ng": "i32", "swn": "i32", "ssn": "i32"}


@requires
def test_lowering_boundary_int4_gemv_odd_group_no_descriptor():
    with pytest.raises(MetalNonRecoverableError):
        _emit_for(_g4, _SIG_G4, {"BN": 32, "BK": 64, "G": 3})


@requires
def test_lowering_boundary_int4_gemv_even_group_descriptor():
    msl, md = _emit_for(_g4, _SIG_G4, {"BN": 32, "BK": 64, "G": 128})
    assert md.get("quant_matmul") is not None


# ---------------------------------------------------------------- B. launch geometry
def _gemv8_case():
    N, K, BN, BK = 64, 128, 32, 32
    torch.manual_seed(7)
    x = torch.randn(K, device=D)
    w = torch.randint(-32, 32, (N, K), device=D, dtype=torch.int8).contiguous()
    s = torch.rand(N, device=D) * 0.05 + 0.02
    z = torch.randint(-4, 5, (N,), device=D).float()
    full = ((w.float() - z[:, None]) * x[None, :]).sum(1) * s
    o = torch.full((N,), float("nan"), device=D)
    launch = lambda grid: _g8[grid](x, w, o, s, z, N, K, w.stride(0), w.stride(1), BN=BN, BK=BK)
    return launch, o, full, (N // BN, 1, 1), "gemv"


def _gemv4_case():
    launch, o, ref = _g4_case(128, K=256)
    return (lambda grid: launch(grid)), o, ref, (2, 1, 1), "gemv"


def _pg_case():
    M, N, K, G, BM, BN, BK = 64, 64, 128, 32, 32, 32, 32
    torch.manual_seed(7)
    a = torch.randn(M, K, device=D)
    w = torch.randint(-8, 8, (K, N), device=D, dtype=torch.int8).contiguous()
    s = (torch.rand(K // G, N, device=D) * 0.1 + 0.02).contiguous()
    z = torch.randint(-3, 4, (K // G, N), device=D).float().contiguous()
    gi = torch.arange(K, device=D) // G
    full = a @ ((w.float() - z[gi]) * s[gi])
    c = torch.full((M, N), float("nan"), device=D)
    st = lambda t: t.stride()
    launch = lambda grid: _pg[grid](a, w, c, s, z, M, N, K, *st(a), *st(w), s.stride(0), s.stride(1), z.stride(0), z.stride(1), *st(c), BM=BM, BN=BN, BK=BK, G=G)
    return launch, c, full, (2, 2, 1), "gemm"


def _sym_case():
    M, N, K, BM, BN, BK = 64, 64, 128, 32, 32, 32
    torch.manual_seed(7)
    a = torch.randn(M, K, device=D)
    w = torch.randint(-127, 127, (K, N), device=D, dtype=torch.int8).contiguous()
    sc = torch.rand(N, device=D) * 0.05 + 0.01
    full = a @ (w.float() * sc)
    c = torch.full((M, N), float("nan"), device=D)
    launch = lambda grid: _sym[grid](a, w, c, sc, M, N, K, a.stride(0), a.stride(1), w.stride(0), w.stride(1), c.stride(0), c.stride(1), BM, BN, BK)
    return launch, c, full, (2, 2, 1), "gemm"


def _pn_case():
    M, N, K, BM, BN, BK = 64, 64, 128, 32, 32, 32
    torch.manual_seed(7)
    a = torch.randn(M, K, device=D)
    w = torch.randint(-8, 8, (K, N), device=D, dtype=torch.int8).contiguous()
    s = torch.rand(N, device=D) * 0.05 + 0.02
    z = torch.randint(-3, 4, (N,), device=D).float()
    full = a @ ((w.float() - z[None, :]) * s[None, :])
    c = torch.full((M, N), float("nan"), device=D)
    launch = lambda grid: _pn[grid](a, w, c, s, z, M, N, K, a.stride(0), a.stride(1), w.stride(0), w.stride(1), c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
    return launch, c, full, (2, 2, 1), "gemm"


def _fd_case(one_d, dtype=torch.float16, mode=0, cpu=False):
    M = N = K = 64
    BM = BN = BK = 32
    dev = "cpu" if cpu else D
    torch.manual_seed(7)
    a = torch.randn(M, K, device=dev, dtype=dtype)
    b = torch.randn(K, N, device=dev, dtype=dtype)
    full = (a.float() @ b.float())
    c = torch.full((M, N), float("nan"), device=dev)
    launch = lambda grid: _fd[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BM, BN, BK, ONE_D=one_d, MODE=mode)
    full_grid = (4, 1, 1) if one_d else (2, 2, 1)
    return launch, c, full, full_grid, ("fast1d" if one_d else "fast2d")


def _partial(full_grid):
    return (1,) + full_grid[1:] if full_grid[0] > 1 else full_grid


def _extra(full_grid):
    return full_grid[:2] + (2,)


def _expected_partial(kind, full, full_grid, grid):
    """What the kernel's own semantics writes under a partial launch (first programs only)."""
    e = torch.full_like(full, float("nan"))
    if kind == "gemv":
        e[: grid[0] * 32] = full[: grid[0] * 32]
    elif kind == "gemm":
        e[: grid[0] * 32, : grid[1] * 32] = full[: grid[0] * 32, : grid[1] * 32]
    elif kind == "fast2d":
        e[: grid[0] * 32, : grid[1] * 32] = full[: grid[0] * 32, : grid[1] * 32]
    elif kind == "fast1d":
        npm = 2
        for pid in range(grid[0]):
            pm, pn = pid % npm, pid // npm
            e[pm * 32:(pm + 1) * 32, pn * 32:(pn + 1) * 32] = full[pm * 32:(pm + 1) * 32, pn * 32:(pn + 1) * 32]
    return e


_FAMILIES = [
    pytest.param(_gemv8_case, id="int8-gemv"),
    pytest.param(_gemv4_case, id="int4-gemv"),
    pytest.param(_pg_case, id="pergroup-int8"),
    pytest.param(_sym_case, id="symmetric-int8"),
    pytest.param(_pn_case, id="perN-int8-fast"),
    pytest.param(lambda: _fd_case(False), id="fast-dot-2d"),
    pytest.param(lambda: _fd_case(True), id="fast-dot-1d-split"),
]


@requires_gpu
@pytest.mark.parametrize("case", _FAMILIES)
def test_full_grid_routes_to_the_replacement_and_computes(cold_gpu_caches, spies, case):
    launch, out, full, full_grid, kind = case()
    launch(full_grid[:2] if kind in ("gemm", "fast2d") else (full_grid[0],))
    torch.mps.synchronize()
    hits = spies[1] if kind.startswith("fast") else spies[0]
    assert hits == [True], f"expected the replacement to run for the full grid (quant {spies[0]}, fast {spies[1]})"
    assert _cmp(out, full) < 5e-2 * max(1.0, full.abs().max().item())


@requires_gpu
@pytest.mark.parametrize("case", _FAMILIES)
def test_partial_grid_never_runs_the_replacement(cold_gpu_caches, spies, case):
    """Bites on ba21e9d / e2330ec: the replacement ran the FULL output. Quantized: refuse
    (output untouched); fast dot: decline to the compiled kernel, which writes exactly the
    launched programs' tiles."""
    launch, out, full, full_grid, kind = case()
    grid = _partial(full_grid)
    state = _launch(lambda: launch(grid[:2] if kind in ("gemm", "fast2d") else (grid[0],)), out)
    hits = spies[1] if kind.startswith("fast") else spies[0]
    assert hits != [True], "the replacement template ran for a partial launch"
    if state == "computed":
        exp = _expected_partial(kind, full, full_grid, grid)
        assert _cmp(out, exp) < 5e-2 * max(1.0, full.abs().max().item()), "wrote tiles the launched programs never map"
    else:
        assert not kind.startswith("fast"), "the fast-dot decline must run the compiled kernel, not refuse"


@requires_gpu
@pytest.mark.parametrize("case", _FAMILIES)
def test_extra_grid_axis_never_runs_the_replacement(cold_gpu_caches, spies, case):
    """An unused grid axis > 1 is not the kernel's program mapping: quantized refuses; the
    fast dot declines to the compiled kernel (its extra programs repeat identical writes)."""
    launch, out, full, full_grid, kind = case()
    grid = _extra(full_grid)
    state = _launch(lambda: launch(grid), out)
    hits = spies[1] if kind.startswith("fast") else spies[0]
    assert hits != [True]
    if state == "computed":
        assert _cmp(out, full) < 5e-2 * max(1.0, full.abs().max().item())
    else:
        assert not kind.startswith("fast")


# ---------------------------------------------------------------- C. bare-template coordinates
_SIG_FD = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "i32", "N": "i32", "K": "i32",
           "sam": "i32", "sak": "constexpr", "sbk": "i32", "sbn": "constexpr", "scm": "i32", "scn": "constexpr"}
_CEX_FD = {"BM": 32, "BN": 32, "BK": 32, "sak": 1, "sbn": 1, "scn": 1, "ONE_D": False}


@requires_gpu
@pytest.mark.parametrize("mode", [31, 37, 38], ids=["pid-coef-2BM", "range(BN,2BN)", "range(BK,2BK)"])
def test_kloop_template_exact_tile_or_refuse(cold_gpu_caches, spies, mode):
    """Bites on e2330ec (and ba21e9d): the compiled K-loop template stored the tiled product."""
    launch, c, full, full_grid, _ = _fd_case(False, dtype=torch.float32, mode=mode)
    state = _launch(lambda: launch((2, 2)), c)
    assert state == "refused" or _cmp(c, full) > 1e-2, "the kernel's semantics differ from the canonical product"


@requires_gpu
def test_kloop_template_untiled_axes_are_tile_zero(cold_gpu_caches, spies):
    """No program_id at all, launched (2, 2): the kernel writes tile (0, 0) in every program;
    the compiled template must not tile by pid (bites on e2330ec: all four tiles written)."""
    launch, c, full, _, _ = _fd_case(False, dtype=torch.float32, mode=30)
    state = _launch(lambda: launch((2, 2)), c)
    if state == "computed":
        exp = torch.full_like(full, float("nan"))
        exp[:32, :32] = full[:32, :32]
        assert _cmp(c, exp) < 1e-2 * max(1.0, full.abs().max().item()), "tiled by program_id although the kernel does not"


@requires
@pytest.mark.parametrize("mode", [31, 37, 38])
def test_lowering_boundary_kloop_exact_tile_refuses(mode):
    with pytest.raises(MetalNonRecoverableError):
        _emit_for(_fd, _SIG_FD, {**_CEX_FD, "MODE": mode})


@requires
def test_lowering_boundary_kloop_untiled_axes_emit_tile_zero():
    msl, _ = _emit_for(_fd, _SIG_FD, {**_CEX_FD, "MODE": 30})
    assert "uint pid_m = 0u;" in msl and "uint pid_n = 0u;" in msl
    assert "pid_m = pid3.x, pid_n = pid3.y" not in msl, "the direct variant still hard-codes a 2-D mapping"


@requires
def test_lowering_boundary_direct_variant_replays_1d_split():
    """The host path's ``__mmdirect`` variant must carry the tutorial's 1-D split, not pid3.y."""
    msl, _ = _emit_for(_fd, _SIG_FD, {**_CEX_FD, "ONE_D": True, "MODE": 0})
    if "__mmdirect" in msl:
        direct = msl[msl.index("__mmdirect"):]
        assert "pid3.x % _npm" in direct and "pid_n = pid3.y" not in direct


# ---------------------------------------------------------------- D. host path (CPU tensors)
@requires_gpu
def test_host_path_1d_split_matches(cold_gpu_caches):
    """CPU tensors take the host-roundtrip path where the aligned-shape direct swap applies."""
    launch, c, full, full_grid, _ = _fd_case(True, dtype=torch.float32, cpu=True)
    launch((4,))
    assert _cmp(c, full) < 1e-2 * max(1.0, full.abs().max().item())


@requires_gpu
def test_host_path_untiled_axes_are_tile_zero(cold_gpu_caches):
    launch, c, full, _, _ = _fd_case(False, dtype=torch.float32, mode=30, cpu=True)
    try:
        launch((2, 2))
    except MetalNonRecoverableError:
        assert bool(c.isnan().all())
        return
    exp = torch.full_like(full, float("nan"))
    exp[:32, :32] = full[:32, :32]
    assert _cmp(c, exp) < 1e-2 * max(1.0, full.abs().max().item())


# ---------------------------------------------------------------- E. MLA dispatch grid gate
class _FakeRT:
    def __init__(self):
        self.calls = []

    def is_unsupported(self, msl):
        return False

    def mark_unsupported(self, msl):
        pass

    def get_library(self, msl):
        return ("lib", msl)

    def dispatch(self, lib, name, kargs, *, threads, group_size):
        self.calls.append((name, threads, group_size))


@requires_gpu
def test_mla_dispatch_refuses_legacy_descriptor_without_source_abi():
    """The MLA launch/ABI contract (grid, source strides, widths, dtypes, q-block) is pinned
    in ``tests/test_mla_dispatch_contract.py`` (packet 108); a descriptor without the source
    ABI ([14..16]) is never provably equivalent."""
    Z, H, N, hd = 2, 2, 64, 64
    t = lambda *s: torch.randn(*s, device=D, dtype=torch.float16)
    kargs = [t(Z, H, N, hd), t(Z, H, N, 32), t(Z, H, N, hd), t(Z, H, N, 32), t(Z, H, N, hd), t(Z, H, N, hd)]
    desc = ("mla", "#include <metal_stdlib>\n// mla", "mla_kernel", 256, 0, 1, 2, 3, 4, 5, 9, 10, 11, 32)
    rt = _FakeRT()
    assert _dispatch_mla(rt, desc, kargs, grid=(2, 4, 1)) is False and not rt.calls
