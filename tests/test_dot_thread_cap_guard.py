"""Actual-dispatch contract for generic cooperative dot staging.

The generic dot path used the output element count as both its emitted loop stride
and ``effective_block_size``. Metal dispatches at most 1024 threads, so an M64xN32
result (2048 elements, with only 14 KiB of shared tiles) silently left indices
1024..2047 unwritten; M64xN64xK16 left 3072/4096 unwritten. These pins prove the
generic cooperative loops stride by the ACTUAL dispatch, retain the fail-closed
guard for an unwired epilogue, and complement the lowering contract with GPU
sentinel checks. Production templates currently hide this path.
"""

import os

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir

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
    def _dot_single_tile(X, Y, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        y = tl.load(Y + ok[:, None] * N + on[None, :])
        tl.store(Z + om[:, None] * N + on[None, :], tl.dot(x, y))

    @triton.jit
    def _dot_plus_one(X, Y, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        y = tl.load(Y + ok[:, None] * N + on[None, :])
        out = tl.dot(x, y) + 1.0
        tl.store(Z + om[:, None] * N + on[None, :], out)

    @triton.jit
    def _dot_transposed_b(X, Y, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        # Load physical Y as [N, K], then present tl.dot with logical [K, N].
        # TTGIR lowers this to local_alloc -> memdesc_trans -> local_load, the
        # transposed staged-operand form admitted by the cooperative eligibility gate.
        y_nk = tl.load(Y + on[:, None] * K + ok[None, :])
        y_kn = tl.trans(y_nk)
        tl.store(Z + om[:, None] * N + on[None, :], tl.dot(x, y_kn))


def _generic_lowerer(m, n, k, fn=None):
    """Compile to TTGIR, then construct the generic lowerer without JIT caches."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from triton_msl.backend.compiler import MetalBackend

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    src = ASTSource(
        fn=fn or _dot_single_tile,
        signature={"X": "*fp32", "Y": "*fp32", "Z": "*fp32"},
        constexprs={"M": m, "N": n, "K": k},
    )
    context = ir.context()
    ir.load_dialects(context)
    mod = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    return generic_lowerer.GenericLowerer(walk_ttgir(mod, options), options)


def _force_generic_dot(monkeypatch):
    """Bypass production template detection to exercise the staging boundary."""
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_detect_matmul_epilogue", lambda self: None)
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_detect_simple_dot", lambda self: None)
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_requires_matmul_template", lambda self: False)


@requires
@pytest.mark.parametrize(
    ("logical_elements", "expected_dispatch"),
    [(1024, 1024), (1025, 1024), (2048, 1024), (4096, 1024)],
)
def test_cooperative_dispatch_boundary(logical_elements, expected_dispatch):
    assert generic_lowerer.GenericLowerer._cooperative_dispatch_threads(logical_elements) == expected_dispatch


@requires
def test_generic_dot_staging_2048_uses_actual_dispatch(monkeypatch):
    # Pre-fix GPU bite: exactly 1024/2048 outputs remained NaN, first index 1024.
    _force_generic_dot(monkeypatch)
    lowerer = _generic_lowerer(64, 32, 16)
    msl = lowerer.lower()
    assert lowerer.effective_block_size == 1024
    assert lowerer._actual_dispatch_threads == 1024
    assert "for (uint _sa = lid; _sa < 1024u; _sa += 1024u)" in msl
    assert "for (uint _sa = lid; _sa < 512u; _sa += 1024u)" in msl
    assert "for (uint _de = lid; _de < 2048u; _de += 1024u)" in msl
    assert "for (uint _st = lid; _st < 2048u; _st += 1024u)" in msl


@requires
def test_generic_dot_staging_4096_uses_actual_dispatch(monkeypatch):
    # Pre-fix GPU bite: 3072/4096 outputs remained NaN, again starting at 1024.
    _force_generic_dot(monkeypatch)
    lowerer = _generic_lowerer(64, 64, 16)
    msl = lowerer.lower()
    assert lowerer.effective_block_size == 1024
    assert lowerer._actual_dispatch_threads == 1024
    assert msl.count("for (uint _sa = lid; _sa < 1024u; _sa += 1024u)") == 2
    assert "for (uint _de = lid; _de < 4096u; _de += 1024u)" in msl
    assert "for (uint _st = lid; _st < 4096u; _st += 1024u)" in msl


@requires
def test_generic_dot_staging_at_1024_stays_allowed(monkeypatch):
    # Equality is the validated boundary: a 32x32 output maps one element to
    # each of the 1024 dispatched threads and must not be caught by the guard.
    _force_generic_dot(monkeypatch)
    lowerer = _generic_lowerer(32, 32, 32)
    msl = lowerer.lower()
    assert lowerer.effective_block_size == 1024
    assert "kernel void" in msl


@requires
def test_generic_dot_unwired_epilogue_above_1024_still_refuses(monkeypatch):
    # Stage 1B wires load/local_alloc/dot/store only. A value-changing epilogue still
    # has one scalar value per lid and must not inherit the cooperative-loop exemption.
    _force_generic_dot(monkeypatch)
    lowerer = _generic_lowerer(64, 32, 16, fn=_dot_plus_one)
    with pytest.raises(MetalNonRecoverableError, match=r"2048.*1024.*unwritten"):
        lowerer.lower()


@pytest.fixture()
def cold_gpu_caches(tmp_path, monkeypatch):
    """Force every GPU pin through this checkout's lowering, not a prior executable."""
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    os.makedirs(tmp_path / "msl", exist_ok=True)
    for fn in (_dot_single_tile, _dot_plus_one, _dot_transposed_b):
        cache = getattr(fn, "device_caches", None)
        assert cache is not None
        cache.clear()
    return tmp_path


@requires_gpu
@pytest.mark.parametrize(("m", "n", "k"), [(64, 32, 16), (64, 64, 16)])
def test_generic_dot_mept_gpu_covers_every_output(monkeypatch, cold_gpu_caches, m, n, k):
    _force_generic_dot(monkeypatch)
    # Prove the compile-time route without relying on a runtime executable
    # cache populated earlier in a randomized session.
    lowerer = _generic_lowerer(m, n, k)
    msl = lowerer.lower()
    assert f"_de < {m * n}u" in msl
    assert f"_st < {m * n}u" in msl
    torch.manual_seed(51000 + m * 100 + n)
    x = torch.randn((m, k), dtype=torch.float32)
    y = torch.randn((k, n), dtype=torch.float32)
    out = torch.full((m, n), float("nan"), dtype=torch.float32)
    _dot_single_tile[(1,)](x, y, out, M=m, N=n, K=k)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, x @ y, atol=1e-4, rtol=1e-4)


@requires_gpu
def test_generic_dot_mept_transposed_operand_lowering_and_gpu(monkeypatch, cold_gpu_caches):
    """Pin the admitted memdesc-transpose operand at both contract boundaries."""
    _force_generic_dot(monkeypatch)

    lowerer = _generic_lowerer(64, 32, 16, fn=_dot_transposed_b)
    msl = lowerer.lower()
    assert lowerer._actual_dispatch_threads == 1024
    assert "for (uint _de = lid; _de < 2048u; _de += 1024u)" in msl
    assert "for (uint _st = lid; _st < 2048u; _st += 1024u)" in msl
    # A transposed B must index its physical [N,K] staging as [col*K + k].
    assert "[_dot_col * 16u + _dk]" in msl

    torch.manual_seed(51064)
    x = torch.randn((64, 16), dtype=torch.float32)
    y_nk = torch.randn((32, 16), dtype=torch.float32)
    out = torch.full((64, 32), float("nan"), dtype=torch.float32)
    _dot_transposed_b[(1,)](x, y_nk, out, M=64, N=32, K=16)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, x @ y_nk.T, atol=1e-4, rtol=1e-4)
