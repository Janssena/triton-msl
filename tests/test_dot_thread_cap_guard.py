"""Fail-closed guard for cooperative staging above Metal's thread cap.

The generic dot path used the output element count as both its emitted loop stride
and ``effective_block_size``. Metal dispatches at most 1024 threads, so an M64xN32
result (2048 elements, with only 14 KiB of shared tiles) silently left indices
1024..2047 unwritten. Production templates currently hide this path; these pins
lower TTGIR directly through the generic path so future envelope widening cannot
expose it again and session-level executable caches cannot bypass the assertion.
"""

import pytest

try:
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir

    from triton_msl.codegen.generic_lowerer import GenericLowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
except ImportError:
    HAS = False


requires = pytest.mark.skipif(not HAS, reason="Triton compiler needed")


if HAS:

    @triton.jit
    def _dot_single_tile(X, Y, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        y = tl.load(Y + ok[:, None] * N + on[None, :])
        tl.store(Z + om[:, None] * N + on[None, :], tl.dot(x, y))


def _generic_lowerer(m, n, k):
    """Compile to TTGIR, then construct the generic lowerer without JIT caches."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from triton_msl.backend.compiler import MetalBackend

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    src = ASTSource(
        fn=_dot_single_tile,
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
    return GenericLowerer(walk_ttgir(mod, options), options)


def _force_generic_dot(monkeypatch):
    """Bypass production template detection to exercise the staging boundary."""
    monkeypatch.setattr(GenericLowerer, "_detect_matmul_epilogue", lambda self: None)
    monkeypatch.setattr(GenericLowerer, "_has_unhandled_matmul_compute_epilogue", lambda self: False)
    monkeypatch.setattr(GenericLowerer, "_detect_simple_dot", lambda self: None)
    monkeypatch.setattr(GenericLowerer, "_requires_matmul_template", lambda self: False)


@requires
def test_generic_dot_staging_above_1024_refuses(monkeypatch):
    # GPU bites proof on the unpatched tree: Metal ran 1024 threads, and exactly
    # 1024/2048 output elements remained NaN. Direct lowering did not raise.
    _force_generic_dot(monkeypatch)
    lowerer = _generic_lowerer(64, 32, 16)
    with pytest.raises(MetalNonRecoverableError, match=r"2048.*1024.*unwritten"):
        lowerer.lower()


@requires
def test_generic_dot_staging_at_1024_stays_allowed(monkeypatch):
    # Equality is the validated boundary: a 32x32 output maps one element to
    # each of the 1024 dispatched threads and must not be caught by the guard.
    _force_generic_dot(monkeypatch)
    lowerer = _generic_lowerer(32, 32, 32)
    msl = lowerer.lower()
    assert lowerer.effective_block_size == 1024
    assert "kernel void" in msl
