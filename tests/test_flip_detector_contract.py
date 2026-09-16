import triton
import triton.language as tl
import pytest
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _xor_plus_one(a, b):
    return (a ^ b) + 1


@triton.jit
def _xor(a, b):
    return a ^ b


@triton.jit
def _flip(X, Z):
    a, b, c = tl.arange(0, 4), tl.arange(0, 4), tl.arange(0, 8)
    off = a[:, None, None] * 32 + b[None, :, None] * 8 + c[None, None, :]
    tl.store(Z + off, tl.flip(tl.load(X + off), 1))


@triton.jit
def _flip_size_two(X, Z):
    a, b, c = tl.arange(0, 4), tl.arange(0, 2), tl.arange(0, 8)
    off = a[:, None, None] * 16 + b[None, :, None] * 8 + c[None, None, :]
    tl.store(Z + off, tl.flip(tl.load(X + off), 1))


@triton.jit
def _flip_axis(X, Z, DIM: tl.constexpr):
    a, b, c = tl.arange(0, 4), tl.arange(0, 4), tl.arange(0, 4)
    off = a[:, None, None] * 16 + b[None, :, None] * 4 + c[None, None, :]
    tl.store(Z + off, tl.flip(tl.load(X + off), DIM))


@triton.jit
def _bad_reducer(X, Z):
    a, b, c = tl.arange(0, 4), tl.arange(0, 2), tl.arange(0, 8)
    off = a[:, None, None] * 16 + b[None, :, None] * 8 + c[None, None, :]
    x = tl.load(X + off)
    red = tl.reduce(x, 1, _xor_plus_one)
    tl.store(Z + off, x ^ red[:, None, :])


@triton.jit
def _extra_output_arithmetic(X, Z):
    a, b, c = tl.arange(0, 4), tl.arange(0, 2), tl.arange(0, 8)
    off = a[:, None, None] * 16 + b[None, :, None] * 8 + c[None, None, :]
    x = tl.load(X + off)
    red = tl.reduce(x, 1, _xor)
    tl.store(Z + off, (x ^ red[:, None, :]) + 1)


@triton.jit
def _strided_flip(X, Z):
    a, b, c = tl.arange(0, 4), tl.arange(0, 4), tl.arange(0, 8)
    off = a[:, None, None] * 40 + b[None, :, None] * 9 + c[None, None, :]
    tl.store(Z + off, tl.flip(tl.load(X + off), 1))


@triton.jit
def _different_store_offset(X, Z):
    a, b, c = tl.arange(0, 4), tl.arange(0, 4), tl.arange(0, 8)
    dense = a[:, None, None] * 32 + b[None, :, None] * 8 + c[None, None, :]
    shifted = dense + 1
    tl.store(Z + shifted, tl.flip(tl.load(X + dense), 1))


@triton.jit
def _masked_flip(X, Z):
    a, b, c = tl.arange(0, 4), tl.arange(0, 4), tl.arange(0, 8)
    off = a[:, None, None] * 32 + b[None, :, None] * 8 + c[None, None, :]
    x = tl.load(X + off, mask=off < 100, other=7)
    tl.store(Z + off, tl.flip(x, 1), mask=off < 100)


@triton.jit
def _permuted_coordinate_flip(X, Z):
    a, b, c = tl.arange(0, 4), tl.arange(0, 4), tl.arange(0, 4)
    # Same extent/coefficient multiset as dense row-major, but the 16 and 1
    # coefficients belong to the wrong tensor coordinates.
    off = c[None, None, :] * 16 + b[None, :, None] * 4 + a[:, None, None]
    tl.store(Z + off, tl.flip(tl.load(X + off), 1))


@triton.jit
def _base_shifted_flip(X, Z):
    a, b, c = tl.arange(0, 4), tl.arange(0, 4), tl.arange(0, 4)
    off = a[:, None, None] * 16 + b[None, :, None] * 4 + c[None, None, :]
    tl.store((Z + 1) + off, tl.flip(tl.load((X + 1) + off), 1))


@triton.jit
def _flip_with_atomic(X, Z, COUNTER):
    a, b, c = tl.arange(0, 4), tl.arange(0, 4), tl.arange(0, 4)
    off = a[:, None, None] * 16 + b[None, :, None] * 4 + c[None, None, :]
    tl.store(Z + off, tl.flip(tl.load(X + off), 1))
    tl.atomic_add(COUNTER, 1)


def _lower(fn, dtype="i32", constants=None):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    ctx = ir.context()
    ir.load_dialects(ctx)
    constants = constants or {}
    signature = {
        name: (f"*{dtype}" if name in ("X", "Z") else "*i32" if name == "COUNTER" else "constexpr")
        for name in fn.arg_names
    }
    module = ASTSource(fn, signature, constants).make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    lowerer = GenericLowerer(walk_ttgir(module, options), options)
    return lowerer, module, backend, options


def test_canonical_flip_variants_retain_direct_template():
    for fn, shape, dtype in (
        (_flip, (4, 4, 8), "i32"),
        (_flip_size_two, (4, 2, 8), "i32"),
        (_flip, (4, 4, 8), "fp32"),
        (_flip_size_two, (4, 2, 8), "fp32"),
    ):
        lowerer, module, backend, options = _lower(fn, dtype)
        info = lowerer._detect_flip()
        assert (info["M"], info["N"], info["K"], info["flip_dim"]) == (*shape, 1)
        assert "uint _src" in backend.make_msl(module, {}, options)
    for axis in (0, 1, 2):
        lowerer, module, backend, options = _lower(_flip_axis, "fp32", {"DIM": axis})
        assert lowerer._detect_flip()["flip_dim"] == axis
        assert "uint _src" in backend.make_msl(module, {}, options)


def test_xor_presence_does_not_bless_wrong_reducer_or_live_post_arithmetic():
    for fn in (_bad_reducer, _extra_output_arithmetic):
        lowerer, module, backend, options = _lower(fn)
        assert lowerer._detect_flip() is None
        with pytest.raises(MetalNonRecoverableError):
            backend.make_msl(module, {}, options)


def test_direct_template_requires_exact_dense_shared_load_store_offset():
    for fn in (
        _strided_flip,
        _different_store_offset,
        _masked_flip,
        _permuted_coordinate_flip,
        _base_shifted_flip,
    ):
        lowerer, module, backend, options = _lower(fn)
        assert lowerer._detect_flip() is None
        try:
            emitted = backend.make_msl(module, {}, options)
        except MetalNonRecoverableError:
            continue
        assert "uint _src" not in emitted


def test_malformed_reducer_return_or_block_operand_refuses():
    lowerer, _module, _backend, _options = _lower(_flip_size_two)
    red = next(op for op in lowerer.graph.ops if op.op == "tt.reduce")
    original_return = red.attrs["return_ids"]
    red.attrs["return_ids"] = [red.id]
    assert lowerer._detect_flip() is None
    red.attrs["return_ids"] = original_return
    red.region_ops[0].operand_ids[0] = red.id
    assert lowerer._detect_flip() is None


def test_observable_atomic_side_effect_cannot_be_dropped():
    lowerer, module, backend, options = _lower(_flip_with_atomic)
    assert any("atomic" in op.op for op in lowerer.graph.ops)
    assert lowerer._detect_flip() is None
    try:
        emitted = backend.make_msl(module, {}, options)
    except MetalNonRecoverableError:
        return
    assert "uint _src" not in emitted
