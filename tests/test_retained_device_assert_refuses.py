"""Issue #8: retained assertions must execute or explicitly refuse, never disappear.
Inductor retains bounds checks by default. Uniform-control entry assertions now
execute with a launch-local host error; device-callee assertions remain unsupported.

The refusal must cover a retained assert that lives inside a NESTED region of a noinline
callee (packet 469): the device-function emitter would otherwise emit it as an
`// UNSUPPORTED in device func: tt.assert` comment and drop it. Both an on-device refusal
and a CPU lowering-boundary pin are kept, and the pins assert the retained OP was seen
(retained-assert count), not merely that some exception was raised.
"""
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl
from triton_msl.errors import MetalNonRecoverableError

_HAS_MPS = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not _HAS_MPS, reason="Metal GPU required")

ROWS = 2


def _observe_launches(monkeypatch, shader):
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_compile_shader_runtime, _get_utils
    calls = []
    def wrapper(_instance, real):
        def observe(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)
        return observe
    patch_live_singleton_method(monkeypatch, _get_compile_shader_runtime if shader == '1' else _get_utils,
                                'dispatch' if shader == '1' else 'launch', wrapper)
    return calls


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------
@triton.jit
def _guarded_load(TABLE, OUT, idx, N: tl.constexpr):
    tl.device_assert(idx < N, "row out of range")
    cols = tl.arange(0, 8)
    tl.store(OUT + cols, tl.load(TABLE + idx * 8 + cols))


@triton.jit
def _guarded_atomic(OUT, valid):
    tl.device_assert(valid != 0, 'atomic guard')
    lane = tl.arange(0, 8)
    tl.atomic_add(OUT + lane, 32)


@triton.jit(noinline=True)
def _callee(x):
    # returns x (NOT x+1): x+1 in debug mode inserts an UNRELATED overflow assert at the
    # callee's top level, which would mask the nested-region assert this pins (469).
    if x > 0:
        tl.device_assert(x < 8, "callee retained")
    return x


@triton.jit
def _guarded_callee(OUT, x):
    tl.store(OUT, _callee(x))


# --------------------------------------------------------------------------------------
# On-device refusals / positive
# --------------------------------------------------------------------------------------
@requires_mps
@pytest.mark.parametrize('shader', ['0', '1'])
def test_retained_assert_debug_true_checks_before_access(monkeypatch, shader):
    monkeypatch.setenv('TRITON_MSL_COMPILE_SHADER', shader)
    calls = _observe_launches(monkeypatch, shader)
    table = torch.arange(ROWS * 8, device="mps", dtype=torch.float32)
    out = torch.full((8,), -99., device="mps")
    from triton_msl.errors import MetalDeviceAssertionError
    _guarded_load[(1,)](table, out, 1, N=ROWS, debug=True)
    assert torch.equal(out.cpu(), torch.arange(8, 16, dtype=torch.float32))
    out.fill_(-99.)
    with pytest.raises(MetalDeviceAssertionError, match='row out of range'):
        _guarded_load[(1,)](table, out, 3, N=ROWS, debug=True)
    assert torch.equal(out.cpu(), torch.full((8,), -99.)), 'guarded access/store ran after a failed assertion'
    _guarded_load[(1,)](table, out, 0, N=ROWS, debug=True)
    assert torch.equal(out.cpu(), torch.arange(8, dtype=torch.float32)), 'prior failure poisoned a valid launch'
    atomic_out = torch.zeros(8, dtype=torch.int32, device='mps')
    with pytest.raises(MetalDeviceAssertionError, match='atomic guard'):
        _guarded_atomic[(1,)](atomic_out, 0, debug=True)
    assert torch.equal(atomic_out.cpu(), torch.zeros(8, dtype=torch.int32)), 'guarded atomic executed after failure'
    _guarded_atomic[(1,)](atomic_out, 2, debug=True)
    assert torch.equal(atomic_out.cpu(), torch.full((8,), 32, dtype=torch.int32)), 'atomic was omitted or replayed'
    assert len(calls) == 5, 'requested runtime path was not witnessed exactly once per invocation'


@requires_mps
def test_retained_assert_in_noinline_callee_region_refuses():
    # The retained assert lives inside the callee's `if` region (469's escape).
    out = torch.empty(1, device="mps", dtype=torch.int32)
    with pytest.raises(MetalNonRecoverableError):
        _guarded_callee.warmup(out, 1, grid=(1,), debug=True)


@requires_mps
def test_same_callee_source_debug_false_computes():
    # SAME nested-callee source, debug off: the frontend removes the assert, so it lowers
    # and runs (this replaces the weaker no-assert control from the first draft).
    out = torch.full((1,), -1, device="mps", dtype=torch.int32)
    _guarded_callee[(1,)](out, 5)
    torch.mps.synchronize()
    assert out.cpu().tolist() == [5], out.cpu().tolist()


# --------------------------------------------------------------------------------------
# Lowering-boundary pin — CPU, no GPU required (protocol 9)
# --------------------------------------------------------------------------------------
def _walk(ops):
    for o in ops or ():
        yield o
        yield from _walk(getattr(o, "region_ops", None))
        yield from _walk(getattr(o, "else_ops", None))


def _compile_guarded_callee(debug):
    """Compile the nested-callee source to a GenericLowerer; return (retained_count, err)
    where err is the MetalNonRecoverableError if lowering refused, else None."""
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"debug": debug})
    src = ASTSource(_guarded_callee, signature={"OUT": "*i32", "x": "i32"}, constexprs={})
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options),
                      backend.get_module_map(), ctx)
    meta = {}
    mod = backend.make_ttir(mod, meta, options)
    mod = backend.make_ttgir(mod, meta, options)
    graph = walk_ttgir(mod, options)
    retained = sum(o.op == "tt.assert" for o in _walk(graph.ops))
    retained += sum(o.op == "tt.assert"
                    for f in (graph.called_funcs or []) for o in _walk(f.ops))
    lower = GenericLowerer(graph, options)
    try:
        lower.lower()
        return retained, None
    except MetalNonRecoverableError as e:
        return retained, e


def test_lowering_boundary_nested_callee_assert_refuses():
    # debug=True: exactly one retained assert (inside the callee region) and lowering
    # refuses BY NAME on tt.assert -- the deterministic prescan refusal, not the generic
    # downstream unsupported-op backstop.
    retained, err = _compile_guarded_callee(debug=True)
    assert retained == 1, f"expected the callee-region assert to be retained; got {retained}"
    assert err is not None, "retained callee-region assert must refuse, not lower"
    assert getattr(err, "op_name", None) == "tt.assert", getattr(err, "op_name", None)


def test_lowering_boundary_nested_callee_no_assert_debug_false_lowers():
    retained, err = _compile_guarded_callee(debug=False)
    assert retained == 0, f"debug off should drop the assert in the frontend; got {retained}"
    assert err is None, f"no retained assert -> must lower; refused: {err}"


@triton.jit
def _masked_loop_assert(TABLE, INDEX, OUT, active, iterations, B: tl.constexpr):
    lane = tl.arange(0, B)
    idx = tl.load(INDEX + lane)
    total = tl.full((B,), 0., tl.float32)
    for step in range(iterations):
        tl.device_assert((idx >= 0) | (lane >= active), 'negative index')
        tl.device_assert((idx < B) | (lane >= active), 'index exceeds table')
        total += tl.load(TABLE + idx, lane < active, other=0.)
    tl.store(OUT + lane, total)


@requires_mps
@pytest.mark.parametrize('shader', ['0', '1'])
def test_masked_loop_assert_group_stop_and_lane_semantics(monkeypatch, shader):
    monkeypatch.setenv('TRITON_MSL_COMPILE_SHADER', shader)
    calls = _observe_launches(monkeypatch, shader)
    from triton_msl.errors import MetalDeviceAssertionError
    b = 64  # Two SIMD groups: the failure must stop the whole threadgroup.
    table = torch.arange(b, dtype=torch.float32, device='mps')
    indices = torch.arange(b, dtype=torch.int32, device='mps')
    indices[-1] = 1000000  # Invalid address, but outside the source's active mask.
    out = torch.full((b,), -99., device='mps')
    compiled = _masked_loop_assert[(1,)](table, indices, out, b-1, 3, B=b, debug=True)
    expected = torch.arange(b, dtype=torch.float32) * 3
    expected[-1] = 0
    assert torch.equal(out.cpu(), expected), 'masked-out invalid lane raised or changed valid values'
    # The emitted stop dominates TABLE access; keeping INDEX in bounds and TABLE
    # distinct makes the guarded read unambiguous, rather than a canary write alone.
    msl = compiled.asm['msl']
    check = msl.index('if (assert_failed_')
    assert check < msl.index('TABLE[')
    assert 'atomic_fetch_max_explicit' in msl
    for bad in (-1, 1000000):
        indices[33] = bad
        out.fill_(-99.)
        message = 'negative index' if bad < 0 else 'index exceeds table'
        with pytest.raises(MetalDeviceAssertionError, match=message):
            _masked_loop_assert[(1,)](table, indices, out, b-1, 3, B=b, debug=True)
        assert torch.equal(out.cpu(), torch.full((b,), -99.)), 'a threadgroup continued after its failed check'
    indices[33] = 33
    _masked_loop_assert[(1,)](table, indices, out, b-1, 3, B=b, debug=True)
    assert torch.equal(out.cpu(), expected)
    assert len(calls) == 4, 'requested runtime path was not witnessed exactly once per invocation'


def _entry_lowerer(fn, signature, constexprs):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    target = GPUTarget('metal', 'apple-m4', 32)
    backend = MetalBackend(target)
    options = backend.parse_options({'debug': True})
    src = ASTSource(fn, signature=signature, constexprs=constexprs)
    ctx = ir.context(); ir.load_dialects(ctx)
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx)
    meta = {}
    mod = backend.make_ttgir(backend.make_ttir(mod, meta, options), meta, options)
    graph = walk_ttgir(mod, options)
    return graph, GenericLowerer(graph, options)


def test_lowering_boundary_real_asserts_precede_guarded_access():
    graph, lowerer = _entry_lowerer(_masked_loop_assert, {'TABLE': '*fp32', 'INDEX': '*i32',
                     'OUT': '*fp32', 'active': 'i32', 'iterations': 'i32'}, {'B': 64})
    retained = [op for op in _walk(graph.ops) if op.op == 'tt.assert']
    msl = lowerer.lower()
    assert [op.attrs['message'] for op in retained] == ['negative index', 'index exceeds table']
    assert msl.count('atomic_fetch_max_explicit') == len(retained) == 2
    assert msl.count('if (assert_failed_') == 2
    assert msl.rindex('if (assert_failed_') < msl.index('TABLE[')
    # Each verdict is sampled, synchronized, and only then used for a return.
    import re
    assert len(re.findall(r'bool (assert_failed_\d+) = atomic_load_explicit[^;]+;\s*threadgroup_barrier\(mem_flags::mem_threadgroup\);\s*if \(\1\) return;', msl)) == 2
    assert lowerer.effective_block_size == 64


@triton.jit
def _mixed_width_predicate(TABLE, OUT):
    small = tl.arange(0, 8)
    tl.device_assert(tl.load(TABLE + small) >= 0, 'narrow predicate')
    wide = tl.arange(0, 128)
    tl.store(OUT + wide, 1.)


def test_lowering_boundary_unproved_predicate_ownership_refuses():
    # Eight source predicate lanes must not accidentally test 128 physical lanes.
    # Until this mixed-width mapping is proved, refuse BEFORE emitting any load.
    _, lowerer = _entry_lowerer(_mixed_width_predicate, {'TABLE': '*fp32', 'OUT': '*fp32'}, {})
    with pytest.raises(MetalNonRecoverableError, match='predicate ownership'):
        lowerer.lower()
