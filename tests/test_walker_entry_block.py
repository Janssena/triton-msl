"""Packet 166 (155 §2) — the walker's function entry-block discovery.

`walk_ttgir` visits the module in post-order, so a nested body is visited BEFORE the op that owns it.
The entry-block discovery took the block of the first VISITED op; when a function's FIRST op owns a
region, that block is the nested body: the function's arguments were read off the body's block
arguments (wrong count, wrong types), the region-owning op and everything after it were filed as
"nested" and dropped, and the body's ops became the function.

Reach: (1) a kernel whose first statement is a `while` over carried values with no literal anywhere
before it (`_k_while_carried`) lowered to a COMPILABLE kernel whose output pointer is a scalar float
and which never stores — a silent-wrong (the output stays untouched); (2) the same with a function
argument in the condition (`_k_while_arg`) refused loudly with a misleading "unresolved value"
message; (3) a scalar `noinline` callee with that shape lost its body the same way; (4) direct TTGIR
with reduction-first functions (GPT's 155 witness) reported CALLEE_MISSING_REDUCTION. Literals are
hoisted to a leading `arith.constant`, and tensor arguments are not allowed on kernels or noinline
functions in Triton 3.7, so `while` is the Python-reachable shape and reductions the TTGIR one.

Now the entry block is the first visited block whose parent region is a FUNCTION BODY (region depth
2: module body = 1), on both the entry and the callee paths.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    import triton
    import triton.language as tl

    import triton_msl
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import IRSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.mlir_walker import walk_ttgir
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


# --------------------------------------------------------------------------- Python-reachable shapes

@triton.jit
def _k_while_carried(out_ptr, x, y):
    # first statement owns a region; its condition reads only carried values, no literal anywhere
    while x < y:
        x = x + x
        y = y - x
    tl.store(out_ptr + tl.program_id(0), x)


@triton.jit
def _k_while_carried_i(out_ptr, n, m):
    while n < m:
        n = n + n
        m = m - n
    tl.store(out_ptr + tl.program_id(0), n)


@triton.jit
def _k_while_arg(out_ptr, x, lim):
    # a function argument (not carried) in the condition: pre-166 an "unresolved value" refusal
    while x < lim:
        x = x + x
    tl.store(out_ptr + tl.program_id(0), x)


@triton.jit
def _k_while_literal(out_ptr, x):
    # control: the literal is hoisted to a leading arith.constant, so the first op owns no region
    while x < 10.0:
        x = x + x
    tl.store(out_ptr + tl.program_id(0), x)


@triton.jit(noinline=True)
def _double_until(x, lim):
    while x < lim:
        x = x + x
    return x


@triton.jit
def _k_callee_while(out_ptr, x, lim):
    tl.store(out_ptr + tl.program_id(0), _double_until(x, lim))


def _ref_carried(x, y, dt):
    x, y = dt(x), dt(y)
    while x < y:
        x = dt(x + x)
        y = dt(y - x)
    return x


def _ref_arg(x, lim):
    x, lim = np.float32(x), np.float32(lim)
    while x < lim:
        x = np.float32(x + x)
    return x


def _walk(ops):
    for op in ops:
        yield op
        yield from _walk(op.region_ops or [])


_SCALAR_SIG = {"out_ptr": "*fp32", "x": "fp32", "y": "fp32", "lim": "fp32"}


def test_walker_files_while_first_entry_under_the_function():
    """Direct lowering (CPU): the kernel's three arguments survive with their types, the loop is a
    top-level op followed by the store and the return, and the MSL stores through the pointer.
    Pre-166: two float arguments named after the loop's carried values, ops = [cmpf, condition],
    an MSL with one comparison and no store."""
    lw = _build_lowerer(_k_while_carried, {k: _SCALAR_SIG[k] for k in ("out_ptr", "x", "y")}, {})
    g = lw.graph
    assert [(a.name, a.type_str) for a in g.args] == [("out_ptr", "!tt.ptr<f32>"), ("x", "f32"), ("y", "f32")]
    top = [o.op for o in g.ops]
    assert top[0] == "scf.while" and top[-2:] == ["tt.store", "tt.return"], top
    msl = lw.lower()
    assert "out_ptr[" in msl and "for (;;)" in msl and "UNKNOWN_" not in msl and "UNSUPPORTED" not in msl


def test_walker_files_while_first_callee_under_the_callee():
    """Direct lowering (CPU): the scalar noinline callee keeps its loop and its return as top-level
    ops. Pre-166 its ops were [cmpf, scf.condition] — the loop's condition block became the callee."""
    g = _build_lowerer(_k_callee_while, {k: _SCALAR_SIG[k] for k in ("out_ptr", "x", "lim")}, {}).graph
    assert g.called_funcs and len(g.called_funcs) == 1
    f = g.called_funcs[0]
    # both arguments, from the function-body block — not the loop's condition block, which also
    # carries block arguments (one: the carried x) and won the unordered "any block with args" scan
    assert [(a.name, a.type_str) for a in f.args] == [("x", "f32"), ("lim", "f32")], f.args
    top = [o.op for o in f.ops]
    assert top == ["scf.while", "tt.return"], top
    assert [o.op for o in _walk(f.ops)] == ["scf.while", "arith.cmpf", "scf.condition", "tt.return"]


@requires_gpu
@pytest.mark.parametrize("kernel,dtype,cases", [
    (_k_while_carried, torch.float32, [(1.0, 100.0), (3.0, 5.0), (0.5, 0.25), (7.0, 7.0)]),
    # no integer argument equal to 1: Triton's JIT specializes it as a constexpr and its own
    # frontend then rejects the loop ("carried variable changed type") on every backend
    (_k_while_carried_i, torch.int32, [(2, 100), (3, 5), (9, 4), (7, 7)]),
])
def test_while_first_kernel_computes(cold_gpu_caches, kernel, dtype, cases):
    """Two-arm, correct-or-refuse: the kernel whose first statement is a `while` over carried values
    must write the Python loop's result. Pre-166 the lowered kernel compiled with no store (its
    output pointer typed as a scalar), so the output stayed at its sentinel — a silent-wrong."""
    dt = np.float32 if dtype == torch.float32 else np.int64
    for a, b in cases:
        out = torch.full((1,), -12345, device=D, dtype=dtype)
        if hasattr(kernel, "device_caches"):
            kernel.device_caches.clear()
        kernel[(1,)](out, a, b)
        torch.mps.synchronize()
        got = out.item()
        want = _ref_carried(a, b, dt)
        assert got == want, f"{kernel.fn.__name__}({a}, {b}) = {got}, expected {want}"


@requires_gpu
def test_while_first_kernel_with_argument_in_condition_computes(cold_gpu_caches):
    """The same shape with a function argument in the condition. Pre-166: the argument fell outside
    the misfiled function and the emitter refused with the unrelated 'unresolved value' message;
    now it runs and matches the Python loop."""
    for a, lim in [(1.0, 100.0), (0.75, 3.0), (5.0, 2.0)]:
        out = torch.full((1,), -1.0, device=D)
        if hasattr(_k_while_arg, "device_caches"):
            _k_while_arg.device_caches.clear()
        _k_while_arg[(1,)](out, a, lim)
        torch.mps.synchronize()
        assert out.item() == _ref_arg(a, lim), (a, lim, out.item())


@requires_gpu
def test_while_first_callee_refuses_honestly(cold_gpu_caches):
    """The device-function emitter does not lower scf.while (ledgered). That must surface as the
    honest 'could not lower' refusal, not the pre-166 'unresolved value' misdiagnosis produced by
    the misfiled callee."""
    out = torch.zeros(1, device=D)
    if hasattr(_k_callee_while, "device_caches"):
        _k_callee_while.device_caches.clear()
    with pytest.raises(MetalNonRecoverableError) as ei:
        _k_callee_while[(1,)](out, 1.0, 100.0)
        torch.mps.synchronize()
    assert "unresolved value" not in str(ei.value), str(ei.value)[:300]


@requires_gpu
def test_while_against_literal_unaffected(cold_gpu_caches):
    """Control on both sides of the fix: the literal is hoisted to a leading constant, the first op
    owns no region, and the kernel computes."""
    g = _build_lowerer(_k_while_literal, {"out_ptr": "*fp32", "x": "fp32"}, {}).graph
    assert [o.op for o in g.ops][:2] == ["arith.constant", "scf.while"]
    out = torch.zeros(1, device=D)
    _k_while_literal[(1,)](out, 0.3)
    torch.mps.synchronize()
    assert out.item() == _ref_arg(0.3, 10.0)


# --------------------------------------------------------------------------- direct TTGIR (155 witness)

_HEADER = '''#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
'''

_CALLEES = _HEADER + '''  tt.func public @entry(%p: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %addr = tt.addptr %ps, %r : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    %x = tt.load %addr : tensor<32x!tt.ptr<f32>, #blocked>
    %a = tt.call @sum_helper(%x) : (tensor<32xf32, #blocked>) -> f32
    %b = tt.call @left_helper(%x) : (tensor<32xf32, #blocked>) -> f32
    %c = arith.addf %a, %b : f32
    tt.store %out, %c : !tt.ptr<f32>
    tt.return
  }
  tt.func private @sum_helper(%x: tensor<32xf32, #blocked>) -> f32 {
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<32xf32, #blocked>) -> f32
    tt.return %r : f32
  }
  tt.func private @left_helper(%x: tensor<32xf32, #blocked>) -> f32 {
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %a : f32
    }) : (tensor<32xf32, #blocked>) -> f32
    tt.return %r : f32
  }
}
'''

_ENTRY_FIRST = _HEADER + '''  tt.func public @entry(%x: tensor<32xf32, #blocked>, %out: !tt.ptr<f32>) {
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<32xf32, #blocked>) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
'''


def _parse(text, tmp_path, name="m.ttgir"):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    path = tmp_path / name
    path.write_text(text)
    ctx = ir.context()
    src = IRSource(str(path), ctx, backend)
    assert src.module.verify()
    return walk_ttgir(src.module, backend.parse_options({}))


def _check_reduce(ops, yielded_is_addf):
    top = [o.op for o in ops]
    assert top == ["tt.reduce", "tt.return"] or top == ["tt.reduce", "tt.store", "tt.return"], top
    red = ops[0]
    assert [o.op for o in red.region_ops] == ["arith.addf"]
    expected = red.region_ops[0].id if yielded_is_addf else red.attrs["block_arg_ids"][0]
    assert red.attrs.get("return_ids") == [expected], (red.attrs.get("return_ids"), expected)


def test_ttgir_reduction_first_callees_keep_their_reduce(tmp_path):
    """GPT's 155 witness verbatim: two private functions whose first op is a tt.reduce. Each callee's
    top-level ops must be [tt.reduce, tt.return] with the 154 return metadata pointing at the
    yielded value (the addf for sum_helper, the first block argument for left_helper). Pre-166:
    CalledFunc.ops == [arith.addf] for both (CALLEE_MISSING_REDUCTION)."""
    g = _parse(_CALLEES, tmp_path)
    assert [o.op for o in g.ops][-3:] == ["arith.addf", "tt.store", "tt.return"]
    funcs = {f.name: f for f in g.called_funcs or []}
    assert set(funcs) == {"sum_helper", "left_helper"}, set(funcs)
    _check_reduce(funcs["sum_helper"].ops, True)
    _check_reduce(funcs["left_helper"].ops, False)


def test_ttgir_reduction_first_callees_prefixed_control(tmp_path):
    """Control (passes on both sides): a constant before each reduce makes the first visited op a
    function-body op, which is why 154's return metadata looked right in that variant."""
    text = _CALLEES.replace('    %r = "tt.reduce"', '    %marker = arith.constant 0.0 : f32\n    %r = "tt.reduce"')
    g = _parse(text, tmp_path)
    funcs = {f.name: f for f in g.called_funcs or []}
    for name, is_addf in (("sum_helper", True), ("left_helper", False)):
        top = [o.op for o in funcs[name].ops]
        assert top == ["arith.constant", "tt.reduce", "tt.return"], top
        _check_reduce(funcs[name].ops[1:], is_addf)


def test_ttgir_reduction_first_entry_function(tmp_path):
    """The ENTRY function with a region-owning first op: its two arguments and its [tt.reduce,
    tt.store, tt.return] body survive, no callee is invented. Pre-166: args read off the reduce
    body (two f32), ops == [arith.addf], the store gone."""
    g = _parse(_ENTRY_FIRST, tmp_path)
    # entry args are named positionally and carry the expanded layout: compare shape / kind
    assert [(a.is_ptr, a.elem_type, a.type_str.split(",")[0]) for a in g.args] == [
        (False, "f32", "tensor<32xf32"), (True, "f32", "!tt.ptr<f32>")], g.args
    _check_reduce(g.ops, True)
    assert g.called_funcs is None


_FOR_REDUCE_FIRST = _HEADER + '''  tt.func public @entry(%x: tensor<32xf32, #blocked>, %out: !tt.ptr<f32>, %lb: index, %ub: index, %st: index, %init: f32) {
    %acc = scf.for %i = %lb to %ub step %st iter_args(%a0 = %init) -> (f32) {
      %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %s = arith.addf %a, %b : f32
        tt.reduce.return %s : f32
      }) : (tensor<32xf32, #blocked>) -> f32
      %n = arith.addf %a0, %r : f32
      scf.yield %n : f32
    }
    tt.store %out, %acc : !tt.ptr<f32>
    tt.return
  }
}
'''

_IF_FIRST_CALLEE = _HEADER + '''  tt.func public @entry(%out: !tt.ptr<f32>, %c: i1, %v: f32) {
    %r = tt.call @pick(%c, %v) : (i1, f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
  tt.func private @pick(%c: i1, %v: f32) -> f32 {
    %r = scf.if %c -> (f32) {
      %d = arith.addf %v, %v : f32
      scf.yield %d : f32
    } else {
      scf.yield %v : f32
    }
    tt.return %r : f32
  }
}
'''


def _walk_all(ops):
    for op in ops:
        yield op
        yield from _walk_all(op.region_ops or [])
        yield from _walk_all(getattr(op, "else_ops", None) or [])


def test_ttgir_two_regions_deep_first_op_entry(tmp_path):
    """The first visited op sits TWO regions deep (an scf.for whose first body op is a tt.reduce):
    the entry function keeps its six arguments and [scf.for, tt.store, tt.return], with the reduce
    nested under the loop. Pre-166: two f32 arguments off the reduce body, ops == [arith.addf]."""
    g = _parse(_FOR_REDUCE_FIRST, tmp_path)
    assert [(a.is_ptr, a.elem_type) for a in g.args] == [
        (False, "f32"), (True, "f32"), (False, "index"), (False, "index"), (False, "index"), (False, "f32")], g.args
    assert [o.op for o in g.ops] == ["scf.for", "tt.store", "tt.return"]
    assert [o.op for o in _walk_all(g.ops)] == [
        "scf.for", "tt.reduce", "arith.addf", "arith.addf", "scf.yield", "tt.store", "tt.return"]


_WHILE_IF_CALLEE = _HEADER + '''  tt.func public @entry(%out: !tt.ptr<f32>, %x: f32, %lim: f32, %c: i1) {
    %r = tt.call @grow(%x, %lim, %c) : (f32, f32, i1) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
  tt.func private @grow(%x: f32, %lim: f32, %c: i1) -> f32 {
    %r = scf.while (%a = %x) : (f32) -> f32 {
      %cond = scf.if %c -> (i1) {
        %lt = arith.cmpf olt, %a, %lim : f32
        scf.yield %lt : i1
      } else {
        %f = arith.constant false
        scf.yield %f : i1
      }
      scf.condition(%cond) %a : f32
    } do {
    ^bb0(%b: f32):
      %n = arith.addf %b, %b : f32
      scf.yield %n : f32
    }
    tt.return %r : f32
  }
}
'''

_PRIVATE_FIRST = _HEADER + '''  tt.func private @twice(%v: f32) -> f32 {
    %d = arith.addf %v, %v : f32
    tt.return %d : f32
  }
  tt.func public @entry(%out: !tt.ptr<f32>, %v: f32) {
    %r = tt.call @twice(%v) : (f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
'''

_TWO_PUBLIC = _HEADER + '''  tt.func public @entry(%out: !tt.ptr<f32>, %v: f32) {
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
  tt.func public @other(%out: !tt.ptr<f32>, %v: f32) {
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}
'''


def test_ttgir_while_first_callee_with_nested_condition(tmp_path):
    """A callee whose first op is an scf.while whose before-region itself owns a region (the
    condition is an scf.if): the first visited op is THREE regions deep. The callee keeps its three
    arguments and [scf.while, tt.return]. Pre-166: args [f32], ops [arith.cmpf, scf.yield]."""
    g = _parse(_WHILE_IF_CALLEE, tmp_path)
    (f,) = g.called_funcs
    assert [a.type_str for a in f.args] == ["f32", "f32", "i1"], f.args
    assert [o.op for o in f.ops] == ["scf.while", "tt.return"]
    assert [o.op for o in _walk_all(f.ops)] == [
        "scf.while", "scf.if", "arith.cmpf", "scf.yield", "arith.constant", "scf.yield",
        "scf.condition", "arith.addf", "scf.yield", "tt.return"]


def test_ttgir_private_function_before_the_public_one(tmp_path):
    """Third site: which FUNCTION a body belongs to. Pre-166 the first function visited was taken as
    the public entry, so a private function preceding the public one in text order swapped the two
    bodies (entry ops == the callee's [arith.addf, tt.return] with one f32 argument; the callee got
    the entry's body). Now the body is routed by the visibility of the function being collected,
    cross-checked by name at each function op."""
    g = _parse(_PRIVATE_FIRST, tmp_path)
    assert g.func_name == "entry"
    assert [(a.is_ptr, a.elem_type) for a in g.args] == [(True, "f32"), (False, "f32")], g.args
    assert [o.op for o in g.ops] == ["tt.call", "tt.store", "tt.return"]
    (f,) = g.called_funcs
    assert f.name == "twice" and [a.type_str for a in f.args] == ["f32"]
    assert [o.op for o in f.ops] == ["arith.addf", "tt.return"]


_MULTI_BLOCK = _HEADER + '''  tt.func public @entry(%out: !tt.ptr<f32>, %v: f32) {
    %d = arith.addf %v, %v : f32
    cf.br ^bb1(%d : f32)
  ^bb1(%a: f32):
    tt.store %out, %a : !tt.ptr<f32>
    tt.return
  }
}
'''


def test_ttgir_multi_block_body_refuses(tmp_path):
    """A function body with more than one block (a cf.br chain — never emitted by Triton at the
    TTGIR level, reachable through IRSource): the walker models single-block bodies and pre-166
    silently dropped the second block's ops (`[arith.addf, cf.br]`, no store, no return). Now it
    refuses, naming the block count."""
    with pytest.raises(MetalNonRecoverableError, match="2 body blocks"):
        _parse(_MULTI_BLOCK, tmp_path)


_IF_THEN_REDUCE = _HEADER + '''  tt.func public @entry(%x: tensor<32xf32, #blocked>, %out: !tt.ptr<f32>, %c: i1) {
    %t = scf.if %c -> (tensor<32xf32, #blocked>) {
      %d = arith.addf %x, %x : tensor<32xf32, #blocked>
      scf.yield %d : tensor<32xf32, #blocked>
    } else {
      scf.yield %x : tensor<32xf32, #blocked>
    }
    %r = "tt.reduce"(%t) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<32xf32, #blocked>) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
'''


def test_ttgir_region_op_feeding_a_region_op_first(tmp_path):
    """Two region-owning ops in a row at the top of the entry (an scf.if whose result feeds a
    tt.reduce): the entry keeps its three arguments and [scf.if, tt.reduce, tt.store, tt.return].
    Pre-166: NO arguments (the then-block has none) and ops == [arith.addf, scf.yield]."""
    g = _parse(_IF_THEN_REDUCE, tmp_path)
    assert [(a.is_ptr, a.elem_type) for a in g.args] == [(False, "f32"), (True, "f32"), (False, "i1")], g.args
    assert [o.op for o in g.ops] == ["scf.if", "tt.reduce", "tt.store", "tt.return"]
    assert [o.op for o in _walk_all(g.ops)] == [
        "scf.if", "arith.addf", "scf.yield", "scf.yield", "tt.reduce", "arith.addf", "tt.store", "tt.return"]


def test_ttgir_two_public_functions_refuse(tmp_path):
    """Two public functions in one module: refused loudly (pre-166 the second was silently filed as
    a callee)."""
    with pytest.raises(MetalNonRecoverableError, match="more than one public function"):
        _parse(_TWO_PUBLIC, tmp_path)


def test_ttgir_if_first_callee(tmp_path):
    """A callee whose first op is an scf.if over a block-argument condition (no op precedes it):
    the callee keeps both arguments and [scf.if, tt.return] with both branches. Pre-166: the
    then-branch's [arith.addf, scf.yield] became the callee."""
    g = _parse(_IF_FIRST_CALLEE, tmp_path)
    assert [o.op for o in g.ops] == ["tt.call", "tt.store", "tt.return"]
    (f,) = g.called_funcs
    assert [(a.type_str) for a in f.args] == ["i1", "f32"], f.args
    assert [o.op for o in f.ops] == ["scf.if", "tt.return"]
    assert [o.op for o in _walk_all(f.ops)] == ["scf.if", "arith.addf", "scf.yield", "scf.yield", "tt.return"]
