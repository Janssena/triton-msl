"""Loop pointers carry addresses, not loaded values or offsets with a frozen base.

Each row checks freshly lowered native IR and a GPU result against exact CPU
indexing. The small spelling table covers distinct representations/phi edges,
not a Cartesian shape sweep; a refusal is not success for these supported rows.
"""

import re

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _pointer_flow(A, B, O, R, BLOCK: tl.constexpr, SCALAR: tl.constexpr, MODE: tl.constexpr):
    x = tl.arange(0, BLOCK)
    p = A if SCALAR else A + x
    q = B + BLOCK if SCALAR else B + BLOCK + x
    acc = tl.zeros((BLOCK,), tl.float32)
    for r in range(R):
        acc += tl.load(p + x) if SCALAR else tl.load(p)
        if MODE == 0:
            p += BLOCK
        elif MODE == 1:
            p = B if SCALAR else B + x
        else:
            p, q = q, p
    tl.store(O + x, acc)
    tl.store(O + BLOCK + x, tl.load(p + x) if SCALAR else tl.load(p))


@triton.jit
def _pointer_only(A, O, R, BLOCK: tl.constexpr, SCALAR: tl.constexpr):
    x = tl.arange(0, BLOCK)
    p = A if SCALAR else A + x
    for r in range(R):
        p += BLOCK
    tl.store(O + x, tl.load(p + x) if SCALAR else tl.load(p))


@triton.jit
def _signed_offset_pointer(A, O, start):
    pointer = A + start
    total = 0.0
    for _ in range(2):
        total += tl.load(pointer)
        pointer += 1
    tl.store(O, total)


@triton.jit
def _signed_offset_pointer_tensor(A, O, start, BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK)
    pointer = A + start + x
    total = tl.zeros((BLOCK,), tl.float32)
    for _ in range(2):
        total += tl.load(pointer)
        pointer += 1
    tl.store(O + x, total)


@triton.jit
def _signed_offset_pointer_mept(A, O, start, BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK)
    pointer = A + x
    for _ in range(1):
        pointer += start
    tl.store(O + x, tl.load(pointer))


@triton.jit
def _signed_offset_pointer_while(A, O, start, R, BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK)
    pointer = A + x
    i = 0
    while i < R:
        pointer += start
        i += 1
    tl.store(O + x, tl.load(pointer))


@triton.jit
def _while_pointer(A, B, O, R, BLOCK: tl.constexpr, SCALAR: tl.constexpr):
    x = tl.arange(0, BLOCK)
    p = A if SCALAR else A + x
    q = B + BLOCK + x
    acc = tl.zeros((BLOCK,), tl.float32)
    i = 0
    while i < R:
        acc += tl.load(p + x) if SCALAR else tl.load(p)
        if SCALAR:
            p = B + BLOCK
        else:
            p, q = q, p
        i += 1
    tl.store(O + x, acc)
    tl.store(O + BLOCK + x, tl.load(p + x) if SCALAR else tl.load(p))


@triton.jit(noinline=True)
def _callee_pointer_sum(p, q, count):
    total = 0.0
    for i in range(count):
        total += tl.load(p)
        p, q = q, p + 1
    return total


@triton.jit
def _pointer_caller(A, B, O, count):
    lane = tl.program_id(0)
    start = lane * count
    tl.store(O + lane, _callee_pointer_sum(A + start, B + start + 1, count))


@triton.jit
def _conditional_pointer(A, B, O, flag, BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK)
    p = A + x
    if flag != 0:
        p = B + x
    tl.store(O + x, tl.load(p))


@triton.jit(noinline=True)
def _callee_conditional_load(p, q, flag):
    if flag != 0:
        p = q
    return tl.load(p)


@triton.jit(noinline=True)
def _callee_conditional_effect(p, q, flag):
    if flag != 0:
        # The observable store prevents canonicalization to arith.select while
        # preserving the input value, forcing the device-function scf.if path.
        tl.store(q, tl.load(q))
        p = q
    return tl.load(p)


@triton.jit
def _conditional_caller(A, B, O, SCF: tl.constexpr):
    lane = tl.program_id(0)
    if SCF:
        value = _callee_conditional_effect(A + lane, B + lane, lane % 2)
    else:
        value = _callee_conditional_load(A + lane, B + lane, lane % 2)
    tl.store(O + lane, value)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
@pytest.mark.parametrize(
    "scalar,mode,block,dtype",
    [
        (False, 1, 32, torch.float32),  # P0: base A -> B, not just an offset update
        (True, 0, 32, torch.float16),  # bare scalar argument must not become float
        (True, 1, 32, torch.int32),  # bare argument also appears on the yield side
        (False, 2, 32, torch.float32),  # simultaneous pointer exchange
        (False, 2, 256, torch.float32),  # per-thread offset arrays + simultaneous exchange
        (False, 3, 32, torch.float32),  # single-result scf.for exit mapping
        (True, 3, 32, torch.bfloat16),
    ],
)
def test_loop_pointer_preserves_address(scalar, mode, block, dtype, monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    fn = _pointer_only if mode == 3 else _pointer_flow
    fn.device_caches.clear()
    dtype_name = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16", torch.int32: "i32"}[dtype]
    # Small exact values keep the reference independent of rounding/reassociation.
    indices = torch.arange(3 * block)
    a = (indices % 32 + 1 + (indices // block) * 16).to(dtype)
    b = a + 64
    actual = torch.full((block if mode == 3 else 2 * block,), -99, dtype=dtype, device="mps")
    sig = dict(A="*" + dtype_name, O="*" + dtype_name, R="i32")
    cex = dict(BLOCK=block, SCALAR=scalar)
    if mode != 3:
        sig = dict(A="*" + dtype_name, B="*" + dtype_name, O="*" + dtype_name, R="i32")
        cex["MODE"] = mode
    # Match the real launch's aligned pointer specialization. The shared
    # no-attrs helper emits a scalar/wrap representation even at BLOCK=256,
    # which is not evidence about the register-array carry.
    from triton._C.libtriton import ir
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    attrs = {(i,): [("tt.divisibility", 16)] for i, typ in enumerate(sig.values()) if typ.startswith("*")}
    src = ASTSource(fn=fn, signature=sig, constexprs=cex, attrs=attrs)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx)
    md = {}
    mod = backend.make_ttgir(backend.make_ttir(mod, md, options), md, options)
    lowerer = GenericLowerer(walk_ttgir(mod, options), options)
    text = lowerer.lower()  # lowering-boundary contract, cannot reuse an executable
    assert "kernel void" in text and "UNKNOWN_" not in text
    if block == 256:
        assert lowerer.env_ptr_array, "must exercise the per-thread pointer-array representation"
    from triton_msl.backend.driver import _get_compile_shader_runtime

    runtime = _get_compile_shader_runtime()
    calls = []
    real = runtime.dispatch

    def dispatch(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(runtime, "dispatch", dispatch)
    if mode == 3:
        compiled = fn[(1,)](a.to("mps"), actual, 2, **cex)
        expected = a[2 * block : 3 * block]
    else:
        compiled = fn[(1,)](a.to("mps"), b.to("mps"), actual, 2, **cex)
        last = b[:block] if mode == 1 else a[2 * block : 3 * block] if mode == 0 else a[:block]
        second = a[block : 2 * block] if mode == 0 else b[block : 2 * block] if mode == 2 else b[:block]
        expected = torch.cat(((a[:block].float() + second.float()).to(dtype), last))
    torch.mps.synchronize()
    assert calls == [1], "must observe the actual dispatched kernel"
    if block == 256:
        assert re.search(r"long off_\d+\[", compiled.asm["msl"]), "executed kernel must carry signed offset arrays"
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
def test_loop_pointer_preserves_negative_runtime_offset(monkeypatch, tmp_path):
    """An i32 -2 is an in-allocation address here, never uint(2**32-2)."""
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    _signed_offset_pointer.device_caches.clear()
    storage = torch.tensor([1.0, 2.0, 4.0, 8.0], device="mps")
    output = torch.full((1,), -99.0, device="mps")
    from triton_msl.backend.driver import _get_compile_shader_runtime

    runtime = _get_compile_shader_runtime()
    calls, real = [], runtime.dispatch

    def dispatch(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(runtime, "dispatch", dispatch)
    compiled = _signed_offset_pointer[(1,)](storage[2:], output, -2)
    torch.mps.synchronize()
    assert calls == [1]
    assert "uint off_" not in compiled.asm["msl"]
    assert output.cpu().item() == 3.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
def test_loop_pointer_tensor_preserves_negative_runtime_offset(monkeypatch, tmp_path):
    """Per-lane pointer math preserves a signed start when combined with uint lid."""
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    _signed_offset_pointer_tensor.device_caches.clear()
    block = 256
    storage = torch.arange(block + 3, dtype=torch.float32)
    output = torch.full((block,), -99.0, device="mps")
    compiled = _signed_offset_pointer_tensor[(1,)](storage.to("mps")[2:], output, -2, BLOCK=block)
    torch.mps.synchronize()
    assert "uint off_" not in compiled.asm["msl"]
    assert re.search(r"A \+ \(int\(start\) \+ int\(lid\)\)", compiled.asm["msl"])
    assert torch.equal(output.cpu(), storage[:block] + storage[1 : block + 1])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
def test_loop_pointer_mept_preserves_negative_runtime_offset(monkeypatch, tmp_path):
    """A signed update stays signed in the actual multi-element pointer array."""
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    _signed_offset_pointer_mept.device_caches.clear()
    block = 256
    storage = torch.arange(block + 2, dtype=torch.float32)
    output = torch.full((block,), -99.0, device="mps")
    from triton_msl.backend.driver import _get_compile_shader_runtime

    runtime = _get_compile_shader_runtime()
    calls, real = [], runtime.dispatch

    def dispatch(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(runtime, "dispatch", dispatch)
    compiled = _signed_offset_pointer_mept[(1,)](storage.to("mps")[2:], output, -2, BLOCK=block)
    torch.mps.synchronize()
    assert calls == [1]
    assert re.search(r"long off_\d+\[2\]", compiled.asm["msl"])
    assert torch.equal(output.cpu(), storage[:block])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
def test_while_pointer_preserves_negative_runtime_offset(monkeypatch, tmp_path):
    """The actual wrapped scf.while route carries a signed device address."""
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    _signed_offset_pointer_while.device_caches.clear()
    block = 32
    storage = torch.arange(block + 2, dtype=torch.float32)
    output = torch.full((block,), -99.0, device="mps")
    compiled = _signed_offset_pointer_while[(1,)](storage.to("mps")[2:], output, -1, 2, BLOCK=block)
    torch.mps.synchronize()
    assert "for (;;)" in compiled.asm["msl"]
    assert re.search(r"auto wh_ptr_\d+ = .*A", compiled.asm["msl"])
    assert "uint wh_off_" not in compiled.asm["msl"]
    assert torch.equal(output.cpu(), storage[:block])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
@pytest.mark.parametrize("scalar", [False, True], ids=["tensor", "bare-scalar"])
def test_while_pointer_preserves_address(scalar, monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    _while_pointer.device_caches.clear()
    block = 32
    indices = torch.arange(3 * block)
    a = (indices % 32 + 1 + (indices // block) * 16).float()
    b = a + 64
    out = torch.full((2 * block,), -99.0, device="mps")
    from triton_msl.backend.driver import _get_compile_shader_runtime

    runtime = _get_compile_shader_runtime()
    calls, real = [], runtime.dispatch

    def dispatch(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(runtime, "dispatch", dispatch)
    compiled = _while_pointer[(1,)](a.to("mps"), b.to("mps"), out, 2, BLOCK=block, SCALAR=scalar)
    torch.mps.synchronize()
    final = b[block : 2 * block] if scalar else a[:block]
    expected = torch.cat((a[:block] + b[block : 2 * block], final))
    assert calls == [1]
    assert "UNKNOWN_" not in compiled.asm["msl"]
    assert torch.equal(out.cpu(), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
def test_noinline_callee_loop_preserves_pointer_arguments(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    _pointer_caller.device_caches.clear()
    a = torch.arange(8, dtype=torch.float32) + 1
    b = a + 32
    out = torch.full((4,), -99.0, device="mps")
    compiled = _pointer_caller[(4,)](a.to("mps"), b.to("mps"), out, 2)
    torch.mps.synchronize()
    assert "UNKNOWN_" not in compiled.asm["msl"]
    assert torch.equal(out.cpu(), a[::2] + b[1::2])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")
@pytest.mark.parametrize("callee", [0, 1, 2], ids=["kernel-if", "noinline-select", "noinline-scf-if"])
def test_conditional_pointer_preserves_address(callee, monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    a = torch.arange(32, dtype=torch.float32) + 1
    b = a + 64
    if callee:
        _conditional_caller.device_caches.clear()
        out = torch.full((32,), -99.0, device="mps")
        compiled = _conditional_caller[(32,)](a.to("mps"), b.to("mps"), out, SCF=callee == 2)
        expected = torch.where(torch.arange(32) % 2 == 0, a, b)
    else:
        _conditional_pointer.device_caches.clear()
        out = torch.full((32,), -99.0, device="mps")
        compiled = _conditional_pointer[(1,)](a.to("mps"), b.to("mps"), out, 2, BLOCK=32)
        expected = b
    torch.mps.synchronize()
    assert "UNKNOWN_" not in compiled.asm["msl"]
    if callee == 2:
        assert "if (" in compiled.asm["msl"]
    assert torch.equal(out.cpu(), expected)


def test_noinline_mixed_if_uses_each_native_result_type():
    """Synthetic boundary: later pointer results must not inherit result zero's scalar type."""
    from triton_msl.codegen._device_func_lowerer import _DeviceFuncLowerer
    from triton_msl.codegen.mlir_walker import CalledFunc, FuncArg, SSAValue
    from triton_msl.codegen.result_metadata import ResultMeta, parse_type_facts

    args = [
        FuncArg(1, "p", "!tt.ptr<f32>", "f32", True, 0),
        FuncArg(2, "q", "!tt.ptr<f32>", "f32", True, 1),
        FuncArg(3, "cond", "i1", "i1", False, 2),
        FuncArg(4, "value", "f32", "f32", False, 3),
    ]
    then = SSAValue(20, "", "scf.yield", [4, 1], {}, "", "", False)
    other = SSAValue(21, "", "scf.yield", [4, 2], {}, "", "", False)
    branch = SSAValue(
        10, "", "scf.if", [3], {}, "f32", "f32", False, region_ops=[then], else_ops=[other], result_ids=[10, 11]
    )
    metadata = {
        10: ResultMeta(10, parse_type_facts("f32"), "result", 10, 0),
        11: ResultMeta(11, parse_type_facts("!tt.ptr<f32>"), "result", 10, 1),
    }
    callee = CalledFunc("mixed", args, [branch], ["f32", "!tt.ptr<f32>"], metadata)
    lines = "\n".join(_DeviceFuncLowerer(callee).lower_body())
    assert "float if_res_" in lines
    assert "volatile device float* if_res_" in lines
    assert "= p;" in lines and "= q;" in lines
