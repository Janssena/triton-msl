"""Native constant proof must not become silent elision of a possible failure."""

import pytest
import triton
import triton.language as tl

from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import IRSource
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen._constant_assertions import discharge_scalar_true
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.codegen.msl_emitter import emit_msl


def _module(tmp_path, body, args="%out: !tt.ptr<i32>", extra=""):
    source = (
        """#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "metal:apple-m4", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @checked("""
        + args
        + """) {
"""
        + body
        + """
%v = arith.constant 17 : i32
tt.store %out, %v : !tt.ptr<i32>
tt.return
}
"""
        + extra
        + "\n}\n"
    )
    path = tmp_path / "input.ttgir"
    path.write_text(source)
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({})
    ctx = ir.context()
    parsed = IRSource(str(path), ctx, backend)
    assert parsed.module.verify()
    return ctx, parsed, options


@pytest.mark.parametrize(
    "body,args,removed,live",
    [
        ('%p = arith.constant true\ntt.assert %p, "true" : i1', "%out: !tt.ptr<i32>", 1, 0),
        ('%p = arith.constant false\ntt.assert %p, "false" : i1', "%out: !tt.ptr<i32>", 0, 1),
        ('tt.assert %p, "runtime" : i1', "%out: !tt.ptr<i32>, %p: i1", 0, 1),
        (
            '%x = arith.constant true\n%p = arith.xori %x, %x : i1\ntt.assert %p, "computed" : i1',
            "%out: !tt.ptr<i32>",
            0,
            1,
        ),
        (
            '%p = arith.constant dense<true> : tensor<8xi1, #b>\ntt.assert %p, "tensor" : tensor<8xi1, #b>',
            "%out: !tt.ptr<i32>",
            0,
            1,
        ),
        (
            '%p = arith.constant dense<[true, false, true, true, true, true, true, true]> : tensor<8xi1, #b>\ntt.assert %p, "mixed tensor" : tensor<8xi1, #b>',
            "%out: !tt.ptr<i32>",
            0,
            1,
        ),
        (
            '%t = arith.constant true\n%f = arith.constant false\ntt.assert %t, "true" : i1\ntt.assert %f, "false" : i1',
            "%out: !tt.ptr<i32>",
            1,
            1,
        ),
        (
            '%t = arith.constant true\nscf.if %condition {\ntt.assert %t, "nested" : i1\n}',
            "%out: !tt.ptr<i32>, %condition: i1",
            0,
            0,
        ),
    ],
)
def test_only_top_level_native_scalar_true_is_discharged(tmp_path, body, args, removed, live):
    ctx, src, options = _module(tmp_path, body, args)
    graph = walk_ttgir(src.module, options)
    before = str(src.module)
    original = tuple(graph.ops)
    candidate = discharge_scalar_true(graph, src.module)
    assert len(original) - len(candidate.ops) == removed
    assert str(src.module) == before
    assert tuple(graph.ops) == original
    # Nested assertions are deliberately left outside this optimization.
    if "nested" in body:
        assert candidate is graph
        assert any(x.op == "tt.assert" for op in candidate.ops for x in (op.region_ops or []))
        return
    if "mixed tensor" in body:
        # This source is already refused by the scalar/splat constant backstop.
        # Discharging a check must not make an unsupported tensor constant pass.
        from triton_msl.errors import MetalNonRecoverableError
        from triton_msl.codegen.generic_lowerer import GenericLowerer

        with pytest.raises(MetalNonRecoverableError, match="nonuniform or unparsed tensor constant") as reference:
            GenericLowerer(graph, options).lower()
        with pytest.raises(MetalNonRecoverableError) as result:
            emit_msl(src.module, {}, options)
        assert type(result.value) is type(reference.value)
        assert str(result.value) == str(reference.value)
        return
    metadata = {}
    msl = emit_msl(src.module, metadata, options)
    descriptor = metadata["device_assert"]
    assert (len(descriptor["messages"]) if descriptor else 0) == live
    assert ("_assert_status" in msl) == bool(live)
    if live:
        assert "threadgroup_barrier" in msl
        assert "return;" in msl
        assert descriptor["messages"][-1] != "true"
    if removed:
        from triton_msl.codegen.generic_lowerer import GenericLowerer

        reference = GenericLowerer(graph, options)
        assert "_assert_status" in reference.lower(), "reference must exercise the old status allocation"


def test_forged_text_graph_and_message_do_not_certify_true(tmp_path):
    ctx, src, options = _module(
        tmp_path, '%p = arith.constant false\n// arith.constant true\ntt.assert %p, "true" : i1'
    )
    graph = walk_ttgir(src.module, options)
    for op in graph.ops:
        if op.op == "arith.constant":
            op.attrs["value"] = -1
    assert discharge_scalar_true(graph, src.module) is graph


def test_callee_assertion_still_refuses(tmp_path):
    from triton_msl.errors import MetalNonRecoverableError

    extra = """tt.func private @callee(%x: i32) -> i32 attributes {noinline = true} {
%p = arith.constant true
tt.assert %p, "callee true" : i1
tt.return %x : i32
}"""
    ctx, src, options = _module(
        tmp_path, "%x = arith.constant 4 : i32\n%y = tt.call @callee(%x) : (i32) -> i32", extra=extra
    )
    graph = walk_ttgir(src.module, options)
    assert discharge_scalar_true(graph, src.module) is graph
    assert any(op.op == "tt.assert" for func in graph.called_funcs for op in func.ops)
    with pytest.raises(MetalNonRecoverableError, match="assert"):
        emit_msl(src.module, {}, options)


@triton.jit
def _constant_guard(OUT, PASS: tl.constexpr):
    tl.device_assert(PASS, "constant guard must hold")
    lane = tl.arange(0, 8)
    tl.store(OUT + lane, 17)


@pytest.mark.parametrize("shader", ["0", "1"])
def test_constant_true_executes_without_flag_false_still_stops(monkeypatch, shader):
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("Metal required")
    from triton_msl.errors import MetalDeviceAssertionError
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_compile_shader_runtime, _get_utils

    calls = []

    def wrap(_instance, real):
        def observed(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        return observed

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", shader)
    patch_live_singleton_method(
        monkeypatch,
        _get_compile_shader_runtime if shader == "1" else _get_utils,
        "dispatch" if shader == "1" else "launch",
        wrap,
    )
    out = torch.full((8,), -99, dtype=torch.int32, device="mps")
    compiled = _constant_guard[(1,)](out, PASS=True, debug=True)
    assert compiled.metadata.device_assert is None
    assert "_assert_status" not in compiled.asm["msl"]
    assert torch.equal(out.cpu(), torch.full((8,), 17, dtype=torch.int32))
    out.fill_(-99)
    with pytest.raises(MetalDeviceAssertionError, match="constant guard must hold"):
        _constant_guard[(1,)](out, PASS=False, debug=True)
    assert torch.equal(out.cpu(), torch.full((8,), -99, dtype=torch.int32)), "false assertion did not stop the store"
    assert len(calls) == 2, "both requested launches must be witnessed without replay"
