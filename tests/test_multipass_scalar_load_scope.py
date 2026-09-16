"""A scalar read must survive a multipass reduction without speculative motion."""

import re

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.codegen.msl_emitter import emit_msl
from tests.test_fa_tail_semantics import executed, _executed_source  # noqa: F401


@triton.jit(do_not_specialize=["MASK"])
def _scalar_scale(X, S, O, MASK, N: tl.constexpr):
    i = tl.arange(0, N)[None, :]
    x = tl.load(X + i)
    scale = tl.load(S, MASK, other=3).to(tl.float32)
    squared = x * x
    denom = tl.sum(squared, axis=1)
    result = x * tl.rsqrt(denom[:, None] / N + 1.0e-6) * scale
    tl.store(O + i, result)


def _module(fn, signature, constants):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 2})
    ctx = ir.context()
    ir.load_dialects(ctx)
    src = ASTSource(fn=fn, signature=signature, constexprs=constants)
    mod = src.make_ir(
        backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx
    )
    mod = backend.make_ttir(mod, {}, options)
    mod = backend.make_ttgir(mod, {}, options)
    mod.context = ctx
    return mod, options


def test_scalar_load_survives_real_reduction_loop(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    mod, options = _module(_scalar_scale, {"X": "*fp32", "S": "*i64", "O": "*fp32", "MASK": "i1"}, {"N": 2048})
    source = emit_msl(mod, {}, options)
    assert source.count("for (uint _loop_e") == 2, source
    read = re.search(r"long (\w+) = [^\n]*static_cast<long>\(S\[0\]\)[^\n]*;", source)
    assert read, source
    assert source.index(read.group(0)) < source.index("for (uint _loop_e"), source
    assert source.count("static_cast<long>(S[0])") == 1
    assert "MASK ?" in read.group(0), "entry-argument load mask was discarded"


def _raw_graph(tmp_path, *, prefix="", volatile=False, mask=False, tensor=False):
    # Native MLIR fixtures: the pointer and mask shapes are checked by the parser.
    layout = "#b = #ttg.blocked<{sizePerThread=[1], threadsPerWarp=[32], warpsPerCTA=[2], order=[0]}>"
    load_type = "tensor<128xi64, #b>" if tensor else "i64"
    pointer = "%ps" if tensor else "%s"
    setup = "%ps = tt.splat %s : !tt.ptr<i64> -> tensor<128x!tt.ptr<i64>, #b>" if tensor else ""
    load_pointer_type = "tensor<128x!tt.ptr<i64>, #b>" if tensor else "!tt.ptr<i64>"
    load_mask = ", %m, %other" if mask else ""
    attrs = " {isVolatile = true}" if volatile else ""
    text = f"""{layout}
module attributes {{"ttg.num-warps"=2:i32, "ttg.threads-per-warp"=32:i32}} {{
tt.func public @scope(%s: !tt.ptr<i64>, %o: !tt.ptr<i64>, %m: i1) {{
%other = arith.constant 7 : i64
{prefix}
{setup}
%v = tt.load {pointer}{load_mask}{attrs} : {load_pointer_type}
tt.return
}}
}}"""
    path = tmp_path / "scope.ttgir"
    path.write_text(text)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = ir.parse_mlir_module(str(path), ctx)
    options = MetalBackend(GPUTarget("metal", "apple-m4", 32)).parse_options({"num_warps": 2})
    graph = walk_ttgir(mod, options)
    return GenericLowerer(graph, options), graph


@pytest.mark.parametrize("mask", [False, True])
def test_readonly_scalar_dependencies_admitted(tmp_path, mask):
    lowerer, graph = _raw_graph(tmp_path, mask=mask)
    (load,) = [op for op in graph.ops if op.op == "tt.load"]
    assert load.attrs["isVolatile"] is False
    assert lowerer._multipass_scalar_load_ids() == {load.id}


def test_entry_mask_uses_native_argument_type_not_defining_op(tmp_path):
    lowerer, graph = _raw_graph(tmp_path, mask=True)
    pointer, output, mask = graph.args
    assert lowerer._is_mask(mask.id)
    assert not lowerer._is_mask(pointer.id)
    assert not lowerer._is_mask(output.id)


@pytest.mark.parametrize("case", ["volatile", "store", "atomic", "assert", "tensor", "missing_volatile_proof"])
def test_scalar_hoist_does_not_cross_effect_or_invent_scalar_shape(tmp_path, case):
    prefix = {
        "store": "tt.store %o, %other : !tt.ptr<i64>",
        "assert": 'tt.assert %m, "keep guarded load behind assertion" : i1',
        "atomic": "%a = tt.atomic_rmw add, acq_rel, gpu, %o, %other, %m : (!tt.ptr<i64>, i64, i1) -> i64",
    }.get(case, "")
    lowerer, graph = _raw_graph(tmp_path, prefix=prefix, volatile=case == "volatile", tensor=case == "tensor")
    if case == "missing_volatile_proof":
        for op in graph.ops:
            if op.op == "tt.load":
                del op.attrs["isVolatile"]
    assert lowerer._multipass_scalar_load_ids() == set()


@pytest.mark.parametrize("mask", [False, True])
def test_scalar_mask_and_value_compute(mask, monkeypatch, tmp_path, executed):
    import torch

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    x = torch.linspace(-2.0, 2.0, 2048, device="mps")
    scale = torch.tensor([17], dtype=torch.int64, device="mps")
    output = torch.full((2052,), 123.0, device="mps")
    kernel = _scalar_scale[(1,)](x, scale, output, N=2048, MASK=mask, num_warps=2)
    torch.mps.synchronize()
    source = _executed_source(executed, kernel)
    assert source == kernel.asm["msl"]
    assert "for (uint _loop_e" in source
    expected = x.cpu().double()
    expected = expected * torch.rsqrt(expected.square().mean() + 1.0e-6) * (17 if mask else 3)
    actual = output.cpu()
    torch.testing.assert_close(actual[:2048].double(), expected, atol=1.0e-5, rtol=1.0e-5)
    assert torch.equal(actual[2048:], torch.full((4,), 123.0))
