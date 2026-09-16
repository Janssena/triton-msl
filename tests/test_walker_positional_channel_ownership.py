"""Text-assisted walker channels bind only real operations, never location text."""

from pathlib import Path

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import IRSource

from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import MLIRWalker, _ModuleTextIndex, _mlir_symbol_identity, walk_ttgir
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _predicates(X, Z, N: tl.constexpr):
    x = tl.arange(0, N)
    value = tl.load(X + x)
    chosen = tl.where(x < 16, value, -value)
    tl.store(Z + x, tl.where(chosen > 0.0, chosen, 0.0))


def _ttgir():
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    module = ASTSource(
        _predicates,
        {"X": "*fp32", "Z": "*fp32", "N": "constexpr"},
        {"N": 32},
    ).make_ir(backend.target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context)
    module = backend.make_ttir(module, {}, options)
    return str(backend.make_ttgir(module, {}, options))


def _parse(text, path):
    path.write_text(text)
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    parsed = IRSource(str(path), context, backend)
    assert parsed.module.verify()
    graph = walk_ttgir(parsed.module, options)
    return graph, GenericLowerer(graph, options).lower()


def test_location_payload_cannot_relabel_native_predicates(tmp_path):
    canonical = _ttgir()
    cmp_line = next(line for line in canonical.splitlines() if "arith.cmpi slt" in line)
    poisoned = (
        '#pred = loc("%fake = arith.cmpi ugt, %x, %y : i32")\n'
        + canonical.replace(cmp_line, cmp_line.rsplit(" loc(", 1)[0] + " loc(#pred)", 1)
    )
    graph, msl = _parse(poisoned, tmp_path / "poison.ttgir")
    comparisons = [op for op in graph.ops if op.op in ("arith.cmpi", "arith.cmpf")]
    assert [(op.attrs["predicate"], op.attrs["predicate_name"]) for op in comparisons] == [
        (2, "slt"),
        (2, "ogt"),
    ]
    assert "(int)lid < (int)16" in msl
    assert "(uint)lid > (uint)16" not in msl


def test_comments_locations_and_private_functions_do_not_create_channel_records():
    text = r'''
#loc = loc("%x = arith.constant 99.0 : f32; %p = arith.cmpi uge, %x, %y : i32; tt.call @bad()")
module {
  // %x = arith.constant 88.0 : f32
  // %p = arith.cmpf uno, %x, %y : f32
  tt.func private @"quoted-helper"() {
    tt.return
  }
  tt.func public @entry() {
    %c = arith.constant 1.25 : f32 loc(#loc)
    %p = arith.cmpf olt, %c, %c : f32
    tt.call @"quoted-helper"() : () -> ()
    tt.return
  }
}
'''
    index = _ModuleTextIndex(text)
    assert index.constants_by_position == [1.25]
    assert index.predicates == {"p": "olt"}
    assert list(index.call_targets.values()) == ["quoted-helper"]


def test_channel_count_mismatch_refuses_after_native_walk(tmp_path):
    text = _ttgir()
    path = tmp_path / "count.ttgir"
    path.write_text(text)
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    context = ir.context()
    ir.load_dialects(context)
    parsed = IRSource(str(path), context, backend)
    walker = MLIRWalker(parsed.module, backend.parse_options({"num_warps": 4}))
    walker._predicates_in_order.insert(0, "ugt")
    with pytest.raises(MetalNonRecoverableError, match="comparison counts disagree|predicate disagrees"):
        walker.walk()


def test_strict_indexes_exclude_fake_extern_and_cond_br_payloads():
    text = r'''
#loc = loc("tt.extern_elementwise %x {symbol = \"bad\", libname = \"bad\"}; cf.cond_br %p, ^bb1, ^bb2")
module {
  tt.func public @entry() {
    // tt.extern_elementwise %x {symbol = "comment", libname = "comment"}
    tt.return loc(#loc)
  }
}
'''
    index = _ModuleTextIndex(text)
    assert index.extern_elementwise_ops == []
    assert index.cond_br_ops == []


def test_full_walk_and_lowering_bind_quoted_escaped_and_plain_callees(tmp_path):
    assert _mlir_symbol_identity(r'"caf\C3\A9"') == "café"
    text = r'''
module attributes {"ttg.num-warps"=4:i32, "ttg.threads-per-warp"=32:i32} {
  tt.func private @"quoted\22helper"(%x: i32) -> i32 {
    %one = arith.constant 1 : i32
    %r = arith.addi %x, %one : i32
    tt.return %r : i32
  }
  tt.func private @plain_helper(%x: i32) -> i32 {
    %one = arith.constant 1 : i32
    %r = arith.addi %x, %one : i32
    tt.return %r : i32
  }
  tt.func public @entry(%out: !tt.ptr<i32>) {
    %zero = arith.constant 0 : i32
    %a = tt.call @"quoted\22helper"(%zero) : (i32) -> i32
    %b = tt.call @plain_helper(%a) : (i32) -> i32
    tt.store %out, %b : !tt.ptr<i32>
    tt.return
  }
}
'''
    graph, msl = _parse(text, tmp_path / "quoted_callees.ttgir")
    assert [callee.name for callee in graph.called_funcs] == ['quoted"helper', "plain_helper"]
    assert [op.attrs["callee"] for op in graph.ops if op.op == "tt.call"] == [
        'quoted"helper',
        "plain_helper",
    ]
    assert "unknown_fn" not in msl
    assert "plain_helper" in msl


def test_full_walk_and_lowering_bind_quoted_escaped_public_entry_arguments(tmp_path):
    text = r'''
#loc = loc("quoted_entry.py":1:1)
module attributes {"ttg.num-warps"=4:i32, "ttg.threads-per-warp"=32:i32} {
  tt.func public @"entry\5Fpoint"(
      %Input.0: !tt.ptr<i32> loc("Input.0"(#loc)),
      %Value: i32 loc("Value"(#loc))) {
    tt.store %Input.0, %Value : !tt.ptr<i32>
    tt.return
  }
}
'''
    graph, msl = _parse(text, tmp_path / "quoted_entry.ttgir")
    assert graph.func_name == "entry_point"
    assert [arg.name for arg in graph.args] == ["Input_0", "Value"]
    assert "device int* Input_0" in msl
    assert "constant int& Value" in msl
