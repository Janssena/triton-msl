"""182/191 stage 1a: facts belong to EACH result, not the first result's op."""
from dataclasses import FrozenInstanceError

import pytest
import triton
import triton.language as tl

from test_fa_bwd_routing import _build_lowerer
from test_scan_multivalue_dtypes import _mixed_scan_kernel
from test_walker_entry_block import _HEADER, _parse


@pytest.mark.parametrize("raw,kind,elem,width,signed,shape", [
    ("f32", "float", "f32", 32, None, ()),
    ("bf16", "float", "bf16", 16, None, ()),
    ("f16", "float", "f16", 16, None, ()),
    ("f64", "float", "f64", 64, None, ()),
    ("f8E4M3FN", "float", "f8E4M3FN", 8, None, ()),
    ("i1", "integer", "i1", 1, None, ()),
    ("i64", "integer", "i64", 64, None, ()),
    ("si32", "integer", "si32", 32, True, ()),
    ("ui8", "integer", "ui8", 8, False, ()),
    ("index", "index", "index", None, None, ()),
    ("tensor<32x1xbf16, #ttg.slice<{dim = 1, parent = #blocked}>>", "float", "bf16", 16, None, (32, 1)),
    ("tensor<32xi64>", "integer", "i64", 64, None, (32,)),
    ("tensor<?x32xf32>", "float", "f32", 32, None, (None, 32)),
    ("tensor<f32>", "float", "f32", 32, None, ()),
])
def test_type_facts_have_no_implicit_f32_or_signed_default(raw, kind, elem, width, signed, shape):
    from triton_msl.codegen.result_metadata import parse_type_facts
    facts = parse_type_facts(raw, {"#blocked": "#ttg.blocked<{sizePerThread = [1], order = [0]}>"})
    assert (facts.kind, facts.elem, facts.width, facts.signed, facts.shape) == (kind, elem, width, signed, shape)
    assert facts.raw == raw and facts.unknown_reason is None
    assert facts.is_tensor == raw.startswith("tensor<")


@pytest.mark.parametrize("raw,shape,address_space", [
    ("!tt.ptr<f32>", (), 1),
    ("!tt.ptr<bf16, 3>", (), 3),
    ("tensor<32x!tt.ptr<i64>, #blocked>", (32,), 1),
    ("tensor<4x8x!tt.ptr<f16, 3>, #blocked>", (4, 8), 3),
])
def test_pointer_bits_are_not_pointee_bits(raw, shape, address_space):
    from triton_msl.codegen.result_metadata import parse_type_facts
    facts = parse_type_facts(raw, {"#blocked": "#ttg.blocked<{sizePerThread = [1], order = [0]}>"})
    assert facts.kind == "pointer" and facts.width is None and facts.signed is None
    assert facts.shape == shape and facts.address_space == address_space
    assert facts.pointee.kind in ("float", "integer")
    assert facts.pointee.width in (16, 32, 64)
    assert facts.unknown_reason is None


@pytest.mark.parametrize("raw", ["", "garbage", "f31", "tensor<watxf32>", "tensor<32xf32", "!tt.ptr<garbage>", "vector<4xf32>", "!ttg.memdesc<32xf32, #blocked, #smem>"])
def test_unknown_types_are_unknown_not_scalar_f32(raw):
    from triton_msl.codegen.result_metadata import parse_type_facts
    facts = parse_type_facts(raw)
    assert facts.unknown_reason and facts.raw == raw
    assert facts.kind == "unknown" and facts.shape is None
    assert facts.width is None and facts.elem is None


def test_layout_aliases_are_resolved_without_erasing_raw_type():
    from triton_msl.codegen.result_metadata import parse_type_facts
    raw = "tensor<32xf32, #slice>"
    aliases = {"#slice": "#ttg.slice<{dim = 0, parent = #blocked}>",
               "#blocked": "#ttg.blocked<{sizePerThread = [1], order = [0]}>"}
    facts = parse_type_facts(raw, aliases)
    assert facts.raw == raw and facts.layout == "#ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1], order = [0]}>}>"
    assert facts.unknown_reason is None
    with pytest.raises(FrozenInstanceError):
        facts.width = 16


@pytest.mark.parametrize("aliases", [{}, {"#slice": "#slice"}, {"#slice": "#other", "#other": "#slice"}])
def test_unresolved_or_cyclic_layout_is_not_a_proved_layout(aliases):
    from triton_msl.codegen.result_metadata import parse_type_facts
    facts = parse_type_facts("tensor<32xf32, #slice>", aliases)
    assert facts.layout is None and facts.unknown_reason
    assert facts.width == 32 and facts.shape == (32,)  # known independent facts survive


@triton.jit
def _mixed_reduce(X, O, I):
    x = tl.load(X + tl.arange(0, 32))
    value, index = tl.max(x, 0, return_indices=True)
    tl.store(O, value)
    tl.store(I, index)


def _all_ops(ops):
    for op in ops:
        yield op
        yield from _all_ops(op.region_ops or [])
        yield from _all_ops(op.else_ops or [])


def _assert_results(graph, op_name, expected):
    op, = [o for o in _all_ops(graph.ops) if o.op == op_name]
    metas = [graph.result_meta[rid] for rid in op.result_ids]
    assert [m.type.elem for m in metas] == expected
    assert [m.result_index for m in metas] == list(range(len(expected)))
    assert all(m.producer_id == op.id and m.kind == "result" for m in metas)
    assert [m.value_id for m in metas] == op.result_ids
    assert len({id(m) for m in metas}) == len(expected)
    return op


def test_mixed_reduce_results_and_block_args_keep_their_own_types():
    graph = _build_lowerer(_mixed_reduce, {"X": "*fp32", "O": "*fp32", "I": "*i32"}, {}).graph
    op = _assert_results(graph, "tt.reduce", ["f32", "i32"])
    args = [graph.result_meta[rid] for rid in op.attrs["block_arg_ids"]]
    assert [m.type.elem for m in args] == ["f32", "i32", "f32", "i32"]
    assert all(m.kind == "block_arg" and m.producer_id is None for m in args)


def test_mixed_scan_results_keep_their_own_types():
    graph = _build_lowerer(_mixed_scan_kernel, {"x_ptr": "*fp32", "c_ptr": "*i32", "o_val": "*fp32", "o_cnt": "*i32"}, {"N": 32}).graph
    _assert_results(graph, "tt.scan", ["f32", "i32"])
    assert all(graph.result_meta[a.id].kind == "entry_arg" for a in graph.args)


MIXED_CONTROL = _HEADER + '''
  tt.func public @entry(%c: i1, %x: f32, %n: i64, %o: !tt.ptr<f32>, %oi: !tt.ptr<i64>) {
    %r:2 = tt.call @helper(%c, %x, %n) : (i1, f32, i64) -> (f32, i64)
    tt.store %o, %r#0 : !tt.ptr<f32>
    tt.store %oi, %r#1 : !tt.ptr<i64>
    tt.return
  }
  tt.func private @helper(%c: i1, %x: f32, %n: i64) -> (f32, i64) {
    %r:2 = scf.if %c -> (f32, i64) {
      scf.yield %x, %n : f32, i64
    } else {
      scf.yield %x, %n : f32, i64
    }
    tt.return %r#0, %r#1 : f32, i64
  }
}
'''


def test_mixed_callee_call_and_control_flow_results(tmp_path):
    graph = _parse(MIXED_CONTROL, tmp_path)
    _assert_results(graph, "tt.call", ["f32", "i64"])
    callee, = graph.called_funcs
    branch, = [o for o in callee.ops if o.op == "scf.if"]
    assert [graph.result_meta[rid].type.elem for rid in branch.result_ids] == ["f32", "i64"]
    assert [graph.result_meta[rid].result_index for rid in branch.result_ids] == [0, 1]
    assert all(graph.result_meta[a.id].kind == "callee_arg" for a in callee.args)
    assert [graph.result_meta[a.id].type.elem for a in callee.args] == ["i1", "f32", "i64"]


@pytest.mark.parametrize("kind,body", [
    ("scf.for", '''
      %r:2 = scf.for %i = %lb to %ub step %st iter_args(%a = %x, %b = %n) -> (f32, i64) {
        scf.yield %a, %b : f32, i64
      }
    '''),
    ("scf.while", '''
      %r:2 = scf.while (%a = %x, %b = %n) : (f32, i64) -> (f32, i64) {
        scf.condition(%c) %a, %b : f32, i64
      } do {
      ^bb0(%a: f32, %b: i64):
        scf.yield %a, %b : f32, i64
      }
    '''),
])
def test_mixed_loop_results_and_every_native_block_arg(tmp_path, kind, body):
    # Type/IR-only: the while body is not a GPU liveness test.
    text = _HEADER + '''
      tt.func public @entry(%lb: index, %ub: index, %st: index, %c: i1, %x: f32, %n: i64, %o: !tt.ptr<f32>, %oi: !tt.ptr<i64>) {
    ''' + body + '''
        tt.store %o, %r#0 : !tt.ptr<f32>
        tt.store %oi, %r#1 : !tt.ptr<i64>
        tt.return
      }
    }
    '''
    graph = _parse(text, tmp_path)
    _assert_results(graph, kind, ["f32", "i64"])
    args = [m for m in graph.result_meta.values() if m.kind == "block_arg"]
    actual = sorted((m.type.elem for m in args))
    assert actual == (sorted(["index", "f32", "i64"]) if kind == "scf.for" else sorted(["f32", "i64"] * 2))
    assert all(m.producer_id is None and m.owner_id is not None for m in args)


def test_signless_bits_remain_signless_for_signed_and_unsigned_consumers(tmp_path):
    text = _HEADER + '''
      tt.func public @entry(%x: i32, %y: i32, %os: !tt.ptr<i32>, %ou: !tt.ptr<i32>) {
        %s = arith.divsi %x, %y : i32
        %u = arith.divui %x, %y : i32
        tt.store %os, %s : !tt.ptr<i32>
        tt.store %ou, %u : !tt.ptr<i32>
        tt.return
      }
    }
    '''
    graph = _parse(text, tmp_path)
    assert all(m.type.signed is None for m in graph.result_meta.values())
    assert {o.op for o in graph.ops} >= {"arith.divsi", "arith.divui"}


def test_metadata_table_covers_all_reachable_operands_and_results():
    graph = _build_lowerer(_mixed_reduce, {"X": "*fp32", "O": "*fp32", "I": "*i32"}, {}).graph
    needed = {a.id for a in graph.args}
    for op in _all_ops(graph.ops):
        needed.update(op.operand_ids)
        needed.update(op.result_ids or ([op.id] if op.type_str else []))
        needed.update(op.attrs.get("block_arg_ids", []))
        needed.update(op.attrs.get("return_ids", []))
    assert needed <= graph.result_meta.keys()
