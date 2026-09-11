"""182/191 stage 1a: facts belong to EACH result, not the first result's op."""
from dataclasses import FrozenInstanceError

import pytest
import torch
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


@triton.jit
def _mixed_reduce_min(X, O, I):
    x = tl.load(X + tl.arange(0, 32))
    value, index = tl.min(x, 0, return_indices=True)
    tl.store(O, value)
    tl.store(I, index)


@triton.jit
def _welford_combine(mean_a, m2_a, weight_a, mean_b, m2_b, weight_b):
    delta = mean_b - mean_a
    weight = weight_a + weight_b
    ratio = tl.where(weight == 0.0, 0.0, weight_b / weight)
    mean = mean_a + delta * ratio
    m2 = m2_a + m2_b + delta * delta * weight_a * ratio
    return mean, m2, weight


@triton.jit
def _unguarded_welford_combine(mean_a, m2_a, weight_a, mean_b, m2_b, weight_b):
    delta = mean_b - mean_a
    weight = weight_a + weight_b
    ratio = weight_b / weight
    mean = mean_a + delta * ratio
    m2 = m2_a + m2_b + delta * delta * weight_a * ratio
    return mean, m2, weight


@triton.jit
def _mixed_welford(X, O_MEAN, O_M2, O_WEIGHT, N: tl.constexpr):
    offset = tl.arange(0, N)
    value = tl.load(X + offset).to(tl.float32)
    m2 = tl.full((N,), 0.0, tl.float32)
    weight = tl.full((N,), 1.0, tl.float32)
    mean, m2, weight = tl.reduce((value, m2, weight), 0, _welford_combine)
    tl.store(O_MEAN, mean)
    tl.store(O_M2, m2)
    tl.store(O_WEIGHT, weight)


@triton.jit
def _unguarded_welford(X, O_MEAN, O_M2, O_WEIGHT, N: tl.constexpr):
    offset = tl.arange(0, N)
    value = tl.load(X + offset).to(tl.float32)
    m2 = tl.full((N,), 0.0, tl.float32)
    weight = tl.full((N,), 1.0, tl.float32)
    mean, m2, weight = tl.reduce(
        (value, m2, weight), 0, _unguarded_welford_combine
    )
    tl.store(O_MEAN, mean)
    tl.store(O_M2, m2)
    tl.store(O_WEIGHT, weight)


@triton.jit
def _weighted_mean_combine(mean_a, extra_a, weight_a, mean_b, extra_b, weight_b):
    weight = weight_a + weight_b
    mean = (mean_a * weight_a + mean_b * weight_b) / weight
    extra = extra_a + extra_b
    return mean, extra, weight


@triton.jit
def _weighted_mean_reduce(X, O_MEAN, O_M2, O_WEIGHT):
    offset = tl.arange(0, 32)
    value = tl.load(X + offset).to(tl.float32)
    extra = tl.full((32,), 0.0, tl.float32)
    weight = tl.full((32,), 1.0, tl.float32)
    mean, extra, weight = tl.reduce((value, extra, weight), 0, _weighted_mean_combine)
    tl.store(O_MEAN, mean)
    tl.store(O_M2, extra)
    tl.store(O_WEIGHT, weight)


@triton.jit
def _max_by_key_combine(value_a, key_a, value_b, key_b):
    take_a = key_a > key_b
    return tl.where(take_a, value_a, value_b), tl.where(take_a, key_a, key_b)


@triton.jit
def _max_by_key_reduce(VALUE, KEY, O_VALUE, O_KEY):
    offset = tl.arange(0, 32)
    value = tl.load(VALUE + offset)
    key = tl.load(KEY + offset)
    value, key = tl.reduce((value, key), 0, _max_by_key_combine)
    tl.store(O_VALUE, value)
    tl.store(O_KEY, key)


@triton.jit
def _metadata_atomic_rmw(X, O):
    offset = tl.arange(0, 8)
    old = tl.atomic_add(X + offset, 1)
    tl.store(O + offset, old)


@triton.jit
def _metadata_atomic_cas(X, O):
    old = tl.atomic_cas(X, 0, 1)
    tl.store(O, old)


@triton.jit
def _metadata_reduce_scalar(X, O):
    value = tl.load(X + tl.arange(0, 8))
    tl.store(O, tl.sum(value, axis=0))


@triton.jit
def _metadata_reduce_rows(X, O):
    row = tl.arange(0, 4)[:, None]
    col = tl.arange(0, 8)[None, :]
    value = tl.load(X + row * 8 + col)
    tl.store(O + tl.arange(0, 4), tl.sum(value, axis=1))


@triton.jit
def _metadata_copy(X, O):
    offset = tl.arange(0, 8)
    tl.store(O + offset, tl.load(X + offset))


@triton.jit
def _metadata_masked_copy(X, O):
    offset = tl.arange(0, 8)
    tl.store(O + offset, tl.load(X + offset), mask=offset < 7)


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


def _multi_reduce_lowerer(kind):
    import os
    from pathlib import Path

    import triton_msl

    if expected_root := os.environ.get("TRITON_MSL_EXPECTED_ROOT"):
        assert Path(triton_msl.__file__).resolve().is_relative_to(Path(expected_root).resolve())
    if kind in ("argmax", "argmin", "max_by_key"):
        return _build_lowerer(
            (
                _max_by_key_reduce
                if kind == "max_by_key"
                else (_mixed_reduce_min if kind == "argmin" else _mixed_reduce)
            ),
            (
                {"VALUE": "*fp32", "KEY": "*i32", "O_VALUE": "*fp32", "O_KEY": "*i32"}
                if kind == "max_by_key"
                else {"X": "*fp16", "O": "*fp16", "I": "*i32"}
            ),
            {},
        )
    return _build_lowerer(
        (
            _weighted_mean_reduce
            if kind == "weighted_mean"
            else (_unguarded_welford if kind == "unguarded_welford" else _mixed_welford)
        ),
        {"X": "*fp32", "O_MEAN": "*fp32", "O_M2": "*fp32", "O_WEIGHT": "*fp32"},
        {} if kind == "weighted_mean" else {"N": 32},
    )


def _multi_reduce_op(lowerer):
    return next(op for op in _all_ops(lowerer.graph.ops) if op.op == "tt.reduce")


def test_multi_result_reduce_contract_uses_native_per_slot_metadata():
    """Legacy result-zero spellings cannot stand in for tuple-slot facts."""
    expected = {
        "argmax": ((32,), (), ["fp16", "i32"]),
        "argmin": ((32,), (), ["fp16", "i32"]),
        "welford": ((32,), (), ["fp32", "fp32", "fp32"]),
    }
    for kind in expected:
        lowerer = _multi_reduce_lowerer(kind)
        reduce = _multi_reduce_op(lowerer)
        pending = list(lowerer.graph.ops)
        while pending:
            op = pending.pop()
            op.type_str = "unusable legacy type"
            pending.extend(op.region_ops or [])
            pending.extend(op.else_ops or [])
        actual = lowerer._prove_reduce_native_contract(reduce)
        assert (actual["shape"], actual["output_shape"], actual["slot_dtypes"]) == expected[kind]


def test_atomic_shape_uses_native_result_metadata_not_legacy_fields():
    """Atomic execution guards must follow the native result, not SSA result zero text."""
    cases = (
        (_metadata_atomic_rmw, {"X": "*i32", "O": "*i32"}, "tensor<1xi64>", False, None, "if (lid == 0)"),
        (_metadata_atomic_cas, {"X": "*i32", "O": "*i32"}, "tensor<8xi32>", True, "lid == 0", "lid < 8u"),
    )
    for kernel, signature, fake_type, fake_is_tensor, expected_guard, forbidden_guard in cases:
        lowerer = _build_lowerer(kernel, signature, {})
        atomic = next(op for op in _all_ops(lowerer.graph.ops) if op.op.startswith("tt.atomic_"))
        atomic.attrs["sem"] = "relaxed"
        atomic.type_str = fake_type
        atomic.elem_type = "i64"
        atomic.is_tensor = fake_is_tensor
        msl = lowerer.lower()
        if expected_guard is not None:
            assert expected_guard in msl
        assert forbidden_guard not in msl


def test_atomic_native_contract_refuses_cross_operand_shape_or_width_damage():
    """A locally well-formed result record cannot contradict its pointer/value operands."""
    from dataclasses import replace

    from triton_msl.errors import MetalNonRecoverableError

    cases = (
        (_metadata_atomic_rmw, {"X": "*i32", "O": "*i32"}, {"shape": (1,)}),
        (
            _metadata_atomic_cas,
            {"X": "*i32", "O": "*i32"},
            {"kind": "float", "elem": "f16", "width": 16},
        ),
    )
    for kernel, signature, type_changes in cases:
        lowerer = _build_lowerer(kernel, signature, {})
        atomic = next(op for op in _all_ops(lowerer.graph.ops) if op.op.startswith("tt.atomic_"))
        atomic.attrs["sem"] = "relaxed"
        meta = lowerer.graph.result_meta[atomic.id]
        lowerer.graph.result_meta[atomic.id] = replace(
            meta, type=replace(meta.type, **type_changes)
        )
        with pytest.raises(MetalNonRecoverableError, match="atomic.*native|native.*atomic"):
            lowerer.lower()


def _single_reduce_lowerer(kind):
    return _build_lowerer(
        _metadata_reduce_scalar if kind == "scalar" else _metadata_reduce_rows,
        {"X": "*fp32", "O": "*fp32"},
        {},
    )


def test_single_result_reduce_uses_native_shape_type_and_layout():
    """One native slot, not the reduce op's mutable legacy fields, owns lowering."""
    expected = {"scalar": ((8,), (), "fp32"), "rows": ((4, 8), (4,), "fp32")}
    for kind, contract in expected.items():
        canonical = _single_reduce_lowerer(kind)
        canonical_msl = canonical.lower()

        lowerer = _single_reduce_lowerer(kind)
        reduce = next(op for op in _all_ops(lowerer.graph.ops) if op.op == "tt.reduce")
        native = lowerer._prove_reduce_native_contract(reduce)
        assert (native["shape"], native["output_shape"], native["result_dtypes"][0]) == contract

        # This mutable summary previously selected narrow-result masking. It may
        # remain a compatibility cache, but it is not proof of the native result.
        reduce.elem_type = "i8"
        assert lowerer.lower() == canonical_msl


def test_single_result_reduce_refuses_cross_boundary_native_damage():
    """A valid-looking result cannot contradict its operand or returned scalar."""
    from dataclasses import replace

    from triton_msl.errors import MetalNonRecoverableError

    for kind, damaged_value, changes in (
        ("scalar", "result", {"shape": (1,)}),
        ("rows", "operand", {"elem": "i32", "kind": "integer"}),
        ("rows", "return", {"elem": "f16", "width": 16}),
    ):
        lowerer = _single_reduce_lowerer(kind)
        reduce = next(op for op in _all_ops(lowerer.graph.ops) if op.op == "tt.reduce")
        value_id = {
            "result": reduce.id,
            "operand": reduce.operand_ids[0],
            "return": reduce.attrs["return_ids"][0],
        }[damaged_value]
        meta = lowerer.graph.result_meta[value_id]
        lowerer.graph.result_meta[value_id] = replace(
            meta, type=replace(meta.type, **changes)
        )
        with pytest.raises(MetalNonRecoverableError, match="reduce.*native|native.*reduce"):
            lowerer.lower()


def test_kernel_extent_prescan_uses_native_result_shapes_not_legacy_text():
    """Top-level and nested printed result summaries cannot resize dispatch."""
    from test_fuzz_reduce import _r1d_inloop_sum

    cases = (
        (_metadata_reduce_scalar, {"X": "*fp32", "O": "*fp32"}, {}, "tt.load"),
        (_r1d_inloop_sum, {"a": "*fp32", "o": "*fp32"}, {"N": 64, "T": 3}, "tt.load"),
    )
    for kernel, signature, constants, target_op in cases:
        canonical = _build_lowerer(kernel, signature, constants)
        canonical_msl = canonical.lower()
        expected_dispatch = canonical.effective_block_size
        expected_extent = canonical._total_elements

        lowerer = _build_lowerer(kernel, signature, constants)
        target = next(
            op
            for op in _all_ops(lowerer.graph.ops)
            if op.op == target_op and lowerer.graph.result_meta.get(op.id, None) is not None
            and lowerer.graph.result_meta[op.id].type.is_tensor
        )
        target.type_str = "tensor<1x999xf32>"
        assert lowerer.lower() == canonical_msl
        assert (lowerer.effective_block_size, lowerer._total_elements) == (
            expected_dispatch,
            expected_extent,
        )


def test_kernel_extent_prescan_uses_native_pointer_shape_not_legacy_text():
    """A store-address summary cannot widen the kernel behind native metadata."""
    canonical = _single_reduce_lowerer("rows")
    canonical_msl = canonical.lower()
    expected_dispatch = canonical.effective_block_size
    expected_extent = canonical._total_elements

    lowerer = _single_reduce_lowerer("rows")
    store = next(op for op in _all_ops(lowerer.graph.ops) if op.op == "tt.store")
    pointer = next(
        op for op in _all_ops(lowerer.graph.ops) if op.id == store.operand_ids[0]
    )
    assert lowerer.graph.result_meta[pointer.id].type.shape == (4,)
    pointer.type_str = "tensor<1x999x!tt.ptr<f32>>"
    assert lowerer.lower() == canonical_msl
    assert (lowerer.effective_block_size, lowerer._total_elements) == (
        expected_dispatch,
        expected_extent,
    )


def test_kernel_extent_prescan_refuses_damaged_native_shape_contract():
    """Missing, dynamic, or internally contradictory tensor facts fail closed."""
    from dataclasses import replace

    from triton_msl.errors import MetalNonRecoverableError

    for damage in ("missing", "dynamic", "false_scalar"):
        lowerer = _build_lowerer(_metadata_copy, {"X": "*fp32", "O": "*fp32"}, {})
        load = next(op for op in _all_ops(lowerer.graph.ops) if op.op == "tt.load")
        meta = lowerer.graph.result_meta[load.id]
        if damage == "missing":
            lowerer.graph.result_meta.pop(load.id)
        elif damage == "dynamic":
            lowerer.graph.result_meta[load.id] = replace(
                meta, type=replace(meta.type, shape=(None,))
            )
        else:
            lowerer.graph.result_meta[load.id] = replace(
                meta, type=replace(meta.type, is_tensor=False, shape=())
            )
        with pytest.raises(MetalNonRecoverableError, match="native.*(missing|shape|tensor-kind)"):
            lowerer.lower()


def test_store_guards_use_native_shapes_not_mutable_replay_cache():
    """A late replay-cache mutation cannot change store coverage or layout."""
    from types import MethodType

    canonical = _single_reduce_lowerer("rows").lower()
    lowerer = _single_reduce_lowerer("rows")
    original = lowerer._lower_store

    def poison_then_store(self, store):
        ptr_id, val_id = store.operand_ids[:2]
        # Keep the poisoned entries fully static and superficially plausible:
        # the boundary under test is authority, not parser error handling.
        self.env_shapes[ptr_id] = (1, 999)
        self.env_shapes[val_id] = (999,)
        return original(store)

    lowerer._lower_store = MethodType(poison_then_store, lowerer)
    assert lowerer.lower() == canonical


def test_store_refuses_native_operand_contract_disagreement():
    """All native store operands are proved before the MEPT early return."""
    from dataclasses import replace
    from types import MethodType

    from triton_msl.errors import MetalNonRecoverableError

    # Independently valid shapes are still invalid when the store boundary
    # does not relate them. Equal element counts do not make ranks equivalent;
    # a mask must additionally remain an i1 value.
    for damage in ("value_shape", "mask_shape", "mask_type"):
        lowerer = _build_lowerer(
            _metadata_masked_copy, {"X": "*fp32", "O": "*fp32"}, {}
        )
        store = next(
            op for op in _all_ops(lowerer.graph.ops) if op.op == "tt.store"
        )
        target_id = store.operand_ids[1 if damage == "value_shape" else 2]
        meta = lowerer.graph.result_meta[target_id]
        changes = (
            {"shape": (2, 4)}
            if damage != "mask_type"
            else {"kind": "integer", "elem": "i32", "width": 32}
        )
        lowerer.graph.result_meta[target_id] = replace(
            meta, type=replace(meta.type, **changes)
        )

        original = lowerer._lower_store

        def force_mept_store(self, candidate):
            ptr_id, candidate_value_id = candidate.operand_ids[:2]
            self.mept_enabled = True
            self.env_ptr_array[ptr_id] = ("out_ptr", "off", 1)
            self.env[candidate_value_id] = "value"
            self.env_array[candidate_value_id] = ("value", 1, "float")
            return original(candidate)

        lowerer._lower_store = MethodType(force_mept_store, lowerer)
        with pytest.raises(MetalNonRecoverableError, match="store.*native"):
            lowerer.lower()


def test_broadcast_layout_propagation_uses_native_operand_shape():
    """A tensor cannot become a scalar merely because replay cache says so."""
    from types import MethodType

    from test_audit_2026_06_21_silent_wrongs import _sort_rows

    def make_lowerer():
        result = _build_lowerer(
            _sort_rows, {"a": "*fp32", "o": "*fp32"}, {"M": 2, "N": 16}
        )
        # Exercise the generic broadcast consumer directly; the production
        # row-sort template otherwise short-circuits this lowering path.
        result._detect_row_wise_sort = lambda: None
        return result

    canonical = make_lowerer().lower()
    lowerer = make_lowerer()
    original = lowerer._propagate_bcast_layout_binary
    hits = []

    def poison_then_propagate(self, binary):
        a_id, b_id = binary.operand_ids[:2]
        a_layout = self._bcast_layout.get(a_id)
        b_layout = self._bcast_layout.get(b_id)
        if (a_layout is None) != (b_layout is None):
            other_id = a_id if a_layout is None else b_id
            replay_shape = self.env_shapes.get(other_id)
            assert self.graph.result_meta[other_id].type.shape
            self.env_shapes[other_id] = ()
            try:
                return original(binary)
            finally:
                hits.append(binary.id)
                if replay_shape is None:
                    self.env_shapes.pop(other_id, None)
                else:
                    self.env_shapes[other_id] = replay_shape
        return original(binary)

    lowerer._propagate_bcast_layout_binary = MethodType(
        poison_then_propagate, lowerer
    )
    assert lowerer.lower() == canonical
    assert hits


def test_make_range_reshape_rewrite_uses_native_result_layout():
    """Neither printed layout nor same-shape inference authorizes a rewrite."""
    from dataclasses import replace
    import re

    from test_audit_2026_06_21_silent_wrongs import _sort_rows
    from triton_msl.codegen.msl_emitter import KernelBuilder

    lowerer = _build_lowerer(
        _sort_rows, {"a": "*fp32", "o": "*fp32"}, {"M": 2, "N": 16}
    )
    ops = list(_all_ops(lowerer.graph.ops))
    op_by_id = {op.id: op for op in ops}
    reshape = next(
        op
        for op in ops
        if op.op == "tt.reshape"
        and sum(
            dim != 1 for dim in lowerer.graph.result_meta[op.id].type.shape
        )
        == 1
        and lowerer._trace_to_make_range(
            op.operand_ids[0], ops, op_by_id
        )
        is not None
    )
    shape = lowerer.graph.result_meta[reshape.id].type.shape
    assert not lowerer.graph.result_meta[reshape.id].type.layout.startswith(
        "#ttg.slice<"
    )
    forged_layout = next(
        lowerer.graph.result_meta[op.id].type.layout
        for op in ops
        if op.op == "tt.reduce"
        and lowerer.graph.result_meta[op.id].type.layout
    )
    reshape.type_str = (
        f"tensor<{'x'.join(map(str, shape))}xi32, {forged_layout}>"
    )
    lowerer._bcast_layouts_by_layout = {forged_layout: (shape, "lid")}
    lowerer.kb = KernelBuilder("metadata_layout_probe", block_size=32)

    assert not lowerer._maybe_rewrite_make_range_reshape(reshape)
    assert reshape.id not in lowerer.env

    # Even a valid-looking native slice layout cannot borrow a different
    # reduction stage merely because the logical shapes are compatible.
    meta = lowerer.graph.result_meta[reshape.id]
    wrong_layout = re.sub(r"dim = \d+", "dim = 99", forged_layout, count=1)
    assert wrong_layout != forged_layout
    lowerer.graph.result_meta[reshape.id] = replace(
        meta, type=replace(meta.type, layout=wrong_layout)
    )
    lowerer._bcast_layouts_by_layout = {}
    lowerer._bcast_layouts_by_shape = {shape: "lid"}
    lowerer.kb = KernelBuilder("metadata_shape_fallback_probe", block_size=32)
    assert not lowerer._maybe_rewrite_make_range_reshape(reshape)
    assert reshape.id not in lowerer.env


def test_reduce_broadcast_layout_registration_refuses_conflicting_native_key():
    """Native keys are exact while same-shape stages remain independently indexed."""
    from types import SimpleNamespace

    from triton_msl.errors import MetalNonRecoverableError

    lowerer = _build_lowerer(
        _mixed_reduce, {"X": "*fp32", "O": "*fp32", "I": "*i32"}, {}
    )
    facts = SimpleNamespace(layout="#ttg.slice<{dim = 1, parent = #blocked}>")

    lowerer._register_bcast_layout_by_native(facts, (2, 16), "lid % 16u")
    # Replaying the same stage is idempotent.
    lowerer._register_bcast_layout_by_native(facts, (2, 16), "lid % 16u")

    other_stage = SimpleNamespace(
        layout="#ttg.slice<{dim = 0, parent = #blocked}>"
    )
    lowerer._register_bcast_layout_by_native(
        other_stage, (2, 16), "(lid / 2u) % 16u"
    )
    assert lowerer._bcast_shapes_by_expr == {
        "lid % 16u": {(2, 16)},
        "(lid / 2u) % 16u": {(2, 16)},
    }

    for shape, projection in (
        ((4, 16), "lid % 16u"),
        ((2, 16), "(lid / 2u) % 16u"),
    ):
        with pytest.raises(MetalNonRecoverableError, match="multiple stages"):
            lowerer._register_bcast_layout_by_native(facts, shape, projection)

    # A result with no native layout key is still retained by expression;
    # sharing a logical shape does not overwrite an earlier stage.
    no_layout = SimpleNamespace(layout=None)
    lowerer._register_bcast_layout_by_native(no_layout, (8, 4), "lid % 4u")
    lowerer._register_bcast_layout_by_native(
        no_layout, (8, 4), "(lid / 2u) % 4u"
    )
    assert lowerer._bcast_shapes_by_expr["lid % 4u"] == {(8, 4)}
    assert lowerer._bcast_shapes_by_expr["(lid / 2u) % 4u"] == {(8, 4)}


def test_multi_result_reduce_refuses_damaged_native_slot_contracts():
    """One node exercises independent ownership, shape, and slot-width boundaries."""
    from dataclasses import replace

    from triton_msl.errors import MetalNonRecoverableError

    for damage in (
        "missing_operand",
        "wrong_result_index",
        "wrong_block_owner",
        "wrong_return_width",
        "mismatched_operand_layout",
        "welford_narrow_slot",
    ):
        lowerer = _multi_reduce_lowerer(
            "welford" if damage == "welford_narrow_slot" else "argmax"
        )
        reduce = _multi_reduce_op(lowerer)
        if damage == "missing_operand":
            lowerer.graph.result_meta.pop(reduce.operand_ids[0])
        elif damage == "wrong_result_index":
            value_id = reduce.result_ids[1]
            meta = lowerer.graph.result_meta[value_id]
            lowerer.graph.result_meta[value_id] = replace(meta, result_index=0)
        elif damage == "wrong_block_owner":
            value_id = reduce.attrs["block_arg_ids"][-1]
            meta = lowerer.graph.result_meta[value_id]
            lowerer.graph.result_meta[value_id] = replace(meta, owner_id=meta.owner_id + 1)
        elif damage == "wrong_return_width":
            value_id = reduce.attrs["return_ids"][1]
            meta = lowerer.graph.result_meta[value_id]
            lowerer.graph.result_meta[value_id] = replace(
                meta, type=replace(meta.type, elem="i64", width=64)
            )
        elif damage == "mismatched_operand_layout":
            value_id = reduce.operand_ids[1]
            meta = lowerer.graph.result_meta[value_id]
            lowerer.graph.result_meta[value_id] = replace(
                meta, type=replace(meta.type, layout="#different")
            )
        elif damage == "welford_narrow_slot":
            value_id = reduce.operand_ids[2]
            meta = lowerer.graph.result_meta[value_id]
            lowerer.graph.result_meta[value_id] = replace(
                meta, type=replace(meta.type, elem="f16", width=16)
            )

        with pytest.raises(MetalNonRecoverableError, match="native|Welford"):
            lowerer.lower()


def test_division_containing_non_welford_tuple_refuses():
    """A division is not proof that all three returned values are Welford."""
    from triton_msl.errors import MetalNonRecoverableError

    with pytest.raises(MetalNonRecoverableError, match="exact Welford"):
        _multi_reduce_lowerer("weighted_mean").lower()

    # Both source spellings are real Triton Welford recurrences.  The emitted
    # ratio must preserve whether the source included the zero-weight guard.
    unguarded = _multi_reduce_lowerer("unguarded_welford").lower()
    assert "float _ratio = _ow / _nw;" in unguarded
    assert "? 0.0f : _ow / _nw" not in unguarded


def test_comparison_containing_non_argminmax_tuple_refuses():
    """A comparison is not proof that slot one is an arg index for slot zero."""
    from triton_msl.errors import MetalNonRecoverableError

    with pytest.raises(MetalNonRecoverableError, match="exact argmin/argmax"):
        _multi_reduce_lowerer("max_by_key").lower()


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)
def test_welford_multi_result_reducer_computes_all_three_slots(tmp_path, monkeypatch):
    """Unique capability control: the exact recurrence still runs on hardware."""
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    torch.manual_seed(329)
    value = torch.randn(32, device="mps", dtype=torch.float32)
    reference = value.cpu()
    expected = (reference.mean(), ((reference - reference.mean()) ** 2).sum(), 32.0)
    for kernel in (_mixed_welford, _unguarded_welford):
        outputs = [torch.empty(1, device="mps") for _ in range(3)]
        kernel[(1,)](value, *outputs, N=32, num_warps=4)
        torch.mps.synchronize()
        assert torch.allclose(outputs[0].cpu(), expected[0].reshape(1), rtol=1e-5, atol=1e-5)
        assert torch.allclose(outputs[1].cpu(), expected[1].reshape(1), rtol=1e-5, atol=1e-5)
        assert outputs[2].item() == expected[2]


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
