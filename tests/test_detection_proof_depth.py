"""Address proofs traverse finite graphs completely and fail closed on cycles."""

import types

import pytest
import triton
import triton.language as tl

from tests.test_codegen_admission_semantics import _emit
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import FuncArg, IRGraph, SSAValue
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _native_repeated_row_term(A, B, C):
    rm = tl.arange(0, 32)
    rn = tl.arange(0, 32)
    rk = tl.arange(0, 32)
    ap = A
    for _ in tl.static_range(10):
        ap += rm[:, None] * 32
    a = tl.load(ap + rk[None, :])
    b = tl.load(B + rk[:, None] * 32 + rn[None, :])
    d = tl.maximum(tl.dot(a, b), 0.0)
    tl.store(C + rm[:, None] * 32 + rn[None, :], d)


def _op(oid, kind, operands=(), *, axis=None, tensor=False):
    attrs = {} if axis is None else {"axis": axis}
    return SSAValue(oid, f"v{oid}", kind, list(operands), attrs,
                    "tensor<32xi32>" if tensor else "i32", "i32", tensor)


def _lowerer(ops):
    args = [FuncArg(i, name, "!tt.ptr<f32>", "f32", True, i)
            for i, name in enumerate(("A", "B", "C"), 1)]
    lowerer = GenericLowerer.__new__(GenericLowerer)
    lowerer.graph = IRGraph("depth", args, ops)
    lowerer._resolve_dot_ptr_roles = types.MethodType(lambda self, dot, ptrs: ptrs, lowerer)
    return lowerer


def _wrapped_term(ops, next_id, axis, count):
    rng = next_id
    ops.append(_op(rng, "tt.make_range", tensor=True))
    cur = rng + 1
    ops.append(_op(cur, "tt.expand_dims", [rng], axis=axis, tensor=True))
    for oid in range(cur + 1, cur + 1 + count):
        ops.append(_op(oid, "ttg.convert_layout", [cur], tensor=True))
        cur = oid
    return cur, cur + 1


def _stride_graph(wrapper_count=0, cycle=False):
    ops = [_op(10, "tt.dot")]
    next_id = 20
    addresses = []
    for base in (1, 2, 3):
        row, next_id = _wrapped_term(ops, next_id, 1, wrapper_count)
        col, next_id = _wrapped_term(ops, next_id, 0, wrapper_count)
        if cycle and base == 1:
            cyc = _op(next_id, "ttg.convert_layout", [next_id], tensor=True)
            ops.append(cyc)
            row = next_id
            next_id += 1
        add = _op(next_id, "arith.addi", [row, col], tensor=True)
        ptr = _op(next_id + 1, "tt.addptr", [base, next_id], tensor=True)
        ops.extend((add, ptr))
        addresses.append(ptr.id)
        next_id += 2
    ops.append(_op(next_id, "tt.store", [addresses[2], 10]))
    return _lowerer(ops)


def _long_addptr_graph(link_count=20):
    ops = [_op(10, "tt.dot")]
    next_id = 20
    addresses = []
    for base in (1, 2, 3):
        row, next_id = _wrapped_term(ops, next_id, 1, 0)
        col, next_id = _wrapped_term(ops, next_id, 0, 0)
        ptr = base
        for n in range(link_count):
            term = row if n % 2 == 0 else col
            ops.append(_op(next_id, "tt.addptr", [ptr, term], tensor=True))
            ptr = next_id
            next_id += 1
        addresses.append(ptr)
    ops.append(_op(next_id, "tt.store", [addresses[2], 10]))
    return _lowerer(ops)


def _batch_graph(kind, wrapper_count):
    ops = [_op(10, "tt.dot")]
    leaf = 20
    ops.append(_op(leaf, kind))
    cur = leaf
    for oid in range(leaf + 1, leaf + 1 + wrapper_count):
        ops.append(_op(oid, "arith.extsi", [cur]))
        cur = oid
    if kind == "cycle":
        ops[-1].operand_ids = [ops[-1].id]
    ops.append(_op(1000, "tt.addptr", [1, cur]))
    return _lowerer(ops)


def _deep_add_batch_graph(with_pid, count=1200):
    ops = [_op(10, "tt.dot"), _op(20, "tt.get_program_id" if with_pid else "arith.constant")]
    cur = 20
    for oid in range(21, 21 + count):
        constant = oid + count + 1
        ops.append(_op(constant, "arith.constant"))
        ops.append(_op(oid, "arith.addi", [cur, constant]))
        cur = oid
    ops.append(_op(5000, "tt.addptr", [1, cur]))
    return _lowerer(ops)


def test_stride_proof_crosses_old_wrapper_boundary():
    assert _stride_graph(wrapper_count=40).infer_dot_strides() == {
        "A": ("1", "1"), "B": ("1", "1"), "C": ("1", "1")
    }


def test_repeated_addptr_terms_do_not_collapse_to_unit_stride():
    # 10*row + 10*col cannot be represented as the unit-stride descriptor.
    assert _long_addptr_graph().infer_dot_strides()["A"] == (None, None)


def test_native_repeated_row_term_does_not_reach_fused_stride_template():
    signature = {"A": "*fp32", "B": "*fp32", "C": "*fp32"}
    msl = _emit(_native_repeated_row_term, signature, {})
    # Native optimization retains the ten additions. The fused stride-only maker
    # declines, while generic cooperative fill replays every source coefficient.
    assert "simdgroup_multiply_accumulate" not in msl
    fill = next(line for line in msl.splitlines() if "smem_0[_sa] = A[" in line)
    assert fill.count("_fill_row * 32") == 10


def test_stride_wrapper_cycle_refuses_affirmative_descriptor():
    assert _stride_graph(wrapper_count=2, cycle=True).infer_dot_strides()["A"] == (None, None)


def test_batch_pid_proof_crosses_old_depth_boundary_and_refuses():
    with pytest.raises(MetalNonRecoverableError, match="batched matmul"):
        _batch_graph("tt.get_program_id", 40)._refuse_batched_matmul_base_offset()


def test_long_finite_non_pid_chain_remains_admitted():
    _batch_graph("arith.constant", 40)._refuse_batched_matmul_base_offset()


def test_deep_additive_pid_chain_is_iterative_and_refuses():
    with pytest.raises(MetalNonRecoverableError, match="batched matmul"):
        _deep_add_batch_graph(True)._refuse_batched_matmul_base_offset()


def test_deep_additive_non_pid_chain_is_iterative_and_admitted():
    _deep_add_batch_graph(False)._refuse_batched_matmul_base_offset()


def test_batch_wrapper_cycle_fails_closed():
    with pytest.raises(MetalNonRecoverableError, match="batched matmul"):
        _batch_graph("cycle", 4)._refuse_batched_matmul_base_offset()
