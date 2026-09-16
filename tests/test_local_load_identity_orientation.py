"""A native local_alloc owns an identity descriptor orientation."""

import pytest

from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import IRGraph, SSAValue
from triton_msl.errors import MetalNonRecoverableError


def _op(oid, kind, operands=()):
    return SSAValue(oid, f"v{oid}", kind, list(operands), {}, "tensor<32x32xf32>", "f32", True)


def _lowerer(source_op, orientation=None):
    lowerer = GenericLowerer.__new__(GenericLowerer)
    lowerer.graph = IRGraph("orientation", [], [source_op])
    lowerer.env = {source_op.id: "shared"}
    lowerer.env_types = {source_op.id: "fp32"}
    lowerer.env_shapes = {source_op.id: (32, 32)}
    lowerer._shared_mem_descs = {source_op.id: ("shared", (32, 32), "fp32")}
    lowerer._shared_mem_orientations = {} if orientation is None else {source_op.id: orientation}
    return lowerer


def test_local_alloc_root_proves_identity_orientation():
    lowerer = _lowerer(_op(1, "ttg.local_alloc", [10]))
    del lowerer._shared_mem_orientations  # specialized allocator has no ordinary helper side effect
    lowerer._lower_local_load(_op(2, "ttg.local_load", [1]))
    assert lowerer._shared_mem_orientations == {1: False, 2: False}
    assert lowerer._shared_mem_descs[2] == ("shared", (32, 32), "fp32")


def test_existing_transposed_orientation_is_preserved():
    lowerer = _lowerer(_op(1, "ttg.memdesc_trans", [10]), True)
    lowerer._lower_local_load(_op(2, "ttg.local_load", [1]))
    assert lowerer._shared_mem_orientations[2] is True


def test_existing_identity_orientation_is_preserved():
    lowerer = _lowerer(_op(1, "ttg.local_alloc", [10]), False)
    lowerer._lower_local_load(_op(2, "ttg.local_load", [1]))
    assert lowerer._shared_mem_orientations == {1: False, 2: False}


def test_nested_local_alloc_source_proves_identity_without_recursion():
    source = _op(1, "ttg.local_alloc", [10])
    container = _op(9, "scf.if")
    container.region_ops = [source]
    lowerer = _lowerer(source)
    lowerer.graph.ops = [container]
    del lowerer._shared_mem_orientations
    lowerer._lower_local_load(_op(2, "ttg.local_load", [1]))
    assert lowerer._shared_mem_orientations == {1: False, 2: False}


@pytest.mark.parametrize("kind", ["ttg.memdesc_trans", "ttg.convert_layout", "unknown"])
def test_unproved_nonallocation_descriptor_still_refuses(kind):
    lowerer = _lowerer(_op(1, kind, [10]))
    with pytest.raises(MetalNonRecoverableError, match="without proved orientation"):
        lowerer._lower_local_load(_op(2, "ttg.local_load", [1]))
