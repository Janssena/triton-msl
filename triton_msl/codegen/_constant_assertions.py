"""Discharge only entry-block assertions of a native scalar i1 literal true.

This is not an assumption about inputs, assertion messages, or successful runs.
The original module and graph are never mutated. Unknown, tensor, computed,
runtime, nested-region and callee predicates retain the existing checked path.
"""

from dataclasses import replace


def discharge_scalar_true(graph, module):
    constants = set()
    native_checks = set()

    def inspect(op):
        name = op.get_name()
        if name == "arith.constant" and op.get_num_results() == 1:
            value = op.get_result(0)
            # The public native getter sign-extends APInt: scalar i1 true is -1.
            # In particular, an i32 -1 or an i1 tensor is not this proof.
            if value.get_type().is_integer(1) and hasattr(op, "get_constant_value") and op.get_constant_value() == -1:
                constants.add(value.id())
        elif name == "tt.assert" and op.get_num_operands() == 1:
            native_checks.add((op.get_operand(0).id(), op.get_block().id()))

    module.walk(inspect)
    kept = []
    for op in graph.ops:
        proven = (
            op.op == "tt.assert"
            and len(op.operand_ids) == 1
            and op.operand_ids[0] in constants
            and (op.operand_ids[0], op.attrs.get("_block_id")) in native_checks
        )
        if not proven:
            kept.append(op)
    return replace(graph, ops=kept) if len(kept) != len(graph.ops) else graph
