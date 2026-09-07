"""Closed value/address vocabulary for the two row-normalization templates.

Recognition is not equivalence: compare the connected source expression with
the expression the maker actually implements. Unknown operations, conversions,
projections, effects and reduction returns cannot supply proof facts.
"""

import math

from triton_msl.codegen.mlir_walker import _extract_shape
from triton_msl.errors import MetalNonRecoverableError


def _comm(op, a, b):
    return (op, *sorted((a, b), key=repr))


class _NotProven(Exception):
    pass


def prove_normalization(graph, info, family):
    """Validate the candidate's whole row contract, or refuse before emission.

    This deliberately does not share a permissive pointer/value peeling helper
    with other detectors. Only equal-shape layout conversions and scalar splats
    are representation-only in this rank-one contract.
    """
    ops = {op.id: op for op in graph.ops}
    args = {arg.id: arg for arg in graph.args}
    cache, active = {}, set()
    block = info["block_size"]
    arg_names = {a.name: a.id for a in graph.args}
    length_id = arg_names[info["n_arg"]]
    length = ("arg", length_id)
    column = ("column", block)
    row = ("pid", 0)
    mask = ("slt", column, length)
    offset = _comm("addi", _comm("muli", row, length), column)

    def require(ok, why):
        if not ok:
            raise _NotProven(why)

    def shape(vid):
        obj = ops.get(vid) or args.get(vid)
        require(obj is not None, "missing operand type")
        return _extract_shape(obj.type_str)

    def elem(vid):
        obj = ops.get(vid) or args.get(vid)
        require(obj is not None, "missing operand element type")
        return obj.elem_type

    def constant(op):
        value = op.attrs.get("value")
        require(op.elem_type in ("f32", "i32", "i1"), "unreplayed constant precision")
        # The walker exposes hexadecimal nonfinite APFloat literals as bits.
        if op.elem_type == "f32" and isinstance(value, int):
            value = {0: 0.0, 0xFF800000: -math.inf, 0x7F800000: math.inf}.get(value)
        require(isinstance(value, (int, float)), "unresolved/non-splat constant")
        require(not math.isnan(value), "NaN constant is not a normalization proof fact")
        return ("constant", op.elem_type, value)

    def expr(vid):
        if vid in cache:
            return cache[vid]
        require(vid not in active, "cyclic value path")
        active.add(vid)
        try:
            result = expression(vid)
            cache[vid] = result
            return result
        finally:
            active.remove(vid)

    def expression(vid):
        if vid in args:
            return ("arg", vid)
        op = ops.get(vid)
        require(op is not None, "unresolved value")
        name, operands = op.op, op.operand_ids or []
        out_shape = shape(vid)
        require(out_shape in ((), (block,)), f"unproved projection at {name}")
        if name == "arith.constant":
            return constant(op)
        if name == "tt.get_program_id":
            require(not operands and out_shape == (), "tensor program id")
            return ("pid", op.attrs.get("axis"))
        if name == "tt.make_range":
            require(op.attrs.get("start") == 0 and op.attrs.get("end") == block, "noncanonical column coordinates")
            return column
        if name in ("ttg.convert_layout", "tt.reshape", "tt.splat"):
            require(len(operands) == 1 and elem(vid) == elem(operands[0]), "layout changes element type")
            expected = () if name == "tt.splat" else out_shape
            require(shape(operands[0]) == expected, "unproved shape-changing layout")
            return expr(operands[0])
        if name == "tt.addptr":
            require(len(operands) == 2 and elem(operands[1]) == "i32", "unproved pointer offset width")
            pointer, delta = map(expr, operands)
            if pointer[0] == "address":
                return ("address", pointer[1], _comm("addi", pointer[2], delta))
            return ("address", pointer, delta)
        if name in ("arith.addi", "arith.muli", "arith.cmpi"):
            require(len(operands) == 2 and all(elem(x) == "i32" for x in operands), "unproved index operands")
            require(all(shape(x) == out_shape for x in operands), "index projection mismatch")
            a, b = map(expr, operands)
            if name == "arith.cmpi":
                return (op.attrs.get("predicate_name"), a, b)
            return _comm(name.removeprefix("arith."), a, b)
        if name == "arith.sitofp":
            require(
                operands == [length_id] and op.elem_type == "f32" and out_shape == (),
                "unproved normalization divisor conversion",
            )
            return ("length_float", length)
        if name == "tt.load":
            require(
                len(operands) == 3 and op.elem_type == "f32" and out_shape == (block,),
                "load precision/mask/fill is not replayed",
            )
            address, condition, fill = map(expr, operands)
            require(
                address == ("address", ("arg", arg_names[info["input_arg"]]), offset),
                "load address is not input + pid(0)*N + column",
            )
            require(condition == mask, "load mask is not column < N")
            want = -math.inf if family == "softmax" else 0.0
            require(fill == ("constant", "f32", want), "load other value is not replayed")
            return ("input",)
        if name == "tt.reduce":
            require(
                len(operands) == 1
                and op.attrs.get("axis") == 0
                and out_shape == ()
                and shape(operands[0]) == (block,)
                and op.elem_type == "f32",
                "reduction axis/precision mismatch",
            )
            body = op.region_ops or []
            bargs = op.attrs.get("block_arg_ids") or []
            require(len(body) == 1 and len(bargs) == 2 and bargs[0] != bargs[1], "noncanonical reduction region")
            combine = body[0]
            require(
                combine.op in ("arith.addf", "arith.maxnumf")
                and combine.elem_type == "f32"
                and sorted(combine.operand_ids or []) == sorted(bargs)
                and op.attrs.get("return_ids") == [combine.id],
                "reduction does not return its plain combiner",
            )
            return ("sum" if combine.op == "arith.addf" else "max", expr(operands[0]))
        if name in ("arith.addf", "arith.mulf", "arith.subf", "arith.divf", "math.exp", "math.rsqrt"):
            arity = 1 if name.startswith("math.") else 2
            require(
                len(operands) == arity
                and op.elem_type == "f32"
                and all(elem(x) == "f32" and shape(x) == out_shape for x in operands),
                "unreplayed arithmetic precision or projection",
            )
            values = [expr(x) for x in operands]
            tag = name.split(".")[1]
            return _comm(tag, *values) if tag in ("addf", "mulf") else (tag, *values)
        if name == "arith.select":
            require(
                len(operands) == 3 and op.elem_type == "f32" and all(shape(x) == out_shape for x in operands),
                "unproved select projection",
            )
            return ("select", *map(expr, operands))
        raise _NotProven(f"unreplayed operation {name}")

    try:
        require(args[length_id].elem_type == "i32", "row length is not i32")
        stores = [op for op in graph.ops if op.op == "tt.store"]
        require(len(stores) == 1, "not exactly one store")
        store = stores[0]
        require(len(store.operand_ids or []) == 3, "store mask missing")
        pointer, value, condition = store.operand_ids
        require(
            expr(pointer) == ("address", ("arg", arg_names[info["output_arg"]]), offset),
            "store address is not output + pid(0)*N + column",
        )
        require(expr(condition) == mask, "store mask is not column < N")
        # Only the final storage conversion is replayed, and its target must be
        # the actual pointer element type. No inner rounding may disappear.
        cast = ops.get(value)
        while cast and cast.op in ("ttg.convert_layout", "tt.reshape"):
            require(
                len(cast.operand_ids) == 1
                and shape(value) == shape(cast.operand_ids[0])
                and elem(value) == elem(cast.operand_ids[0]),
                "unproved store layout",
            )
            value = cast.operand_ids[0]
            cast = ops.get(value)
        output_type = args[arg_names[info["output_arg"]]].elem_type
        if cast and cast.op == "arith.truncf":
            require(
                cast.elem_type == output_type
                and output_type in ("f16", "bf16")
                and len(cast.operand_ids) == 1
                and elem(cast.operand_ids[0]) == "f32"
                and shape(value) == shape(cast.operand_ids[0]),
                "unreplayed store cast",
            )
            cache[value] = ("output_cast",)  # visited effect-free conversion
            value = cast.operand_ids[0]
        else:
            require(output_type == "f32", "output pointer conversion is not proved")
        x = ("input",)
        if family == "softmax":
            exponential = ("exp", ("subf", x, ("max", x)))
            expected = ("divf", exponential, ("sum", exponential))
        else:
            divisor = ("length_float", length)
            centered = ("subf", x, ("divf", ("sum", x), divisor))
            centered = ("select", mask, centered, ("constant", "f32", 0.0))
            variance = ("divf", ("sum", _comm("mulf", centered, centered)), divisor)
            eps = ("constant", "f32", info["eps"])
            expected = _comm("mulf", centered, ("rsqrt", _comm("addf", variance, eps)))
        require(expr(value) == expected, "stored value is not the source-equivalent normalization DAG")
        # No unaccounted side effect, call, region, second store or value-changing
        # operation can hitchhike on an otherwise matching output expression.
        for op in graph.ops:
            if op.op == "tt.return":
                require(not op.operand_ids, "entry returns a value")
            elif op is not store and op.id not in cache:
                expr(op.id)
    except _NotProven as exc:
        raise MetalNonRecoverableError(
            f"Normalization template ({family}) cannot prove its whole value/address/effect path: {exc}. "
            "Refusing rather than discard source semantics.",
            op_name="tt.reduce",
        ) from None
