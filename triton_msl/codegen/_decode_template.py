"""Propose small-query dense FA bindings; the existing whole-value proof decides admission."""


def detect(lowerer):
    graph = lowerer.graph
    if graph.called_funcs:
        return None
    loops = [o for o in graph.ops if o.op == "scf.for"]
    if len(loops) != 1:
        return None
    loop = loops[0]
    ops = list(graph.ops) + list(loop.region_ops or [])
    dots = [o for o in ops if o.op == "tt.dot"]
    if len(dots) != 2 or any(d not in loop.region_ops for d in dots):
        return None
    qshape = lowerer._native_shape(dots[0].operand_ids[0], op_name="tt.dot")
    if qshape not in ((8, 64), (16, 64)) or lowerer._native_shapes_for_op(dots[0]) != ((qshape[0], 32),):
        return None
    if lowerer._native_shapes_for_op(dots[1]) != (qshape,):
        return None
    if any(a.elem_type not in ("f16", "f32") if a.is_ptr else a.elem_type != "i32" for a in graph.args):
        return None
    if len({a.elem_type for a in graph.args if a.is_ptr}) != 1:
        return None
    for dot in dots:
        if any(lowerer._native_value_facts(v, op_name="tt.dot").elem not in ("f16", "f32") for v in dot.operand_ids):
            return None
    # This recovery covers the reporter's widened P/V dot. A narrowed
    # probability recurrence needs its own rounding qualification.
    if any(lowerer._native_value_facts(v, op_name="tt.dot").elem != "f32" for v in dots[1].operand_ids):
        return None
    by = {o.id: o for o in ops}
    args = {a.id: a for a in graph.args}
    if (
        len(loop.operand_ids) < 3
        or lowerer._const_int(loop.operand_ids[0], by) != 0
        or lowerer._const_int(loop.operand_ids[2], by) != 32
    ):
        return None
    key_length = args.get(loop.operand_ids[1])
    if key_length is None or key_length.is_ptr or key_length.elem_type != "i32":
        return None
    stores = [o for o in ops if o.op == "tt.store"]
    if len(stores) != 1 or len(stores[0].operand_ids) != 3:
        return None

    def peel(vid):
        seen = set()
        while vid not in seen:
            seen.add(vid)
            o = by.get(vid)
            if o is None or o.op not in ("tt.broadcast", "tt.splat", "tt.expand_dims", "ttg.convert_layout"):
                return vid
            if len(o.operand_ids) != 1:
                return None
            vid = o.operand_ids[0]
        return None

    cmp = by.get(peel(stores[0].operand_ids[2]))
    if cmp is None or cmp.op != "arith.cmpi" or cmp.attrs.get("predicate_name") not in ("slt", "ult"):
        return None
    rhs = peel(cmp.operand_ids[1])
    query_arg = args.get(rhs)
    if query_arg is not None and not query_arg.is_ptr and query_arg.elem_type == "i32":
        query_length = query_arg.index
    elif lowerer._const_int(rhs, by) == 1:
        query_length = "c1"
    else:
        return None
    # This is only a binding proposal. Original masks, pointers, recurrence,
    # loop range, casts and every effect are proved at the template boundary.
    info = lowerer._detect_flash_attention(sequence_index=key_length.index)
    if info is None or info.get("is_mla") or info["head_dim"] != 64 or info.get("v_head_dim", 64) != 64:
        return None
    info["query_length"] = query_length
    return info
