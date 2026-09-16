"""Retained assertions on the generic, threadgroup-uniform execution path.

An assertion is a uniform rendezvous: failing lanes record a launch-local error,
then ALL threads in that group return before any subsequent source operation.
No lane-dependent early return may cross a threadgroup barrier. Consequently the
control-flow proof is positive and closed, not a whitelist of assertion messages.
"""

from triton_msl.errors import MetalNonRecoverableError


def walk(ops):
    for op in ops or ():
        yield op
        yield from walk(op.region_ops)
        yield from walk(op.else_ops)


def refuse(reason):
    raise MetalNonRecoverableError("retained device assertion: " + reason, op_name="tt.assert")


def prepare(lowerer):
    graph = lowerer.graph
    ops = list(walk(graph.ops))
    checks = [op for op in ops if op.op == "tt.assert"]
    # Callee checks remain owned by the recursive refusal-catalog backstop.
    if not checks:
        return None
    if graph.called_funcs:
        refuse("assert-carrying kernels with device callees are not supported")
    by_id = {op.id: op for op in ops}
    uniform_ids = {arg.id for arg in graph.args}
    uniform_ops = {
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.divsi",
        "arith.divui",
        "arith.remsi",
        "arith.remui",
        "arith.cmpi",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.select",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.index_cast",
        "tt.splat",
        "tt.expand_dims",
        "tt.broadcast",
        "ttg.convert_layout",
    }

    def uniform(value, seen=frozenset()):
        if value in uniform_ids:
            return True
        if value in seen:
            return False
        op = by_id.get(value)
        if op is None:
            return False
        if op.op == "arith.constant":
            return type(op.attrs.get("value")) in (int, float, bool)
        if op.op in ("tt.get_program_id", "tt.get_num_programs"):
            return True
        return op.op in uniform_ops and bool(op.operand_ids) and all(uniform(v, seen | {value}) for v in op.operand_ids)

    def control(sequence):
        for op in sequence:
            if op.op == "scf.for":
                if len(op.operand_ids) < 3 or not all(uniform(v) for v in op.operand_ids[:3]):
                    refuse("loop bounds/step are not proved threadgroup-uniform")
                args = op.attrs.get("block_arg_ids", [])
                if not args:
                    refuse("missing loop induction-variable identity")
                uniform_ids.add(args[0])
            elif op.op == "scf.if":
                if len(op.operand_ids) != 1 or not uniform(op.operand_ids[0]):
                    refuse("branch condition is not proved threadgroup-uniform")
            elif (op.region_ops or op.else_ops) and op.op != "tt.reduce":
                refuse("unsupported control region " + op.op)
            if op.op == "tt.reduce" and any(x.op == "tt.assert" for x in walk(op.region_ops)):
                refuse("assertions in reduction combiners are not supported")
            control(op.region_ops or [])
            control(op.else_ops or [])

    control(graph.ops)
    allowed = {
        "tt.assert",
        "tt.load",
        "tt.store",
        "tt.addptr",
        "tt.make_range",
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "ttg.convert_layout",
        "tt.get_program_id",
        "tt.get_num_programs",
        "tt.reduce",
        "tt.reduce.return",
        "tt.return",
        "tt.bitcast",
        "tt.fp_to_fp",
        "tt.precise_divf",
        "tt.precise_sqrt",
        "tt.extern_elementwise",
        "scf.for",
        "scf.if",
        "scf.yield",
        "ttg.barrier",
        "tt.atomic_rmw",
        "tt.atomic_cas",
    }
    for op in ops:
        if op.op not in allowed and not op.op.startswith(("arith.", "math.")):
            refuse("operation family not qualified with assertions: " + op.op)
        if op.op == "tt.extern_elementwise" and op.attrs.get("pure") is not True:
            refuse("extern elementwise operation lacks a pure-operation proof")
    messages = []
    for op in checks:
        if len(op.operand_ids) != 1:
            refuse("expected one predicate value")
        facts = lowerer._native_value_facts(op.operand_ids[0], op_name="tt.assert")
        if facts.kind != "integer" or facts.elem != "i1" or facts.width != 1:
            refuse("predicate must have native i1 assertion type")
        message = op.attrs.get("message")
        if type(message) is not str:
            refuse("missing native assertion message")
        messages.append(message)
    if len(graph.args) >= 31:
        refuse("no spare Metal buffer binding for the launch-local error flag")
    return {"messages": messages, "ids": tuple(op.id for op in checks)}


def emit(lowerer, op):
    plan = lowerer._assert_plan
    if plan is None or op.id not in plan["ids"]:
        refuse("assertion was not admitted by the uniform-control proof")
    value = op.operand_ids[0]
    if value in lowerer.env_array:
        refuse("register-array assertion predicates are not qualified")
    code = plan["ids"].index(op.id) + 1
    predicate = lowerer._lookup(value)
    lowerer.kb.raw_line(f"    if (!({predicate})) {{")
    lowerer.kb.raw_line("        atomic_store_explicit(&_assert_group_failed, 1u, memory_order_relaxed);")
    lowerer.kb.raw_line(
        f"        atomic_fetch_max_explicit((device atomic_uint*)_assert_status, {code}u, memory_order_relaxed);"
    )
    lowerer.kb.raw_line("    }")
    lowerer.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    failed = lowerer._next_var("assert_failed")
    lowerer.kb.raw_line(f"    bool {failed} = atomic_load_explicit(&_assert_group_failed, memory_order_relaxed) != 0u;")
    # Everyone must sample this check's verdict BEFORE a faster thread can
    # publish a failure at the next check. Otherwise some threads could return
    # here while others reach the next rendezvous and deadlock.
    lowerer.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    lowerer.kb.raw_line(f"    if ({failed}) return;")
