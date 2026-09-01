"""Fail-closed K-chunked cooperative lowering for oversized generic dots.

The generic dot fallback keeps its output in a persistent fp32 threadgroup tile.
At S=64, materializing both full input tiles as fp32 would require 48 KiB total.
This mixin proves one of two complete value graphs before allocating anything:

* one dot with a zero or load-derived matrix/row/column accumulator; or
* two chained dots where the first result is the sole left operand of the second.

Both use KC=16 input slices.  The chain form additionally computes one MxKC
intermediate slice at a time and consumes it immediately, so the peak declared
threadgroup storage is calculated before admission (28 KiB at S=64) instead of
being inferred from a known test shape.  Four KiB is reserved under the 32 KiB
project limit for future/small auxiliary allocations.
Anything outside the complete graphs below returns ``None`` and remains behind
the existing correct-or-refuse guards.
"""

from triton_msl.codegen.mlir_walker import _extract_shape
from triton_msl.errors import MetalNonRecoverableError


class _DotKChunkMixin:
    _DOT_KC = 16
    _DOT_TG_LIMIT = 32 * 1024
    _DOT_TG_RESERVE = 4 * 1024

    @staticmethod
    def _dot_kchunk_all_ops(ops):
        for op in ops:
            yield op
            if op.region_ops:
                yield from _DotKChunkMixin._dot_kchunk_all_ops(op.region_ops)
            if op.else_ops:
                yield from _DotKChunkMixin._dot_kchunk_all_ops(op.else_ops)

    def _dot_kchunk_plan(self):
        """Return a complete, budgeted K-chunk plan, or ``None`` (default deny).

        This is deliberately a whole-graph proof.  Local checks at ``tt.dot``
        cannot establish that an unreplayed accumulator/epilogue does not exist,
        which is the same recognize-then-drop class fixed on the FA value paths.
        """
        if hasattr(self, "_dot_kchunk_plan_cached"):
            return self._dot_kchunk_plan_cached
        self._dot_kchunk_plan_cached = None
        graph = getattr(self, "graph", None)
        if graph is None:
            return None

        # No structured control flow or nested regions in this capability.
        if any(op.region_ops or op.else_ops for op in graph.ops):
            return None
        ops = list(graph.ops)
        by_id = {op.id: op for op in ops}
        dots = [op for op in ops if op.op == "tt.dot"]
        if len(dots) not in (1, 2):
            return None

        # Default-deny the op spelling too: these are the address/layout ops
        # Triton emits for the frozen add/chain surface.  A value-changing op
        # outside a dot accumulator cannot be replayed by this mechanism.
        allowed_ops = {
            "arith.constant",
            "arith.extf",
            "arith.muli",
            "arith.truncf",
            "tt.make_range",
            "tt.expand_dims",
            "tt.splat",
            "tt.broadcast",
            "tt.addptr",
            "tt.load",
            "tt.store",
            "tt.dot",
            "tt.return",
            "ttg.convert_layout",
            "ttg.local_alloc",
            "ttg.local_load",
        }
        if any(op.op not in allowed_ops for op in ops):
            return None
        stores = [op for op in ops if op.op == "tt.store"]
        if len(stores) != 1:
            return None

        def _dot_shapes(dot):
            if len(dot.operand_ids or []) < 3:
                return None
            at = self._find_op_type_str(dot.operand_ids[0])
            bt = self._find_op_type_str(dot.operand_ids[1])
            a = _extract_shape(at) if at else None
            b = _extract_shape(bt) if bt else None
            if not a or not b or len(a) != 2 or len(b) != 2:
                return None
            if a[1] != b[0]:
                return None
            # The generic primitive is independently guarded at max dim 64.
            # This capability is for cooperative (>1024-output) square tiles;
            # smaller outputs stay on the established stage-1A path.
            if not (a[0] == b[1] and a[0] * b[1] > 1024):
                return None
            if not all(1 <= dim <= 64 for dim in (a[0], a[1], b[1])):
                return None
            return (a[0], a[1], b[1])

        shapes = [_dot_shapes(dot) for dot in dots]
        if any(shape is None for shape in shapes):
            return None
        M, K, N = shapes[0]

        passthrough = {
            "ttg.convert_layout",
            "arith.extf",
            "arith.truncf",
        }

        def _terminal_store_from(value_id):
            seen = set()
            frontier = [value_id]
            terminals = set()
            while frontier:
                current = frontier.pop()
                if current in seen:
                    continue
                seen.add(current)
                consumers = [op for op in ops if current in (op.operand_ids or [])]
                if not consumers:
                    return False
                for consumer in consumers:
                    if consumer.op == "tt.store":
                        if len(consumer.operand_ids) < 2 or consumer.operand_ids[1] != current:
                            return False
                        terminals.add(consumer.id)
                    elif (
                        consumer.op in passthrough
                        and consumer.operand_ids
                        and consumer.operand_ids[0] == current
                    ):
                        frontier.append(consumer.id)
                    else:
                        return False
            return len(terminals) == 1

        if not _terminal_store_from(dots[-1].id):
            return None

        def _literal_zero(value_id):
            op = by_id.get(value_id)
            if op is None or op.op != "arith.constant":
                return False
            try:
                return float(op.attrs.get("value")) == 0.0
            except (TypeError, ValueError):
                return False

        def _staged_alloc(value_id):
            local_load = by_id.get(value_id)
            if local_load is None or local_load.op != "ttg.local_load" or not local_load.operand_ids:
                return None
            alloc = by_id.get(local_load.operand_ids[0])
            if alloc is None or alloc.op != "ttg.local_alloc" or not alloc.operand_ids:
                return None
            return alloc

        def _trace_to_load(value_id):
            seen = set()
            current = value_id
            while current not in seen:
                seen.add(current)
                op = by_id.get(current)
                if op is None:
                    return None
                if op.op == "tt.load":
                    return op
                if op.op in passthrough and op.operand_ids:
                    current = op.operand_ids[0]
                    continue
                return None
            return None

        def _trace_acc_load(value_id):
            seen = set()
            current = value_id
            wrappers = passthrough | {"tt.broadcast", "tt.expand_dims"}
            expand_axis = None
            while current not in seen:
                seen.add(current)
                op = by_id.get(current)
                if op is None:
                    return None
                if op.op == "tt.load":
                    shape = tuple(_extract_shape(op.type_str or "") or ())
                    if shape == (M, N):
                        return op, "matrix"
                    if shape == (M,) and str(expand_axis) == "1":
                        return op, "row"
                    if shape == (N,) and str(expand_axis) == "0":
                        return op, "col"
                    return None
                if op.op in wrappers and op.operand_ids:
                    if op.op == "tt.expand_dims":
                        expand_axis = op.attrs.get("axis")
                    current = op.operand_ids[0]
                    continue
                return None
            return None

        alloc_roles = {}
        first = dots[0]
        first_a = _staged_alloc(first.operand_ids[0])
        first_b = _staged_alloc(first.operand_ids[1])
        if first_a is None or first_b is None:
            return None
        if _trace_to_load(first_a.operand_ids[0]) is None or _trace_to_load(first_b.operand_ids[0]) is None:
            return None
        alloc_roles[first_a.id] = "first_a"
        alloc_roles[first_b.id] = "first_b"

        if len(dots) == 1:
            acc_id = first.operand_ids[2]
            acc_info = None if _literal_zero(acc_id) else _trace_acc_load(acc_id)
            if not _literal_zero(acc_id):
                if acc_info is None or not acc_info[0].operand_ids or len(acc_info[0].operand_ids) != 1:
                    return None
                acc_shape = _extract_shape(acc_info[0].type_str or "")
                if tuple(acc_shape or ()) not in ((M, N), (M,), (N,)):
                    return None
            if set(op.id for op in ops if op.op == "ttg.local_alloc") != set(alloc_roles):
                return None
            resource_bytes = 4 * (M * self._DOT_KC + self._DOT_KC * N + M * N)
            if resource_bytes > self._DOT_TG_LIMIT - self._DOT_TG_RESERVE:
                return None
            plan = {
                "kind": "single",
                "dots": dots,
                "shapes": shapes,
                "alloc_roles": alloc_roles,
                "acc_load": acc_info[0] if acc_info is not None else None,
                "acc_kind": acc_info[1] if acc_info is not None else "zero",
                "acc_zero": _literal_zero(acc_id),
                "resource_bytes": resource_bytes,
            }
            self._dot_kchunk_plan_cached = plan
            return plan

        # Chain dot: both accumulator inits are literal zero; the first result
        # reaches the second left operand only through optional narrowing and a
        # local_alloc/local_load pair.  The second right operand is a staged load.
        second = dots[1]
        M2, K2, N2 = shapes[1]
        if (M2, K2, N2) != (M, N, N):
            return None
        if not _literal_zero(first.operand_ids[2]) or not _literal_zero(second.operand_ids[2]):
            return None
        second_a = _staged_alloc(second.operand_ids[0])
        second_b = _staged_alloc(second.operand_ids[1])
        if second_a is None or second_b is None or _trace_to_load(second_b.operand_ids[0]) is None:
            return None

        current = second_a.operand_ids[0]
        seen = set()
        while current not in seen:
            seen.add(current)
            op = by_id.get(current)
            if op is None:
                return None
            if op.id == first.id:
                break
            if op.op in passthrough and op.operand_ids:
                current = op.operand_ids[0]
                continue
            return None
        else:
            return None
        if by_id.get(current) is None or by_id[current].id != first.id:
            return None

        # The first dot may feed only the narrowing/local_alloc chain that
        # becomes second_a.  Any other consumer would be silently dropped.
        allowed_first_path = {second_a.id}
        cursor = second_a.operand_ids[0]
        while cursor != first.id:
            allowed_first_path.add(cursor)
            op = by_id[cursor]
            cursor = op.operand_ids[0]
        for op in ops:
            if first.id in (op.operand_ids or []) and op.id not in allowed_first_path:
                return None

        alloc_roles[second_a.id] = "chain_intermediate"
        alloc_roles[second_b.id] = "second_b"
        if set(op.id for op in ops if op.op == "ttg.local_alloc") != set(alloc_roles):
            return None

        intermediate_dtype = second_a.elem_type or first.elem_type or "f32"
        if intermediate_dtype not in ("f16", "bf16", "f32"):
            return None
        resource_bytes = 4 * (
            M * self._DOT_KC  # first A slice
            + self._DOT_KC * N  # shared first-B / second-B slice
            + M * self._DOT_KC  # quantized first-dot intermediate slice
            + M * N  # persistent second result
        )
        if resource_bytes > self._DOT_TG_LIMIT - self._DOT_TG_RESERVE:
            return None
        plan = {
            "kind": "chain",
            "dots": dots,
            "shapes": shapes,
            "alloc_roles": alloc_roles,
            "intermediate_dtype": intermediate_dtype,
            "resource_bytes": resource_bytes,
        }
        self._dot_kchunk_plan_cached = plan
        return plan

    def _lower_dot_kchunk_alloc(self, ssa):
        """Handle a local_alloc owned by the proven K-chunk plan."""
        plan = self._dot_kchunk_plan()
        if plan is None:
            return False
        role = plan["alloc_roles"].get(ssa.id)
        if role is None:
            return False
        if not hasattr(self, "_dot_kchunk_alloc_meta"):
            self._dot_kchunk_alloc_meta = {}

        shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[0]) or "")
        if not shape or len(shape) != 2:
            raise MetalNonRecoverableError(
                "K-chunked dot local_alloc lost its proven 2-D source shape; refusing.",
                op_name="ttg.local_alloc",
            )

        if role == "chain_intermediate":
            # The chain emitter materializes only an MxKC slice, not the full
            # first result.  The intervening scalar/layout ops are placeholders
            # ignored by that whole-chain emitter.
            self.env[ssa.id] = "_dot_chain_intermediate_pending"
            self.env_types[ssa.id] = ssa.elem_type or plan["intermediate_dtype"]
            self.env_shapes[ssa.id] = tuple(shape)
            self._dot_kchunk_alloc_meta[ssa.id] = {"role": role, "shape": tuple(shape)}
            return True

        ops = list(self._dot_kchunk_all_ops(self.graph.ops))
        by_id = {op.id: op for op in ops}
        current = ssa.operand_ids[0]
        load = None
        for _ in range(8):
            op = by_id.get(current)
            if op is None:
                break
            if op.op == "tt.load":
                load = op
                break
            if op.op in ("ttg.convert_layout", "arith.extf", "arith.truncf") and op.operand_ids:
                current = op.operand_ids[0]
                continue
            break
        if load is None or not load.operand_ids or len(load.operand_ids) != 1:
            raise MetalNonRecoverableError(
                "K-chunked dot staging requires an unmasked, structurally reconstructible load; refusing.",
                op_name="ttg.local_alloc",
            )
        ptr_id = load.operand_ids[0]
        ptr_info = self.env_is_ptr.get(ptr_id)
        if ptr_info is None:
            raise MetalNonRecoverableError(
                "K-chunked dot staging could not resolve its load pointer; refusing.",
                op_name="ttg.local_alloc",
            )
        base = ptr_info[0]
        offset = self._rebuild_staged_fill_offset(ptr_id, by_id, base, shape[0], shape[1])

        kc = self._DOT_KC
        if role == "first_a":
            declared = shape[0] * kc
            shared_name = f"smem_kca_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(shared_name, dtype="fp32", size=declared)
        elif role == "first_b":
            declared = kc * shape[1]
            shared_name = f"smem_kcb_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(shared_name, dtype="fp32", size=declared)
        elif role == "second_b":
            first_b_id = next(i for i, r in plan["alloc_roles"].items() if r == "first_b")
            prior = self._dot_kchunk_alloc_meta.get(first_b_id)
            if prior is None:
                raise MetalNonRecoverableError(
                    "K-chunked chain-dot could not reuse its first B staging buffer; refusing.",
                    op_name="ttg.local_alloc",
                )
            declared = prior["declared"]
            shared_name = prior["name"]
        else:  # pragma: no cover - plan construction owns the enum
            return False

        meta = {
            "role": role,
            "shape": tuple(shape),
            "name": shared_name,
            "declared": declared,
            "base": base,
            "offset": offset,
            "dtype": load.elem_type or "f32",
        }
        self._dot_kchunk_alloc_meta[ssa.id] = meta
        self.env[ssa.id] = shared_name
        self.env_types[ssa.id] = load.elem_type or "fp32"
        self.env_shapes[ssa.id] = tuple(shape)
        if not hasattr(self, "_shared_mem_descs"):
            self._shared_mem_descs = {}
        self._shared_mem_descs[ssa.id] = (shared_name, tuple(shape), "fp32")
        return True

    @staticmethod
    def _dot_kchunk_quantized(expr, dtype):
        if dtype == "f16":
            return f"static_cast<float>(static_cast<half>({expr}))"
        if dtype == "bf16":
            return f"static_cast<float>(bfloat({expr}))"
        return expr

    def _dot_kchunk_acc_expr(self, plan, by_id):
        if plan["acc_zero"]:
            return "0.0f"
        load = plan["acc_load"]
        ptr_id = load.operand_ids[0]
        ptr_info = self.env_is_ptr.get(ptr_id)
        if ptr_info is None:
            raise MetalNonRecoverableError(
                "K-chunked dot accumulator load pointer became unresolved; refusing.",
                op_name="tt.dot",
            )
        base = ptr_info[0]
        M, _K, N = plan["shapes"][0]
        axis_hint = {"matrix": None, "row": 0, "col": 1}[plan["acc_kind"]]
        offset = self._rebuild_staged_fill_offset(
            ptr_id, by_id, base, M, N, axis_dim_hint=axis_hint
        )
        return f"{base}[{offset}]"

    def _lower_dot_kchunk(self, ssa):
        """Lower a dot owned by the plan; return True when handled."""
        plan = self._dot_kchunk_plan()
        if plan is None or ssa.id not in {dot.id for dot in plan["dots"]}:
            return False
        if plan["kind"] == "chain" and ssa.id == plan["dots"][0].id:
            placeholder = self._next_var("dot_pending")
            self.kb.raw_line(f"    float {placeholder} = 0.0f;")
            self.env[ssa.id] = placeholder
            self.env_types[ssa.id] = "fp32"
            M, _K, N = plan["shapes"][0]
            self.env_shapes[ssa.id] = (M, N)
            return True
        if plan["kind"] == "chain":
            self._lower_dot_kchunk_chain(ssa, plan)
        else:
            self._lower_dot_kchunk_single(ssa, plan)
        return True

    def _dot_kchunk_meta_by_role(self, plan, role):
        alloc_id = next(i for i, r in plan["alloc_roles"].items() if r == role)
        try:
            return self._dot_kchunk_alloc_meta[alloc_id]
        except (AttributeError, KeyError) as exc:
            raise MetalNonRecoverableError(
                f"K-chunked dot reached emission before its {role} staging metadata; refusing.",
                op_name="tt.dot",
            ) from exc

    def _lower_dot_kchunk_single(self, ssa, plan):
        M, K, N = plan["shapes"][0]
        self._dot_kchunk_declared_bytes = plan["resource_bytes"]
        kc = self._DOT_KC
        dispatch = getattr(self, "_actual_dispatch_threads", self.effective_block_size)
        a = self._dot_kchunk_meta_by_role(plan, "first_a")
        b = self._dot_kchunk_meta_by_role(plan, "first_b")
        ops = list(self._dot_kchunk_all_ops(self.graph.ops))
        by_id = {op.id: op for op in ops}
        acc = self._dot_kchunk_acc_expr(plan, by_id)

        result = f"smem_dot_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(result, dtype="fp32", size=M * N)
        self.kb.raw_line(f"    for (uint _de = lid; _de < {M * N}u; _de += {dispatch}u) {{")
        self.kb.raw_line(f"        uint _fill_row = _de / {N}u;")
        self.kb.raw_line(f"        uint _fill_col = _de % {N}u;")
        self.kb.raw_line(f"        {result}[_de] = {acc};")
        self.kb.raw_line("    }")

        self.kb.raw_line(f"    for (uint _kc = 0u; _kc < {K}u; _kc += {kc}u) {{")
        self.kb.raw_line(f"        for (uint _sa = lid; _sa < {M * kc}u; _sa += {dispatch}u) {{")
        self.kb.raw_line(f"            uint _fill_row = _sa / {kc}u;")
        self.kb.raw_line(f"            uint _fill_col = _kc + (_sa % {kc}u);")
        self.kb.raw_line(
            f"            {a['name']}[_sa] = (_fill_col < {K}u) ? {a['base']}[{a['offset']}] : 0.0f;"
        )
        self.kb.raw_line("        }")
        self.kb.raw_line(f"        for (uint _sa = lid; _sa < {kc * N}u; _sa += {dispatch}u) {{")
        self.kb.raw_line(f"            uint _fill_row = _kc + (_sa / {N}u);")
        self.kb.raw_line(f"            uint _fill_col = _sa % {N}u;")
        self.kb.raw_line(
            f"            {b['name']}[_sa] = (_fill_row < {K}u) ? {b['base']}[{b['offset']}] : 0.0f;"
        )
        self.kb.raw_line("        }")
        self.kb.raw_line("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.kb.raw_line(f"        for (uint _de = lid; _de < {M * N}u; _de += {dispatch}u) {{")
        self.kb.raw_line(f"            uint _dot_row = _de / {N}u;")
        self.kb.raw_line(f"            uint _dot_col = _de % {N}u;")
        self.kb.raw_line(f"            float _dot_sum = {result}[_de];")
        self.kb.raw_line(f"            for (uint _dk = 0u; _dk < {kc}u; ++_dk) {{")
        self.kb.raw_line(
            f"                _dot_sum += {a['name']}[_dot_row * {kc}u + _dk] * "
            f"{b['name']}[_dk * {N}u + _dot_col];"
        )
        self.kb.raw_line("            }")
        self.kb.raw_line(f"            {result}[_de] = _dot_sum;")
        self.kb.raw_line("        }")
        self.kb.raw_line("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.kb.raw_line("    }")

        if ssa.elem_type in ("f16", "bf16"):
            quant = self._dot_kchunk_quantized(f"{result}[_de]", ssa.elem_type)
            self.kb.raw_line(f"    for (uint _de = lid; _de < {M * N}u; _de += {dispatch}u) {{")
            self.kb.raw_line(f"        {result}[_de] = {quant};")
            self.kb.raw_line("    }")
            self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")

        result_var = self._next_var("dot")
        self.kb.raw_line(f"    float {result_var} = (lid < {M * N}u) ? {result}[lid] : 0.0f;")
        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = "fp32"
        self.env_shapes[ssa.id] = (M, N)
        if not hasattr(self, "_shared_mem_descs"):
            self._shared_mem_descs = {}
        self._shared_mem_descs[ssa.id] = (result, (M, N), "fp32")

    def _lower_dot_kchunk_chain(self, ssa, plan):
        M, K, Q = plan["shapes"][0]
        Q2, K2, N = plan["shapes"][1]
        if Q != Q2 or K2 != Q:
            raise MetalNonRecoverableError(
                "K-chunked chain-dot dimensions no longer match the proven chain; refusing.",
                op_name="tt.dot",
            )
        self._dot_kchunk_declared_bytes = plan["resource_bytes"]
        kc = self._DOT_KC
        dispatch = getattr(self, "_actual_dispatch_threads", self.effective_block_size)
        a = self._dot_kchunk_meta_by_role(plan, "first_a")
        b = self._dot_kchunk_meta_by_role(plan, "first_b")
        w = self._dot_kchunk_meta_by_role(plan, "second_b")

        mid = f"smem_dot_mid_{self._shared_counter}"
        self._shared_counter += 1
        result = f"smem_dot_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(mid, dtype="fp32", size=M * kc)
        self.kb.declare_threadgroup_array(result, dtype="fp32", size=M * N)

        self.kb.raw_line(f"    for (uint _de = lid; _de < {M * N}u; _de += {dispatch}u) {result}[_de] = 0.0f;")
        self.kb.raw_line(f"    for (uint _q0 = 0u; _q0 < {Q}u; _q0 += {kc}u) {{")
        self.kb.raw_line(f"        for (uint _mi = lid; _mi < {M * kc}u; _mi += {dispatch}u) {mid}[_mi] = 0.0f;")
        self.kb.raw_line(f"        for (uint _kc = 0u; _kc < {K}u; _kc += {kc}u) {{")
        self.kb.raw_line(f"            for (uint _sa = lid; _sa < {M * kc}u; _sa += {dispatch}u) {{")
        self.kb.raw_line(f"                uint _fill_row = _sa / {kc}u;")
        self.kb.raw_line(f"                uint _fill_col = _kc + (_sa % {kc}u);")
        self.kb.raw_line(
            f"                {a['name']}[_sa] = (_fill_col < {K}u) ? {a['base']}[{a['offset']}] : 0.0f;"
        )
        self.kb.raw_line("            }")
        self.kb.raw_line(f"            for (uint _sa = lid; _sa < {kc * kc}u; _sa += {dispatch}u) {{")
        self.kb.raw_line(f"                uint _fill_row = _kc + (_sa / {kc}u);")
        self.kb.raw_line(f"                uint _fill_col = _q0 + (_sa % {kc}u);")
        self.kb.raw_line(
            f"                {b['name']}[_sa] = (_fill_row < {K}u && _fill_col < {Q}u) ? "
            f"{b['base']}[{b['offset']}] : 0.0f;"
        )
        self.kb.raw_line("            }")
        self.kb.raw_line("            threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.kb.raw_line(f"            for (uint _mi = lid; _mi < {M * kc}u; _mi += {dispatch}u) {{")
        self.kb.raw_line(f"                uint _dot_row = _mi / {kc}u;")
        self.kb.raw_line(f"                uint _dot_col = _mi % {kc}u;")
        self.kb.raw_line(f"                float _dot_sum = {mid}[_mi];")
        self.kb.raw_line(f"                for (uint _dk = 0u; _dk < {kc}u; ++_dk)")
        self.kb.raw_line(
            f"                    _dot_sum += {a['name']}[_dot_row * {kc}u + _dk] * "
            f"{b['name']}[_dk * {kc}u + _dot_col];"
        )
        self.kb.raw_line(f"                {mid}[_mi] = _dot_sum;")
        self.kb.raw_line("            }")
        self.kb.raw_line("            threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.kb.raw_line("        }")

        quant = self._dot_kchunk_quantized(f"{mid}[_mi]", plan["intermediate_dtype"])
        self.kb.raw_line(f"        for (uint _mi = lid; _mi < {M * kc}u; _mi += {dispatch}u) {mid}[_mi] = {quant};")
        self.kb.raw_line("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.kb.raw_line(f"        for (uint _sa = lid; _sa < {kc * N}u; _sa += {dispatch}u) {{")
        self.kb.raw_line(f"            uint _fill_row = _q0 + (_sa / {N}u);")
        self.kb.raw_line(f"            uint _fill_col = _sa % {N}u;")
        self.kb.raw_line(
            f"            {w['name']}[_sa] = (_fill_row < {Q}u) ? {w['base']}[{w['offset']}] : 0.0f;"
        )
        self.kb.raw_line("        }")
        self.kb.raw_line("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.kb.raw_line(f"        for (uint _de = lid; _de < {M * N}u; _de += {dispatch}u) {{")
        self.kb.raw_line(f"            uint _dot_row = _de / {N}u;")
        self.kb.raw_line(f"            uint _dot_col = _de % {N}u;")
        self.kb.raw_line(f"            float _dot_sum = {result}[_de];")
        self.kb.raw_line(f"            for (uint _dk = 0u; _dk < {kc}u; ++_dk)")
        self.kb.raw_line(
            f"                _dot_sum += {mid}[_dot_row * {kc}u + _dk] * "
            f"{w['name']}[_dk * {N}u + _dot_col];"
        )
        self.kb.raw_line(f"            {result}[_de] = _dot_sum;")
        self.kb.raw_line("        }")
        self.kb.raw_line("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.kb.raw_line("    }")

        if ssa.elem_type in ("f16", "bf16"):
            quant_out = self._dot_kchunk_quantized(f"{result}[_de]", ssa.elem_type)
            self.kb.raw_line(f"    for (uint _de = lid; _de < {M * N}u; _de += {dispatch}u) {result}[_de] = {quant_out};")
            self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_threadgroup);")

        result_var = self._next_var("dot")
        self.kb.raw_line(f"    float {result_var} = (lid < {M * N}u) ? {result}[lid] : 0.0f;")
        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = "fp32"
        self.env_shapes[ssa.id] = (M, N)
        if not hasattr(self, "_shared_mem_descs"):
            self._shared_mem_descs = {}
        self._shared_mem_descs[ssa.id] = (result, (M, N), "fp32")
