"""Frozen635 behavior oracle; functions copied verbatim before specialization.

Parent freeze: 548655456678d21312c11061976deb84bb3e4fe9ae9b6b7056627a47d310e644.
Loaded by the CPU test harness with the real SubmissionState class.
"""

from triton_msl.autotuning._submission import SubmissionState


def _launch_grid_ok(grid, expected):
    """Packet 105 B: a replacement template computes the FULL output from its descriptor;
    it may stand in for the compiled kernel only when the caller launched EXACTLY the
    program grid the kernel maps (unused axes 1). No grid / no contract -> not equivalent."""
    if grid is None or expected is None:
        return False
    try:
        g = tuple(int(x) for x in grid)
        e = tuple(int(x) for x in expected)
    except (TypeError, ValueError):
        return False
    return len(g) == 3 and len(e) == 3 and g == e


def _dispatch_mla(
    rt, descriptor, kargs, *, grid=None, launch_exit_hook=None, launch_metadata=None, submission_state=None
):
    """MLA (nope/rope) dispatch: concat the split QK tensors and run the qk=head_dim /
    v=v_head_dim kernel. descriptor = ('mla', msl, name, tg, q_nope, q_rope, k_nope,
    k_rope, v, out, Z, H, N) with the last 9 being indices into kargs. FAIL-CLOSED in the
    caller: a False here (non-MPS / bad shape / error) makes the driver refuse, since the
    qk=head_dim kernel's ABI differs from the @jit kernel's (the host path would mis-run)."""
    _submission = submission_state if submission_state is not None else SubmissionState()
    try:
        import torch

        # Packet 107: the source kernel's semantics are its POINTERS plus its explicit scalar
        # STRIDES and compile-time widths — never a host tensor's shape or .stride(). The
        # descriptor carries the source ABI ([12..16]); this dispatch resolves it from kargs,
        # proves the host tensors can hold exactly those views, builds them with
        # ``as_strided`` (Q/K parts, then a contiguous concat of exactly DN + DR values per
        # logical row) and hands V / Out to the template with the SOURCE strides.
        if len(descriptor) < 17:
            return False  # a descriptor without the source ABI is not provably equivalent
        msl, name, tg = descriptor[1], descriptor[2], int(descriptor[3])
        qn_i, qr_i, kn_i, kr_i, v_i, o_i = descriptor[4:10]
        z_ref, h_ref, n_ref, src_bm = descriptor[10], descriptor[11], descriptor[12], descriptor[13]
        dims, srefs, elems = descriptor[14], descriptor[15], descriptor[16]
        if msl is None or not name or rt.is_unsupported(msl):
            return False
        if src_bm != 32:
            return False  # the emitted template is fixed at BLOCK_M = 32 (packet 107 §4)
        DN, DR, DV = (int(x) for x in dims)
        if DN <= 0 or DR <= 0 or DV <= 0:
            return False

        def _res(ref):
            """A source scalar: a kernel-arg index into kargs, or the folded constant 1."""
            if ref == "c1":
                return 1
            if isinstance(ref, bool) or not isinstance(ref, int) or ref < 0 or ref >= len(kargs):
                return None
            val = kargs[ref]
            if hasattr(val, "data_ptr"):
                return None  # a pointer where a scalar was expected
            return int(val)

        Z, H, N = _res(z_ref), _res(h_ref), _res(n_ref)
        if None in (Z, H, N) or Z <= 0 or H <= 0 or N <= 0:
            return False
        n_qb = (N + 31) // 32
        if not _launch_grid_ok(grid, (n_qb, Z * H, 1)):
            return False
        # Packet 109 A: the template declares ONE pointer type for Q/K/V/Out — all six roles
        # must share it (f16 or f32; the template has no bf16 variant), and each host
        # tensor must carry exactly that dtype.
        _dt = {
            "f16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "f32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        if len(elems) != 6 or len({_dt.get(str(e)) for e in elems}) != 1 or _dt.get(str(elems[0])) is None:
            return False
        roles = (
            ("q", qn_i, DN),
            ("q_rope", qr_i, DR),
            ("k", kn_i, DN),
            ("k_rope", kr_i, DR),
            ("v", v_i, DV),
            ("out", o_i, DV),
        )
        views = {}
        for (role, idx, last), elem in zip(roles, elems):
            t = kargs[idx] if 0 <= int(idx) < len(kargs) else None
            want = _dt.get(str(elem))
            if (
                t is None
                or want is None
                or not hasattr(t, "data_ptr")
                or not str(getattr(t, "device", "")).startswith("mps")
            ):
                return False
            if t.dtype != want:
                return False
            st = tuple(_res(r) for r in srefs[role])
            if len(st) != 4 or any(s is None or s < 0 for s in st):
                return False
            shape = (Z, H, N, last)
            # the logical view must lie inside the tensor's storage from its data pointer
            reach = sum((d - 1) * s for d, s in zip(shape, st)) + 1
            cap = t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()
            if reach > cap:
                return False
            views[role] = (torch.as_strided(t, shape, st), st, t)
        # exactly DN + DR values per logical row, from the source views
        q = torch.cat([views["q"][0], views["q_rope"][0]], dim=-1).contiguous()
        k = torch.cat([views["k"][0], views["k_rope"][0]], dim=-1).contiguous()
        v_t, v_st = views["v"][2], views["v"][1]
        o_t, o_st = views["out"][2], views["out"][1]
        buffers = [q, k, v_t, o_t] + list(q.stride()) + list(k.stride()) + list(v_st) + list(o_st) + [Z, H, N]
        lib = rt.get_library(msl)
        _submission.begin()
        rt.dispatch(lib, name, buffers, threads=(n_qb * tg, Z * H, 1), group_size=(tg, 1, 1))
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)
        return True
    except Exception as _error:
        _submission.reraise_if_attempted(_error)
        try:
            rt.mark_unsupported(descriptor[1])
        except Exception as _error:
            _submission.reraise_if_attempted(_error)
            pass
        return False
