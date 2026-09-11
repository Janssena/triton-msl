# triton_msl/autotuning/_fa_dispatch.py
"""FlashAttention zero-copy dispatch (compile_shader, native 2-D grid).

The simdgroup FlashAttention kernel (head_dim=128, BM=32/BN=64) is emitted with a
2-D threadgroup grid ``(n_q_blocks, Z*H, 1)``. That 2-D grid disqualifies it from
the generic 1-D compile_shader fast-path in the driver (which hard-requires
``gridY == gridZ == 1``), so without this it falls through to the host-roundtrip
metallib path -- measured ~2.5-4.2x SLOWER than dispatching the SAME kernel via
compile_shader (the entire gap is dispatch overhead, not the kernel: cs-2d == cs-1d
in cold A/B). On the zero-copy path the kernel beats PyTorch SDPA up to 2.27x
(fp16 full), median 1.29x across the dtype x causal x N matrix.

Unlike the quantized path (fail-CLOSED: the compiled kernel IS the fast dequant
kernel and the host path can't run it), FlashAttention is fail-OPEN: the host
metallib path produces the SAME correct result, just slower. So any miss here
(non-MPS, compile_shader unavailable, opt-out, or a pre-invocation error) returns
False. Once caller-workload invocation is attempted, errors propagate: replay
could apply side effects twice, even when submission/completion is uncertain.

Signature:
    dispatch_flash_attention(rt, descriptor, kernel_name, kargs, gridX, gridY, gridZ,
                             *, launch_exit_hook=None, launch_metadata=None) -> bool

descriptor = ("flash_attention", msl_src, threadgroup_size). kargs is the ordered
non-constexpr arg list (matches the kernel's [[buffer(i)]] order: Q,K,V,Out, 16
strides, Z,H,N). The 2-D dispatch is threads=(gridX*tg, gridY, gridZ),
group_size=(tg, 1, 1) -- exactly the (validated) native-grid launch.
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


def _dispatch_mla(rt, descriptor, kargs, *, grid=None, launch_exit_hook=None, launch_metadata=None, submission_state=None):
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
        _dt = {"f16": torch.float16, "fp16": torch.float16, "half": torch.float16, "f32": torch.float32, "fp32": torch.float32, "float": torch.float32}
        if len(elems) != 6 or len({_dt.get(str(e)) for e in elems}) != 1 or _dt.get(str(elems[0])) is None:
            return False
        roles = (("q", qn_i, DN), ("q_rope", qr_i, DR), ("k", kn_i, DN), ("k_rope", kr_i, DR), ("v", v_i, DV), ("out", o_i, DV))
        views = {}
        for (role, idx, last), elem in zip(roles, elems):
            t = kargs[idx] if 0 <= int(idx) < len(kargs) else None
            want = _dt.get(str(elem))
            if t is None or want is None or not hasattr(t, "data_ptr") or not str(getattr(t, "device", "")).startswith("mps"):
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


def dispatch_flash_attention(
    rt,
    descriptor,
    kernel_name,
    kargs,
    gridX,
    gridY,
    gridZ,
    *,
    launch_exit_hook=None,
    launch_metadata=None,
    submission_state=None,
):
    _submission = submission_state if submission_state is not None else SubmissionState()
    try:
        if isinstance(descriptor, (tuple, list)) and len(descriptor) >= 10 and descriptor[0] == "mla":
            return _dispatch_mla(
                rt, descriptor, kargs, grid=(gridX, gridY, gridZ),
                launch_exit_hook=launch_exit_hook, launch_metadata=launch_metadata, submission_state=_submission,
            )
        if not (isinstance(descriptor, (tuple, list)) and len(descriptor) >= 3 and descriptor[0] == "flash_attention"):
            return False
        msl = descriptor[1]
        tg = int(descriptor[2])
        if msl is None or not kernel_name or rt.is_unsupported(msl):
            return False
        gx, gy, gz = int(gridX), int(gridY), int(gridZ)
        if gx <= 0 or gy <= 0 or gz <= 0 or tg <= 0:
            return False

        lib = rt.get_library(msl)
        # >31 args -> the emitted MSL packs its overflow SCALARS into one buffer after
        # the pointers (issue #4.7); pack the dispatch args to match (the biased 3-D
        # backward kernels have ~40 args). <=31 -> positional, unchanged.
        from triton_msl.backend.driver import _pack_overflow_scalars, _MAX_METAL_BUFFERS
        _dk = _pack_overflow_scalars(kargs) if len(kargs) > _MAX_METAL_BUFFERS else kargs
        # Native 2-D/3-D grid: gx*gy*gz threadgroups, tg threads each (in x).
        # threadgroup_position_in_grid -> (q_block, zh, 0); thread_index -> 0..tg-1.
        _submission.begin()
        rt.dispatch(lib, kernel_name, _dk, threads=(gx * tg, gy, gz), group_size=(tg, 1, 1))
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
