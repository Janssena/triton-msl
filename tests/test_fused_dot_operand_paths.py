"""Round 2.1 F1 + packet 102: the fused matmul+softmax / matmul+epilogue template must not
stage a dot operand from its BUFFER when the kernel transformed the loaded value.

On clean ``ba21e9d`` both fused forms with ``tl.load(X_f32).to(tl.float16)`` before ``tl.dot``
routed to ``_lower_matmul_softmax_template`` and stored the raw-fp32 result (err vs raw 0 /
6e-8; err vs the IR-ordered oracle 1.8e-3 / 2.0e-4) — the 079/W1 class on a route P0 (packet
090) had not covered; a square in-tile ``tl.trans`` on A or B stored the raw product too (err
0.5 / 0.8 / 4.8 / 4.4), and the canonical transposed-B layout (Y stored [N,K]) produced neither
the raw nor the intended result. Now ``_dot_operand_paths_reason`` — the SAME proof the bare
templates use — runs in ``_detect_matmul_softmax`` and ``_detect_matmul_epilogue`` with
``allow_trans=False``: inside the generic dot envelope the kernel declines to the op-by-op
lowerer (probe-verified to replay the casts), outside it refuses with a role-naming reason.
Roles are pinned INDEPENDENTLY (packet 101 §3.4): a dot needs equal operand dtypes, so the
A-only / B-only forms are fp32→fp16→fp32 round-trips (value-changing, dtype-preserving), the
both-operands form is the original fp16 dot. fp32 fused kernels stay on the template.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir

    from triton_msl.backend.compiler import MetalBackend
    import triton_msl.codegen.generic_lowerer as generic_lowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False

requires = pytest.mark.skipif(not HAS, reason="Triton compiler needed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")

if HAS:

    @triton.jit
    def _fused_softmax(X, Y, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       CAST_A: tl.constexpr, CAST_B: tl.constexpr, CAST_AB16: tl.constexpr, TRANS_A: tl.constexpr, TRANS_B: tl.constexpr, Y_NK: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        if Y_NK:
            y = tl.load(Y + on[:, None] * K + ok[None, :])
        else:
            y = tl.load(Y + ok[:, None] * N + on[None, :])
        if CAST_A:
            x = x.to(tl.float16).to(tl.float32)  # role-isolated: rounds A, keeps the dot fp32
        if CAST_B:
            y = y.to(tl.float16).to(tl.float32)
        if CAST_AB16:
            x = x.to(tl.float16)  # the original F1 form: both operands fp16 at the dot
            y = y.to(tl.float16)
        if TRANS_A:
            x = tl.trans(x)
        if TRANS_B:
            y = tl.trans(y)
        z = tl.dot(x, y)
        z = z - tl.max(z, 1)[:, None]
        num = tl.exp(z)
        tl.store(C + om[:, None] * N + on[None, :], num / tl.sum(num, 1)[:, None])

    @triton.jit
    def _fused_epilogue(X, Y, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        CAST_A: tl.constexpr, CAST_B: tl.constexpr, CAST_AB16: tl.constexpr, TRANS_A: tl.constexpr, TRANS_B: tl.constexpr, Y_NK: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        if Y_NK:
            y = tl.load(Y + on[:, None] * K + ok[None, :])
        else:
            y = tl.load(Y + ok[:, None] * N + on[None, :])
        if CAST_A:
            x = x.to(tl.float16).to(tl.float32)  # role-isolated: rounds A, keeps the dot fp32
        if CAST_B:
            y = y.to(tl.float16).to(tl.float32)
        if CAST_AB16:
            x = x.to(tl.float16)  # the original F1 form: both operands fp16 at the dot
            y = y.to(tl.float16)
        if TRANS_A:
            x = tl.trans(x)
        if TRANS_B:
            y = tl.trans(y)
        z = tl.dot(x, y) + tl.load(B + on)[None, :]
        tl.store(C + om[:, None] * N + on[None, :], tl.maximum(z, 0.0))

    def _direct_lowerer_for(fn, signature, constexprs):
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        target = GPUTarget("metal", "apple-m4", 32)
        backend = MetalBackend(target)
        options = backend.parse_options({"num_warps": 4})
        src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
        context = ir.context()
        ir.load_dialects(context)
        mod = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context)
        metadata = {}
        mod = backend.make_ttir(mod, metadata, options)
        mod = backend.make_ttgir(mod, metadata, options)
        return generic_lowerer.GenericLowerer(walk_ttgir(mod, options), options)


_OFF = dict(CAST_A=False, CAST_B=False, CAST_AB16=False, TRANS_A=False, TRANS_B=False, Y_NK=False)
_VARIANTS = {
    "castA": dict(CAST_A=True),  # fp32->fp16->fp32 round-trip on A only (a dot needs equal dtypes)
    "castB": dict(CAST_B=True),
    "castAB": dict(CAST_AB16=True),  # both operands fp16 at the dot (the original F1 form)
    "transA": dict(TRANS_A=True),
    "transB": dict(TRANS_B=True),
    "transB-canonical": dict(TRANS_B=True, Y_NK=True),
}


def _flags(name):
    return {**_OFF, **_VARIANTS[name]}


def _shape(name, inside):
    if inside:
        return 32, 32, 32
    if name in ("transB", "transB-canonical"):
        return 64, 32, 32  # B square at 32x32; outside the envelope (M != N)
    return 32, 64, 32  # A square at 32x32; outside the envelope


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    for fn in (_fused_softmax, _fused_epilogue):
        if hasattr(fn, "device_caches"):
            fn.device_caches.clear()


@pytest.fixture
def route_spies(monkeypatch):
    fused, generic = [], []
    of = generic_lowerer.GenericLowerer._lower_matmul_softmax_template
    og = generic_lowerer.GenericLowerer._lower_dot
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_matmul_softmax_template", lambda self, info: (fused.append(1), of(self, info))[1])
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", lambda self, ssa: (generic.append(1), og(self, ssa))[1])
    return fused, generic


def _case(form, name, inside):
    M, N, K = _shape(name, inside)
    fl = _flags(name)
    torch.manual_seed(3)
    x = torch.randn(M, K, device="mps") * 0.5
    y = torch.randn(N, K, device="mps") * 0.5 if fl["Y_NK"] else torch.randn(K, N, device="mps") * 0.5
    b = torch.randn(N, device="mps") * 0.5
    xi = x.half().float() if (fl["CAST_A"] or fl["CAST_AB16"]) else x
    yi = y.half().float() if (fl["CAST_B"] or fl["CAST_AB16"]) else y
    xi = xi.T if fl["TRANS_A"] else xi
    yi = yi.T if fl["TRANS_B"] else yi
    yraw = y.reshape(K, N) if fl["Y_NK"] else y  # a template ignoring the transpose reads the buffer as [K,N]
    c = torch.full((M, N), float("nan"), device="mps")
    if form == "softmax":
        raw, intended = torch.softmax(x @ yraw, 1), torch.softmax(xi @ yi, 1)
        launch = lambda: _fused_softmax[(1,)](x, y, c, M=M, N=N, K=K, **fl, num_warps=4)
    else:
        raw, intended = torch.relu(x @ yraw + b), torch.relu(xi @ yi + b)
        launch = lambda: _fused_epilogue[(1,)](x, y, b, c, M=M, N=N, K=K, **fl, num_warps=4)
    return launch, c, raw, intended


def _witness(launch, c, raw, intended):
    """Refuse (touching nothing) or match the IR-ordered oracle; never the raw result."""
    sep = (raw - intended).abs().max().item()
    assert sep > 0
    try:
        launch()
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        assert bool(c.isnan().all()), "a refused kernel must touch nothing"
        return "refused"
    err = (c - intended).abs().max().item()
    assert err == err and err <= sep / 10, f"computed the raw result, not the intended one (err {err}, separation {sep})"
    return "computed"


@requires_gpu
@pytest.mark.parametrize("name", list(_VARIANTS))
@pytest.mark.parametrize("form", ["softmax", "epilogue"])
def test_fused_transform_outside_envelope_never_raw(cold_gpu_caches, form, name):
    """Bites on ba21e9d: outside the generic envelope the template must refuse (A-only,
    B-only and both casts; in-tile transposes; the canonical transposed-B layout)."""
    launch, c, raw, intended = _case(form, name, inside=False)
    _witness(launch, c, raw, intended)


@requires_gpu
@pytest.mark.parametrize("name", ["castA", "castB", "castAB"])
@pytest.mark.parametrize("form", ["softmax", "epilogue"])
def test_fused_cast_inside_envelope_declines_and_computes(cold_gpu_caches, route_spies, form, name):
    """Bites on ba21e9d (template computed raw): 32^3 is generic-eligible — the kernel must
    decline to the generic lowerer, which replays the casts and matches the oracle."""
    launch, c, raw, intended = _case(form, name, inside=True)
    assert _witness(launch, c, raw, intended) == "computed"
    assert route_spies[1] and not route_spies[0], "expected the generic route, not the fused template"


@requires_gpu
@pytest.mark.parametrize("form", ["softmax", "epilogue"])
def test_fused_fp32_positive_keeps_template(cold_gpu_caches, route_spies, form):
    """Positive: fp32 fused kernels (no transform on the operand paths) stay on the template."""
    M, N, K = 32, 64, 32
    torch.manual_seed(3)
    x = torch.randn(M, K, device="mps") * 0.5
    y = torch.randn(K, N, device="mps") * 0.5
    b = torch.randn(N, device="mps") * 0.5
    c = torch.full((M, N), float("nan"), device="mps")
    if form == "softmax":
        _fused_softmax[(1,)](x, y, c, M=M, N=N, K=K, **_OFF, num_warps=4)
        ref = torch.softmax(x @ y, 1)
    else:
        _fused_epilogue[(1,)](x, y, b, c, M=M, N=N, K=K, **_OFF, num_warps=4)
        ref = torch.relu(x @ y + b)
    torch.mps.synchronize()
    assert route_spies[0] and not route_spies[1]
    torch.testing.assert_close(c, ref, rtol=1e-4, atol=1e-4)


_ROLE_MATCH = {"castA": "dot operand A", "castB": "dot operand B", "castAB": "dot operand A",
               "transA": "dot operand A .*transposed", "transB": "dot operand B .*transposed",
               "transB-canonical": "dot operand B .*transposed"}


@requires
@pytest.mark.parametrize("name", list(_VARIANTS))
@pytest.mark.parametrize("form", ["softmax", "epilogue"])
def test_lowering_boundary_fused_transform_refuses(form, name):
    """Rule 9 (bites on ba21e9d): fresh TTGIR, outside the envelope, the fused detectors refuse
    at lowering with the role-naming reason — B-only forms must name B, transposes must say so."""
    M, N, K = _shape(name, inside=False)
    cex = {"M": M, "N": N, "K": K, **_flags(name)}
    if form == "softmax":
        lw = _direct_lowerer_for(_fused_softmax, {"X": "*fp32", "Y": "*fp32", "C": "*fp32"}, cex)
    else:
        lw = _direct_lowerer_for(_fused_epilogue, {"X": "*fp32", "Y": "*fp32", "B": "*fp32", "C": "*fp32"}, cex)
    with pytest.raises(MetalNonRecoverableError, match=_ROLE_MATCH[name]):
        lw.lower()
