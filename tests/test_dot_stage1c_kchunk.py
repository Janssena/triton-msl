"""Stage 1C characterization for the remaining S=64 ``test_dot`` rows.

This surface was frozen before the K-chunked implementation.  The recovery rows
are the exact 13 failures in the 2026-08-31 upstream ratchet: nine accumulator
adds and four chain dots.  A recovered row must both execute the generic dot
lowerer (one hit for an add, two for a chain) and match an independent torch
reference; a specialized fallback is not recovery credit.  The three S=64
softmax rows are already supported and pin route preservation separately.

The direct-lowering contract complements the GPU checks: a cached executable
cannot hide a refusal or an over-budget emitted kernel.  The 28 KiB ceiling
reserves 4 KiB of headroom under the project's 32 KiB threadgroup policy; the
planned fp32 KC=16 input chunks plus persistent fp32 result use 24 KiB before
small epilogue temporaries.
"""

import os
import re

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
    def _dot_s64(X, Y, W, Z, S: tl.constexpr, EP: tl.constexpr, OUT_HALF: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        if OUT_HALF:
            z = tl.dot(x, y, out_dtype=tl.float16)
        else:
            z = tl.dot(x, y, out_dtype=tl.float32)
        if EP == 1:  # add-matrix, aliasing the output exactly like upstream
            z += tl.load(Z + om[:, None] * S + on[None, :])
        if EP == 2:  # add-rows, aliasing column zero of the output
            z += tl.load(Z + om * S)[:, None]
        if EP == 3:  # add-cols, aliasing row zero of the output
            z += tl.load(Z + on)[None, :]
        if EP == 4:  # canonical softmax: already template-backed at S=64
            zmax = tl.max(z, 1)
            z = z - zmax[:, None]
            num = tl.exp(z.to(tl.float32)).to(zmax.dtype)
            den = tl.sum(num, 1)
            z = num / den[:, None]
        if EP == 5:  # chain-dot, W is deliberately distinct from X/Y/Z
            w = tl.load(W + on[:, None] * S + on[None, :])
            if OUT_HALF:
                z = tl.dot(z.to(w.dtype), w, out_dtype=tl.float16)
            else:
                z = tl.dot(z.to(w.dtype), w, out_dtype=tl.float32)
        tl.store(Z + om[:, None] * S + on[None, :], z)

    @triton.jit
    def _dot_k_holdout(X, Y, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        y = tl.load(Y + ok[:, None] * N + on[None, :])
        acc = tl.load(Z + om[:, None] * N + on[None, :])
        tl.store(Z + om[:, None] * N + on[None, :], tl.dot(x, y, acc))

    @triton.jit
    def _dot_extra_after_output(X, Y, Z, EXTRA, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        acc = tl.load(Z + om[:, None] * S + on[None, :])
        tl.store(Z + om[:, None] * S + on[None, :], tl.dot(x, y, acc))

    @triton.jit
    def _dot_physical_b_transpose(X, Y_NK, Z, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        # Logical [K,N] B from physical [N,K], expressed in address dataflow
        # rather than through a tt.trans operation.
        y = tl.load(Y_NK + on[None, :] * S + ok[:, None])
        acc = tl.load(Z + om[:, None] * S + on[None, :])
        tl.store(Z + om[:, None] * S + on[None, :], tl.dot(x, y, acc))

    @triton.jit
    def _dot_plus_one_s64(X, Y, Z, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(Z + om[:, None] * S + on[None, :], tl.dot(x, y) + 1.0)

    @triton.jit
    def _chain_with_intermediate_store(X, Y, W, TMP, Z, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        first = tl.dot(x, y)
        tl.store(TMP + om[:, None] * S + on[None, :], first)
        w = tl.load(W + on[:, None] * S + on[None, :])
        second = tl.dot(first.to(w.dtype), w)
        tl.store(Z + om[:, None] * S + on[None, :], second)


_RECOVERY_ROWS = [
    pytest.param(ep, in_dt, out_dt, id=f"{name}-{in_dt}-to-{out_dt}")
    for ep, name, dtypes in (
        (1, "add-matrix", (("float16", "float16"), ("float16", "float32"), ("float32", "float32"))),
        (2, "add-rows", (("float16", "float16"), ("float16", "float32"), ("float32", "float32"))),
        (3, "add-cols", (("float16", "float16"), ("float16", "float32"), ("float32", "float32"))),
        (
            5,
            "chain-dot",
            (
                ("bfloat16", "float32"),
                ("float16", "float16"),
                ("float16", "float32"),
                ("float32", "float32"),
            ),
        ),
    )
    for in_dt, out_dt in dtypes
]

_SOFTMAX_ROWS = [
    pytest.param("float16", "float16", id="softmax-float16-to-float16"),
    pytest.param("float16", "float32", id="softmax-float16-to-float32"),
    pytest.param("float32", "float32", id="softmax-float32-to-float32"),
]


def _reference(x, y, w, z0, ep, out_dt):
    # Match tl.dot's declared output precision before the chain/epilogue.  All
    # arithmetic thereafter is evaluated independently in torch.
    z = x.float() @ y.float()
    # Upstream deliberately keeps the SOFTMAX dot in fp32 even when Z is fp16
    # (test_core.py excludes softmax from its float16-dot out_dtype branch).
    if out_dt == "float16" and ep != 4:
        z = z.half()
    if ep == 1:
        z = z + z0
    elif ep == 2:
        z = z + z0[:, 0][:, None]
    elif ep == 3:
        z = z + z0[0, :][None, :]
    elif ep == 4:
        z = torch.softmax(z.float(), dim=-1).to(z0.dtype)
    elif ep == 5:
        z = z.to(w.dtype).float() @ w.float()
        if out_dt == "float16":
            z = z.half()
    return z.to(z0.dtype)


@pytest.fixture()
def cold_gpu_caches(tmp_path, monkeypatch):
    """Force this session through this checkout's compiler and lowerer."""
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    os.makedirs(tmp_path / "msl", exist_ok=True)
    if HAS:
        functions = (
            _dot_s64,
            _dot_k_holdout,
            _dot_extra_after_output,
            _dot_physical_b_transpose,
        )
        cache = getattr(_dot_s64, "device_caches", None)
        assert cache is not None
        for fn in functions:
            fn.device_caches.clear()
    return tmp_path


def _inputs(in_dt, out_dt, *, seed):
    torch.manual_seed(seed)
    idt = getattr(torch, in_dt)
    x = (torch.randn(64, 64, device="mps") * 0.1).to(idt)
    y = (torch.randn(64, 64, device="mps") * 0.1).to(idt)
    w = (torch.randn(64, 64, device="mps") * 0.1).to(idt)
    # Upstream's out_dtype parameter controls tl.dot's result type; the Z
    # allocation remains in_dtype.  In particular fp16->fp32 rows load an
    # fp16 accumulator, extend it for the dot, then store back to fp16.
    z = (1.0 + torch.randn(64, 64, device="mps") * 0.1).to(idt)
    return x, y, w, z


def _tolerances(in_dt, out_dt):
    if in_dt == "bfloat16":
        return 8e-2, 8e-2
    if in_dt == "float16" or out_dt == "float16":
        return 2e-2, 2e-2
    return 3e-4, 3e-4


@requires_gpu
@pytest.mark.parametrize(("ep", "in_dt", "out_dt"), _RECOVERY_ROWS)
def test_s64_upstream_recovery_rows_compute_on_generic_route(
    monkeypatch, cold_gpu_caches, ep, in_dt, out_dt
):
    hits = []
    original = generic_lowerer.GenericLowerer._lower_dot

    def _spy(self, ssa):
        hits.append(ssa.id)
        return original(self, ssa)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", _spy)
    # Route is a compile-time contract: assert it at direct lowering so an
    # executable cached earlier in a randomized session cannot hide or fake it.
    _direct_lowerer(ep, in_dt, out_dt).lower()
    assert len(hits) == (2 if ep == 5 else 1), "row escaped the intended generic dot route"
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", original)

    x, y, w, z = _inputs(in_dt, out_dt, seed=64000 + ep * 100)
    z0 = z.clone()
    _dot_s64[(1,)](x, y, w, z, S=64, EP=ep, OUT_HALF=out_dt == "float16", num_warps=4)
    torch.mps.synchronize()

    assert bool(torch.isfinite(z).all()), "NaN/Inf or an unwritten sentinel reached the output"
    ref = _reference(x, y, w, z0, ep, out_dt)
    atol, rtol = _tolerances(in_dt, out_dt)
    torch.testing.assert_close(z, ref, atol=atol, rtol=rtol)


@requires_gpu
@pytest.mark.parametrize(("in_dt", "out_dt"), _SOFTMAX_ROWS)
def test_s64_softmax_preserves_specialized_route(
    monkeypatch, cold_gpu_caches, in_dt, out_dt
):
    generic_hits = []
    specialized_hits = []
    original_generic = generic_lowerer.GenericLowerer._lower_dot
    original_specialized = generic_lowerer.GenericLowerer._lower_matmul_softmax_template

    def _generic_spy(self, ssa):
        generic_hits.append(ssa.id)
        return original_generic(self, ssa)

    def _specialized_spy(self, info):
        specialized_hits.append(dict(info))
        return original_specialized(self, info)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", _generic_spy)
    monkeypatch.setattr(
        generic_lowerer.GenericLowerer, "_lower_matmul_softmax_template", _specialized_spy
    )
    # As above, prove routing at lowering rather than through a possibly
    # precompiled runtime executable.
    _direct_lowerer(4, in_dt, out_dt).lower()
    assert generic_hits == []
    assert len(specialized_hits) == 1
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", original_generic)
    monkeypatch.setattr(
        generic_lowerer.GenericLowerer, "_lower_matmul_softmax_template", original_specialized
    )

    x, y, w, z = _inputs(in_dt, out_dt, seed=64400)
    z0 = z.clone()
    _dot_s64[(1,)](x, y, w, z, S=64, EP=4, OUT_HALF=False, num_warps=4)
    torch.mps.synchronize()

    assert bool(torch.isfinite(z).all())
    ref = _reference(x, y, w, z0, 4, out_dt)
    atol, rtol = _tolerances(in_dt, out_dt)
    torch.testing.assert_close(z, ref, atol=atol, rtol=rtol)


def _direct_lowerer(ep, in_dt, out_dt):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    dtype = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    src = ASTSource(
        fn=_dot_s64,
        signature={
            "X": f"*{dtype[in_dt]}",
            "Y": f"*{dtype[in_dt]}",
            "W": f"*{dtype[in_dt]}",
            "Z": f"*{dtype[in_dt]}",
        },
        # Upstream keeps the softmax dot in fp32 even when its parameterized
        # output/storage dtype is fp16.
        constexprs={"S": 64, "EP": ep, "OUT_HALF": ep != 4 and out_dt == "float16"},
    )
    context = ir.context()
    ir.load_dialects(context)
    mod = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    return generic_lowerer.GenericLowerer(walk_ttgir(mod, options), options)


def _direct_lowerer_for(fn, signature, constexprs):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    context = ir.context()
    ir.load_dialects(context)
    mod = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    return generic_lowerer.GenericLowerer(walk_ttgir(mod, options), options)


def _threadgroup_bytes(msl):
    widths = {"half": 2, "bfloat": 2, "float": 4, "int": 4, "uint": 4}
    total = 0
    for ty, count in re.findall(r"threadgroup\s+(half|bfloat|float|int|uint)\s+\w+\[(\d+)\]", msl):
        total += widths[ty] * int(count)
    return total


@requires
@pytest.mark.parametrize(
    ("ep", "in_dt", "out_dt"),
    [
        pytest.param(1, "float32", "float32", id="add-matrix-fp32"),
        pytest.param(2, "float16", "float32", id="add-rows-fp16"),
        pytest.param(3, "float16", "float16", id="add-cols-fp16"),
        pytest.param(5, "bfloat16", "float32", id="chain-dot-bf16"),
        pytest.param(5, "float32", "float32", id="chain-dot-fp32"),
    ],
)
def test_s64_recovery_lowering_is_cooperative_and_keeps_budget_headroom(ep, in_dt, out_dt):
    lowerer = _direct_lowerer(ep, in_dt, out_dt)
    msl = lowerer.lower()
    assert lowerer._actual_dispatch_threads == 1024
    assert "_de < 4096u" in msl and "_de += 1024u" in msl
    assert "_st < 4096u" in msl and "_st += 1024u" in msl
    assert _threadgroup_bytes(msl) == lowerer._dot_kchunk_declared_bytes
    assert lowerer._dot_kchunk_declared_bytes <= 28 * 1024
    # All participating threads reach every barrier: this deliberately
    # branch-free surface has no emitted conditional around the uniform
    # constant-bound K-chunk loops.  The core chain path emits at least five
    # synchronization points and a single dot at least two; dtype-conversion
    # paths may add another uniform barrier.
    assert "if (" not in msl
    assert msl.count("threadgroup_barrier") >= (5 if ep == 5 else 2)


@requires_gpu
@pytest.mark.parametrize("k", [8, 32])
def test_non64_k_holdout_uses_budgeted_cooperative_route(monkeypatch, cold_gpu_caches, k):
    """Non-64 K, including K<KC, proves semantic admission and zero-fill."""
    hits = []
    original = generic_lowerer.GenericLowerer._lower_dot

    def _spy(self, ssa):
        hits.append(ssa.id)
        return original(self, ssa)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", _spy)
    _direct_lowerer_for(
        _dot_k_holdout,
        {"X": "*fp32", "Y": "*fp32", "Z": "*fp32"},
        {"M": 64, "N": 64, "K": k},
    ).lower()
    assert len(hits) == 1
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", original)

    torch.manual_seed(6400 + k)
    x = torch.randn(64, k, device="mps") * 0.1
    y = torch.randn(k, 64, device="mps") * 0.1
    z = 1.0 + torch.randn(64, 64, device="mps") * 0.1
    z0 = z.clone()
    _dot_k_holdout[(1,)](x, y, z, M=64, N=64, K=k, num_warps=4)
    torch.mps.synchronize()
    assert bool(torch.isfinite(z).all())
    torch.testing.assert_close(z, x @ y + z0, atol=3e-4, rtol=3e-4)


@requires_gpu
def test_output_pointer_need_not_be_last_argument(monkeypatch, cold_gpu_caches):
    hits = []
    original = generic_lowerer.GenericLowerer._lower_dot

    def _spy(self, ssa):
        hits.append(ssa.id)
        return original(self, ssa)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", _spy)
    _direct_lowerer_for(
        _dot_extra_after_output,
        {"X": "*fp32", "Y": "*fp32", "Z": "*fp32", "EXTRA": "*fp32"},
        {"S": 64},
    ).lower()
    assert len(hits) == 1
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", original)

    torch.manual_seed(6404)
    x = torch.randn(64, 64, device="mps") * 0.1
    y = torch.randn(64, 64, device="mps") * 0.1
    z = 1.0 + torch.randn(64, 64, device="mps") * 0.1
    z0 = z.clone()
    extra = torch.full((64, 64), 7.0, device="mps")
    _dot_extra_after_output[(1,)](x, y, z, extra, S=64, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(z, x @ y + z0, atol=3e-4, rtol=3e-4)
    assert bool((extra == 7.0).all()), "trailing pointer canary was mutated"


@requires_gpu
def test_physical_b_transpose_is_resolved_from_address_dataflow(monkeypatch, cold_gpu_caches):
    hits = []
    original = generic_lowerer.GenericLowerer._lower_dot

    def _spy(self, ssa):
        hits.append(ssa.id)
        return original(self, ssa)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", _spy)
    _direct_lowerer_for(
        _dot_physical_b_transpose,
        {"X": "*fp32", "Y_NK": "*fp32", "Z": "*fp32"},
        {"S": 64},
    ).lower()
    assert len(hits) == 1
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", original)

    torch.manual_seed(6464)
    x = torch.randn(64, 64, device="mps") * 0.1
    y_nk = torch.randn(64, 64, device="mps") * 0.1
    z = 1.0 + torch.randn(64, 64, device="mps") * 0.1
    z0 = z.clone()
    _dot_physical_b_transpose[(1,)](x, y_nk, z, S=64, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(z, x @ y_nk.T + z0, atol=3e-4, rtol=3e-4)


@requires
def test_value_changing_epilogue_stays_outside_kchunk_plan():
    lowerer = _direct_lowerer_for(
        _dot_plus_one_s64,
        {"X": "*fp32", "Y": "*fp32", "Z": "*fp32"},
        {"S": 64},
    )
    plan = getattr(lowerer, "_dot_kchunk_plan", lambda: None)()
    assert plan is None
    # Triton folds the trailing +1 into tt.dot's accumulator operand; the
    # existing non-zero-accumulator refusal is therefore the correct boundary.
    with pytest.raises(MetalNonRecoverableError, match="non-zero accumulator"):
        lowerer.lower()


@requires
def test_chain_intermediate_with_second_store_stays_outside_kchunk_plan():
    lowerer = _direct_lowerer_for(
        _chain_with_intermediate_store,
        {"X": "*fp32", "Y": "*fp32", "W": "*fp32", "TMP": "*fp32", "Z": "*fp32"},
        {"S": 64},
    )
    plan = getattr(lowerer, "_dot_kchunk_plan", lambda: None)()
    assert plan is None
    with pytest.raises(MetalNonRecoverableError, match="chain-dot"):
        lowerer.lower()
