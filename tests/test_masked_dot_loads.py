"""Correct-or-refuse surface for masked loads staged into ``tt.dot``.

The three upstream rows use a single flattened bounds comparison and differ
only by source dtype.  Recovery credit requires the generic ``tt.dot`` route:
the inline/prebuilt templates do not replay load masks.  Padded-stride,
reversed-comparison, scalar-``other``, shape, and argument-order holdouts keep
the implementation semantic rather than tailored to those three rows.

Unknown mask algebra is a negative control. It must continue to refuse rather
than reach a template that drops it.
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
    def _masked_dot(
        A,
        B,
        C,
        stride_a,
        stride_b,
        limit_a,
        limit_b,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        MASK_MODE: tl.constexpr,
        OTHER_MODE: tl.constexpr,
    ):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rk = tl.arange(0, K)
        off_a = rm[:, None] * stride_a + rk[None, :]
        off_b = rk[:, None] * stride_b + rn[None, :]
        if MASK_MODE == 0:
            mask_a = off_a < limit_a
            mask_b = off_b < limit_b
        elif MASK_MODE == 1:
            mask_a = limit_a > off_a
            mask_b = limit_b > off_b
        else:
            # Deliberately outside the single-comparison replay envelope.
            mask_a = (off_a & 1) == 0
            mask_b = (off_b & 1) == 0
        if OTHER_MODE == 0:
            a = tl.load(A + off_a, mask=mask_a)
            b = tl.load(B + off_b, mask=mask_b)
        elif OTHER_MODE == 1:
            a = tl.load(A + off_a, mask=mask_a, other=0.0)
            b = tl.load(B + off_b, mask=mask_b, other=0.0)
        else:
            a = tl.load(A + off_a, mask=mask_a, other=2.0)
            b = tl.load(B + off_b, mask=mask_b, other=2.0)
        out = tl.dot(a, b, out_dtype=tl.float32)
        tl.store(C + rm[:, None] * N + rn[None, :], out)

    @triton.jit
    def _masked_dot_output_first(
        C,
        EXTRA,
        B,
        A,
        stride_b,
        stride_a,
        limit_b,
        limit_a,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
    ):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rk = tl.arange(0, K)
        off_a = rm[:, None] * stride_a + rk[None, :]
        off_b = rk[:, None] * stride_b + rn[None, :]
        a = tl.load(A + off_a, mask=off_a < limit_a, other=0.0)
        b = tl.load(B + off_b, mask=off_b < limit_b, other=0.0)
        tl.store(C + rm[:, None] * N + rn[None, :], tl.dot(a, b, out_dtype=tl.float32))

    @triton.jit
    def _unmasked_dot(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rk = tl.arange(0, K)
        a = tl.load(A + rm[:, None] * K + rk[None, :])
        b = tl.load(B + rk[:, None] * N + rn[None, :])
        tl.store(C + rm[:, None] * N + rn[None, :], tl.dot(a, b, out_dtype=tl.float32))


_RECOVERY_ROWS = [
    # Exact upstream shape/spelling, including implicit other.
    pytest.param(32, 32, 16, "bfloat16", 0, 0, False, id="upstream-bf16"),
    pytest.param(32, 32, 16, "float16", 0, 0, False, id="upstream-fp16"),
    pytest.param(32, 32, 16, "float32", 0, 0, False, id="upstream-fp32"),
    # The guard-removal bite: runtime padding makes many lanes false.
    pytest.param(32, 32, 16, "float32", 0, 1, True, id="padded-explicit-zero"),
    # Same value path, distinct replay details.
    pytest.param(32, 32, 16, "float32", 0, 2, True, id="padded-scalar-other"),
    pytest.param(32, 32, 16, "float32", 1, 1, True, id="reversed-comparison"),
    # Shape + pointer-order holdout: no exact-upstream argument/extent allowlist.
    pytest.param(16, 32, 8, "float32", 0, 1, True, id="shape-and-output-order-holdout"),
]


def _dtype_sig(dtype):
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}[dtype]


def _direct_lowerer(
    *,
    m=32,
    n=32,
    k=16,
    dtype="float32",
    mask_mode=0,
    other_mode=1,
    output_first=False,
    unmasked=False,
):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    ptr = f"*{_dtype_sig(dtype)}"
    if unmasked:
        fn = _unmasked_dot
        signature = {"A": ptr, "B": ptr, "C": ptr}
        constexprs = {"M": m, "N": n, "K": k}
    elif output_first:
        fn = _masked_dot_output_first
        signature = {
            "C": ptr,
            "EXTRA": ptr,
            "B": ptr,
            "A": ptr,
            "stride_b": "i32",
            "stride_a": "i32",
            "limit_b": "i32",
            "limit_a": "i32",
        }
        constexprs = {"M": m, "N": n, "K": k}
    else:
        fn = _masked_dot
        signature = {
            "A": ptr,
            "B": ptr,
            "C": ptr,
            "stride_a": "i32",
            "stride_b": "i32",
            "limit_a": "i32",
            "limit_b": "i32",
        }
        constexprs = {
            "M": m,
            "N": n,
            "K": k,
            "MASK_MODE": mask_mode,
            "OTHER_MODE": other_mode,
        }
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


@pytest.fixture()
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    os.makedirs(tmp_path / "msl", exist_ok=True)
    if HAS:
        for fn in (_masked_dot, _masked_dot_output_first, _unmasked_dot):
            cache = getattr(fn, "device_caches", None)
            assert cache is not None
            cache.clear()
    return tmp_path


def _logical_tile(flat, rows, cols, stride, limit, mask_mode, other):
    rr = torch.arange(rows, device=flat.device)[:, None]
    cc = torch.arange(cols, device=flat.device)[None, :]
    offsets = rr * stride + cc
    if mask_mode in (0, 1):
        valid = offsets < limit
    else:
        valid = (offsets & 1) == 0
    safe = torch.where(valid, offsets, torch.zeros_like(offsets))
    fill = torch.full((rows, cols), other, dtype=flat.dtype, device=flat.device)
    return torch.where(valid, flat[safe], fill)


@requires_gpu
@pytest.mark.parametrize(
    ("m", "n", "k", "dtype", "mask_mode", "other_mode", "padded"),
    _RECOVERY_ROWS,
)
def test_replayable_masked_dot_uses_generic_route_and_computes(
    monkeypatch,
    cold_gpu_caches,
    m,
    n,
    k,
    dtype,
    mask_mode,
    other_mode,
    padded,
):
    output_first = m == 16
    hits = []
    original = generic_lowerer.GenericLowerer._lower_dot

    def _spy(self, ssa):
        hits.append(ssa.id)
        return original(self, ssa)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", _spy)
    lowerer = _direct_lowerer(
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        mask_mode=mask_mode,
        other_mode=other_mode,
        output_first=output_first,
    )
    msl = lowerer.lower()
    assert len(hits) == 1, "masked dot escaped to a template that does not replay its mask"
    assert re.search(r"smem_\d+\[_sa\] = .*\?.*:", msl)
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", original)

    torch.manual_seed(5900 + m + n + k + other_mode)
    tdt = getattr(torch, dtype)
    stride_a = k + 4 if padded else k
    stride_b = n + 8 if padded else n
    len_a = (m - 1) * stride_a + k
    len_b = (k - 1) * stride_b + n
    a = (torch.randn(len_a, device="mps") * 0.1).to(tdt)
    b = (torch.randn(len_b, device="mps") * 0.1).to(tdt)
    # Make a dropped mask loudly different while keeping all physical reads in-bounds.
    if padded:
        limit_a = m * k
        limit_b = k * n
        a[limit_a:] = 7
        b[limit_b:] = -5
    else:
        limit_a = m * k
        limit_b = k * n
    out = torch.full((m, n), float("nan"), dtype=tdt, device="mps")
    extra = torch.full((17,), 23.0, dtype=tdt, device="mps")
    if output_first:
        _masked_dot_output_first[(1,)](
            out,
            extra,
            b,
            a,
            stride_b,
            stride_a,
            limit_b,
            limit_a,
            M=m,
            N=n,
            K=k,
            num_warps=4,
        )
    else:
        _masked_dot[(1,)](
            a,
            b,
            out,
            stride_a,
            stride_b,
            limit_a,
            limit_b,
            M=m,
            N=n,
            K=k,
            MASK_MODE=mask_mode,
            OTHER_MODE=other_mode,
            num_warps=4,
        )
    torch.mps.synchronize()

    other = 2.0 if other_mode == 2 else 0.0
    ref_a = _logical_tile(a, m, k, stride_a, limit_a, mask_mode, other)
    ref_b = _logical_tile(b, k, n, stride_b, limit_b, mask_mode, other)
    ref = (ref_a.float() @ ref_b.float()).to(tdt)
    assert bool(torch.isfinite(out).all())
    atol, rtol = (5e-2, 5e-2) if dtype == "bfloat16" else (2e-2, 2e-2) if dtype == "float16" else (3e-4, 3e-4)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    assert bool((extra == 23).all()), "trailing argument canary was mutated"


@requires
def test_unreplayable_mask_refuses():
    lowerer = _direct_lowerer(mask_mode=2, other_mode=1)
    with pytest.raises(
        MetalNonRecoverableError,
        match=r"masked.*(not supported|cannot be safely reconstructed|per-element)",
    ):
        lowerer.lower()


@requires
def test_unmasked_dot_keeps_inline_template_route(monkeypatch):
    generic_hits = []
    inline_hits = []
    original_generic = generic_lowerer.GenericLowerer._lower_dot
    original_inline = generic_lowerer.GenericLowerer._lower_simple_dot_inline

    def _generic_spy(self, ssa):
        generic_hits.append(ssa.id)
        return original_generic(self, ssa)

    def _inline_spy(self, info):
        inline_hits.append(dict(info))
        return original_inline(self, info)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", _generic_spy)
    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_simple_dot_inline", _inline_spy)
    _direct_lowerer(unmasked=True).lower()
    assert generic_hits == []
    assert len(inline_hits) == 1
