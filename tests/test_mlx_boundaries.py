"""Packet 176 (158 row 8) — the MLX route fails closed at every boundary it cannot honour.

`triton_msl.mlx.triton_call` re-launches this emitter's MSL through mx.fast.metal_kernel, so it
inherits every codegen fix of the campaign. Its launcher, though, binds the Triton arguments
positionally against the parsed kernel signature and allocates fresh outputs — which is silently
wrong for six shapes that now refuse before any dispatch:

  1. a lowering that set a dispatch descriptor the torch.mps driver honours (flash_attention,
     mm_two_kernel, fast_matmul, quant_matmul, batched_dot_bounds);
  2. an MSL holding more than one kernel (the body cut would swallow the second);
  3. a parsed signature whose argument count differs from the Triton signature (a packed or
     template-ordered ABI);
  4. an output pointer the kernel also reads or atomically updates (MLX outputs are fresh,
     uninitialised arrays) — and the old output_arg_indices=None mode, which made every pointer
     an output, refuses as soon as the kernel reads any pointer;
  5. an MLX array dtype the signature map does not know (it became *fp32 silently);
  6. a Python int outside int32 (it became i32 silently).

Not checkable from Python with MLX 0.32 (no strides / flags on mx.array): the caller's stride
arguments must describe the ROW-CONTIGUOUS layout MLX hands the kernel (ensure_row_contiguous). That
is a documented contract (docs row), not a refusal.
"""

import pytest

mx = pytest.importorskip("mlx.core")

import triton
import triton.language as tl

import triton_msl
import triton_msl.mlx as tmlx
from triton_msl.errors import MetalNonRecoverableError
from triton_msl.mlx.msl_extractor import extract_msl_for_mlx


@pytest.fixture
def cold_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    tmlx._compile_cache.clear()


@pytest.fixture
def no_launch(monkeypatch):
    """Any dispatch reaching the launcher is a failure of the boundary under test."""
    from triton_msl.mlx import mlx_launcher

    def _boom(self, grid, *args):
        raise AssertionError("the launcher was reached; the boundary did not refuse")

    monkeypatch.setattr(mlx_launcher.MLXLauncher, "__call__", _boom)
    monkeypatch.setattr(tmlx, "MLXLauncher", mlx_launcher.MLXLauncher)


# ---------------------------------------------------------------- 1. dispatch descriptors


@triton.jit
def _k_matmul(
    a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * sam + (k0 + rk)[None, :] * sak)
        b = tl.load(b_ptr + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc)


def _mlx_args_for(fn, cex):
    """Placeholder MLX arrays / scalars in signature order (a compile-only refusal never launches)."""
    args = []
    for n in fn.arg_names:
        if n in cex:
            continue
        if n.endswith("_ptr"):
            args.append(mx.zeros((8,), dtype=mx.uint8 if ("mask" in n or n == "m_ptr") else mx.float32))
        elif n in ("sm_scale", "neg_inf"):
            args.append(0.5)
        else:
            args.append(1)
    return args


def test_descriptor_kernels_refuse_before_launch(cold_caches, no_launch):
    """trifast's biased forward lowers to the flash_attention template (a route-only ABI with a
    packed scalar buffer on the torch.mps path); the MLX route refuses it by name, before the
    extractor and before any launch. Pre-176: the descriptor was ignored and the arguments bound
    positionally against the template's ABI."""
    import sys

    sys.path.insert(0, "tests")
    from test_fa_biased_routing import _biased_tri_fa

    fn = _biased_tri_fa
    cex = {n: (32 if n in ("DIM", "BLOCK_J", "BLOCK_K") else 256) for n in [fn.arg_names[i] for i in fn.constexprs]}
    with pytest.raises(MetalNonRecoverableError, match="'flash_attention' dispatch descriptor"):
        tmlx.triton_call(fn, *_mlx_args_for(fn, cex), grid=(1, 1, 1), **cex)


def test_matmul_refuses_on_the_read_output_boundary(cold_caches, no_launch):
    """A plain fp32 matmul refuses too: its compile metadata carries no output list on this
    lowering path (`output_arg_indices` is None), so the extractor's conservative mode makes every
    pointer an output — and the kernel reads two of them. Refused, not bound blind (pre-176 the
    kernel would have run against fresh, uninitialised copies of A and B)."""
    a = mx.zeros((64, 64))
    b = mx.zeros((64, 64))
    c = mx.zeros((64, 64))
    with pytest.raises(MetalNonRecoverableError, match="MLX route"):
        tmlx.triton_call(_k_matmul, a, b, c, 64, 64, 64, 64, 1, 64, 1, 64, 1, grid=(1, 1), BM=64, BN=64, BK=32)


class _Meta:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.mark.parametrize(
    "name", ["flash_attention", "mm_two_kernel", "fast_matmul", "quant_matmul", "batched_dot_bounds", "device_assert"]
)
def test_each_unsupported_descriptor_is_named(monkeypatch, cold_caches, no_launch, name):
    """Each descriptor the table names refuses, with the descriptor in the message (the compile is
    stubbed to return metadata carrying just that descriptor)."""
    msl = "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(volatile device float* x [[buffer(0)]], volatile device float* o [[buffer(1)]], uint tid [[thread_position_in_grid]]) {\n    o[tid] = x[tid];\n}\n"
    meta = _Meta(block_size=32, output_arg_indices=[1], needs_2d_grid=False, **{name: {"any": 1}})
    monkeypatch.setattr(tmlx, "_compile_kernel", lambda fn, sig, cex: (msl, meta, None))

    @triton.jit
    def _k(x_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(o_ptr + offs, tl.load(x_ptr + offs))

    with pytest.raises(MetalNonRecoverableError, match=name):
        tmlx.triton_call(_k, mx.zeros((32,)), mx.zeros((32,)), grid=(1,), BLOCK=32)


# ---------------------------------------------------------------- 2–4. the extractor

_ONE = """#include <metal_stdlib>
using namespace metal;

kernel void k(
    volatile device float* x [[buffer(0)]],
    volatile device float* o [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) { o[tid] = x[tid]; }
}
"""


def test_two_kernels_refuse():
    two = _ONE + _ONE.replace("kernel void k(", "kernel void k2(")
    with pytest.raises(MetalNonRecoverableError, match="holds 2 kernels"):
        extract_msl_for_mlx(two, output_arg_indices=[1])


def test_argument_count_mismatch_refuses():
    """The Triton signature says 4 runtime arguments; the parsed MSL has 2 pointers + 1 scalar."""
    with pytest.raises(MetalNonRecoverableError, match="3 scalar|1 scalar"):
        extract_msl_for_mlx(_ONE, output_arg_indices=[1], expected_args=4)
    ext = extract_msl_for_mlx(_ONE, output_arg_indices=[1], expected_args=3)  # control
    assert ext.output_names == ["o"] and ext.input_names == ["x"]


@pytest.mark.parametrize(
    "body,what",
    [
        ("    o[tid] = o[tid] + x[tid];", "accumulating store"),
        (
            "    device atomic_uint* p = (device atomic_uint*)(o + tid); atomic_fetch_add_explicit(p, 1u, memory_order_relaxed);",
            "atomic on the output",
        ),
        ("    float prev = o[tid];\n    o[tid] = prev * x[tid];", "in-place read then store"),
        ("    if (o[tid] > 0.0f) { o[tid] = x[tid]; }", "output read in a condition"),
    ],
)
def test_output_that_is_read_refuses(body, what):
    msl = _ONE.replace("    if (tid < n) { o[tid] = x[tid]; }", body)
    with pytest.raises(MetalNonRecoverableError, match="reads .* output pointer 'o'"):
        extract_msl_for_mlx(msl, output_arg_indices=[1])


def test_vectorized_store_through_an_alias_is_accepted():
    """Control: the softmax template stores through `device float4* o4 = (device float4*)(o + base);`
    — an alias used only for stores is a store (the first cut of the rule refused it)."""
    msl = _ONE.replace(
        "    if (tid < n) { o[tid] = x[tid]; }",
        "    device float4* o4 = (device float4*)(o + tid * 4u);\n    o4[0] = float4(x[tid]);\n    o[tid] = x[tid];",
    )
    ext = extract_msl_for_mlx(msl, output_arg_indices=[1])
    assert ext.output_names == ["o"]


def test_read_through_an_alias_refuses():
    msl = _ONE.replace(
        "    if (tid < n) { o[tid] = x[tid]; }",
        "    device float* p = o + tid;\n    float prev = p[0];\n    p[0] = prev + x[tid];",
    )
    with pytest.raises(MetalNonRecoverableError, match="reads .* output pointer 'o'"):
        extract_msl_for_mlx(msl, output_arg_indices=[1])


def test_plain_store_output_is_accepted():
    """Control: `o[...] = ...` stores only (several, masked, in a comment mentioning o) pass."""
    msl = _ONE.replace(
        "    if (tid < n) { o[tid] = x[tid]; }",
        "    // o is the output\n    if (tid < n) { o[tid] = x[tid]; }\n    if (tid == 0u) { o[0] = 1.0f; }",
    )
    ext = extract_msl_for_mlx(msl, output_arg_indices=[1])
    assert ext.output_names == ["o"]


def test_conservative_mode_refuses_when_any_pointer_is_read():
    """output_arg_indices=None makes every pointer an output; `x` is read, so refuse (the old
    behaviour handed the kernel an uninitialised `x`)."""
    with pytest.raises(MetalNonRecoverableError, match="output pointer 'x'"):
        extract_msl_for_mlx(_ONE, output_arg_indices=None)


# ---------------------------------------------------------------- 5–6. the signature map


@pytest.mark.parametrize("dtype", [mx.int64, mx.uint64, mx.complex64])
def test_unknown_dtypes_refuse(dtype):
    with pytest.raises(MetalNonRecoverableError, match="no Triton signature mapping"):
        tmlx._mlx_dtype_to_triton_sig(dtype)


def test_known_dtypes_map():
    assert tmlx._mlx_dtype_to_triton_sig(mx.bfloat16) == "*bf16" and tmlx._mlx_dtype_to_triton_sig(mx.uint8) == "*u8"


@pytest.mark.parametrize("val", [2**31, -(2**31) - 1, 10**12])
def test_ints_beyond_int32_refuse(val):
    with pytest.raises(MetalNonRecoverableError, match="does not fit int32"):
        tmlx._scalar_to_triton_sig(val)


def test_ints_within_int32_map():
    assert (
        tmlx._scalar_to_triton_sig(2**31 - 1) == "i32"
        and tmlx._scalar_to_triton_sig(-(2**31)) == "i32"
        and tmlx._scalar_to_triton_sig(True) == "i1"
    )


# ---------------------------------------------------------------- the route still works


@triton.jit
def _k_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


@pytest.mark.skipif(not tmlx.mlx_available(), reason="MLX metal_kernel needed")
def test_plain_kernel_still_launches(cold_caches):
    """Control: a kernel with no descriptor, one kernel, matching arguments, write-only output."""
    x = mx.arange(1024, dtype=mx.float32)
    y = mx.ones((1024,))
    out = mx.zeros((1024,))
    (r,) = tmlx.triton_call(_k_add, x, y, out, 1024, grid=(4,), BLOCK=256)
    mx.eval(r)
    assert mx.allclose(r, x + 1.0).item()
