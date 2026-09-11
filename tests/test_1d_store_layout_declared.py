"""Every producer of a 1-D value in a 2-D kernel must DECLARE its thread-to-element
layout, so the store never has to guess.

Background (2026-08-26/27). Three separate silent-wrongs had one root cause: the store
inferred a 1-D value's layout from its SIZE.

  * axis=0 argmin/argmax on a square tile -- size N vs size M is ambiguous when M == N,
    so the store used the axis=1 rule and broadcast column 0's index (reduce-probe #6)
  * axis=1 argmin/argmax with both results consumed -- the value was redistributed by a
    convert_layout and the index was not, but the store read a KERNEL-WIDE flag, so both
    took the simple layout and the index broadcast row 0's (reduce-probe #5)
  * tt.split of an (N,2) input -- size N equals the input's first dim, so the store used
    the blocked rule and output[k] got source element 2k (re-audit #8)

All three were closed by REFUSING. They are now fixed by having the producer declare
(``_register_1d_layout``) and the store honour that declaration.

Refusing an undeclared layout is now the DEFAULT (flipped 2026-08-27). The first flip
attempt was measured with ``TRITON_MSL_WARN_UNDECLARED_LAYOUT=1``, which UNDERCOUNTED
(pytest captures stdout/stderr for passing tests, so the warn print reads as zero even
when the path is hit) and regressed 51 upstream ``test_reduce`` cases -- traced to the
multipass-reduce and 1-D argminmax full-reduce paths, plus atomic_rmw's per-thread
result (which never got an ``env_shapes`` entry at all). All now declare. Re-measured
with the refusal live (an exception cannot be captured away): the full upstream
conformance suite (9,342 tests) and the project suite both hit ZERO undeclared cases,
so refusing costs nothing observable. ``TRITON_MSL_INFER_LAYOUT=1`` is the opt-out
for a kernel this project's test surface doesn't cover (mirrors ``TRITON_MSL_LEGACY=1``).

These tests run under the DEFAULT (refuse), so a producer that stops declaring fails
here directly -- no special mode needed.

Measuring note, preserved because it bit twice: count undeclared cases with STRICT,
never with the WARN print -- pytest captures stdout/stderr for passing tests, so a
warn-based count reads as zero even when the path is hit thousands of times.
"""

import os

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@pytest.fixture()
def strict_layout():
    """No-op: refusing an undeclared layout is now the DEFAULT.

    Kept as a marker on the tests that specifically exercise that contract, and it
    clears any inherited opt-out so the default under test is the real default.
    """
    prev = os.environ.pop("TRITON_MSL_INFER_LAYOUT", None)
    try:
        yield
    finally:
        if prev is not None:
            os.environ["TRITON_MSL_INFER_LAYOUT"] = prev


@triton.jit
def _reduce_axis(x, o, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    v = tl.load(x + rm[:, None] * N + rn[None, :])
    s = tl.sum(v, axis=AXIS)
    tl.store(o + (rm if AXIS == 1 else rn), s)


@requires_mps
@pytest.mark.parametrize("M,N", [(16, 32), (32, 16), (32, 32)])
@pytest.mark.parametrize("axis", [0, 1])
def test_2d_reduce_declares_layout(M, N, axis, strict_layout):
    o_size = M if axis == 1 else N
    x = torch.randn(M, N, device="mps", dtype=torch.float32)
    o = torch.zeros(o_size, device="mps", dtype=torch.float32)
    _reduce_axis[(1,)](x, o, M=M, N=N, AXIS=axis)
    torch.mps.synchronize()
    assert torch.allclose(o.cpu(), x.cpu().sum(dim=axis), atol=1e-3), (
        f"2-D reduce axis={axis} wrong at {M}x{N}")


@triton.jit
def _argmax_axis(x, oi, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    v = tl.load(x + rm[:, None] * N + rn[None, :])
    idx = tl.argmax(v, axis=AXIS)
    tl.store(oi + (rm if AXIS == 1 else rn), idx)


@requires_mps
@pytest.mark.parametrize("M,N", [(16, 32), (32, 16), (32, 32)])
@pytest.mark.parametrize("axis", [0, 1])
def test_2d_argmax_declares_layout(M, N, axis, strict_layout):
    # Includes the square (M == N) tile that used to refuse for axis=0.
    o_size = M if axis == 1 else N
    x = torch.randn(M, N, device="mps", dtype=torch.float32)
    oi = torch.zeros(o_size, device="mps", dtype=torch.int32)
    _argmax_axis[(1,)](x, oi, M=M, N=N, AXIS=axis)
    torch.mps.synchronize()
    assert oi.cpu().tolist() == x.cpu().argmax(dim=axis).tolist(), (
        f"argmax axis={axis} index wrong at {M}x{N}")


@triton.jit
def _argmax_both(x, ov, oi, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    v, i = tl.max(tl.load(x + rm[:, None] * N + rn[None, :]), axis=1, return_indices=True)
    tl.store(ov + rm, v)
    tl.store(oi + rm, i)


@requires_mps
@pytest.mark.parametrize("M,N", [(32, 16), (32, 32)])
def test_argmax_both_results_declare_layout(M, N, strict_layout):
    # The value is redistributed by a convert_layout and the index is not; both must
    # still resolve their own layout rather than sharing a kernel-wide flag.
    x = torch.randn(M, N, device="mps", dtype=torch.float32)
    ov = torch.zeros(M, device="mps", dtype=torch.float32)
    oi = torch.zeros(M, device="mps", dtype=torch.int32)
    _argmax_both[(1,)](x, ov, oi, M=M, N=N)
    torch.mps.synchronize()
    rv, ri = x.cpu().max(dim=1)
    assert torch.allclose(ov.cpu(), rv, atol=1e-5), "value wrong"
    assert oi.cpu().tolist() == ri.tolist(), "index wrong"


@triton.jit
def _split_pairs(x, a, b, N: tl.constexpr):
    i = tl.arange(0, N)
    lo, hi = tl.split(tl.load(x + i[:, None] * 2 + tl.arange(0, 2)[None, :]))
    tl.store(a + i, lo)
    tl.store(b + i, hi)


@requires_mps
@pytest.mark.parametrize("N", [8, 32, 256])
def test_split_declares_layout(N, strict_layout):
    x = torch.arange(2 * N, device="mps", dtype=torch.float32)
    a = torch.zeros(N, device="mps", dtype=torch.float32)
    b = torch.zeros(N, device="mps", dtype=torch.float32)
    _split_pairs[(1,)](x, a, b, N=N)
    torch.mps.synchronize()
    assert torch.allclose(a.cpu(), x.cpu()[0::2]), f"split even wrong at N={N}"
    assert torch.allclose(b.cpu(), x.cpu()[1::2]), f"split odd wrong at N={N}"


@triton.jit
def _hist_2d(x, o, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    v = tl.load(x + rm[:, None] * N + rn[None, :])
    tl.store(o + tl.arange(0, B), tl.histogram(tl.reshape(v, (M * N,)), B))


@requires_mps
def test_histogram_declares_layout(strict_layout):
    M, N, B = 8, 8, 8
    x = torch.full((M, N), 3, device="mps", dtype=torch.int32)
    o = torch.zeros(B, device="mps", dtype=torch.int32)
    # This supported source must reach the layout/store and compute. An error
    # elsewhere cannot supply evidence that the producer declared its layout.
    _hist_2d[(1,)](x, o, M=M, N=N, B=B)
    torch.mps.synchronize()
    exp = [0] * B
    exp[3] = M * N
    assert o.cpu().tolist() == exp


def test_undeclared_layout_default_raises_and_infer_opt_out_does_not():
    """Pin both directions of the (now-default) refusal.

    Exercised directly rather than through a kernel, because there is deliberately no
    undeclared producer left in either test suite -- a test that needed one would stop
    testing anything the moment the last straggler was fixed.
    """
    import triton_msl.codegen.generic_lowerer as gl

    from triton_msl.errors import MetalNonRecoverableError

    def _probe():
        lowerer = gl.GenericLowerer.__new__(gl.GenericLowerer)
        lowerer.graph = type("G", (), {"func_name": "probe"})()
        lowerer._effective_2d_shape = (8, 8)
        return lowerer

    prev = os.environ.get("TRITON_MSL_INFER_LAYOUT")
    try:
        os.environ.pop("TRITON_MSL_INFER_LAYOUT", None)
        with pytest.raises(MetalNonRecoverableError):
            _probe()._report_undeclared_layout("size-inference", 8)
        # The opt-out must NOT raise -- it exists precisely for a kernel this project's
        # test surface does not cover.
        os.environ["TRITON_MSL_INFER_LAYOUT"] = "1"
        _probe()._report_undeclared_layout("size-inference", 8)
    finally:
        if prev is None:
            os.environ.pop("TRITON_MSL_INFER_LAYOUT", None)
        else:
            os.environ["TRITON_MSL_INFER_LAYOUT"] = prev


@triton.jit
def _mixed_direct_blocked_layout(pairs, matrix, out, N: tl.constexpr, K: tl.constexpr):
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    pair_cols = tl.arange(0, 2)
    pair_values = tl.load(pairs + rn[:, None] * 2 + pair_cols[None, :])
    direct, _ = tl.split(pair_values)
    tile = tl.load(matrix + rn[:, None] * K + rk[None, :])
    blocked = tl.sum(tile, axis=1)
    tl.store(out + rn, direct + blocked)


@requires_mps
def test_mixed_direct_and_blocked_layout_refuses(strict_layout):
    """A known DIRECT + BLOCKED elementwise result has no coherent mapping.

    Pre-fix, the resolver returned None for the conflict but convert_layout overrode
    that result from the one visible reduce provenance, staged the whole expression as
    BLOCKED, and silently used pair row 0 for every output row (GPU err 1.8633).
    """
    torch.manual_seed(20260830)
    N, K = 16, 32
    pairs = torch.randn(N, 2, device="mps")
    matrix = torch.randn(N, K, device="mps")
    out = torch.zeros(N, device="mps")
    with pytest.raises(MetalNonRecoverableError, match="source layout is unresolved"):
        _mixed_direct_blocked_layout[(1,)](pairs, matrix, out, N=N, K=K)


@triton.jit
def _blocked_reduce_with_larger_2d_shape(
    small,
    wide,
    reduced,
    wide_out,
    M: tl.constexpr,
    K: tl.constexpr,
    W: tl.constexpr,
):
    rm = tl.arange(0, M)
    rk = tl.arange(0, K)
    rw = tl.arange(0, W)
    blocked = tl.sum(tl.load(small + rm[:, None] * K + rk[None, :]), axis=1)
    wide_values = tl.load(wide + rm[:, None] * W + rw[None, :])
    tl.store(wide_out + rm[:, None] * W + rw[None, :], wide_values + 0.25)
    tl.store(reduced + rm, blocked)


@requires_mps
def test_blocked_convert_uses_reduce_inner_dim_not_effective_shape(strict_layout):
    """The blocked divisor comes from its reduce input, not the dominant 2-D tile."""
    torch.manual_seed(20260830)
    M, K, W = 16, 8, 32
    small = torch.randn(M, K, device="mps")
    wide = torch.randn(M, W, device="mps")
    reduced = torch.zeros(M, device="mps")
    wide_out = torch.zeros_like(wide)
    _blocked_reduce_with_larger_2d_shape[(1,)](
        small, wide, reduced, wide_out, M=M, K=K, W=W
    )
    torch.mps.synchronize()
    torch.testing.assert_close(reduced, small.sum(dim=1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(wide_out, wide + 0.25, rtol=0, atol=0)


@triton.jit
def _scalar_atomic_result_splat(x, counter, out, M: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    tile = tl.load(x + rows[:, None] * N + cols[None, :])
    total = tl.sum(tl.sum(tile, axis=1), axis=0)
    old = tl.atomic_add(counter, total)
    tl.store(out + rows, old)


@requires_mps
def test_scalar_atomic_result_is_broadcast_before_direct_declaration(strict_layout):
    """A scalar atomic runs on lid 0, but its scalar return is uniform after splat.

    Registering the atomic result as DIRECT without first broadcasting lid 0's old
    value suppresses the undeclared-layout refusal and silently stores
    ``[old, 0, 0, ...]``.  This 2-D shape deliberately reaches that ambiguity gate.
    """
    M, N = 8, 8
    x = torch.ones((M, N), device="mps", dtype=torch.float32)
    counter = torch.tensor([7.0], device="mps", dtype=torch.float32)
    out = torch.full((M,), -1.0, device="mps", dtype=torch.float32)
    _scalar_atomic_result_splat[(1,)](x, counter, out, M=M, N=N)
    torch.mps.synchronize()
    assert counter.cpu().tolist() == [71.0]
    assert out.cpu().tolist() == [7.0] * M
