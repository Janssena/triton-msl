"""Regression: EVERY stored-to pointer arg must be in the host copy-back metadata.

Background (2026-08-29, GPT packet 036 root cause + packet 037 generality proof).
``_prescan_stores`` resolved store pointers to func args with a private walk that only
linked ``tt.addptr``/``tt.splat`` and gave up after 5 hops. A 2-D output address crosses
``tt.broadcast`` and runs deeper, so it never resolved. In a MULTI-output kernel that
produced a PARTIAL ``output_arg_indices`` — and the host dispatch path faithfully obeyed
it: the 2-D output was computed correctly on the GPU and then never copied back. The
caller saw whatever was in host memory before the launch (GPU-measured: 1-D output at
1e-6, 2-D output UNTOUCHED).

Hidden for so long because it needs the intersection of: multi-output AND the host path
(zero-copy shares memory, so no copy-back exists to skip) AND a broadcast-crossing
chain. Single-output kernels were safe only because TOTAL resolution failure yields an
empty set -> None -> copy-everything. It is also what made trifast #6b's "O err 0.848"
look like an unexplained codegen fault: O was always computed correctly, just never
copied back.

The fix routes resolution through ``_trace_ptr_source`` (cycle-safe, all pass-throughs)
and treats PARTIAL knowledge as NO knowledge (any unresolved store pointer -> copy-all
fallback). The tracer is more permissive than the old walk, so the input-safety pin
below guards the opposite failure: an INPUT misidentified as an output would have the
driver copy device memory back over the caller's input tensor.
"""

import pytest
import torch
import triton
import triton.language as tl

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _two_outputs(X, O2D, O1D, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(X + rm[:, None] * N + rn[None, :])
    # 2-D store: pointer chain crosses expand_dims + broadcast (deep chain)
    tl.store(O2D + rm[:, None] * N + rn[None, :], x * 2.0)
    # 1-D store: short chain (this one always resolved)
    tl.store(O1D + rm, tl.sum(x, axis=1))


def _metadata_indices():
    """The output_arg_indices the lowerer just produced, via a lowering spy."""
    import triton_msl.codegen.generic_lowerer as gl

    box = {}
    real = gl.GenericLowerer.get_output_arg_indices

    def spy(self):
        out = real(self)
        box["indices"] = out
        return out

    return box, spy


@requires_mps
@pytest.mark.parametrize("host_path", [False, True], ids=["zero-copy", "host"])
def test_both_outputs_copied_back(monkeypatch, host_path):
    # Pre-fix on the host path: O2D stayed at its -1.0 fill (never copied back)
    # while O1D was correct — the partial-metadata silent-wrong.
    if host_path:
        monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    dev = "mps"
    M, N = 16, 32
    torch.manual_seed(0)
    x = torch.randn(M, N, device=dev)
    o2d = torch.full((M, N), -1.0, device=dev)
    o1d = torch.full((M,), -1.0, device=dev)
    _two_outputs[(1,)](x, o2d, o1d, M=M, N=N)
    torch.mps.synchronize()
    assert not bool((o2d == -1.0).all()), "2-D output was never copied back (stale fill)"
    assert (o2d - x * 2.0).abs().max().item() < 1e-5, "2-D output wrong"
    assert (o1d - x.sum(dim=1)).abs().max().item() < 1e-4, "1-D output wrong"


@requires_mps
def test_metadata_lists_every_stored_arg(monkeypatch):
    # Structural pin: the metadata itself names BOTH stored args (indices 1 and 2),
    # and does NOT name the input (index 0). Guards the mechanism, not just the
    # observable, so a future partial-resolution regression fails even on a lucky path.
    import triton_msl.codegen.generic_lowerer as gl

    box, spy = _metadata_indices()
    monkeypatch.setattr(gl.GenericLowerer, "get_output_arg_indices", spy)
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")

    @triton.jit
    def _two_outputs_fresh(X, A, B, M: tl.constexpr, N: tl.constexpr):
        # Private clone -> unique cache key -> the lowerer actually runs (a cache hit
        # would skip lowering and leave the spy empty; the order-dependence lesson).
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(X + rm[:, None] * N + rn[None, :])
        tl.store(A + rm[:, None] * N + rn[None, :], x + 1.0)
        tl.store(B + rm, tl.max(x, axis=1))

    dev = "mps"
    M, N = 16, 32
    x = torch.randn(M, N, device=dev)
    a = torch.zeros(M, N, device=dev)
    b = torch.zeros(M, device=dev)
    _two_outputs_fresh[(1,)](x, a, b, M=M, N=N)
    torch.mps.synchronize()
    idx = box.get("indices")
    assert idx is not None, "lowering did not run (cache hit?) — pin proves nothing"
    assert sorted(idx) == [1, 2], f"output_arg_indices must be [1, 2], got {idx}"


@requires_mps
def test_input_not_copied_back_over(monkeypatch):
    # Input safety: the more-permissive tracer must never mark a LOADED-only arg as an
    # output. If it did, the driver would copy the device buffer back over the caller's
    # input tensor. Assert the input is byte-identical after a host-path launch.
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    dev = "mps"
    M, N = 16, 32
    torch.manual_seed(1)
    x = torch.randn(M, N, device=dev)
    x_before = x.clone()
    o2d = torch.zeros(M, N, device=dev)
    o1d = torch.zeros(M, device=dev)
    _two_outputs[(1,)](x, o2d, o1d, M=M, N=N)
    torch.mps.synchronize()
    assert torch.equal(x, x_before), "input tensor was modified by copy-back"
