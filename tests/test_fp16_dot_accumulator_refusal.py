"""fp16 bias widened into a float32 dot accumulator must not be silently dropped.

Originally a REFUSAL pin: Triton inserts ``arith.extf`` between the fp16 bias load
and the float32 dot init, the guard's walk stopped at that cast, and all three
upstream forms executed while SILENTLY DROPPING the bias (9 failures across
N=16/32/64, 100% mismatched). ``_acc_init_is_bias`` was taught to see through
numeric casts, which made the guard refuse them.

CONVERTED 2026-08-30 (dot-recovery stage 1a): these single-tile N=16 kernels now
sit inside the generic dot path's PROVEN envelope, so the guard falls through and
the generic lowerer seeds each output element's accumulator from the bias and
COMPUTES all three forms (~5e-7). The tests keep their original purpose by
asserting the bias is actually applied — each checks the result is far from the
bias-DROPPED value, which is exactly the silent-wrong this file was written for.
Outside the envelope the guard still refuses (see
tests/test_dot_epilogue_generic.py::test_nonuniform_fused_accumulator_still_refuses).
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
def _fp16_bias_f32_dot(a, b, bias, out, MODE: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, N)
    cols = tl.arange(0, N)
    depth = tl.arange(0, N)
    av = tl.load(a + rows[:, None] * N + depth[None, :])
    bv = tl.load(b + depth[:, None] * N + cols[None, :])
    acc = tl.dot(av, bv, out_dtype=tl.float32)
    if MODE == 0:
        acc += tl.load(bias + rows[:, None] * N + cols[None, :])
    elif MODE == 1:
        acc += tl.load(bias + rows)[:, None]
    else:
        acc += tl.load(bias + cols)[None, :]
    tl.store(out + rows[:, None] * N + cols[None, :], acc)


@requires_mps
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["matrix", "rows", "cols"])
def test_widened_fp16_dot_accumulator_applies_bias(mode):
    # The widened (extf'd) fp16 bias must reach the accumulator. NOTE the kernel
    # indexes `bias + rows` / `bias + cols` — FLAT offsets 0..N-1, i.e. bias ROW 0
    # in both cases, not a column.
    torch.manual_seed(3)
    n = 16
    a = torch.randn((n, n), device="mps", dtype=torch.float16) * 0.1
    b = torch.randn((n, n), device="mps", dtype=torch.float16) * 0.1
    bias = torch.randn((n, n), device="mps", dtype=torch.float16)
    out = torch.empty((n, n), device="mps", dtype=torch.float32)
    _fp16_bias_f32_dot[(1,)](a, b, bias, out, MODE=mode, N=n, num_warps=1)
    torch.mps.synchronize()

    bare = a.float() @ b.float()
    if mode == 0:
        ref = bare + bias.float()
    elif mode == 1:
        ref = bare + bias.float()[0, :][:, None]
    else:
        ref = bare + bias.float()[0, :][None, :]
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    # The original silent-wrong: bias dropped entirely (result == bare matmul).
    assert (out - bare).abs().max().item() > 1e-2, "bias was dropped — the silent-wrong is back"
