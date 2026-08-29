"""fp16 bias widened into a float32 dot accumulator must not evade the bias guard."""

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError


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
def test_widened_fp16_dot_accumulator_refuses(mode):
    # Triton inserts arith.extf between the fp16 load and the float32 dot init.
    # The old guard stopped at that cast, so all three upstream forms executed and
    # silently dropped the bias (9 failures across N=16/32/64, 100% mismatched).
    n = 16
    a = torch.randn((n, n), device="mps", dtype=torch.float16) * 0.1
    b = torch.randn((n, n), device="mps", dtype=torch.float16) * 0.1
    bias = torch.randn((n, n), device="mps", dtype=torch.float16)
    out = torch.empty((n, n), device="mps", dtype=torch.float32)
    with pytest.raises(MetalNonRecoverableError, match="non-zero accumulator init"):
        _fp16_bias_f32_dot[(1,)](a, b, bias, out, MODE=mode, N=n, num_warps=1)
        torch.mps.synchronize()
