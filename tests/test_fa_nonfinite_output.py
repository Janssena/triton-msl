"""Row scaling must not invent cross-row 0*Inf terms (packet 401)."""

import pytest
import torch

from tests.test_fa_tail_semantics import _executed_source, _tail_attention, executed
from tests.test_varlen_fa_routing import _varlen_fwd


@pytest.mark.parametrize("family,n", [
    ("dense", 17), ("dense", 65), ("varlen", 8), ("varlen", 65),
    ("tiled", 65), ("dense_bf16", 17), ("varlen_scalar", 65),
])
def test_nonfinite_value_outputs_remain_rowwise(family, n, executed):
    # Q=K=0 makes every live probability positive and uniform. Thus +Inf,
    # -Inf, NaN in V's first three columns stay in those columns; all other
    # columns and the independent second head remain 1. No softmax oracle.
    if not torch.backends.mps.is_available():
        pytest.skip("requires Metal")
    dense = family in ("dense", "dense_bf16", "tiled")
    dim = 128 if family in ("tiled", "varlen_scalar") else 64
    shape = (1, 2, n, dim) if dense else (n, 2, dim)
    dtype = torch.bfloat16 if family == "dense_bf16" else torch.float32
    q = torch.zeros(shape, device="mps", dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    if dense:
        v[0, 0, 7, :3] = torch.tensor([float("inf"), -float("inf"), float("nan")], device="mps")
    else:
        v[7, 0, :3] = torch.tensor([float("inf"), -float("inf"), float("nan")], device="mps")
    out = torch.full_like(q, float("nan"))
    if dense:
        if family == "tiled":
            # A genuine noncontiguous Q selects the tiled production route.
            backing = torch.zeros(1, 2, n, 2 * dim, device="mps")
            q = backing[..., ::2]
        compiled = _tail_attention[((n + 31) // 32, 2)](
            q, k, v, out, *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            1, 2, n, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=dim, MASK=1,
            START=0, STEP=32, FILL_K=0., FILL_V=0., PLAIN_K=False, PLAIN_V=False,
        )
        marker = ("Head-dim-tiled FlashAttention-2"
                  if family in ("tiled", "dense_bf16")
                  else "simdgroup-MMA FlashAttention-2")
    else:
        cu = torch.tensor([0, n], device="mps", dtype=torch.int32)
        compiled = _varlen_fwd[((n + 31) // 32, 2)](
            q, k, v, out, cu, cu, *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            2, n, .125, 32, 32, dim,
        )
        # fp32 D128 exceeds MMA's scratch budget, but fits the scalar maker.
        marker = "fp32 compute), packed cu_seqlens" if family == "varlen_scalar" else "via simdgroup_matrix MMA"
    torch.mps.synchronize()
    source = _executed_source(executed, compiled)
    assert source == compiled.asm["msl"]
    assert marker in source
    assert ("simdgroup_multiply_accumulate" in source) == (family in ("dense", "varlen"))
    actual = out.cpu() if dense else out.cpu().permute(1, 0, 2).unsqueeze(0)
    expected = torch.ones_like(actual)
    expected[0, 0, :, 0] = float("inf")
    expected[0, 0, :, 1] = -float("inf")
    expected[0, 0, :, 2] = float("nan")
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    assert finite.any()
    torch.testing.assert_close(actual[finite], expected[finite], atol=1e-6, rtol=1e-6)
