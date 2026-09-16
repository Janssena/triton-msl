"""On-device computation pins for the rank-3 argmin repair (not refusal credit)."""

import pytest
import torch

from tests.test_codegen_admission_semantics import _reduce3
from tests.test_fa_tail_semantics import executed, _executed_source  # noqa: F401


@pytest.mark.parametrize("kind", [5, 6], ids=["argmin", "argmax"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32], ids=["i32", "f32"])
def test_reduce3_argminmax_gpu_computes_with_canaries(kind, axis, dtype, executed):
    # i32 values above 2**24 also catch an accidental float accumulator.
    original = (torch.arange(512, dtype=torch.int32) % 13).reshape(4, 8, 16)
    if dtype == torch.int32:
        original += 2**25
        original[0, 0, 0] = -(2**31)
        original[-1, -1, -1] = 2**31 - 1
    else:
        original = original.to(dtype) / 8
        original[0, 0, 0] = -float("inf")
        original[-1, -1, -1] = float("inf")
    reference = (original.argmin(axis) if kind == 5 else original.argmax(axis)).to(torch.int32)
    inp = original.to("mps")
    storage = torch.full((reference.numel() + 8,), -1234567, dtype=torch.int32, device="mps")
    out = storage[4:-4]
    handle = _reduce3[(1,)](inp, out, AXIS=axis, KIND=kind, num_warps=4)
    torch.mps.synchronize()
    assert _executed_source(executed, handle) == handle.asm["msl"], "executed shader differs from compiled source"
    assert "_result_idx[_r] = best_i" in handle.asm["msl"], "expected rank-3 template did not emit"
    assert torch.equal(out.cpu().reshape(reference.shape), reference), "rank-3 index does not match source"
    assert torch.equal(inp.cpu(), original), "input modified"
    retained = storage.cpu()
    assert bool((retained[:4] == -1234567).all()) and bool((retained[-4:] == -1234567).all()), "output canary modified"
