"""Atomic return values with one logical element must be uniform before broadcast.

The Metal lowerer executes scalar and underfilled tensor atomics only on their logical
lanes.  Triton's later splat/broadcast is metadata-only in the SIMT lowering, so the
old value must be distributed explicitly; otherwise inactive lanes retain the local
zero initializer (or, for unguarded CAS, race on the same address and observe NEW).
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
def _tensor1_rmw_then_broadcast(counter, out, N: tl.constexpr):
    one_offset = tl.arange(0, 1)
    one_value = tl.full((1,), 1.0, tl.float32)
    old = tl.atomic_add(counter + one_offset, one_value)
    tl.store(out + tl.arange(0, N), old.broadcast_to(N))


@triton.jit
def _scalar_cas_then_splat(counter, out, N: tl.constexpr):
    old = tl.atomic_cas(counter, 7, 11)
    tl.store(out + tl.arange(0, N), old)


@triton.jit
def _tensor1_cas_then_broadcast(counter, out, N: tl.constexpr):
    one_offset = tl.arange(0, 1)
    cmp = tl.full((1,), 7, tl.int32)
    val = tl.full((1,), 11, tl.int32)
    old = tl.atomic_cas(counter + one_offset, cmp, val)
    tl.store(out + tl.arange(0, N), old.broadcast_to(N))


@requires_mps
def test_tensor1_rmw_return_broadcasts_old_value():
    # Before the fix: counter was correct (8), but output was [7, 0, ..., 0].
    counter = torch.tensor([7.0], device="mps", dtype=torch.float32)
    out = torch.full((32,), -1.0, device="mps", dtype=torch.float32)
    _tensor1_rmw_then_broadcast[(1,)](counter, out, N=32, num_warps=1)
    torch.mps.synchronize()
    assert counter.cpu().tolist() == [8.0]
    assert out.cpu().tolist() == [7.0] * 32


@requires_mps
def test_scalar_cas_return_splats_old_value():
    # Before the fix: only lid 0 held the return, yielding [7, 0, ..., 0].
    counter = torch.tensor([7], device="mps", dtype=torch.int32)
    out = torch.full((32,), -1, device="mps", dtype=torch.int32)
    _scalar_cas_then_splat[(1,)](counter, out, N=32, num_warps=1)
    torch.mps.synchronize()
    assert counter.cpu().tolist() == [11]
    assert out.cpu().tolist() == [7] * 32


@requires_mps
def test_tensor1_cas_is_guarded_and_broadcasts_old_value():
    # Before the fix: every lane raced the same address, then the metadata-only
    # broadcast exposed [7, 11, ..., 11].
    counter = torch.tensor([7], device="mps", dtype=torch.int32)
    out = torch.full((32,), -1, device="mps", dtype=torch.int32)
    _tensor1_cas_then_broadcast[(1,)](counter, out, N=32, num_warps=1)
    torch.mps.synchronize()
    assert counter.cpu().tolist() == [11]
    assert out.cpu().tolist() == [7] * 32
