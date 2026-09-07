"""Packet 168 (158 row 3) — the weak-CAS contract of `tt.atomic_cas`.

MSL's only compare-exchange is `atomic_compare_exchange_weak_explicit`, which the language allows to
fail SPURIOUSLY: the location still holds `cmp`, `expected` is reloaded with that same value, and the
call returns false without storing. The emitter issued ONE such call and reported `expected` as the
old value, so a spurious failure would have reported old == cmp — "swapped" — with nothing stored: a
silent-wrong for every lock / claim / once pattern built on CAS. The emitter now emulates the strong
CAS: retry while the failure is spurious (reloaded value == cmp, compared as bits, as the CAS itself
compares), stop on a genuine mismatch.

A spurious failure cannot be forced on hardware, so the pins are (1) the emitted MSL spelling for the
scalar-int, scalar-float and tensor paths — a one-shot call is refused by the pin — and (2) GPU
semantic controls that hold on both sides of the fix: a 256-program claim on one word has exactly
one winner and every loser observes a non-cmp old value; float CAS compares bits (-0.0 does not match
+0.0, as CUDA's atomicCAS behaves); the tensor path swaps every lane independently.
"""

import re
import sys

import pytest

try:
    import torch
    import triton
    import triton.language as tl

    import triton_msl

    sys.path.insert(0, "tests")
    from test_fa_bwd_routing import _build_lowerer

    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")


@triton.jit
def _k_claim(lock_ptr, out_ptr):
    # 256 programs race to claim one word: exactly one sees old == 0
    old = tl.atomic_cas(lock_ptr, 0, 1)
    tl.store(out_ptr + tl.program_id(0), old)


@triton.jit
def _k_cas_float(ptr, out_ptr, cmp, val):
    old = tl.atomic_cas(ptr, cmp, val)
    tl.store(out_ptr + tl.program_id(0), old)


@triton.jit
def _k_cas_tensor(ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    cmp = offs.to(tl.int32)               # lane i expects value i
    val = tl.full((N,), -1, tl.int32)
    old = tl.atomic_cas(ptr + offs, cmp, val)
    tl.store(out_ptr + offs, old)


_ONE_SHOT = re.compile(r"^\s*atomic_compare_exchange_weak_explicit\(", re.M)   # a bare statement = one shot
_LOOP = re.compile(r"while \(!atomic_compare_exchange_weak_explicit\([^\n]*\n[^\n]*\n\s*if \((expected_\d+) != (cmp_(?:bits|val)_\d+)\) break;")


def _msl(fn, sig, cex):
    return _build_lowerer(fn, sig, cex).lower()


@pytest.mark.parametrize("fn,sig,cex,n_cas", [
    (_k_claim, {"lock_ptr": "*i32", "out_ptr": "*i32"}, {}, 1),
    (_k_cas_float, {"ptr": "*fp32", "out_ptr": "*fp32", "cmp": "fp32", "val": "fp32"}, {}, 1),
    (_k_cas_tensor, {"ptr": "*i32", "out_ptr": "*i32"}, {"N": 32}, 1),
], ids=["scalar-int", "scalar-float", "tensor-int"])
def test_cas_emits_the_strong_emulation_loop(fn, sig, cex, n_cas):
    """Direct lowering (CPU): every tt.atomic_cas lowers to the retry loop that stops only on a
    genuine mismatch; no bare one-shot weak call remains. Pre-168: one-shot call, no loop."""
    msl = _msl(fn, sig, cex)
    assert "kernel void" in msl and "UNSUPPORTED" not in msl
    assert not _ONE_SHOT.search(msl), "one-shot weak compare-exchange (spurious failure = 'swapped', nothing stored)"
    loops = _LOOP.findall(msl)
    assert len(loops) == n_cas, msl
    for expected, cmp in loops:
        assert expected.split("_")[-1] == cmp.split("_")[-1]     # the loop compares against ITS cmp


@requires_gpu
def test_claim_has_exactly_one_winner(cold_gpu_caches):
    """Control on both sides: 256 programs CAS the same word 0→1; exactly one observes old == 0 and
    every other observes the value a winner stored (1). Holds pre-168 too on hardware whose weak
    CAS never fails spuriously — the contract is what the pin above fixes."""
    lock = torch.zeros(1, device=D, dtype=torch.int32)
    out = torch.full((256,), -5, device=D, dtype=torch.int32)
    _k_claim[(256,)](lock, out)
    torch.mps.synchronize()
    o = out.cpu().tolist()
    assert lock.item() == 1
    assert o.count(0) == 1 and o.count(1) == 255, (o.count(0), o.count(1), set(o))


@requires_gpu
@pytest.mark.parametrize("stored,cmp,val,swaps", [
    (0.0, -0.0, 5.0, False),      # bit comparison: -0.0 != +0.0 (CUDA atomicCAS semantics)
    (0.0, 0.0, 5.0, True),
    (2.5, 2.5, -1.0, True),
    (2.5, 2.0, -1.0, False),
])
def test_float_cas_compares_bits(cold_gpu_caches, stored, cmp, val, swaps):
    """Control: the float CAS compares the 32-bit patterns; the old value comes back either way."""
    x = torch.tensor([stored], device=D, dtype=torch.float32)
    out = torch.zeros(1, device=D, dtype=torch.float32)
    _k_cas_float[(1,)](x, out, cmp, val)
    torch.mps.synchronize()
    assert out.item() == stored and torch.equal(out.cpu(), torch.tensor([stored]))
    assert x.item() == (val if swaps else stored)


@requires_gpu
def test_tensor_cas_swaps_each_lane_independently(cold_gpu_caches):
    """Control: 32 lanes CAS their own words; lanes whose word holds the expected value swap to -1,
    the others keep their value, and every lane returns its old value."""
    N = 32
    x = torch.arange(N, device=D, dtype=torch.int32)
    x[::2] += 100                       # even lanes hold a non-matching value
    before = x.clone()
    out = torch.zeros(N, device=D, dtype=torch.int32)
    _k_cas_tensor[(1,)](x, out, N=N, num_warps=1)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), before.cpu())
    want = before.clone()
    want[1::2] = -1
    assert torch.equal(x.cpu(), want.cpu())
