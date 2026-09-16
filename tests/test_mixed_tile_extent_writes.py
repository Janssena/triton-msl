"""Regressions for the mask/extent sweep (2026-08-26).

The staged-fill out-of-bounds READ (tests/test_staged_fill_masked_load.py) was one
instance of a wider class: emitted code whose element loop runs over the KERNEL's tile
rather than the tile of the tensor it is actually touching. Sweeping the other emission
sites turned up four more, all silent, all fixed here rather than refused:

1. ``tl.histogram`` re-read its input flatly as ``src[_h]``, discarding the source
   ``tl.load``'s ADDRESS arithmetic — ``tl.load(a + OFF + i)`` binned ``a[0:N]`` instead
   of the offset window.

2. The same re-read discarded the load's ``mask``/``other``, so a masked histogram read
   past the end of the tensor and counted masked-out lanes into the bins (measured: 512
   counts for a 300-element tensor, drifting with num_warps because the overhang raced).

   1 and 2 are fixed by preferring the per-thread loaded value — which already carries
   address, mask and ``other`` — and only re-reading when one thread does not hold
   exactly one element, where the flat form is verified before use and refused otherwise.

3. A store whose own tile is SMALLER than the kernel's element loop wrote past the end of
   its buffer — measured 504 float32 words past an 8-element output. The store's own
   elements were correct, so nothing looked wrong; it just corrupted whatever followed.
   Now bounded by the store's own extent.

4. The kernel-wide block size was only widened to the largest tensor when that was
   ``<= 1024``; above it the widening was skipped entirely and the SMALLER tile's size
   was kept, truncating every wider tensor — measured ``tl.sum`` over a 2048-element load
   returning 16. Sizes above 1024 already had a wrapping path; they just never reached
   it. Fixing 3 is what makes this safe, since the narrow store now sits inside a wide
   loop.

Each overrun test uses a canary allocation after the output so corruption is detected
directly rather than inferred.
"""

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


def _canary(n_out, extra=4096, dtype=torch.float32):
    """Return (out_view, whole_pack, saved_tail) for overrun detection."""
    pack = torch.zeros(n_out + extra, device="mps", dtype=dtype)
    return pack[:n_out], pack, pack[n_out:].clone()


def _overrun(pack, n_out, saved_tail):
    return int((pack[n_out:] != saved_tail).sum())


# ---------------------------------------------------------------- 1. histogram mask


@triton.jit
def _masked_hist(a, o, N: tl.constexpr, B: tl.constexpr, n_real):
    i = tl.arange(0, N)
    x = tl.load(a + i, mask=i < n_real, other=0)
    tl.store(o + tl.arange(0, B), tl.histogram(x, B))


@requires_mps
@pytest.mark.parametrize("nw", [1, 4, 8])
@pytest.mark.parametrize("N,n_real", [(2048, 1000), (512, 300), (4096, 100)])
def test_masked_histogram_honours_mask_and_other(N, n_real, nw):
    # The tensor is exactly n_real long, so the tile overhangs it. Pre-fix the re-read
    # dropped the load's mask: for (512, 300) all 512 lanes landed in the value bin
    # instead of 300, the other= fill was missing entirely, and the counts drifted with
    # num_warps (1033/1045/1093) because the overhang was also racing.
    #
    # Correct behaviour: the n_real real elements bin by value, and the (N - n_real)
    # masked-out lanes bin as the `other=0` fill, exactly as the tile holds them.
    a = torch.full((n_real,), 3, device="mps", dtype=torch.int32)
    o = torch.zeros(8, device="mps", dtype=torch.int32)
    _masked_hist[(1,)](a, o, N=N, B=8, n_real=n_real, num_warps=nw)
    torch.mps.synchronize()
    exp = [0] * 8
    exp[0] = N - n_real  # other=0
    exp[3] = n_real  # the real values
    assert o.cpu().tolist() == exp, f"masked histogram wrong for N={N} n_real={n_real} nw={nw}"


@triton.jit
def _hist_offset(a, o, OFF: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    tl.store(o + tl.arange(0, B), tl.histogram(tl.load(a + OFF + tl.arange(0, N)), B))


@requires_mps
@pytest.mark.parametrize("nw", [1, 4])
@pytest.mark.parametrize("N", [64, 512])
def test_histogram_honours_the_load_offset(N, nw):
    # The re-read used a flat src[_h], dropping the load's address arithmetic entirely,
    # so `tl.load(a + OFF + i)` binned a[0:N] instead of the offset window — measured
    # as 64 counts in the wrong bin. Values before OFF differ from those after, so a
    # dropped offset is visible in the bins.
    OFF, B = N, 8
    a = torch.cat([torch.full((OFF,), 1, dtype=torch.int32), torch.full((N,), 5, dtype=torch.int32)]).to("mps")
    o = torch.zeros(B, device="mps", dtype=torch.int32)
    _hist_offset[(1,)](a, o, OFF=OFF, N=N, B=B, num_warps=nw)
    torch.mps.synchronize()
    exp = [0] * B
    exp[5] = N
    assert o.cpu().tolist() == exp, f"histogram ignored the load offset (N={N} nw={nw})"


@requires_mps
def test_wide_histogram_with_load_offset_refuses_not_miscomputes():
    # Above 1024 elements the histogram must re-read the input cooperatively. A chained
    # base+OFF pointer is not the verified flat base+arange form: flattening it to
    # base[_h] drops OFF and bins the wrong window. The N<=512 cases above compute from
    # their already-loaded per-thread values; this wider case must refuse until the
    # complete chained address can be reconstructed.
    N, B, OFF = 2048, 8, 2048
    a = torch.cat(
        [
            torch.full((OFF,), 1, dtype=torch.int32),
            torch.full((N,), 5, dtype=torch.int32),
        ]
    ).to("mps")
    o = torch.zeros(B, device="mps", dtype=torch.int32)
    with pytest.raises(MetalNonRecoverableError, match="not a plain"):
        _hist_offset[(1,)](a, o, OFF=OFF, N=N, B=B, num_warps=4)


@triton.jit
def _plain_hist(a, o, N: tl.constexpr, B: tl.constexpr):
    tl.store(o + tl.arange(0, B), tl.histogram(tl.load(a + tl.arange(0, N)), B))


@requires_mps
@pytest.mark.parametrize("N", [64, 256, 512, 2048])
def test_unmasked_histogram_still_computes_and_stays_in_bounds(N):
    # Must NOT over-refuse: an unmasked histogram is still supported. And its result
    # store must not run past the bin array (pre-fix it wrote N words into B).
    a = torch.full((N,), 3, device="mps", dtype=torch.int32)
    pack = torch.zeros(8 + 4096, device="mps", dtype=torch.int32)
    o, saved = pack[:8], pack[8:].clone()
    _plain_hist[(1,)](a, o, N=N, B=8)
    torch.mps.synchronize()
    exp = [0] * 8
    exp[3] = N
    assert o.cpu().tolist() == exp, f"histogram wrong for N={N}"
    assert int((pack[8:] != saved).sum()) == 0, f"histogram store overran its output (N={N})"


# ------------------------------------------------------- 2. small store beside big tile


@triton.jit
def _big_in_small_out(a, o, N: tl.constexpr, B: tl.constexpr):
    x = tl.load(a + tl.arange(0, N))
    s = tl.sum(x, 0)
    tl.store(o + tl.arange(0, B), tl.zeros([B], tl.float32) + s)


@requires_mps
@pytest.mark.parametrize("nw", [1, 4])
@pytest.mark.parametrize("N,B", [(512, 8), (256, 8), (1024, 32)])
def test_small_store_beside_big_tile_stays_in_bounds(N, B, nw):
    out, pack, saved = _canary(B)
    _big_in_small_out[(1,)](a := torch.ones(N, device="mps", dtype=torch.float32), out, N=N, B=B, num_warps=nw)
    torch.mps.synchronize()
    assert a is not None
    got = out.cpu().tolist()
    assert all(abs(g - N) < 1e-3 for g in got), f"wrong sum for N={N} B={B} nw={nw}: {got[0]}"
    assert _overrun(pack, B, saved) == 0, f"store overran its {B}-element output for N={N} nw={nw}"


# ------------------------------------------------------------- 3. mixed tile widths


@requires_mps
@pytest.mark.parametrize("nw", [1, 4])
@pytest.mark.parametrize("N,B", [(2048, 16), (2048, 8), (4096, 32), (2048, 1024)])
def test_wide_tile_beside_narrow_store_computes_not_truncates(N, B, nw):
    # Pre-fix this returned 16.0 for a 2048-element sum. The kernel-wide block size was
    # only widened to the largest tensor when that was <= 1024; above it the widening
    # was skipped entirely and the SMALLER tile's size was kept, truncating the load to
    # the threadgroup. Sizes above 1024 already had a wrapping path — they just never
    # reached it. These must compute, not refuse and not truncate.
    out, pack, saved = _canary(B)
    a = torch.ones(N, device="mps", dtype=torch.float32)
    _big_in_small_out[(1,)](a, out, N=N, B=B, num_warps=nw)
    torch.mps.synchronize()
    got = out.cpu().tolist()
    assert all(abs(g - N) < 1e-3 for g in got), f"wide tile truncated: N={N} B={B} nw={nw} gave {got[0]}, expected {N}"
    assert _overrun(pack, B, saved) == 0, f"store overran its output (N={N} B={B} nw={nw})"


@requires_mps
@pytest.mark.parametrize("N", [64, 256, 1024])
def test_uniform_tile_width_not_over_refused(N):
    # The refusal must be narrow: a kernel whose tiles agree still compiles and runs.
    a = torch.ones(N, device="mps", dtype=torch.float32)
    o = torch.zeros(N, device="mps", dtype=torch.float32)
    _big_in_small_out[(1,)](a, o, N=N, B=N)
    torch.mps.synchronize()
    assert abs(o[0].item() - N) < 1e-3, f"uniform-width kernel regressed at N={N}"
