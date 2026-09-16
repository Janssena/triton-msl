"""Issue #9 regression: when two broadcasts carry the SAME source range to DIFFERENT
target shapes, each must get its own per-axis stride, not share one. Pre-fix, the second
broadcast silently reused the first's stride (a silent-wrong).

The original recovery handled bare ranges and value-arithmetic/gather chains
(``range*3+7``, ``tl.load(X+range)``). This file also pins the residual cases Astra
reported in packet 465: a comparison-derived (``(range>0).to(int32)``) and a
select-derived (``tl.where(range==0, 7, 3)``) value before the broadcast. These are
coordinate-preserving — the make_range still indexes the same axis — so they RECOVER
(compute the source), they are not refused. There is no conflict-refusal path: a range
that feeds multiple distinct broadcast shapes simply records no per-range target and
falls back to the pre-existing heuristic.

Two kinds of pin here:
  * on-device numeric pins (require MPS), and
  * a lowering-boundary (emission) pin that proves the corrected per-range stride map
    directly from GenericLowerer on CPU, per collaboration protocol 9 (not GPU-only).
"""

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl

_HAS_MPS = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not _HAS_MPS, reason="Metal GPU required")


# --------------------------------------------------------------------------------------
# On-device numeric pins
# --------------------------------------------------------------------------------------
@triton.jit
def _two_broadcasts(OFLIP, OOTHER):
    a = tl.arange(0, 2)[None, :, None]
    flip = tl.reshape(tl.broadcast_to(a, [1, 2, 4]), [8])  # middle stride 4 -> [0,0,0,0,1,1,1,1]
    b = tl.arange(0, 2)[None, :, None]
    other = tl.reshape(tl.broadcast_to(b, [4, 2, 1]), [8])  # middle stride 1 -> [0,1,0,1,0,1,0,1]
    idx = tl.arange(0, 8)
    tl.store(OFLIP + idx, flip)
    tl.store(OOTHER + idx, other)


@requires_mps
def test_two_broadcasts_same_source_different_targets_each_correct():
    of = torch.full((8,), -1, device="mps", dtype=torch.int32)
    oo = torch.full((8,), -1, device="mps", dtype=torch.int32)
    _two_broadcasts[(1,)](of, oo)
    torch.mps.synchronize()
    assert of.cpu().tolist() == [0, 0, 0, 0, 1, 1, 1, 1], of.cpu().tolist()
    # the second broadcast must NOT reuse the first's stride (the #9 silent-wrong)
    assert oo.cpu().tolist() == [0, 1, 0, 1, 0, 1, 0, 1], oo.cpu().tolist()


@triton.jit
def _one_source_two_broadcasts(OFLIP, OOTHER):
    a = tl.arange(0, 2)[None, :, None]  # ONE make_range spelling feeding two broadcasts
    flip = tl.reshape(tl.broadcast_to(a, [1, 2, 4]), [8])
    other = tl.reshape(tl.broadcast_to(a, [4, 2, 1]), [8])
    idx = tl.arange(0, 8)
    tl.store(OFLIP + idx, flip)
    tl.store(OOTHER + idx, other)


@requires_mps
def test_one_source_two_broadcasts_each_correct():
    # Triton gives each broadcast its own make_range, so a single-source spelling also
    # resolves to per-broadcast strides (no shared index variable in practice). Both
    # outputs compute the source.
    of = torch.full((8,), -1, device="mps", dtype=torch.int32)
    oo = torch.full((8,), -1, device="mps", dtype=torch.int32)
    _one_source_two_broadcasts[(1,)](of, oo)
    torch.mps.synchronize()
    assert of.cpu().tolist() == [0, 0, 0, 0, 1, 1, 1, 1], of.cpu().tolist()
    assert oo.cpu().tolist() == [0, 1, 0, 1, 0, 1, 0, 1], oo.cpu().tolist()


@triton.jit
def _cmp_derived_broadcasts(OA, OB):
    # comparison-derived value before the broadcast (packet 465 residual, mode 3)
    r = (tl.arange(0, 2) > 0).to(tl.int32)
    s = (tl.arange(0, 2) > 0).to(tl.int32)
    a = tl.reshape(tl.broadcast_to(r[None, :, None], [1, 2, 4]), [8])
    b = tl.reshape(tl.broadcast_to(s[None, :, None], [4, 2, 1]), [8])
    idx = tl.arange(0, 8)
    tl.store(OA + idx, a)
    tl.store(OB + idx, b)


@requires_mps
def test_comparison_derived_broadcasts_recover():
    oa = torch.full((8,), -1, device="mps", dtype=torch.int32)
    ob = torch.full((8,), -1, device="mps", dtype=torch.int32)
    _cmp_derived_broadcasts[(1,)](oa, ob)
    torch.mps.synchronize()
    # A is the outer axis (stride 4); B is the inner axis (stride 1). Pre-fix, A silently
    # copied B's alternating stride: [0,1,0,1,...].
    assert oa.cpu().tolist() == [0, 0, 0, 0, 1, 1, 1, 1], oa.cpu().tolist()
    assert ob.cpu().tolist() == [0, 1, 0, 1, 0, 1, 0, 1], ob.cpu().tolist()


@triton.jit
def _select_derived_broadcasts(OA, OB):
    # select-derived value before the broadcast (packet 465 residual, mode 4)
    r = tl.where(tl.arange(0, 2) == 0, 7, 3)
    s = tl.where(tl.arange(0, 2) == 0, 7, 3)
    a = tl.reshape(tl.broadcast_to(r[None, :, None], [1, 2, 4]), [8])
    b = tl.reshape(tl.broadcast_to(s[None, :, None], [4, 2, 1]), [8])
    idx = tl.arange(0, 8)
    tl.store(OA + idx, a)
    tl.store(OB + idx, b)


@requires_mps
def test_select_derived_broadcasts_recover():
    oa = torch.full((8,), -1, device="mps", dtype=torch.int32)
    ob = torch.full((8,), -1, device="mps", dtype=torch.int32)
    _select_derived_broadcasts[(1,)](oa, ob)
    torch.mps.synchronize()
    assert oa.cpu().tolist() == [7, 7, 7, 7, 3, 3, 3, 3], oa.cpu().tolist()
    assert ob.cpu().tolist() == [7, 3, 7, 3, 7, 3, 7, 3], ob.cpu().tolist()


# --------------------------------------------------------------------------------------
# Lowering-boundary (emission) pin — CPU, no GPU required (protocol 9)
# --------------------------------------------------------------------------------------
def _lower(kernel):
    """Compile ``kernel`` to TTGIR and run GenericLowerer; return (msl, lowerer)."""
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    src = ASTSource(kernel, signature={"OA": "*i32", "OB": "*i32"}, constexprs={})
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx)
    meta = {}
    mod = backend.make_ttir(mod, meta, options)
    mod = backend.make_ttgir(mod, meta, options)
    graph = walk_ttgir(mod, options)
    lower = GenericLowerer(graph, options)
    msl = lower.lower()
    return msl, lower


@pytest.mark.parametrize("kernel", [_cmp_derived_broadcasts, _select_derived_broadcasts], ids=["comparison", "select"])
def test_coordinate_preserving_broadcast_targets_emit_distinct_strides(kernel):
    """The corrected per-range map must record exactly two distinct broadcast targets
    with strides {1, 4}, and the emitted MSL must carry both index expressions — proving
    the two ranges do NOT share one index variable (the #9 residual silent-wrong)."""
    msl, lower = _lower(kernel)

    targets = set(getattr(lower, "_make_range_bcast_target", {}).values())
    strides = sorted(getattr(lower, "_make_range_stride_below", {}).values())
    # exactly the two per-axis broadcast targets, with the outer (stride 4) and
    # inner (stride 1) axes distinguished
    assert (1, 2, 4) in targets and (4, 2, 1) in targets, targets
    assert strides == [1, 4], strides

    # emitted index expressions: one outer stride-4 term and one inner stride-1 term,
    # never two copies of the same expression
    assert "/ 4u) % 2u" in msl, msl
    assert msl.count("% 2u") >= 2, msl
