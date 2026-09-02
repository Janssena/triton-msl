"""Dot-recovery 2B (2026-09-01): reshape-provenance dots route to the generic path.

Upstream ``test_dot_multidim`` rank-2 stages each operand as
``tt.load(1-D) -> tt.reshape(32x32) -> ttg.local_alloc -> [ttg.memdesc_trans]
-> ttg.local_load`` — no 2-D addptr arithmetic exists, so the template route's
stride tracer can only refuse ("operand stride could not be inferred").  The
generic path stages exactly this chain correctly (bf16 + memdesc_trans
included), so ``_detect_simple_dot`` and ``_requires_matmul_template`` now
DECLINE when ``_dot_has_reshape_provenance() and _dot_generic_eligible()``,
and lower() falls through.  Outside the envelope the loud refusal remains the
authority; these pins hold both sides of that boundary.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir

    from triton_msl.backend.compiler import MetalBackend
    import triton_msl.codegen.generic_lowerer as generic_lowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False


requires = pytest.mark.skipif(not HAS, reason="Triton compiler needed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")


if HAS:

    @triton.jit
    def _rs_dot(X, Y, Z, TRANS_A: tl.constexpr, TRANS_B: tl.constexpr):
        # Upstream test_dot_multidim rank-2, verbatim shape.
        x = tl.load(X + tl.arange(0, 1024)).reshape([32, 32])
        y = tl.load(Y + tl.arange(0, 1024)).reshape([32, 32])
        if TRANS_A:
            x = tl.trans(x)
        if TRANS_B:
            y = tl.trans(y)
        z = tl.dot(x, y)
        tl.store(Z + tl.arange(0, 1024), z.reshape([1024]))

    @triton.jit
    def _rs_dot64(X, Y, Z):
        x = tl.load(X + tl.arange(0, 4096)).reshape([64, 64])
        y = tl.load(Y + tl.arange(0, 4096)).reshape([64, 64])
        tl.store(Z + tl.arange(0, 4096), tl.dot(x, y).reshape([4096]))

    @triton.jit
    def _rs_dot_nonsq(X, Y, Z):
        x = tl.load(X + tl.arange(0, 512)).reshape([16, 32])
        y = tl.load(Y + tl.arange(0, 1024)).reshape([32, 32])
        tl.store(Z + tl.arange(0, 512), tl.dot(x, y).reshape([512]))

    @triton.jit
    def _flat_acc_dot(X, Y, A, Z, S: tl.constexpr):
        # 2-D addptr OPERANDS (no reshape provenance) but a FLAT-reshaped
        # ACCUMULATOR: packet 063's shared-predicate-leak holdout.
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        acc = tl.load(A + tl.arange(0, S * S)).reshape([S, S])
        tl.store(Z + om[:, None] * S + on[None, :], tl.dot(x, y, acc))

    @triton.jit
    def _plain_dot(X, Y, Z, S: tl.constexpr):
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        tl.store(Z + om[:, None] * S + on[None, :], tl.dot(x, y))

    def _direct_lowerer_for(fn, signature, constexprs):
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        target = GPUTarget("metal", "apple-m4", 32)
        backend = MetalBackend(target)
        options = backend.parse_options({"num_warps": 4})
        src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
        context = ir.context()
        ir.load_dialects(context)
        mod = src.make_ir(
            target,
            options,
            backend.get_codegen_implementation(options),
            backend.get_module_map(),
            context,
        )
        metadata = {}
        mod = backend.make_ttir(mod, metadata, options)
        mod = backend.make_ttgir(mod, metadata, options)
        return generic_lowerer.GenericLowerer(walk_ttgir(mod, options), options)


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    for fn in (_rs_dot, _rs_dot64, _rs_dot_nonsq):
        fn.device_caches.clear() if hasattr(fn, "device_caches") else None


@requires_gpu
@pytest.mark.parametrize("ta", [False, True])
@pytest.mark.parametrize("tb", [False, True])
def test_reshape_dot_computes_on_generic_route(monkeypatch, cold_gpu_caches, ta, tb):
    """The four upstream rank-2 variants: generic route, full coverage, exact."""
    hits = []
    original = generic_lowerer.GenericLowerer._lower_dot

    def _spy(self, ssa):
        hits.append(1)
        return original(self, ssa)

    monkeypatch.setattr(generic_lowerer.GenericLowerer, "_lower_dot", _spy)
    torch.manual_seed(31)
    a = torch.randint(-4, 5, (32, 32), dtype=torch.bfloat16, device="mps")
    b = torch.randint(-4, 5, (32, 32), dtype=torch.bfloat16, device="mps")
    c = torch.full((32, 32), float("nan"), dtype=torch.float32, device="mps")
    _rs_dot[(1,)](a, b, c, ta, tb)
    torch.mps.synchronize()
    assert len(hits) == 1, "reshape dot did not take the generic _lower_dot route"
    assert not bool(c.isnan().any()), "unwritten sentinel elements"
    aa = a.T if ta else a
    bb = b.T if tb else b
    torch.testing.assert_close(c, aa.float() @ bb.float(), rtol=1e-3, atol=1e-2)


@requires
def test_reshape_dot_lowering_boundary():
    """Rule 9: fresh TTGIR lowers to MSL (no refusal) via the generic staging."""
    lowerer = _direct_lowerer_for(
        _rs_dot,
        {"X": "*bf16", "Y": "*bf16", "Z": "*fp32"},
        {"TRANS_A": 0, "TRANS_B": 0},
    )
    assert lowerer._dot_has_reshape_provenance() is True
    # The flat (S*S,) admission is OPT-IN at the two reshape sites only; the
    # shared predicate's default must NOT have widened (rule 8, packet 063).
    assert lowerer._dot_generic_eligible(allow_flat=True) is True
    assert lowerer._dot_generic_eligible() is False
    msl = lowerer.lower()
    assert "UNSUPPORTED" not in msl
    assert "smem_dot" in msl, "generic dot staging absent — wrong route claimed the kernel"


@requires
def test_plain_dot_has_no_reshape_provenance():
    """The decline must not leak onto stride-traceable kernels: a plain 2-D
    addptr dot reports NO reshape provenance, so template routing is untouched."""
    lowerer = _direct_lowerer_for(
        _plain_dot,
        {"X": "*fp32", "Y": "*fp32", "Z": "*fp32"},
        {"S": 32},
    )
    probe = getattr(lowerer, "_dot_has_reshape_provenance", None)
    if probe is None:
        pytest.skip("pre-2B tree: helper absent, decline cannot exist either")
    assert probe() is False


@requires_gpu
def test_reshape_dot_outside_envelope_still_refuses(cold_gpu_caches):
    """S=64 flat form (4096 elements) is OUTSIDE the envelope: loud refusal,
    never a silent guess at the un-inferable strides."""
    x = torch.randn(4096, device="mps") * 0.1
    y = torch.randn(4096, device="mps") * 0.1
    z = torch.full((4096,), float("nan"), device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _rs_dot64[(1,)](x, y, z)
        torch.mps.synchronize()


@requires_gpu
def test_reshape_dot_nonsquare_still_refuses(cold_gpu_caches):
    """Non-square reshape (16x32 @ 32x32) is outside the envelope: loud refusal."""
    x = torch.randn(512, device="mps") * 0.1
    y = torch.randn(1024, device="mps") * 0.1
    z = torch.full((512,), float("nan"), device="mps")
    with pytest.raises(MetalNonRecoverableError):
        _rs_dot_nonsq[(1,)](x, y, z)
        torch.mps.synchronize()


@requires
def test_flat_accumulator_without_operand_provenance_keeps_old_route():
    """Packet 063 control: a flat-reshaped ACCUMULATOR with ordinary 2-D operands
    has no dot-operand reshape provenance, so the default shared eligibility must
    be unchanged (False) and the pre-2B full-2-D-accumulator refusal must stand.
    On the first 062 candidate the unparameterized widening flipped this to
    generic emission — the leak this test exists to catch."""
    lowerer = _direct_lowerer_for(
        _flat_acc_dot,
        {"X": "*fp32", "Y": "*fp32", "A": "*fp32", "Z": "*fp32"},
        {"S": 32},
    )
    probe = getattr(lowerer, "_dot_has_reshape_provenance", None)
    if probe is None:
        pytest.skip("pre-2B tree: helper absent")
    assert probe() is False
    assert lowerer._dot_generic_eligible() is False
    with pytest.raises(MetalNonRecoverableError):
        lowerer.lower()
