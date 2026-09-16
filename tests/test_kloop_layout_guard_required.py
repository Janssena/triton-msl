"""Unproved offset widths must never disable the K-loop safety boundary."""
import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.backend.compiler import MetalBackend
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _mixed_fused(A, B, C, M, N, K, sam, sbk, scm,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + (rm[:, None] * sam + rk[None, :])
    bp = B + rk[:, None] * sbk + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(ap), tl.load(bp))
        ap += BK
        bp += BK * sbk
    tl.store(C + rm[:, None] * scm + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


def _emit(width, metadata):
    backend = MetalBackend(GPUTarget('metal', 'apple-m4', 32))
    options = backend.parse_options({'num_warps': 4})
    context = ir.context()
    ir.load_dialects(context)
    signature = {n: '*fp32' for n in ('A', 'B', 'C')}
    signature.update({n: 'i32' for n in ('M', 'N', 'K', 'sbk', 'scm')})
    signature['sam'] = width
    constants = {'BM': 32, 'BN': 32, 'BK': 32}
    signature.update({n: 'constexpr' for n in constants})
    source = ASTSource(_mixed_fused, signature, constants)
    module = source.make_ir(backend.target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    assert module.verify()
    return backend.make_msl(module, metadata, options)


def test_mixed_fused_width_cannot_bypass_guard(tmp_path, monkeypatch):
    monkeypatch.setenv('TRITON_MSL_CACHE_DIR', str(tmp_path / 'cache'))
    with pytest.raises(MetalNonRecoverableError, match='unmasked K-loop source address widths are unresolved; mask the K tail') as error:
        _emit('i64', {})
    assert error.value.op_name == 'tt.dot'


def test_proven_fused_i32_keeps_guarded_main_and_direct(tmp_path, monkeypatch):
    monkeypatch.setenv('TRITON_MSL_CACHE_DIR', str(tmp_path / 'cache'))
    metadata = {}
    msl = _emit('i32', metadata)
    assert '__mmdirect' in msl
    assert msl.count('bool _simd_ok =') == 2
    assert msl.count('if (!_simd_ok) {') == 2
    assert 'simdgroup_multiply_accumulate' in msl


def test_missing_stride_proof_never_emits_unguarded_kernel(tmp_path, monkeypatch):
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    monkeypatch.setenv('TRITON_MSL_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(GenericLowerer, '_dot_offset_arith_widths', lambda self: None)
    with pytest.raises(MetalNonRecoverableError, match='unmasked K-loop source address widths are unresolved; mask the K tail') as error:
        _emit('i32', {})
    assert error.value.op_name == 'tt.dot'


@pytest.mark.parametrize('width', ['i32', 'i64'])
def test_existing_layout_guard_refuses_when_tail_plan_check_is_isolated(width, tmp_path, monkeypatch):
    """The earlier tail refusal must not hide removal of the existing guard."""
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    monkeypatch.setenv('TRITON_MSL_CACHE_DIR', str(tmp_path / 'cache'))
    # Isolate the downstream guard deliberately; no shader is ever launched.
    monkeypatch.setattr(GenericLowerer, '_record_full_block_tail_bounds', lambda *args: None)
    if width == 'i32':
        monkeypatch.setattr(GenericLowerer, '_dot_offset_arith_widths', lambda self: None)
    with pytest.raises(MetalNonRecoverableError, match='requires proven operand strides and source integer widths') as error:
        _emit(width, {})
    assert error.value.op_name == 'tt.dot'
