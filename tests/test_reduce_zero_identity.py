"""Compiler-added sum identities must preserve the sign of a source's zero."""
import pytest
import torch

from tests.test_fa_tail_semantics import executed, _executed_source  # noqa: F401
from tests.test_fuzz_reduce import _r1d_sum, _r2d_sum_ax1, _r3d_sum_ax2
from triton_msl.codegen.msl_emitter import KernelBuilder


def test_threadgroup_sum_identity_preserves_both_zero_signs():
    """Lowering boundary: float padding differs from integer padding."""
    for dtype, expected in [('float', '-0.0f'), ('int', '(int)0')]:
        builder = KernelBuilder('zero_contract', block_size=64)
        builder.threadgroup_reduce('sum', 'value', 'shared', 'total', reduce_ty=dtype)
        assert f'? shared[tiisg] : {expected}' in builder.build()


@pytest.mark.parametrize('family', ['simd64', 'register4096', 'multipass4096', 'rows', 'rank3'])
def test_sum_zero_sign_survives_compiler_padding(family, monkeypatch, tmp_path, executed):
    monkeypatch.setenv('TRITON_MSL_COMPILE_SHADER', '1')
    monkeypatch.setenv('TRITON_MSL_USE_CPP', '0')
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path / 'triton'))
    monkeypatch.setenv('TRITON_MSL_CACHE_DIR', str(tmp_path / 'metal'))
    if family == 'multipass4096':
        monkeypatch.setenv('TRITON_MSL_MEPT', '0')
    if family in ('simd64', 'register4096', 'multipass4096'):
        shape = (64 if family == 'simd64' else 4096,)
        kernel, constants = _r1d_sum, dict(N=shape[0])
        outputs = 1
    elif family == 'rows':
        shape = (2, 32)
        kernel, constants = _r2d_sum_ax1, dict(M=2, N=32)
        outputs = 2
    else:
        shape = (2, 2, 32)
        kernel, constants = _r3d_sum_ax2, dict(B=2, R=2, C=32)
        outputs = 4
    x = torch.full(shape, -0., device='mps')
    output = torch.full((outputs,), 123., device='mps')
    compiled = kernel[(1,)](x, output, **constants)
    torch.mps.synchronize()
    source = _executed_source(executed, compiled)
    assert source == compiled.asm['msl']
    bits = output.cpu().view(torch.int32)
    assert bool((bits == -(2**31)).all()), 'all-negative-zero source sum lost its sign'
    # One positive zero per source reduction changes the source result to +0.
    # The repair must not force every zero-valued sum negative.
    x.reshape(outputs, -1)[:, 0] = 0.
    executed.clear()
    compiled = kernel[(1,)](x, output, **constants)
    torch.mps.synchronize()
    assert _executed_source(executed, compiled) == source
    assert bool((output.cpu().view(torch.int32) == 0).all()), 'positive-zero source contribution was lost'


@pytest.mark.parametrize('axis', ['full', 'row', 'column'])
def test_public_sum_makers_preserve_zero_and_empty_identities(axis):
    from triton_msl.codegen._msl_templates import (
        make_reduce_kernel, make_row_reduce_kernel, make_col_reduce_kernel,
    )
    maker = {'full': make_reduce_kernel, 'row': make_row_reduce_kernel,
             'column': make_col_reduce_kernel}[axis]
    source = maker('sum_zero_identity', 'sum', block_size=64)
    library = torch.mps.compile_shader(source)
    for value in (-0., 0.):
        x = torch.full((128,), value, device='mps')
        output = torch.full((2,), 123., device='mps')
        args = (64,) if axis == 'full' else (2, 64) if axis == 'row' else (64, 2)
        library.sum_zero_identity(x, output, *args, threads=128, group_size=64)
        torch.mps.synchronize()
        expected_bits = -(2**31) if torch.signbit(torch.tensor(value)) else 0
        assert bool((output.cpu().view(torch.int32) == expected_bits).all()), 'direct sum maker lost zero sign'
    # The callable makers accept a runtime zero extent; preserve their prior
    # empty-sum +0 convention while changing only the nonempty accumulator identity.
    args = (0,) if axis == 'full' else (2, 0) if axis == 'row' else (0, 2)
    output.fill_(123.)
    library.sum_zero_identity(x, output, *args, threads=128, group_size=64)
    torch.mps.synchronize()
    assert bool((output.cpu().view(torch.int32) == 0).all()), 'empty direct sum changed its +0 convention'


def test_legacy_direct_sum_preserves_zero_and_empty_identities():
    from tests.test_ttgir_parser import SUM_REDUCE_TTGIR, FakeOptions
    from triton_msl.codegen.ttgir_parser import parse_ttgir

    source = parse_ttgir(SUM_REDUCE_TTGIR, FakeOptions()).build()
    library = torch.mps.compile_shader(source)
    for value, n, expected_bits in [(-0., 256, -(2**31)), (0., 256, 0), (-0., 0, 0)]:
        x = torch.full((256,), value, device='mps')
        output = torch.full((1,), 123., device='mps')
        library.sum_kernel(x, output, n, threads=256, group_size=256)
        torch.mps.synchronize()
        assert output.cpu().view(torch.int32).item() == expected_bits
