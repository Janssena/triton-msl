"""A stride descriptor and cooperative fill must not erase a narrowing cast."""
import pytest
import triton
import triton.language as tl

from tests.test_codegen_admission_semantics import _emit
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _address_cast(A, B, C, stride, N, K, MODE: tl.constexpr, WIDTH: tl.constexpr):
    rm = tl.arange(0, 32)
    rn = tl.arange(0, 32)
    rk = tl.arange(0, 32)
    ro = rm[:, None] * stride
    if WIDTH == 8:
        ro = ro.to(tl.int8).to(tl.int32)
    elif WIDTH == 16:
        ro = ro.to(tl.int16).to(tl.int32)
    elif WIDTH == 64:
        ro = ro.to(tl.int64).to(tl.int32)
    elif WIDTH == -8:
        ro = (rm + stride).to(tl.int8).to(tl.int32)[:, None] * K
    elif WIDTH == -16:
        ro = (rm + stride).to(tl.int16).to(tl.int32)[:, None] * K
    a = tl.load(A + ro + rk[None, :])
    b = tl.load(B + rk[:, None] * N + rn[None, :])
    d = tl.dot(a, b)
    if MODE == 1:
        e = tl.exp(d - tl.max(d, 1)[:, None])
        d = e / tl.sum(e, 1)[:, None]
    elif MODE == 2:
        d = tl.maximum(d, 0.0)
    tl.store(C + rm[:, None] * N + rn[None, :], d)


def _source(mode, width):
    constants = dict(MODE=mode, WIDTH=width)
    signature = dict(A='*fp32', B='*fp32', C='*fp32', stride='i32', N='i32', K='i32',
                     MODE='constexpr', WIDTH='constexpr')
    return _emit(_address_cast, signature, constants)


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('width', [8, 16, -8, -16])
def test_value_changing_address_cast_is_not_erased(mode, width):
    # Narrow plain dot formerly kept the template; fused cases may decline to
    # generic, whose cooperative staging must independently reject the same loss.
    with pytest.raises(MetalNonRecoverableError, match='stride could not be inferred|cannot structurally resolve an offset|arith.trunci'):
        _source(mode, width)


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_proven_widen_then_restore_keeps_canonical_emission(mode):
    assert _source(mode, 64) == _source(mode, 0)


def test_narrow_address_witness_uses_different_in_bounds_locations():
    # A pointer bound inside a larger allocation can make both addresses legal;
    # this is a changed source address, not just an out-of-bounds artifact.
    row, stride, base = 8, 4096, 65536
    product = row * stride
    source_offset = ((product + 32768) % 65536) - 32768
    assert base + source_offset == 32768
    assert base + product == 98304
    assert 0 <= base + source_offset < 200000
    assert 0 <= base + product < 200000
