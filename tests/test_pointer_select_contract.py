"""Do not turn selected addresses into loaded data and cast it back to a pointer.

Baseline bites are LOWERING ONLY: its emitted atomic addresses are unsafe to
execute. The GPU witness was already recorded in review203. Numerical select
controls remain required capability; pointer selection is an explicit refusal.
"""
import importlib
from pathlib import Path
import sys

import pytest
import torch
import triton
import triton.language as tl
from triton_msl.errors import MetalNonRecoverableError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fa_bwd_routing import _build_lowerer


@triton.jit
def _selected_pointer(X, Y, O, N: tl.constexpr, SCALAR: tl.constexpr, CONSUMER: tl.constexpr):
    if SCALAR:
        c = tl.program_id(0)
    else:
        c = tl.arange(0, N)
    p = tl.where(c % 2 == 0, X + c, Y + c)
    if CONSUMER == 0:
        old = tl.load(p)
        tl.store(O + c, old)
    elif CONSUMER == 1:
        tl.store(p, 7)
    elif CONSUMER == 2:
        old = tl.atomic_add(p, 3)
        tl.store(O + c, old)
    else:
        compare = (c * 0 + 1).to(X.dtype.element_ty)
        desired = (c * 0 + 2).to(X.dtype.element_ty)
        old = tl.atomic_cas(p, compare, desired)
        tl.store(O + c, old)


@triton.jit
def _selected_value(X, Y, O, N: tl.constexpr):
    c = tl.arange(0, N)
    x = tl.load(X + c)
    y = tl.load(Y + c)
    tl.store(O + c, tl.where(c % 2 == 0, x, y))


@pytest.mark.parametrize('dtype', ['i32', 'fp32'])
@pytest.mark.parametrize('scalar', [False, True], ids=['tensor', 'scalar'])
@pytest.mark.parametrize('consumer', range(4), ids=['load', 'store', 'rmw', 'cas'])
def test_pointer_select_lowering_refuses(dtype, scalar, consumer):
    signature = {name: '*' + dtype for name in ('X', 'Y', 'O')}
    lowerer = _build_lowerer(_selected_pointer, signature, {'N': 8, 'SCALAR': scalar, 'CONSUMER': consumer})
    with pytest.raises(MetalNonRecoverableError, match='pointer-valued arith.select'):
        lowerer.lower()


@pytest.mark.parametrize('dtype', ['i32', 'fp32'])
def test_value_select_lowering_stays_supported(dtype):
    signature = {name: '*' + dtype for name in ('X', 'Y', 'O')}
    text = _build_lowerer(_selected_value, signature, {'N': 8}).lower()
    assert 'kernel void' in text and 'UNSUPPORTED' not in text


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path / 'triton'))
    monkeypatch.setenv('TRITON_MSL_CACHE_DIR', str(tmp_path / 'msl'))
    monkeypatch.setenv('TRITON_ALWAYS_COMPILE', '1')
    for fn in (_selected_pointer, _selected_value):
        fn.device_caches.clear()
    hits = []
    cls = importlib.import_module('triton_msl.codegen.generic_lowerer').GenericLowerer
    real = cls._lower_select
    def spy(self, *args, **kwargs):
        hits.append('select')
        return real(self, *args, **kwargs)
    monkeypatch.setattr(cls, '_lower_select', spy)
    return hits


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Metal GPU required')
@pytest.mark.parametrize('dtype', [torch.int32, torch.float32])
def test_value_select_gpu_computes_exactly(fresh, dtype):
    x_cpu = torch.arange(8, dtype=dtype)
    y_cpu = x_cpu + 20
    out = torch.full((8,), -99, dtype=dtype, device='mps')
    _selected_value[(1,)](x_cpu.to('mps'), y_cpu.to('mps'), out, N=8)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.where(torch.arange(8) % 2 == 0, x_cpu, y_cpu))
    assert fresh == ['select']
