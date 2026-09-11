"""Ordinary K-loop matmul epilogues across input precision and tile layouts."""
import pytest
import torch
import triton
import triton.language as tl
from tests.test_template_scalar_abi import _lower


@triton.jit
def _epilogue_mm(A, B, Bias, C, M, N, K, SA, SB, SC, alpha, beta,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    nt = tl.cdiv(N, BN)
    m = (pid // nt) * BM + tl.arange(0, BM)
    n = (pid % nt) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + m[:, None] * SA + (k + kk)[None, :])
        b = tl.load(B + (k + kk)[:, None] * SB + n[None, :])
        acc = tl.dot(a, b, acc)
    bi = tl.load(Bias + n, n < N, other=0).to(tl.float32)
    value = alpha * acc + beta * bi[None, :]
    tl.store(C + m[:, None] * SC + n[None, :], value, (m[:, None] < M) & (n[None, :] < N))


CASES = [
    ('fp32', 32, 32, 32), ('fp16', 32, 32, 32), ('bf16', 32, 32, 32),
    ('fp16', 16, 32, 32), ('fp16', 32, 16, 32),
    ('fp16', 32, 64, 32), ('fp16', 64, 32, 32),
    ('bf16', 32, 64, 32), ('fp16', 64, 64, 32), ('fp16', 32, 32, 64),
]


def compile_case(dtype, bm, bn, bk):
    sig = {n: '*' + dtype if n in ('A', 'B', 'Bias', 'C') else
           'fp32' if n in ('alpha', 'beta') else 'i32'
           for n in _epilogue_mm.arg_names if n not in ('BM', 'BN', 'BK')}
    return _lower(_epilogue_mm, sig, {'BM': bm, 'BN': bn, 'BK': bk})


@pytest.mark.parametrize('dtype,bm,bn,bk', CASES)
def test_epilogue_native_lowering(dtype, bm, bn, bk):
    msl = compile_case(dtype, bm, bn, bk).lower()
    assert 'Source-replayed' in msl
    assert 'constant float& alpha' in msl and 'constant float& beta' in msl
    if bm * bn > 1024:
        assert f'_st < {bm * bn}u; _st += 1024u' in msl
        assert 'smem_iter_' in msl and 'epilogue_' in msl
    elif bm * bn < max(bm * bk, bk * bn):
        assert f'lid < {bm * bn}u' in msl


@pytest.mark.parametrize('dtype', ['fp16', 'bf16', 'fp32'])
def test_epilogue_budget_refuses_excess_but_admits_exact_boundary(dtype):
    import re
    from triton_msl.errors import MetalResourceError
    with pytest.raises(MetalResourceError) as caught:
        compile_case(dtype, 64, 64, 64).lower()
    assert (caught.value.required, caught.value.limit, caught.value.resource) == (
        49152, 32768, 'threadgroup memory bytes')
    lo = compile_case(dtype, 64, 64, 32)
    msl = lo.lower()
    # Independently inspect actual emitted declarations, not just the checker.
    arrays = re.findall(r'^\s*threadgroup (\w+) \w+\[(\d+)\];', msl, re.M)
    assert len(arrays) == 3 and all(ty == 'float' for ty, _ in arrays)
    assert sum(4 * int(n) for _, n in arrays) == lo._replay_shared_bytes == 32768


def test_replay_budget_uses_typed_physical_slots_and_refuses_unknown_facts():
    from triton_msl.codegen._lowerer_helpers import _alias_shared_memory, _check_replay_shared_memory_budget
    from triton_msl.codegen.msl_emitter import KernelBuilder
    from triton_msl.errors import MetalNonRecoverableError, MetalResourceError
    kb = KernelBuilder('budget_control')
    kb.declare_threadgroup_array('a', 'fp16', 8192)
    kb.declare_threadgroup_array('b', 'fp16', 8192)
    kb.raw_line('    a[lid] = 1.0f;')
    kb.raw_line('    threadgroup_barrier(mem_flags::mem_threadgroup);')
    kb.raw_line('    b[lid] = 2.0f;')
    aliases = {}
    _alias_shared_memory(kb.build(), allocation_aliases=aliases)
    # fp16 logical storage is emitted as FLOAT scratch; two logical arrays
    # reuse one physical slot. Counting logical arrays would wrongly refuse.
    assert aliases['a'] == aliases['b']
    assert _check_replay_shared_memory_budget(kb._threadgroup_arrays, aliases) == 32768
    with pytest.raises(MetalResourceError) as caught:
        _check_replay_shared_memory_budget(kb._threadgroup_arrays, aliases, extra_bytes=4)
    assert caught.value.required == 32772
    assert _check_replay_shared_memory_budget([('wide', 'i64', 4096)], {}) == 32768
    with pytest.raises(MetalNonRecoverableError):
        _check_replay_shared_memory_budget([('a', 'fp32', 8)], {'a': 'unknown'})
    with pytest.raises(MetalNonRecoverableError):
        _check_replay_shared_memory_budget([('a', 'fp32', 8), ('b', 'i64', 8)], {'a': 'b'})


def test_replay_capacity_never_masks_a_store_integrity_failure(monkeypatch):
    from triton_msl.errors import MetalNonRecoverableError
    lo = compile_case('fp16', 64, 64, 64)
    original = lo._lower_store
    error = MetalNonRecoverableError('injected source-store integrity failure', op_name='tt.store')
    visited = []
    def reject(store):
        visited.append(store.id)
        raise error
    monkeypatch.setattr(lo, '_lower_store', reject)
    with pytest.raises(MetalNonRecoverableError) as caught:
        lo.lower()
    assert visited and caught.value is error
    assert callable(original)


@pytest.mark.parametrize('dtype,bm,bn,bk', CASES)
def test_epilogue_runtime(dtype, bm, bn, bk, monkeypatch):
    assert torch.backends.mps.is_available(), 'This capability witness requires MPS'
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_utils
    monkeypatch.setenv('TRITON_MSL_USE_CPP', '0')
    monkeypatch.setenv('TRITON_MSL_COMPILE_SHADER', '0')
    dt = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[dtype]
    m, n, k = bm + 3, bn + 5, bk * 2
    mp, np = bm * 2, bn * 2
    g = torch.Generator().manual_seed(509)
    # Exactly representable operands and scalars keep an independent double oracle
    # insensitive to legal dot reassociation; nontrivial signs and column bias remain.
    a = (torch.randint(-8, 9, (mp, k), generator=g).float() / 32).to(dt)
    b = (torch.randint(-8, 9, (k, np), generator=g).float() / 32).to(dt)
    bi = (torch.randint(-8, 9, (n,), generator=g).float() / 32).to(dt)
    out = torch.full((mp, np), -2048., device='mps', dtype=dt)
    calls = []
    def observe(instance, real):
        def wrapper(pipeline, grid, group, buffers, **kwargs):
            calls.append((pipeline, tuple(grid), tuple(group)))
            return real(pipeline, grid, group, buffers, **kwargs)
        return wrapper
    patch_live_singleton_method(monkeypatch, _get_utils, 'launch', observe)
    h = _epilogue_mm[(4,)](a.to('mps'), b.to('mps'), bi.to('mps'), out,
                           m, n, k, k, np, np, -0.5, 1.25, BM=bm, BN=bn, BK=bk)
    torch.mps.synchronize()
    assert len(calls) == 1 and calls[0][0] is h.function and calls[0][1] == (4, 1, 1)
    assert calls[0][2] == (min(max(bm * bk, bk * bn, bm * bn), 1024), 1, 1)
    assert 'Source-replayed' in h.asm.get('msl', h.asm.get('metal', ''))
    actual = out.cpu()
    expected = (-0.5 * (a.double() @ b.double())[:m, :n] + 1.25 * bi.double()[None, :]).to(dt)
    torch.testing.assert_close(actual[:m, :n], expected, rtol=0, atol=0)
    assert torch.equal(actual[m:, :], torch.full_like(actual[m:, :], -2048.))
    assert torch.equal(actual[:m, n:], torch.full_like(actual[:m, n:], -2048.))


def test_wide_epilogue_nonfinite_column_and_finite_siblings(monkeypatch):
    assert torch.backends.mps.is_available()
    monkeypatch.setenv('TRITON_MSL_USE_CPP', '0')
    monkeypatch.setenv('TRITON_MSL_COMPILE_SHADER', '0')
    a = torch.ones((32, 64), device='mps', dtype=torch.float16)
    b = torch.ones((64, 64), dtype=torch.float16)
    b[:, 37] = float('inf')
    bias = torch.ones((64,), device='mps', dtype=torch.float16)
    out = torch.empty((32, 64), device='mps', dtype=torch.float16)
    h = _epilogue_mm[(1,)](a, b.to('mps'), bias, out, 32, 64, 64, 64, 64, 64,
                           -0.5, 1.25, BM=32, BN=64, BK=32)
    torch.mps.synchronize()
    assert 'Source-replayed' in h.asm.get('msl', h.asm.get('metal', ''))
    actual = out.cpu()
    assert not actual.isnan().any()
    assert torch.isneginf(actual[:, 37]).all()
    siblings = torch.cat((actual[:, :37], actual[:, 38:]), dim=1)
    assert torch.equal(siblings, torch.full_like(siblings, -30.75))
