"""Grouped tutorial coordinates: prove the source mapping before replaying it.

Reconstructs the grouped spelling described in PR6, not a reporter-exact kernel.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _grouped(
    A,
    B,
    C,
    M,
    N,
    K,
    SAM,
    SAK,
    SBK,
    SBN,
    SCM,
    SCN,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    MUT: tl.constexpr,
):
    pid = tl.program_id(1 if MUT == 5 else 0)
    nm = tl.cdiv(M.to(tl.int16).to(tl.int32) if MUT == 7 else M, BM)
    nn = tl.cdiv(N, BN)
    group_id = pid // (GM * nn)
    first_m = group_id * (GM + 1 if MUT == 1 else GM)
    gm = tl.minimum((nn if MUT == 2 else nm) - first_m, GM)
    pm = first_m + (pid % (GM * nn)) % gm
    pn = (pid % (GM * nn)) // (GM if MUT == 3 else gm)
    am = (pm * BM + tl.arange(0, BM)) % M
    if MUT == 6:
        am = am.to(tl.int16).to(tl.int32)
    bn = (pn * BN + tl.arange(0, BN)) % N
    kk = tl.arange(0, BK)
    ap = A + am[:, None] * SAM + kk[None, :] * SAK
    bp = B + kk[:, None] * SBK + bn[None, :] * SBN
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap, kk[None, :] < K - k * BK, other=0.0)
        b = tl.load(bp, kk[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(a, b, acc=acc)
        ap += BK * SAK
        bp += BK * SBK
    cm = (pm + 1 if MUT == 4 else pm) * BM + tl.arange(0, BM)
    cn = pn * BN + tl.arange(0, BN)
    tl.store(C + cm[:, None] * SCM + cn[None, :] * SCN, acc, (cm[:, None] < M) & (cn[None, :] < N))


def _lower(gm=8, mutation=0):
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    signature = {
        n: "*fp16" if n in ("A", "B", "C") else "i32"
        for n in _grouped.arg_names
        if n not in ("BM", "BN", "BK", "GM", "MUT")
    }
    source = ASTSource(_grouped, signature=signature, constexprs=dict(BM=32, BN=32, BK=32, GM=gm, MUT=mutation))
    context = ir.context()
    ir.load_dialects(context)
    mod = source.make_ir(
        target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context
    )
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    lowerer = GenericLowerer(walk_ttgir(mod, options), options)
    return lowerer, str(mod)


@pytest.mark.parametrize("gm", [4, 8])
def test_grouped_coordinates_are_proved_and_replayed(gm):
    lowerer, ttgir = _lower(gm)
    assert "arith.minsi" in ttgir
    msl = lowerer.lower()
    assert "grouped source tile mapping" in msl
    assert f"* {gm}u" in msl
    assert "min(as_type<int>(as_type<uint>(_npm) - as_type<uint>(_first_m))" in msl
    assert "pid_m = as_type<uint>(_first_m) + as_type<uint>(_within % _group_m)" in msl
    assert "pid_n = as_type<uint>(_within / _group_m)" in msl


@pytest.mark.parametrize("mutation", [1, 2, 3, 4, 5, 6, 7])
def test_unreplayed_grouped_near_miss_refuses(mutation):
    lowerer, _ = _lower(mutation=mutation)
    with pytest.raises(MetalNonRecoverableError):
        lowerer.lower()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize(
    "m,n,programs,overflow", [(256, 64, None, False), (288, 70, None, False), (288, 96, 25, False), (288, 64, 9, True)]
)
def test_grouped_full_tail_and_partial_launch_compute_source(monkeypatch, m, n, programs, overflow):
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_utils

    gen = torch.Generator().manual_seed(459)
    a_cpu = torch.randint(-2, 3, (m, 32), generator=gen).to(torch.float16)
    b_cpu = torch.randint(-2, 3, (32, n), generator=gen).to(torch.float16)
    a, b = a_cpu.to("mps"), b_cpu.to("mps")
    out = torch.full((m, n), -8192.0, dtype=torch.float16, device="mps")
    cm, cn = triton.cdiv(m, 32), triton.cdiv(n, 32)
    count = cm * cn if programs is None else programs
    launches = []

    def observe(instance, real):
        def wrapper(pipeline, grid, group, buffers, **kw):
            result = real(pipeline, grid, group, buffers, **kw)
            launches.append((pipeline, grid, group))
            return result

        return wrapper

    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe)
    logical_m = 2147483647 if overflow else m
    compiled = _grouped[(count,)](a, b, out, logical_m, n, 32, 32, 1, n, 1, n, 1, BM=32, BN=32, BK=32, GM=8, MUT=0)
    torch.mps.synchronize()
    assert len(launches) == 1
    assert launches[0][0] is compiled.function
    assert tuple(launches[0][1]) == (count, 1, 1)
    assert any("grouped source tile mapping" in value for value in compiled.asm.values() if isinstance(value, str))
    # Enumerate rows in groups independently of the shader's quotient/remainder DAG.
    tile_order = [
        (row, col) for first in range(0, cm, 8) for col in range(cn) for row in range(first, min(first + 8, cm))
    ]
    ref = a_cpu.double() @ b_cpu.double()
    if overflow:
        # Source i32 cdiv numerator wraps negative; gm is negative. For these
        # nine programs remsi(0..8, gm) is 0..8 and divsi(0..8, gm) is zero.
        # Only allocated rows 0..287 / columns 0..31 are accessed; all in bounds.
        tile_order = [(row, 0) for row in range(count)]
    expected = torch.full_like(out.cpu(), -8192.0)
    for row, col in tile_order[:count]:
        section = (slice(row * 32, (row + 1) * 32), slice(col * 32, (col + 1) * 32))
        expected[section] = ref[section].half()
    assert torch.equal(out.cpu(), expected), "computed tiles and untouched sentinel tiles must both match"
