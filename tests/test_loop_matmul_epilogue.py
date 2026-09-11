"""Issue11's K-loop alpha/beta workload, not merely its scalar ABI.

Adapted from Hocine Benkelaya / NeuroBrix's bledden/triton-msl issue11
reproducer. MODE supplies explicit proof-boundary controls only.
"""

import pytest
import torch
import triton
import triton.language as tl
from tests.test_template_scalar_abi import _lower
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _scaled_mm(A, B, Bias, C, M, N, K, SA, SB, SC, alpha, beta, MODE: tl.constexpr):
    pid = tl.program_id(0)
    nt = tl.cdiv(N, 32)
    pm, pn = pid // nt, pid % nt
    m = pm * 32 + tl.arange(0, 32)
    n = pn * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 32)
    acc = tl.full((32, 32), 1.0 if MODE == 1 else 0.0, tl.float32)
    for k in range(0, K, 32):
        a = tl.load(A + m[:, None] * SA + (k + kk)[None, :])
        b = tl.load(B + (k + kk)[:, None] * SB + n[None, :])
        acc = tl.dot(a, b, acc)
        if MODE == 2:
            acc = acc * 2.0
    bi = tl.load(Bias + (m if MODE == 3 else n), mask=(m < M) if MODE == 3 else (n < N), other=0.0)
    result = alpha * acc + beta * bi[None, :]
    tl.store(C + m[:, None] * SC + n[None, :], result, (m[:, None] < M) & (n[None, :] < N))


def _compile(mode):
    sig = {
        n: "*fp32" if n in ("A", "B", "Bias", "C") else "fp32" if n in ("alpha", "beta") else "i32"
        for n in _scaled_mm.arg_names
        if n != "MODE"
    }
    return _lower(_scaled_mm, sig, {"MODE": mode})


def test_loop_epilogue_admitted_at_native_lowering_boundary():
    obj = _compile(0)
    msl = obj.lower()
    assert "Source-replayed K-loop epilogue" in msl
    assert "constant float& alpha" in msl and "constant float& beta" in msl
    assert "alpha *" in msl and "beta *" in msl
    assert obj._fast_matmul is None


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_loop_epilogue_does_not_bypass_unproved_carry_or_bias(mode):
    with pytest.raises(MetalNonRecoverableError):
        _compile(mode).lower()


@pytest.mark.parametrize(
    "m,n,k,alpha,beta,programs",
    [
        (64, 64, 64, 1.0, 1.0, 4),
        (64, 96, 96, -0.5, 2.0, 6),
        (70, 97, 64, 1.25, -0.75, 8),
    ],
)
def test_loop_epilogue_computes_source_and_preserves_unlaunched_tiles(monkeypatch, m, n, k, alpha, beta, programs):
    if not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_utils

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    g = torch.Generator().manual_seed(493)
    mp, np = triton.cdiv(m, 32) * 32, triton.cdiv(n, 32) * 32
    ac = torch.randn(mp, k, generator=g) * 0.03
    bc = torch.randn(k, np, generator=g) * 0.03
    bias = torch.randn(n, generator=g) * 0.1
    a, b, bi = ac.to("mps"), bc.to("mps"), bias.to("mps")
    out = torch.full((mp, np), -8192.0, device="mps")
    calls = []

    def observe(instance, real):
        def wrapper(pipeline, grid, group, buffers, **kwargs):
            calls.append((pipeline, grid, group))
            return real(pipeline, grid, group, buffers, **kwargs)

        return wrapper

    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe)
    h = _scaled_mm[(programs,)](a, b, bi, out, m, n, k, k, np, np, alpha, beta, MODE=0)
    torch.mps.synchronize()
    assert len(calls) == 1 and calls[0][0] is h.function and tuple(calls[0][1]) == (programs, 1, 1)
    assert "Source-replayed K-loop epilogue" in h.asm.get("msl", h.asm.get("metal", ""))
    ref = (alpha * (ac.double() @ bc.double())[:m, :n] + beta * bias.double()[None, :]).float()
    expected = torch.full((mp, np), -8192.0)
    written = torch.zeros((mp, np), dtype=torch.bool)
    for pid in range(programs):
        r, c = divmod(pid, triton.cdiv(n, 32))
        rows = slice(r * 32, min((r + 1) * 32, m))
        cols = slice(c * 32, min((c + 1) * 32, n))
        expected[rows, cols] = ref[rows, cols]
        written[rows, cols] = True
    actual = out.cpu()
    assert torch.equal(actual[~written], expected[~written])
    torch.testing.assert_close(actual[written], expected[written], atol=2e-6, rtol=2e-5)
