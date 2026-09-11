"""PR7's bias and independent-length workload, including folded Q=1.

Reconstructed from the forward/decode algorithm reported by Hocine Benkelaya
(NeuroBrix) in bledden/triton-msl PR7; not a verbatim submitted bias reproducer.
"""

import pytest
import torch
import triton
import triton.language as tl
from tests.test_template_scalar_abi import _lower


@triton.jit
def _biased_decode(
    Q,
    K,
    V,
    Out,
    Bias,
    qz,
    qh,
    qm,
    qd,
    kz,
    kh,
    kn,
    kd,
    vz,
    vh,
    vn,
    vd,
    oz,
    oh,
    om,
    od,
    bz,
    bh,
    bm,
    bn,
    heads,
    nq,
    nk,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    MODE: tl.constexpr,
):
    row = tl.program_id(0) * 32 + tl.arange(0, 32)
    hz = tl.program_id(1)
    z = hz // heads
    h = hz % heads
    col = tl.arange(0, 32)
    d = tl.arange(0, D)
    q = tl.load(Q + z * qz + h * qh + row[:, None] * qm + d[None, :] * qd, row[:, None] < nq, other=0)
    q = (q * (1.0 / tl.sqrt(float(D)))).to(q.dtype)
    maximum = tl.full((32,), float("-inf"), tl.float32)
    denom = tl.zeros((32,), tl.float32)
    acc = tl.zeros((32, D), tl.float32)
    for start in range(0, nk, 32):
        key = start + col
        k = tl.load(K + z * kz + h * kh + key[:, None] * kn + d[None, :] * kd, key[:, None] < nk, other=0)
        bias = tl.load(
            Bias + z * bz + h * bh + row[:, None] * bm + key[None, :] * bn,
            (row[:, None] < nq) & (key[None, :] < nk),
            other=0,
        )
        if MODE == 1:
            score = tl.dot(q, tl.trans(k).to(q.dtype)) + bias.to(tl.float32)
        else:
            score = tl.dot(q, tl.trans(k).to(q.dtype), bias.to(tl.float32))
        if MODE == 2:
            score = score * 0.5
        if CAUSAL:
            score = tl.where(row[:, None] >= key[None, :], score, float("-inf"))
        if MODE == 3:
            score = score * 0.5
        newmax = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.exp(maximum - newmax)
        p = tl.exp(score - newmax[:, None])
        denom = denom * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + z * vz + h * vh + key[:, None] * vn + d[None, :] * vd, key[:, None] < nk, other=0)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        maximum = newmax
    out = acc / denom[:, None]
    tl.store(
        Out + z * oz + h * oh + row[:, None] * om + d[None, :] * od, out.to(Out.dtype.element_ty), row[:, None] < nq
    )


def _compile(dtype="fp32", d=32, mode=0):
    const = dict(D=d, CAUSAL=True, MODE=mode, nq=1)
    sig = {
        n: "*" + dtype if n in ("Q", "K", "V", "Out") else "*fp32" if n == "Bias" else "i32"
        for n in _biased_decode.arg_names
        if n not in const
    }
    return _lower(_biased_decode, sig, const)


def test_bias_without_lse_replays_native_graph():
    lo = _compile()
    msl = lo.lower()
    assert "Source-replayed biased attention" in msl
    assert lo._flash_attention is None
    assert "Bias[" in msl and "threadgroup_barrier" in msl


@pytest.mark.parametrize("d,mode", [(128, 0), (32, 2), (32, 3)])
def test_unqualified_shape_or_score_scale_is_not_admitted(d, mode):
    from triton_msl.errors import MetalNonRecoverableError

    with pytest.raises(MetalNonRecoverableError):
        _compile(d=d, mode=mode).lower()


@pytest.mark.parametrize(
    "dtype,nq,nk,d,causal,mode",
    [
        (torch.float32, 1, 64, 32, False, 0),
        (torch.float16, 1, 65, 64, True, 0),
        (torch.float32, 8, 48, 32, False, 1),
        (torch.float16, 32, 96, 64, False, 0),
        (torch.float32, 49, 65, 64, True, 0),
        (torch.float16, 33, 48, 32, True, 1),
    ],
)
def test_biased_decode_computes_and_keeps_masked_rows(monkeypatch, dtype, nq, nk, d, causal, mode):
    if not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_utils

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    gen = torch.Generator().manual_seed(493)
    np = triton.cdiv(nq, 32) * 32
    # Different batches and heads; non-unit bias column stride is a real address test.
    qc = (torch.randn(2, 2, np, d, generator=gen) * 0.2).to(dtype)
    qc[:, :, nq:] = float("nan")
    kc = (torch.randn(2, 2, nk, d, generator=gen) * 0.2).to(dtype)
    vc = (torch.randn(2, 2, nk, d, generator=gen) * 0.3).to(dtype)
    bc = torch.randn(2, 2, nq, nk * 2, generator=gen) * 0.4
    q, k, v = [x.to("mps") for x in (qc, kc, vc)]
    b = bc.to("mps")[..., ::2]
    o = torch.full_like(q, -8192)
    calls = []

    def observe(instance, real):
        def launch(pipeline, grid, group, buffers, **kwargs):
            calls.append((pipeline, grid))
            return real(pipeline, grid, group, buffers, **kwargs)

        return launch

    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe)
    handle = _biased_decode[(triton.cdiv(nq, 32), 4)](
        q,
        k,
        v,
        o,
        b,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        *b.stride(),
        2,
        nq,
        nk,
        D=d,
        CAUSAL=causal,
        MODE=mode,
    )
    torch.mps.synchronize()
    assert len(calls) == 1 and calls[0][0] is handle.function
    assert tuple(calls[0][1]) == (triton.cdiv(nq, 32), 4, 1)
    assert handle.metadata.flash_attention is None
    assert "Source-replayed biased attention" in handle.asm.get("msl", handle.asm.get("metal", ""))
    # Literal source has zero K/V/bias padding, NOT a -inf logits padding mask.
    pad = triton.cdiv(nk, 32) * 32 - nk
    ks = torch.nn.functional.pad(kc.double(), (0, 0, 0, pad))
    vs = torch.nn.functional.pad(vc.double(), (0, 0, 0, pad))
    bs = torch.nn.functional.pad(bc[..., ::2].double(), (0, pad))
    qs = (qc[:, :, :nq].float() * (d**-0.5)).to(dtype).double()
    score = qs @ ks.transpose(-1, -2) + bs
    if causal:
        score = score.masked_fill(torch.arange(nq)[:, None] < torch.arange(nk + pad)[None, :], float("-inf"))
    ref = torch.softmax(score, -1) @ vs
    actual = o.cpu()
    assert torch.equal(actual[:, :, nq:], torch.full_like(actual[:, :, nq:], -8192))
    torch.testing.assert_close(
        actual[:, :, :nq].double(),
        ref,
        atol=3e-4 if dtype == torch.float16 else 2e-6,
        rtol=1e-3 if dtype == torch.float16 else 2e-5,
    )


@pytest.mark.parametrize("poison", ["query_nan", "value_inf", "bias_negative_inf"])
def test_biased_replay_nonfinite_rows_do_not_contaminate_siblings(monkeypatch, poison):
    if not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_utils

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    q = torch.zeros((1, 2, 32, 32), device="mps")
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    b = torch.zeros_like(q)
    out = torch.empty_like(q)
    calls = []

    def observe(instance, real):
        def launch(pipeline, *args, **kwargs):
            calls.append(pipeline)
            return real(pipeline, *args, **kwargs)

        return launch

    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe)

    def run():
        h = _biased_decode[(1, 2)](
            q,
            k,
            v,
            out,
            b,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            *b.stride(),
            2,
            32,
            32,
            D=32,
            CAUSAL=False,
            MODE=0,
        )
        torch.mps.synchronize()
        assert calls[-1] is h.function and "Source-replayed biased attention" in h.asm.get(
            "msl", h.asm.get("metal", "")
        )
        return out.cpu().clone()

    clean = run()
    assert torch.equal(clean, torch.ones_like(clean))
    if poison == "query_nan":
        q[0, 0, 0, 0] = float("nan")
    elif poison == "value_inf":
        v[0, 0, :, 0] = float("inf")
    else:
        b[0, 0, 0, :] = float("-inf")
    actual = run()
    assert len(calls) == 2
    assert torch.equal(actual[:, 1].view(torch.int32), clean[:, 1].view(torch.int32))
    if poison == "value_inf":
        assert torch.isposinf(actual[0, 0, :, 0]).all()
        assert torch.equal(actual[0, 0, :, 1:], clean[0, 0, :, 1:])
    else:
        assert torch.isnan(actual[0, 0, 0]).all()
        assert torch.equal(actual[0, 0, 1:].view(torch.int32), clean[0, 0, 1:].view(torch.int32))
