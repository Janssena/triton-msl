"""Gated DeltaNet / Kimi Delta Attention (KDA) prefill kernel.

Linear/delta-rule attention (per-key-dim gate + delta correction), the variant the
2026 frontier models use. It is NOT softmax attention and cannot be expressed as a
single ``@triton.jit`` kernel (the chunked form needs a UT-transform triangular solve),
so it ships as a direct compile_shader op, validated here against a recurrent gated-delta
ground truth. See ``make_kda_kernel`` for the algorithm.
"""

import pytest

# Import triton FIRST so backend discovery completes before triton_msl.backend is touched.
import triton  # noqa: F401

from triton_msl.codegen._msl_templates import make_kda_kernel

try:
    import torch

    import Metal
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None and CompileShaderRuntime().available()
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + compile_shader needed")


def _gdn_recurrent(q, k, v, a, beta):
    """Recurrent gated-delta ground truth: Sg=diag(a)S; u=k^T Sg; S=Sg+b k(v-u)^T; o=q^T S."""
    ZH, T, d = q.shape
    O = torch.zeros_like(v)
    for h in range(ZH):
        S = torch.zeros(d, d)
        for t in range(T):
            Sg = a[h, t].unsqueeze(1) * S
            u = k[h, t] @ Sg
            S = Sg + beta[h, t] * torch.outer(k[h, t], v[h, t] - u)
            O[h, t] = q[h, t] @ S
    return O


@requires
@pytest.mark.parametrize("T", [64, 512])
def test_kda_prefill_matches_recurrent(T):
    ZH, D = 8, 64  # the kernel is fixed at D=64, C=8; one threadgroup per head
    torch.manual_seed(0)
    q = torch.randn(ZH, T, D)
    k = torch.nn.functional.normalize(torch.randn(ZH, T, D), dim=-1)
    v = torch.randn(ZH, T, D)
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn(ZH, T, D))  # per-key-dim gate in (0.9, 1)
    beta = torch.sigmoid(torch.randn(ZH, T))
    ref = _gdn_recurrent(q, k, v, a, beta)

    rt = CompileShaderRuntime()
    lib = rt.get_library(make_kda_kernel())
    args = [t.contiguous().to("mps") for t in (q, k, v, a, beta)]
    out = torch.empty(ZH, T, D, device="mps")
    rt.dispatch(lib, "kda_prefill", args + [out, T], threads=(ZH * 256, 1, 1), group_size=(256, 1, 1))
    torch.mps.synchronize()

    rel = (out.cpu() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-4, f"KDA prefill rel err {rel:.2e} (T={T})"


@requires
def test_kda_attention_op_matches_recurrent():
    """The public op wraps the dispatch (reshape, cached library) and stays correct."""
    from triton_msl.kda import kda_attention

    ZH, D, T = 8, 64, 256
    torch.manual_seed(1)
    q = torch.randn(ZH, T, D)
    k = torch.nn.functional.normalize(torch.randn(ZH, T, D), dim=-1)
    v = torch.randn(ZH, T, D)
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn(ZH, T, D))
    beta = torch.sigmoid(torch.randn(ZH, T))
    ref = _gdn_recurrent(q, k, v, a, beta)

    out = kda_attention(q.to("mps"), k.to("mps"), v.to("mps"), a.to("mps"), beta.to("mps"))
    torch.mps.synchronize()
    rel = (out.cpu() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-4, f"kda_attention op rel err {rel:.2e}"


@requires
def test_kda_attention_op_rejects_bad_shape():
    """T not divisible by the chunk size, and wrong head dim, fail loudly (not silently wrong)."""
    from triton_msl.kda import kda_attention

    z = lambda *s: torch.zeros(*s, device="mps")
    with pytest.raises(ValueError):
        kda_attention(z(8, 60, 64), z(8, 60, 64), z(8, 60, 64), z(8, 60, 64), z(8, 60))  # T=60 not %8
    with pytest.raises(ValueError):
        kda_attention(z(8, 64, 32), z(8, 64, 32), z(8, 64, 32), z(8, 64, 32), z(8, 64))  # head dim != 64


@requires
def test_kda_decode_step_matches_recurrent(monkeypatch):
    """Autoregressive decode (state threaded across steps in place) matches the recurrent form."""
    from tests.cache_helpers import patch_live_singleton_method
    import triton_msl.kda as op
    kda_decode_step = op.kda_decode_step

    ZH, D, T = 8, 64, 48
    torch.manual_seed(2)
    q = torch.randn(ZH, T, D)
    k = torch.nn.functional.normalize(torch.randn(ZH, T, D), dim=-1)
    v = torch.randn(ZH, T, D)
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn(ZH, T, D))
    beta = torch.sigmoid(torch.randn(ZH, T))
    ref = _gdn_recurrent(q, k, v, a, beta)

    S = torch.zeros(ZH, D, D, device="mps")
    out = torch.empty(ZH, T, D)
    poisoned_state = torch.zeros_like(S)
    poisoned_out = torch.empty_like(out)
    poison_at = T // 2
    runtime, library = op._decode_kernel()
    hits = []

    def observe(instance, dispatch):
        assert instance is runtime

        def wrapped(lib, name, args, **kwargs):
            assert lib is library and name == "kda_decode"
            result = dispatch(lib, name, args, **kwargs)
            hits.append(name)
            return result

        return wrapped

    patch_live_singleton_method(monkeypatch, lambda: op._decode_kernel()[0], "dispatch", observe)
    for t in range(T):
        args = [x[:, t].to("mps") for x in (q, k, v, a, beta)]
        o = kda_decode_step(*args, S)
        poisoned_q = args[0].clone()
        if t == poison_at:
            poisoned_q[0, 0] = float("nan")
        p = kda_decode_step(poisoned_q, *args[1:], poisoned_state)
        torch.mps.synchronize()
        out[:, t] = o.cpu()
        poisoned_out[:, t] = p.cpu()
        # Q only reads state: even the poisoned step cannot change it.
        assert torch.equal(S.cpu(), poisoned_state.cpu())
    assert hits == ["kda_decode"] * (2 * T)
    assert torch.equal(poisoned_out[:, :poison_at], out[:, :poison_at])
    assert torch.equal(poisoned_out[1:], out[1:])
    assert torch.equal(poisoned_out[0, poison_at + 1:], out[0, poison_at + 1:])
    assert torch.isnan(poisoned_out[0, poison_at]).all()
    rel = (out - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-4, f"kda_decode_step rel err {rel:.2e}"


@requires
def test_kda_attention_fp16():
    """fp16 I/O (fp32 accumulate + state) routes to the half kernel; fp16-grade tolerance."""
    from triton_msl.kda import kda_attention

    ZH, D, T = 8, 64, 256
    torch.manual_seed(3)
    q = torch.randn(ZH, T, D)
    k = torch.nn.functional.normalize(torch.randn(ZH, T, D), dim=-1)
    v = torch.randn(ZH, T, D)
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn(ZH, T, D))
    beta = torch.sigmoid(torch.randn(ZH, T))
    ref = _gdn_recurrent(q, k, v, a, beta)

    args = [t.half().to("mps") for t in (q, k, v, a, beta)]
    out = kda_attention(*args)
    torch.mps.synchronize()
    assert out.dtype == torch.float16
    rel = (out.float().cpu() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 2e-2, f"fp16 KDA rel {rel:.2e}"


@requires
@pytest.mark.parametrize("case,dtype", [
    ("query_inf", torch.float32), ("query_inf", torch.float16),
    ("future_value_nan", torch.float32), ("gate_underflow", torch.float32),
    ("chunk_dot_overflow", torch.float32), ("key_quotient_overflow", torch.float32),
])
def test_kda_prefill_preserves_exceptional_recurrence(case, dtype, monkeypatch):
    """Chunk algebra must not invent Inf*0 or read a future poisoned token."""
    from tests.cache_helpers import patch_live_singleton_method
    import triton_msl.kda as op

    q = torch.full((2, 16, 64), 1 / 64, dtype=dtype)
    k, v = q.clone(), torch.full_like(q, 1 / 8)
    a = torch.full_like(q, 1 / 2)
    beta = torch.full((2, 16), 1 / 4, dtype=dtype)
    clean_inputs = [x.clone() for x in (q, k, v, a, beta)]
    if case == "query_inf":
        q[0, 7, 0] = float("inf")
    elif case == "future_value_nan":
        v[0, 7, 0] = float("nan")
    elif case == "gate_underflow":
        a[0] = 2 ** -20  # Valid (0,1) gates; the eight-term product underflows.
    else:
        k[0] = 1e20 if case == "chunk_dot_overflow" else 1e38
        v[0] = 0  # State stays exactly zero; no source k*k or k/B exists.
    ref = _gdn_recurrent(*(x.float() for x in (q, k, v, a, beta))).to(dtype)
    if case == "query_inf":
        # Analytic witness: state is strictly positive; q is not a state input.
        assert torch.isposinf(ref[0, 7]).all()
        assert torch.isfinite(ref[0, :7]).all() and torch.isfinite(ref[0, 8:]).all()
    if case == "future_value_nan":
        assert torch.isfinite(ref[0, :7]).all()
    if case in ("chunk_dot_overflow", "key_quotient_overflow"):
        assert torch.equal(ref[0], torch.zeros_like(ref[0]))
    rt, lib = op._kernel(fp16=dtype == torch.float16)
    hits = []

    def observe(runtime, dispatch):
        assert runtime is rt

        def wrapped(actual_lib, name, args, **kwargs):
            assert actual_lib is lib and name == "kda_prefill"
            assert kwargs == dict(threads=(512, 1, 1), group_size=(256, 1, 1))
            result = dispatch(actual_lib, name, args, **kwargs)
            hits.append(name)
            return result

        return wrapped

    patch_live_singleton_method(monkeypatch, lambda: op._kernel(dtype == torch.float16)[0], "dispatch", observe)
    clean_gpu = [x.to("mps") for x in clean_inputs]
    control = op.kda_attention(*clean_gpu).cpu()
    repeated_control = op.kda_attention(*clean_gpu).cpu()
    assert torch.equal(control, repeated_control), "all-finite repeated execution changed bits"
    control_ref = _gdn_recurrent(*(x.float() for x in clean_inputs)).to(dtype)
    torch.testing.assert_close(control, control_ref, atol=2e-5, rtol=2e-5)
    actual = op.kda_attention(*(x.to("mps") for x in (q, k, v, a, beta))).cpu()
    assert hits == ["kda_prefill"] * 3
    for classify in (torch.isnan, torch.isposinf, torch.isneginf):
        assert torch.equal(classify(actual), classify(ref)), case
    # Only head 0 changes algorithm. Do not let the approximate source check
    # excuse even a one-ULP perturbation of the unreplayed sibling head.
    assert torch.equal(actual[1], control[1]), "unreplayed sibling head changed bits"
    finite = torch.isfinite(ref[0])
    assert finite.any() and torch.isfinite(ref[1]).all()
    torch.testing.assert_close(actual[0][finite], ref[0][finite], atol=2e-5, rtol=2e-5)
    if case in ("chunk_dot_overflow", "key_quotient_overflow"):
        assert torch.equal(actual[0], torch.zeros_like(actual[0]))
