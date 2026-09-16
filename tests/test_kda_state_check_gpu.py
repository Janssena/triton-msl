"""KDA state-only replay trigger and finite bits across check relocation."""
import pytest
import triton  # noqa: F401 — complete backend discovery before backend imports

from triton_msl.codegen._msl_templates import make_kda_kernel

try:
    import torch
    import Metal
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    HAS = Metal.MTLCreateSystemDefaultDevice() is not None and CompileShaderRuntime().available()
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + compile_shader needed")
_STATE_CHECK = "if(!isfinite(S[l*D+j])) atomic_store_explicit(&replay,1u,memory_order_relaxed);"


def _recurrent(q, k, v, a, beta):
    out = torch.zeros_like(v)
    for head in range(q.shape[0]):
        state = torch.zeros(64, 64)
        for token in range(q.shape[1]):
            state = a[head, token, :, None] * state
            u = k[head, token] @ state
            state = state + beta[head, token] * torch.outer(k[head, token], v[head, token] - u)
            out[head, token] = q[head, token] @ state
    return out


def _dispatch(runtime, library, inputs):
    zh, length, _ = inputs[0].shape
    output = torch.empty_like(inputs[0])
    runtime.dispatch(library, "kda_prefill", [*inputs, output, length],
                     threads=(zh * 256, 1, 1), group_size=(256, 1, 1))
    torch.mps.synchronize()
    return output.cpu()


@requires
def test_state_overflow_alone_replays_and_preserves_sibling(monkeypatch):
    from tests.cache_helpers import patch_live_singleton_method
    import triton_msl.kda as op

    q = torch.zeros(2, 8, 64)
    k = torch.full_like(q, 1e10)
    v = torch.zeros_like(q)
    v[0, -1] = 1e30
    a = torch.full_like(q, 0.5)
    beta = torch.full((2, 8), 0.25)
    # The other diagnostics cannot cover this case: all their operands and
    # chunk output are finite, but the final state accumulation overflows.
    B = a.cumprod(1)
    QT, KT, KH = q * B, k / B, k * B
    M = torch.tril(KH @ KT.transpose(1, 2), diagonal=-1) * beta[:, :, None]
    W = beta[:, :, None] * v
    for i in range(8):
        for j in range(i):
            W[:, i] -= M[:, i, j, None] * W[:, j]
    chunk_output = torch.tril(QT @ KT.transpose(1, 2)) @ W
    chunk_state = B[:, -1, :, None] * (KT.transpose(1, 2) @ W)
    assert all(torch.isfinite(x).all() for x in (q, k, v, a, beta, B, QT, KT, KH, M, W, chunk_output))
    assert torch.equal(chunk_output, torch.zeros_like(chunk_output))
    assert torch.isinf(chunk_state[0]).all()
    assert torch.equal(chunk_state[1], torch.zeros_like(chunk_state[1]))
    reference = _recurrent(q, k, v, a, beta)
    assert torch.isnan(reference[0, -1]).all()
    assert torch.equal(reference[:, :-1], torch.zeros_like(reference[:, :-1]))
    assert torch.equal(reference[1], torch.zeros_like(reference[1]))

    values = [x.to("mps") for x in (q, k, v, a, beta)]
    clean = [x.clone() for x in values]
    clean[2].zero_()
    runtime, library = op._kernel(fp16=False)
    hits = []

    def observe(instance, dispatch):
        assert instance is runtime

        def wrapped(actual_library, name, args, **kwargs):
            assert actual_library is library and name == "kda_prefill"
            assert kwargs == dict(threads=(512, 1, 1), group_size=(256, 1, 1))
            result = dispatch(actual_library, name, args, **kwargs)
            hits.append(name)
            return result

        return wrapped

    with monkeypatch.context() as context:
        patch_live_singleton_method(context, lambda: op._kernel(False)[0], "dispatch", observe)
        control = op.kda_attention(*clean).cpu()
        repeated = op.kda_attention(*clean).cpu()
        actual = op.kda_attention(*values).cpu()
    assert hits == ["kda_prefill"] * 3
    assert torch.equal(control.view(torch.uint8), repeated.view(torch.uint8))
    assert torch.equal(actual[1].view(torch.uint8), control[1].view(torch.uint8))
    for classify in (torch.isnan, torch.isposinf, torch.isneginf):
        assert torch.equal(classify(actual), classify(reference))
    finite = torch.isfinite(reference)
    assert torch.equal(actual[finite], reference[finite])

    # Test-only negative control: missing this predicate must produce the wrong
    # zero output. Never run this shader as a production path or time it.
    source = make_kda_kernel()
    assert source.count(_STATE_CHECK) == 1
    broken = runtime.get_library(source.replace(_STATE_CHECK, ""))
    wrong = _dispatch(runtime, broken, values)
    assert torch.equal(wrong, torch.zeros_like(wrong))
    assert not torch.equal(torch.isnan(wrong), torch.isnan(reference))


@requires
@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("length", [64, 256, 512])
def test_state_check_relocation_preserves_finite_output_bits(fp16, length):
    torch.manual_seed(759)
    q = torch.randn(2, length, 64)
    k = torch.nn.functional.normalize(torch.randn(2, length, 64), dim=-1)
    v = torch.randn(2, length, 64)
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn(2, length, 64))
    beta = torch.sigmoid(torch.randn(2, length))
    cpu = [x.to(torch.float16 if fp16 else torch.float32) for x in (q, k, v, a, beta)]
    reference = _recurrent(*(x.float() for x in cpu))
    inputs = [x.to("mps") for x in cpu]
    source = make_kda_kernel(fp16=fp16)
    # Reconstruct only the previous, fully checked diagnostic placement. This
    # control retains all exceptional-value checks and the same barriers.
    producer = "float Bl=B[(C-1u)*D+l]; S[l*D+j]=Bl*(S[l*D+j]+d);"
    assert source.count(producer + "\n      " + _STATE_CHECK + " }") == 1
    previous = source.replace(producer + "\n      " + _STATE_CHECK + " }", producer + " }")
    anchor = "    for(uint idx=lid;idx<C*D;idx+=NT)\n      if(!isfinite(W[idx])"
    assert previous.count(anchor) == 1
    previous = previous.replace(anchor,
        "    for(uint idx=lid;idx<D*D;idx+=NT)\n"
        "      if(!isfinite(S[idx])) atomic_store_explicit(&replay,1u,memory_order_relaxed);\n" + anchor)
    runtime = CompileShaderRuntime()
    baseline = runtime.get_library(previous)
    current = runtime.get_library(source)
    outputs = [_dispatch(runtime, library, inputs) for library in (baseline, current, baseline, current)]
    assert all(torch.equal(output.view(torch.uint8), outputs[0].view(torch.uint8)) for output in outputs)
    assert torch.isfinite(outputs[0]).all()
    relative = (outputs[0].float() - reference).abs().max().item() / max(reference.abs().max().item(), 1e-30)
    assert relative < (2e-2 if fp16 else 1e-4)
