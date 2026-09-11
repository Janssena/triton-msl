"""Q1 decode recovery. Kernel adapted from contributor PR7
(reporter _fa_two_lengths), retained as a source-semantic regression witness.
"""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _q1_attention(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    seqlen_q,
    seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """FA2 forward with independent query / key lengths.

    The scale is applied to Q and cast BACK to Q's dtype, which is what the
    reference implementations do and what puts an `arith.truncf` between the
    multiply and the dot on a fp16 kernel.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    qk_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    q = (q * qk_scale).to(q.dtype)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, seqlen_k, BLOCK_N):
        k_ptrs = (
            K
            + off_z * stride_kz
            + off_h * stride_kh
            + (start_n + offs_n)[:, None] * stride_kn
            + offs_d[None, :] * stride_kk
        )
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)

        qk = tl.dot(q, tl.trans(k).to(q.dtype))

        if IS_CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = (
            V
            + off_z * stride_vz
            + off_h * stride_vh
            + (start_n + offs_n)[:, None] * stride_vn
            + offs_d[None, :] * stride_vk
        )
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)

        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


def _kernel(mutation=None):
    fn = triton.jit(_q1_attention.fn)
    if mutation == "store":
        old = "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)"
        new = "tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < (seqlen_q - 1))"
    elif mutation == "score":
        old = "qk = tl.dot(q, tl.trans(k).to(q.dtype))"
        new = old + "\n        qk = qk * 0.5"
    elif mutation == "query_bound":
        old = "q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)"
        new = old.replace("seqlen_q", "seqlen_k")
    elif mutation == "key_bound":
        old = "k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)"
        new = old.replace("seqlen_k", "seqlen_q")
    elif mutation == "recurrence":
        old = "m_new = tl.maximum(m_i, m_ij)"
        new = "m_new = tl.minimum(m_i, m_ij)"
    elif mutation == "narrow_p":
        old = "tl.dot(p.to(tl.float32), v.to(tl.float32))"
        new = "tl.dot(p.to(q.dtype), v.to(q.dtype))"
    else:
        return fn
    assert fn.src.count(old) == 1
    fn._unsafe_update_src(fn.src.replace(old, new))
    return fn


def _lower(dtype="fp16", causal=False, mutation=None, block_m=32, query_one=True):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.generic_lowerer import GenericLowerer
    from triton_msl.codegen.mlir_walker import walk_ttgir

    kernel = _kernel(mutation)
    const = dict(
        BLOCK_M=block_m,
        BLOCK_N=32,
        HEAD_DIM=64,
        IS_CAUSAL=causal,
        seqlen_q=1,
        stride_qk=1,
        stride_kk=1,
        stride_vk=1,
        stride_ok=1,
        Z=1,
    )
    if not query_one:
        del const["seqlen_q"]
    signature = {n: "*" + dtype if n in ("Q", "K", "V", "Out") else "i32" for n in kernel.arg_names if n not in const}
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = ASTSource(kernel, signature=signature, constexprs=const).make_ir(
        target, options, backend.get_codegen_implementation(options), backend.get_module_map(), ctx
    )
    meta = {}
    mod = backend.make_ttir(mod, meta, options)
    mod = backend.make_ttgir(mod, meta, options)
    return GenericLowerer(walk_ttgir(mod, options), options)


@pytest.mark.parametrize(
    "nq,nk,bm,dtype,causal",
    [
        (1, 128, 8, torch.float16, False),
        (1, 128, 16, torch.float16, False),
        (8, 128, 16, torch.float32, False),
        (16, 128, 16, torch.float32, True),
        (33, 65, 16, torch.float32, True),
        (17, 65, 8, torch.float16, False),
    ],
)
def test_reporter_small_query_tiles_compute(monkeypatch, nq, nk, bm, dtype, causal):
    lo = _lower("fp16" if dtype == torch.float16 else "fp32", causal, block_m=bm, query_one=nq == 1)
    msl = lo.lower()
    assert "Head-dim-tiled FlashAttention-2" in msl and f"const uint BM = {bm}u;" in msl
    assert lo._flash_attention is None
    assert "q_row < 1u" in msl if nq == 1 else "q_row < fa_arg" in msl
    if not torch.backends.mps.is_available():
        pytest.skip("GPU proof requires MPS; lowering ran")
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_utils

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    gen = torch.Generator().manual_seed(493)
    qc = torch.randn(1, 2, triton.cdiv(nq, bm) * bm, 64, generator=gen).to(dtype)
    qc[:, :, nq:] = float("nan")
    kc = torch.randn(1, 2, nk, 64, generator=gen).to(dtype)
    vc = torch.randn(1, 2, nk, 64, generator=gen).to(dtype)
    q, k, v = [x.to("mps") for x in (qc, kc, vc)]
    out = torch.full_like(q, -8192)
    calls = []

    def observe(instance, real):
        def launch(pipeline, grid, group, buffers, **kwargs):
            calls.append((pipeline, grid, group))
            return real(pipeline, grid, group, buffers, **kwargs)

        return launch

    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe)
    h = _kernel()[(triton.cdiv(nq, bm), 2)](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        1,
        2,
        nq,
        nk,
        BLOCK_M=bm,
        BLOCK_N=32,
        HEAD_DIM=64,
        IS_CAUSAL=causal,
    )
    torch.mps.synchronize()
    assert len(calls) == 1 and calls[0][0] is h.function
    assert tuple(calls[0][1]) == (triton.cdiv(nq, bm), 2, 1) and tuple(calls[0][2]) == (bm * 32, 1, 1)
    qs = (qc[:, :, :nq].float() * 0.125).to(dtype).double()
    pad = triton.cdiv(nk, 32) * 32 - nk
    kp = torch.nn.functional.pad(kc.double(), (0, 0, 0, pad))
    vp = torch.nn.functional.pad(vc.double(), (0, 0, 0, pad))
    scores = qs @ kp.transpose(-1, -2)
    if causal:
        scores = scores.masked_fill(torch.arange(nq)[:, None] < torch.arange(nk + pad)[None, :], float("-inf"))
    ref = torch.softmax(scores, -1) @ vp
    actual = out.cpu()
    assert torch.equal(actual[:, :, nq:], torch.full_like(actual[:, :, nq:], -8192))
    torch.testing.assert_close(actual[:, :, :nq].double(), ref, atol=4e-3 if dtype == torch.float16 else 1e-5, rtol=0)


@pytest.mark.parametrize("mutation", ["query_bound", "key_bound", "recurrence", "narrow_p"])
def test_small_query_template_proves_each_role_and_recurrence(mutation):
    from triton_msl.errors import MetalNonRecoverableError

    with pytest.raises(MetalNonRecoverableError):
        _lower(mutation=mutation, block_m=16).lower()


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("causal", [False, True])
def test_folded_query_declines_dense_template_at_lowering(monkeypatch, dtype, causal):
    lower = _lower(dtype, causal)
    calls = []
    real = lower._lower_generic

    def generic():
        calls.append("generic")
        return real()

    def forbidden(*args, **kwargs):
        pytest.fail("single-length template must not claim the folded two-length source")

    monkeypatch.setattr(lower, "_lower_generic", generic)
    monkeypatch.setattr(lower, "_lower_flash_attention_template", forbidden)
    assert "kernel void" in lower.lower()
    assert calls == ["generic"]


@pytest.mark.parametrize("mutation", ["store", "score"])
def test_tighter_output_or_result_scale_stays_protected(mutation):
    from triton_msl.errors import MetalNonRecoverableError

    with pytest.raises(MetalNonRecoverableError, match="output store mask"):
        _lower(mutation=mutation).lower()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("dtype,tolerance", [(torch.float16, 4e-3), (torch.float32, 1e-5)])
@pytest.mark.parametrize("causal", [False, True])
def test_q1_source_and_untouched_rows_on_gpu(monkeypatch, dtype, tolerance, causal):
    from triton_msl.backend.driver import _get_utils
    from tests.cache_helpers import patch_live_singleton_method

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    gen = torch.Generator().manual_seed(485)
    q_cpu = torch.randn(1, 2, 32, 64, generator=gen).to(dtype)
    q_cpu[:, :, 1:] = float("nan")  # masked-away query rows must not poison row0
    k_cpu = torch.randn(1, 2, 128, 64, generator=gen).to(dtype)
    v_cpu = torch.randn(1, 2, 128, 64, generator=gen).to(dtype)
    q, k, v = [x.to("mps") for x in (q_cpu, k_cpu, v_cpu)]
    out = torch.full_like(q, -8192.0)
    launches = []

    def observe(instance, real):
        def launch(pipeline, grid, group, buffers, **kwargs):
            result = real(pipeline, grid, group, buffers, **kwargs)
            launches.append((pipeline, grid))
            return result

        return launch

    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe)
    compiled = _kernel()[(1, 2)](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        1,
        2,
        1,
        128,
        BLOCK_M=32,
        BLOCK_N=32,
        HEAD_DIM=64,
        IS_CAUSAL=causal,
    )
    torch.mps.synchronize()
    assert len(launches) == 1 and launches[0][0] is compiled.function
    assert tuple(launches[0][1]) == (1, 2, 1)
    assert compiled.metadata.flash_attention is None
    # Source rounds scaled Q back to its input dtype BEFORE the dot.
    qs = (q_cpu[:, :, :1].double() * 0.125).to(dtype).double()
    scores = qs @ k_cpu.double().transpose(-1, -2)
    if causal:
        scores[..., 1:] = -float("inf")  # source top-left causal alignment, query0
    ref = torch.softmax(scores, dim=-1) @ v_cpu.double()
    actual = out.cpu()
    assert torch.isfinite(actual[:, :, :1]).all()
    torch.testing.assert_close(actual[:, :, :1].double(), ref, atol=tolerance, rtol=0)
    assert torch.equal(actual[:, :, 1:], torch.full_like(actual[:, :, 1:], -8192.0)), (
        "masked output rows must stay bit-identical"
    )
