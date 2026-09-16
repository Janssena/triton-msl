"""Attention combinations adjacent to the recovered PR7 source family.

Own reconstruction of the additive-bias/independent-length feature, not the
reporter's unavailable full application. The oracle preserves Q and P rounding.
"""

import pytest
import torch
import triton
import triton.language as tl
from tests.test_template_scalar_abi import _lower


@triton.jit
def _attention(
    Q,
    K,
    V,
    Bias,
    Out,
    NQ,
    NK,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NARROW: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    ds = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * D + ds[None, :], rows[:, None] < NQ, other=0)
    q = (q * (1.0 / tl.sqrt(float(D)))).to(q.dtype)
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for start in range(0, NK, BN):
        keys = start + cols
        k = tl.load(K + keys[:, None] * D + ds[None, :], keys[:, None] < NK, other=0)
        bias = tl.load(Bias + rows[:, None] * NK + keys[None, :], (rows[:, None] < NQ) & (keys[None, :] < NK), other=0)
        score = tl.dot(q.to(tl.float32), tl.trans(k).to(tl.float32), bias)
        score = tl.where(keys[None, :] < NK, score, float("-inf"))
        if CAUSAL:
            score = tl.where(rows[:, None] >= keys[None, :], score, float("-inf"))
        newmax = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.exp(maximum - newmax)
        p = tl.exp(score - newmax[:, None])
        denom = denom * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + keys[:, None] * D + ds[None, :], keys[:, None] < NK, other=0)
        if NARROW:
            acc += tl.dot(p.to(v.dtype), v)
        else:
            acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        maximum = newmax
    tl.store(Out + rows[:, None] * D + ds[None, :], acc / denom[:, None], rows[:, None] < NQ)


PROBED_CASES = [
    ("fp16", 32, 32, 32, 1, 65, False, False),
    ("fp16", 16, 32, 32, 1, 65, False, False),
    ("fp16", 8, 32, 64, 1, 65, False, True),
    ("fp16", 16, 32, 64, 19, 65, False, False),
    ("fp16", 32, 32, 64, 19, 65, True, True),
    ("bf16", 32, 32, 32, 19, 65, False, False),
    ("fp16", 32, 64, 64, 19, 97, False, False),
    ("fp16", 64, 32, 64, 65, 97, False, True),
]

# Keep the original eight-row discovery matrix visible. The final two rows use
# a separately proved full-row replay because their score tiles exceed Metal's
# physical 1024-thread threadgroup limit.
CASES = PROBED_CASES[:6]
OPEN_CASES = PROBED_CASES[6:]

# Adjacent complete/tail key blocks and a second query program on both wide
# forms. Same source, oracle, numerical tolerance and canary checks below.
WIDE_REUSE_CASES = [
    (dtype, bm, bn, d, bm + 1, keys, narrow, causal)
    for dtype, bm, bn, d, _nq, _nk, narrow, causal in OPEN_CASES
    for keys in (bn - 1, bn, bn + 1)
]


def compile_case(case, fn=_attention):
    dtype, bm, bn, d, nq, nk, narrow, causal = case
    sig = {
        n: "*" + dtype if n in ("Q", "K", "V") else "*fp32" if n in ("Bias", "Out") else "i32"
        for n in fn.arg_names
        if n not in ("D", "BM", "BN", "NARROW", "CAUSAL")
    }
    return _lower(fn, sig, {"D": d, "BM": bm, "BN": bn, "NARROW": narrow, "CAUSAL": causal})


def reference(q, k, v, bias, bn, narrow, causal):
    nq, d = q.shape
    # The source multiplies the loaded Q in its storage precision, then widens.
    q = (q.float() * (d**-0.5)).to(q.dtype).double()
    k, vv = k.double(), v.double()
    maximum = torch.full((nq,), -float("inf"), dtype=torch.float64)
    denom = torch.zeros(nq, dtype=torch.float64)
    acc = torch.zeros((nq, d), dtype=torch.float64)
    for start in range(0, len(k), bn):
        end = min(start + bn, len(k))
        score = q @ k[start:end].T + bias[:, start:end].double()
        if causal:
            score = score.masked_fill(torch.arange(nq)[:, None] < torch.arange(start, end)[None, :], -float("inf"))
        newmax = torch.maximum(maximum, score.max(1).values)
        alpha = (maximum - newmax).exp()
        p = (score - newmax[:, None]).exp()
        denom = denom * alpha + p.sum(1)
        pp = p.to(v.dtype).double() if narrow else p
        acc = acc * alpha[:, None] + pp @ vv[start:end]
        maximum = newmax
    return (acc / denom[:, None]).float()


@pytest.mark.parametrize("case", CASES)
def test_attention_native(case):
    import re

    msl = compile_case(case).lower()
    assert "Source-replayed biased attention" in msl
    # The reduction must complete EVERY row's input reads before any result
    # write can reuse that storage. This checks the emitted synchronization,
    # not just the success of a timing-dependent GPU witness.
    assert re.search(
        r"reduced_\d+ = acc;\s*}\s*threadgroup_barrier\(mem_flags::mem_threadgroup\);\s*if \(lid < \d+u\)", msl
    )
    # A scalar transpose read also needs a completion barrier before a later
    # staging operation can reuse the same allocation.
    assert re.search(r"float trans_\d+ = [^;]+;\s*threadgroup_barrier\(mem_flags::mem_threadgroup\);", msl)


@pytest.mark.parametrize("case", OPEN_CASES)
def test_wider_score_tiles_use_zero_scratch_row_replay(case):
    msl = compile_case(case).lower()
    assert "Source-replayed wide biased attention" in msl
    assert "one complete query row per lane" in msl
    assert "threadgroup float " not in msl
    assert "float _p523_output[64];" in msl
    result = msl.index("_p523_output[_p515_od] = _p515_acc / _p515_denom;")
    barrier = msl.index("threadgroup_barrier(mem_flags::mem_device);")
    store = msl.index("Out[_p515_row * 64u + _p523_od] = _p523_output[_p523_od];")
    assert result < barrier < store
    assert "Out[" not in msl[:barrier]


@pytest.mark.parametrize("case", OPEN_CASES)
def test_wide_score_reuse_is_outside_output_column_loop(case):
    """Pin the work reduction and its ordering at the real lowering boundary."""
    import re

    obj = compile_case(case)
    msl = obj.lower()
    _, bm, bn, d, *_ = case
    assert obj._actual_dispatch_threads == bm
    assert obj._replay_shared_bytes == 0
    assert not obj.kb._threadgroup_arrays
    assert f"float _p553_block_values[{bn}];" in msl
    # One emitted dot loop, nested only in the key/block loops. The previous
    # emitter had four sites, two of them inside the output-column loop.
    assert len(re.findall(r"for \(uint _p515_kd_", msl)) == 1
    assert msl.count("float _p515_score_d0 = Bias[") == 1
    block = msl.index("for (int _p515_start_d")
    score = msl.index("float _p515_score_d0")
    maximum = msl.index("float _p515_newmax_d")
    probability = msl.index("float _p553_p = exp(")
    denom = msl.index("_p515_denom = _p515_denom * _p515_alpha_d + _p515_p_sum_d;")
    output_loop = msl.index(f"for (uint _p515_od = 0u; _p515_od < {d}u; ++_p515_od)", denom)
    rescale = msl.index("_p515_acc *= _p515_alpha_d;")
    valid_key = msl.index("if (_p515_key_a1 < NK)", rescale)
    accumulate = msl.index("_p515_acc += _p553_block_values[_p515_j_a1]", valid_key)
    normalize = msl.index("_p523_output[_p515_od] = _p515_acc / _p515_denom;")
    barrier = msl.index("threadgroup_barrier(mem_flags::mem_device);")
    assert (
        block
        < score
        < maximum
        < probability
        < denom
        < output_loop
        < rescale
        < valid_key
        < accumulate
        < normalize
        < barrier
    )
    assert "Bias[" not in msl[output_loop:]
    assert "Q[" not in msl[output_loop:]
    assert "K[" not in msl[output_loop:]
    assert "return;" not in msl
    assert "Out[" not in msl[:barrier]
    assert re.search(r"}\s*threadgroup_barrier\(mem_flags::mem_device\);\s*if \(_p523_active\)", msl)


def test_finite_f32_tail_sentinel_has_distinct_analytic_semantics():
    # In the second BN64 block, 33 valid keys score zero while 31 padded
    # positions score 64512. The valid probabilities underflow and padded V
    # values are zero, unlike the canonical -Inf result of one.
    valid = torch.exp(torch.full((33,), -64512.0, dtype=torch.float64)).sum()
    padded = torch.ones(31, dtype=torch.float64).sum()
    source_result = valid / (valid + padded)
    assert source_result.item() == 0.0
    assert source_result.item() != 1.0


@pytest.mark.parametrize("case", OPEN_CASES)
def test_finite_f32_tail_sentinel_does_not_match_negative_infinity(case):
    from triton_msl.errors import MetalNonRecoverableError

    fn = triton.JITFunction(_attention.fn)
    needle = 'score, float("-inf"))'
    assert needle in fn.src
    fn._unsafe_update_src(fn.src.replace(needle, "score, 64512.0)"))
    with pytest.raises(MetalNonRecoverableError):
        compile_case(case, fn).lower()


@pytest.mark.parametrize(
    "case,poison",
    [(case, False) for case in PROBED_CASES]
    + [(CASES[2], True), (CASES[4], True)]
    + [(case, True) for case in OPEN_CASES]
    + [(case, False) for case in WIDE_REUSE_CASES]
    + [(case, poison) for case in OPEN_CASES for poison in ("masked_row", "nan_bias", "inf_bias")],
)
def test_attention_runtime(case, poison, monkeypatch):
    assert torch.backends.mps.is_available()
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_utils

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    dtype, bm, bn, d, nq, nk, narrow, causal = case
    dt = torch.float16 if dtype == "fp16" else torch.bfloat16
    g = torch.Generator().manual_seed(511)
    q = torch.randn((nq, d), generator=g).to(dt)
    k = torch.randn((nk, d), generator=g).to(dt)
    v = torch.randn((nk, d), generator=g).to(dt)
    bias = torch.randn((nq, nk), generator=g) * 0.2
    if poison is True:
        q.zero_()
        k.zero_()
        bias.zero_()
        v.fill_(1)
        v[0, 7] = float("inf")
    elif poison == "masked_row":
        bias[3, :] = -float("inf")
    elif poison == "nan_bias":
        bias[3, 0] = float("nan")
    elif poison == "inf_bias":
        bias[3, 0] = float("inf")
    out = torch.full((triton.cdiv(nq, bm) * bm + 1, d), -8192.0, device="mps")
    calls = []

    def observe(instance, real):
        def wrapper(pipeline, grid, group, buffers, **kwargs):
            calls.append(pipeline)
            return real(pipeline, grid, group, buffers, **kwargs)

        return wrapper

    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe)
    h = _attention[(triton.cdiv(nq, bm),)](
        q.to("mps"),
        k.to("mps"),
        v.to("mps"),
        bias.to("mps"),
        out,
        nq,
        nk,
        D=d,
        BM=bm,
        BN=bn,
        NARROW=narrow,
        CAUSAL=causal,
    )
    torch.mps.synchronize()
    assert calls == [h.function]
    actual = out.cpu()
    if poison is True:
        # Every valid query attends key zero with positive probability. Only
        # column seven can be infinite; all other columns average exact ones.
        expected = torch.ones((nq, d))
        expected[:, 7] = float("inf")
    else:
        expected = reference(q, k, v, bias, bn, narrow, causal)
    if narrow and not poison:
        widened = reference(q, k, v, bias, bn, False, causal)
        # The inputs make probability rounding observable at the unchanged
        # numeric bar. An implementation discarding the cast cannot pass.
        with pytest.raises(AssertionError):
            torch.testing.assert_close(widened, expected, atol=3e-5, rtol=3e-5)
    assert torch.equal(torch.isnan(actual[:nq]), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual[:nq]), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual[:nq]), torch.isneginf(expected))
    torch.testing.assert_close(
        actual[:nq],
        expected,
        atol=3e-5,
        rtol=3e-5,
        equal_nan=poison in ("masked_row", "nan_bias", "inf_bias"),
    )
    assert torch.equal(actual[nq:], torch.full_like(actual[nq:], -8192.0))


@pytest.mark.parametrize("case", OPEN_CASES)
@pytest.mark.parametrize("bias_out_alias", [False, True])
def test_wide_attention_preserves_input_reads_before_output_stores(case, bias_out_alias, monkeypatch):
    assert torch.backends.mps.is_available()
    from tests.cache_helpers import patch_live_singleton_method
    from triton_msl.backend.driver import _get_utils

    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    dtype, bm, bn, d, _case_nq, nk, narrow, causal = case
    assert dtype == "fp16" and d == 64 and not narrow
    nq = 4
    q = torch.zeros((nq, d), dtype=torch.float16)
    k = torch.zeros((nk, d), dtype=torch.float16)
    v = (torch.arange(nk)[:, None].float() / 32 + torch.arange(d)[None, :].float() / 64).half()
    bias = torch.zeros((nq, nk), dtype=torch.float32)
    expected = reference(q, k, v, bias, bn, narrow, causal)
    device_bias = torch.cat((bias.flatten(), torch.full((d,), -8192.0))).to("mps")
    if bias_out_alias:
        out_storage = None
        out = device_bias[: nq * d].view(nq, d)
    else:
        out_storage = torch.full((nq + 1, d), -8192.0, device="mps")
        out = out_storage[:nq]
    assert (device_bias.data_ptr() == out.data_ptr()) is bias_out_alias
    calls = []

    def observe(instance, real):
        def wrapper(pipeline, grid, group, buffers, **kwargs):
            calls.append((pipeline, grid, group))
            return real(pipeline, grid, group, buffers, **kwargs)

        return wrapper

    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe)
    handle = _attention[(1,)](
        q.to("mps"),
        k.to("mps"),
        v.to("mps"),
        device_bias,
        out,
        nq,
        nk,
        D=d,
        BM=bm,
        BN=bn,
        NARROW=narrow,
        CAUSAL=causal,
    )
    torch.mps.synchronize()
    assert len(calls) == 1 and calls[0][0] == handle.function
    assert str(calls[0][1]) == "(1, 1, 1)"
    assert str(calls[0][2]) == f"({bm}, 1, 1)"
    assert "Source-replayed wide biased attention" in handle.asm["msl"]
    assert torch.equal(out.cpu(), expected)
    assert torch.equal(device_bias[-d:].cpu(), torch.full((d,), -8192.0))
    if out_storage is not None:
        assert torch.equal(out_storage[nq:].cpu(), torch.full((1, d), -8192.0))
