"""Each backward store replays its own conversion, not a sibling output's width.

All inputs are fp32 so only output storage differs. Direct-lowering pins inspect the
contract before executable caching; GPU rows use source-faithful independent math.
"""
import itertools
import re

import pytest
import torch
import triton

import triton_msl.codegen._msl_templates as makers
from triton_msl.errors import MetalNonRecoverableError
from test_fa_bwd_rounding_replay import (
    NEG, _bwd_b, _bwd_kv, _bwd_q, _build_lowerer,
    _oracle_b, _oracle_kv, _oracle_q, _problem, _sig,
)


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
CASTS = {"fp32": None, "fp16": "half", "bf16": "bfloat"}
PAIRS = list(itertools.product(DTYPES, repeat=2))
requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")


def _kernel(kind):
    return {"kv": _bwd_kv, "q": _bwd_q, "b": _bwd_b}[kind]


def _roles(kind):
    return {"kv": ("dk", "dv"), "q": ("dq", "delta"), "b": ("db",)}[kind]


def _signature(kind, widths, dim):
    fn = _kernel(kind)
    sig, cex = _sig(fn, "fp32")
    cex["DIM"] = dim
    for role, width in zip(_roles(kind), widths, strict=True):
        sig[("d" if role == "delta" else role) + "_ptr"] = "*" + width
    return fn, sig, cex


@pytest.mark.parametrize("kind", ["kv", "q"])
@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("widths", PAIRS)
def test_lowering_replays_each_output_role(kind, dim, widths):
    fn, sig, cex = _signature(kind, widths, dim)
    low = _build_lowerer(fn, sig, cex)
    msl = low.lower()
    for role, width in zip(_roles(kind), widths, strict=True):
        ptr = "Delta" if role == "delta" else role.upper()
        rows = re.findall(rf"\b{ptr}\[.*?\] = ([^;]+);", msl)
        assert len(rows) == 1, (role, rows)
        cast = CASTS[width]
        explicit = next((t for t in ("half", "bfloat") if rows[0].startswith(t + "(")), None)
        assert explicit is None or explicit == cast, (role, width, rows)
        # An implicit assignment conversion to this role's typed pointer is also correct.
        elem = cast or "float"
        assert re.search(rf"device\s+{elem}\s*\*\s*{ptr}\b", msl)


@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("width", DTYPES)
def test_bias_output_role_is_independent_of_input_width(dim, width):
    fn, sig, cex = _signature("b", (width,), dim)
    msl = _build_lowerer(fn, sig, cex).lower()
    rhs, = re.findall(r"\bDB\[.*?\] = ([^;]+);", msl)
    expected = "db_acc[i]" if width == "fp32" else CASTS[width] + "(db_acc[i])"
    assert rhs == expected


@pytest.mark.parametrize("kind", ["kv", "q", "b"])
@pytest.mark.parametrize("corruption", ["absent", "partial", "wrong_width", "extra"])
def test_incomplete_output_proof_cannot_fall_back_to_global_dtype(kind, corruption):
    fn, sig, cex = _signature(kind, ("fp32",) * len(_roles(kind)), 32)
    low = _build_lowerer(fn, sig, cex)
    info = low._detect_biased_fa_backward()
    casts = dict.fromkeys(_roles(kind))
    if corruption == "absent":
        casts = None
    elif corruption == "partial":
        casts.pop(_roles(kind)[0])
    elif corruption == "wrong_width":
        casts[_roles(kind)[0]] = "half"
    else:
        casts["unproved"] = None
    info["rounding"]["output_casts"] = casts
    with pytest.raises(MetalNonRecoverableError, match="output (role|pointer)"):
        low._lower_biased_fa_backward(info)


@pytest.mark.parametrize("kind", ["kv", "q", "b"])
def test_integer_output_does_not_reuse_float_store_cast(kind):
    fn, sig, cex = _signature(kind, ("fp32",) * len(_roles(kind)), 32)
    sig[_roles(kind)[0] + "_ptr"] = "*i32"
    with pytest.raises(MetalNonRecoverableError):
        _build_lowerer(fn, sig, cex).lower()


@pytest.fixture
def cold(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    for fn in (_bwd_kv, _bwd_q, _bwd_b):
        fn.device_caches.clear()


def _exact_problem(kind, dim):
    """Dyadic inputs: P=1, and every dot/reduce fits exactly in fp32.

    Output values retain low bits that half/bfloat MUST round. This isolates the
    store contract without assuming that two fp32 reduction orders round alike.
    Both outputs are nonzero and rounding-sensitive, including dQ and Delta.
    """
    p = _problem(torch.float32, DIM=dim, I=64 if kind == "b" else 2)
    for name in ("q", "k", "v", "do", "bias", "lse", "mask", "mh", "o"):
        p[name].zero_()
    p["sm"] = 0.5
    p["delta"].fill_(0.03125)
    if kind == "kv":
        col = (torch.arange(dim, device="mps") + 1).float() / 32768
        p["q"].copy_(0.125 + col)
        p["do"].copy_(0.0625 + col)
    elif kind == "q":
        row = (torch.arange(p["N"], device="mps") + 1).float() / 32768
        p["k"].fill_(0.25)
        p["do"].copy_((0.0625 + row)[:, None])
        p["o"].fill_(0.03125)
    else:
        row = (torch.arange(p["N"], device="mps") + 1).float() / 32768
        p["delta"].copy_(0.03125 + row)
    return p


@requires_gpu
@pytest.mark.parametrize("kind", ["kv", "q"])
@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("widths", PAIRS)
def test_gpu_outputs_match_each_source_storage_type(kind, dim, widths, cold, monkeypatch):
    p = _exact_problem(kind, dim)
    n, i, hc, hh = p["N"], p["I"], p["Hc"], p["Hh"]
    first = torch.zeros(hc, i, n, dim, dtype=DTYPES[widths[0]], device="mps")
    second_shape = (hc, i, n, dim) if kind == "kv" else (hc, i, n)
    second = torch.zeros(second_shape, dtype=DTYPES[widths[1]], device="mps")
    name = "make_flash_attention_bwd_" + kind + "_kernel" + ("_simd" if dim == 32 else "")
    real = getattr(makers, name)
    calls = []
    def spy(*args, **kw):
        calls.append(kw)
        return real(*args, **kw)
    monkeypatch.setattr(makers, name, spy)
    args = []
    delta = p["delta"] if kind == "kv" else second
    tensors = [delta, p["q"], p["k"], p["v"], p["bias"], p["lse"], p["mask"]]
    tensors += [p["do"], first, second] if kind == "kv" else [p["o"], p["do"], first]
    for tensor in tensors:
        args.extend((tensor, *tensor.stride()))
    _kernel(kind)[(triton.cdiv(n, 32), i, hc)](*args, p["sm"], NEG, n, hh, dim, n, 32, 32)
    torch.mps.synchronize()
    assert len(calls) == 1, (name, calls)
    refs = (_oracle_kv if kind == "kv" else _oracle_q)(p, torch.float32)
    for role, actual, ref, width in zip(_roles(kind), (first, second), refs, widths, strict=True):
        intended = ref.to(DTYPES[width]).float()
        error = (actual.float() - intended).abs().max().item()
        print("OUTPUT_ROLE", kind, dim, widths, role, "MAX_ERROR", error, "MAKER", name, flush=True)
        torch.testing.assert_close(actual.float(), intended, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("width", DTYPES)
def test_gpu_bias_output_matches_source_storage(dim, width, cold, monkeypatch):
    p = _exact_problem("b", dim)
    n, hc, hh = p["N"], p["Hc"], p["Hh"]
    out = torch.zeros(hc, n, n, dtype=DTYPES[width], device="mps")
    real = makers.make_flash_attention_bwd_b_kernel
    calls = []
    def spy(*args, **kw):
        calls.append(kw)
        return real(*args, **kw)
    monkeypatch.setattr(makers, "make_flash_attention_bwd_b_kernel", spy)
    args = []
    for tensor in [p[k] for k in ("delta", "q", "k", "v", "bias", "lse", "mask", "do")] + [out]:
        args.extend((tensor, *tensor.stride()))
    _bwd_b[(triton.cdiv(n, 32), triton.cdiv(n, 32), hc)](*args, p["sm"], NEG, hh, n, dim, n, 32, 32)
    torch.mps.synchronize()
    assert len(calls) == 1
    intended = _oracle_b(p, torch.float32).to(DTYPES[width]).float()
    torch.testing.assert_close(out.float(), intended, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("width", ["fp32", "fp16", "bf16"])
def test_random_dv_float32_is_not_rounded_to_dk_width(width, cold, monkeypatch):
    """211's original random witness, unchanged source oracle and 2e-6 bound."""
    p = _problem(torch.float32)
    n, dim, i, hc, hh = (p[k] for k in ("N", "DIM", "I", "Hc", "Hh"))
    dk = torch.zeros(hc, i, n, dim, device="mps", dtype=DTYPES[width])
    dv = torch.zeros(hc, i, n, dim, device="mps", dtype=torch.float32)
    real = makers.make_flash_attention_bwd_kv_kernel_simd
    calls = []
    def spy(*args, **kw):
        calls.append(kw)
        return real(*args, **kw)
    monkeypatch.setattr(makers, "make_flash_attention_bwd_kv_kernel_simd", spy)
    args = []
    for tensor in [p[k] for k in ("delta", "q", "k", "v", "bias", "lse", "mask", "do")] + [dk, dv]:
        args.extend((tensor, *tensor.stride()))
    _bwd_kv[(triton.cdiv(n, 32), i, hc)](*args, p["sm"], NEG, n, hh, dim, n, 32, 32)
    torch.mps.synchronize()
    _, expected = _oracle_kv(p, torch.float32)
    assert len(calls) == 1
    error = (dv - expected).abs().max().item()
    print("RANDOM_211", width, "DV_ERROR", error, flush=True)
    assert error < 2e-6
