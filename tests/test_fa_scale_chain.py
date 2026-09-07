"""Scalar conversion is independent of Q-product and probability rounding."""
import struct

import pytest
import torch
import triton

import triton_msl.codegen._msl_templates as makers
from triton_msl.errors import MetalNonRecoverableError
import test_fa_forward_rounding as source
from test_fa_bwd_routing import _build_lowerer

CHAINS = [("half",), ("bfloat",), ("half", "bfloat")]
DTYPES = {"half": torch.float16, "bfloat": torch.bfloat16}
requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal GPU")


def _variant(chain):
    fn = triton.JITFunction(source._biased_tri_fa.fn)
    types = {"half": "tl.float16", "bfloat": "tl.bfloat16"}
    expression = f"tl.full([1], value=sm_scale, dtype={types[chain[0]]})"
    for item in chain[1:]:
        expression += f".to({types[item]})"
    expression += ".to(tl.float32)"
    old = "tl.full([1], value=sm_scale, dtype=q_block.type.element_ty)"
    assert fn.src.count(old) == 1
    fn._unsafe_update_src(fn.src.replace(old, expression))
    return fn


def _lower(fn):
    cex = {"DIM": 32, "BLOCK_J": 32, "BLOCK_K": 32}
    sig = {n: ("*u8" if n == "mask_ptr" else "*fp32") if n.endswith("_ptr") else
           "fp32" if n in ("sm_scale", "neg_inf") else "i32" for n in fn.arg_names if n not in cex}
    return _build_lowerer(fn, sig, cex)


@pytest.mark.parametrize("chain", CHAINS)
def test_scale_chain_is_preserved_at_lowering_boundary(chain):
    low = _lower(_variant(chain))
    msl = low.lower()
    assert low._fa_rounding["scale_chain"] == chain
    assert "round_q" not in low._fa_rounding  # product remains fp32
    if "half" in chain:
        assert "float(half(" in msl
    if "bfloat" in chain:
        assert "0x7FFFu" in msl and "isnan(" in msl
    canonical = _lower(source._biased_tri_fa).lower()
    assert msl != canonical


def test_unproved_scalar_chain_refuses_before_maker(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unproved conversion reached a forward maker")
    monkeypatch.setattr(makers, "make_flash_attention_kernel_simdgroup", forbidden)
    monkeypatch.setattr(makers, "make_flash_attention_kernel_tiled", forbidden)
    with pytest.raises(MetalNonRecoverableError, match="scalar.*(chain|conversion)"):
        _lower(_variant(("bfloat", "half"))).lower()


@pytest.fixture
def cold(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")


@requires_gpu
@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("route", ["simdgroup", "tiled"])
def test_gpu_scalar_chain_matches_source(chain, route, cold, monkeypatch):
    monkeypatch.setattr(source, "_biased_tri_fa", _variant(chain))
    p = source._problem(torch.float32, N=64, seed=5)
    if route == "tiled":
        backing = torch.empty(*p["q"].shape[:-1], p["DIM"] * 2, device="mps")
        q = backing[..., ::2]
        q.copy_(p["q"])
        p["q"] = q  # exact same values, non-contiguous head axis selects tiled
    expected_problem = dict(p)
    scalar = torch.tensor(p["sm"], dtype=torch.float32)
    for item in chain:
        scalar = scalar.to(DTYPES[item]).float()
    expected_problem["sm"] = scalar.item()
    ref, _ = source._faithful(expected_problem, torch.float32)
    calls = []
    name = "make_flash_attention_kernel_" + route
    real = getattr(makers, name)
    def spy(*args, **kw):
        calls.append(kw)
        return real(*args, **kw)
    monkeypatch.setattr(makers, name, spy)
    actual, _ = source._launch(p, torch.float32)
    error = (actual - ref).abs().max().item()
    print("SCALAR_CHAIN", chain, route, "MAX_ERROR", error, "HITS", len(calls), flush=True)
    assert len(calls) == 1 and tuple(calls[0]["scale_chain"]) == chain
    assert error < 2e-6  # unchanged211 source bound


@requires_gpu
@pytest.mark.parametrize("bits", [0x7FFFFFFF, 0xFFC00001])
def test_gpu_noncanonical_nan_scalar_stays_nan(bits, cold, monkeypatch):
    monkeypatch.setattr(source, "_biased_tri_fa", _variant(("bfloat",)))
    p = source._problem(torch.float32, N=64, seed=5)
    p["sm"] = struct.unpack("<f", struct.pack("<I", bits))[0]
    actual, _ = source._launch(p, torch.float32)
    ref, _ = source._faithful(p, torch.float32)
    assert torch.equal(torch.isnan(actual), torch.isnan(ref))
    assert bool(torch.isnan(actual).any())
