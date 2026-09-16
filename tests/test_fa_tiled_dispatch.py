"""Cooperative tiled FA must cover every logical element at the actual dispatch.

Numeric pins launch exactly 1024 threads for a 2048-score tile. The independent
reference never uses the emitted shader or its indexing. No fallback earns credit.
"""

import math
import struct

import pytest
import torch

from tests.cache_helpers import patch_live_singleton_method

from triton_msl.codegen import _msl_templates as makers

gpu = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="requires Metal hardware and compile_shader",
)


@pytest.mark.parametrize("bn", [16, 32, 64])
def test_tiled_stride_matches_dispatch(bn):
    src = makers.make_flash_attention_kernel_tiled(32, 32, bn, Dc=32)
    assert f"const uint TPG = {min(32 * bn, 1024)}u;" in src
    # All seven cooperative loops share the dispatch stride; row-only work is
    # separately owned by lid < BM. Barriers are outside those loops/branches.
    assert src.count("i += TPG") == 7


@pytest.mark.parametrize("bm,bn", [(0, 64), (32, 0), (1025, 1), (True, 32), (32, 1.5)])
def test_tiled_row_ownership_requires_physical_threads(bm, bn):
    from triton_msl.errors import MetalNonRecoverableError

    with pytest.raises(MetalNonRecoverableError, match="physical thread owner"):
        makers.make_flash_attention_kernel_tiled(32, bm, bn, Dc=32)


@pytest.mark.parametrize("bn", [16, 32, 64])
def test_biased_lowering_descriptor_matches_shader(bn):
    from test_fa_biased_routing import _biased_tri_fa as fn
    from test_fa_bwd_routing import _build_lowerer

    cex = {
        fn.arg_names[i]: (bn if fn.arg_names[i] == "BLOCK_K" else 32 if fn.arg_names[i] in ("DIM", "BLOCK_J") else 256)
        for i in fn.constexprs
    }
    sig = {
        n: ("*u8" if "mask" in n or n == "m_ptr" else "*fp32" if n.startswith("lse") else "*bf16")
        if n.endswith("_ptr")
        else "fp32"
        if n in ("sm_scale", "neg_inf")
        else "i32"
        for n in fn.arg_names
        if n not in cex
    }
    lw = _build_lowerer(fn, sig, cex)
    src = lw.lower()
    assert lw.effective_block_size == min(32 * bn, 1024)
    assert lw._flash_attention[2] == lw.effective_block_size
    assert f"const uint TPG = {lw.effective_block_size}u;" in src


@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bn", [32, 64])
@pytest.mark.parametrize("causal,strided,n", [(False, False, 64), (True, True, 77)])
def test_tiled_finite_scores_cover_tail(dtype, bn, causal, strided, n):
    torch.manual_seed(231)
    d = 32

    def operand():
        x = torch.randn(1, 1, n, d * (2 if strided else 1), device="mps", dtype=dtype)
        return x[..., ::2] if strided else x

    q, k, v = operand(), operand(), operand()
    storage = torch.full_like(q if not strided else torch.empty(1, 1, n, d * 2, device="mps", dtype=dtype), 123)
    out = storage[..., ::2] if strided else storage
    src = makers.make_flash_attention_kernel_tiled(
        d,
        32,
        bn,
        Dc=d,
        causal=causal,
        out_dtype={torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype],
    )
    lib = torch.mps.compile_shader(src)
    tpg = min(32 * bn, 1024)
    lib.flash_attention(
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        1,
        1,
        n,
        threads=(math.ceil(n / 32) * tpg, 1),
        group_size=(tpg, 1),
    )
    torch.mps.synchronize()
    scores = (q.float() / math.sqrt(d)) @ k.float().transpose(-1, -2)
    if causal:
        rows = torch.arange(n, device="mps")
        scores = scores.masked_fill(rows[None, :] > rows[:, None], -float("inf"))
    expected = (scores.softmax(-1) @ v.float()).to(dtype).float()
    err = (out.float() - expected).abs().max().item()
    print("TILED_FINITE", dtype, bn, causal, strided, n, "MAX_ERR", err, flush=True)
    torch.testing.assert_close(
        out.float(), expected, atol={torch.float32: 2e-6, torch.float16: 1e-3, torch.bfloat16: 8e-3}[dtype], rtol=0
    )
    if strided:
        assert bool((storage[..., 1::2] == 123).all())


@gpu
@pytest.mark.parametrize("bits", [0x7FFFFFFF, 0xFFC00001])
def test_biased_bf16_nan_scale_covers_all_rows(bits, monkeypatch, tmp_path):
    import test_fa_forward_rounding as source
    from triton_msl.backend.driver import _get_compile_shader_runtime

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    p = source._problem(torch.bfloat16)
    p["BK"] = 64
    p["sm"] = struct.unpack("<f", struct.pack("<I", bits))[0]
    expected, expected_lse = source._faithful(p, torch.bfloat16)
    assert bool(torch.isnan(expected).all()) and bool(torch.isnan(expected_lse).all())
    hits = []
    dispatches = []
    real = makers.make_flash_attention_kernel_tiled

    def spy(*args, **kwargs):
        src = real(*args, **kwargs)
        hits.append(src)
        return src

    def observe_dispatch(_runtime, real_dispatch):
        def dispatch_spy(lib, kernel_name, args, *, threads, group_size):
            dispatches.append((threads, group_size))
            return real_dispatch(lib, kernel_name, args, threads=threads, group_size=group_size)

        return dispatch_spy

    monkeypatch.setattr(makers, "make_flash_attention_kernel_tiled", spy)
    # Patch the exact singleton consumed by MetalLauncher. Earlier randomized
    # tests may leave an instance-bound dispatch attribute, in which case a
    # class-level patch observes nothing even though the real dispatch occurs.
    patch_live_singleton_method(monkeypatch, _get_compile_shader_runtime, "dispatch", observe_dispatch)
    out, lse = source._launch(p, torch.bfloat16)
    assert len(hits) == 1
    assert "const uint TPG = 1024u;" in hits[0]  # logical score tile is 32 * 64 = 2048
    assert len(dispatches) == 1
    threads, group_size = dispatches[0]
    assert group_size == (1024, 1, 1)
    assert threads[0] % group_size[0] == 0
    print("TILED_NAN", hex(bits), int(torch.isnan(out).sum()), out.numel(), flush=True)
    assert bool(torch.isnan(out).all()) and bool(torch.isnan(lse).all())
