"""Nonfinite input data must stay in its source row across FA replacements.

The source kernels use one online-softmax state per query row.  If a whole Q row
is NaN, that row's denominator and output are NaN, while every sibling row stays
finite.  SIMD replacement templates use 8x8 diagonal MMAs for rescaling; an
unprotected NaN diagonal entry can contaminate all eight rows through ``0*NaN``.
The scalar/tiled replacement must likewise distinguish a NaN denominator from a
genuine empty-row denominator instead of mapping both to zero.
"""

import math

import pytest
import torch

import triton_msl.autotuning._fa_dispatch as fa_dispatch
import triton_msl.codegen._msl_templates as msl_templates
from tests.test_flash_attention import _flash_attn_fwd
from tests.test_mla_value_paths import (
    DN,
    DR,
    DV,
    H as MLA_H,
    N as MLA_N,
    Z as MLA_Z,
    _mla_value_path,
)
from tests.test_varlen_fa_routing import _varlen_fwd


requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")


@pytest.fixture
def fa_spy(monkeypatch):
    hits = []
    real = fa_dispatch.dispatch_flash_attention

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        hits.append(bool(result))
        return result

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)
    return hits


@pytest.fixture
def maker_spy(monkeypatch):
    hits = []
    for name, label in (
        ("make_flash_attention_kernel_simdgroup", "simd"),
        ("make_flash_attention_kernel_tiled", "tiled"),
        ("make_varlen_flash_attention_mma", "varlen_mma"),
        ("make_varlen_flash_attention", "varlen_scalar"),
    ):
        real = getattr(msl_templates, name)

        def spy(*args, _real=real, _label=label, **kwargs):
            hits.append(_label)
            return _real(*args, **kwargs)

        monkeypatch.setattr(msl_templates, name, spy)
    return hits


def _assert_only_first_row_nan(got, finite_ref, label):
    expected = torch.zeros(got.shape[:-1], dtype=torch.bool, device=got.device)
    expected[..., 0] = True
    actual = torch.isnan(got).all(dim=-1)
    assert torch.equal(actual, expected), f"{label}: NaN rows differ"
    assert torch.isfinite(got[~expected]).all(), f"{label}: a sibling row was contaminated"
    assert (got[~expected].float() - finite_ref[~expected].float()).abs().max().item() < 1e-3


def _clear_kernel_cache(kernel):
    if hasattr(kernel, "device_caches"):
        kernel.device_caches.clear()


@requires_mps
@pytest.mark.parametrize(
    "route,d,force_tiled,expected_maker",
    [
        ("generic", 32, False, []),
        ("simd", 128, False, ["simd"]),
        ("tiled", 128, True, ["tiled"]),
    ],
)
def test_dense_nan_q_row_is_row_local(cold_gpu_caches, maker_spy, route, d, force_tiled, expected_maker):
    z = h = 1
    n = 64
    torch.manual_seed(11501)
    q = torch.randn(z, h, n, d, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q[..., 0, :] = float("nan")
    if force_tiled:
        backing = torch.empty(z, h, n, d * 2, device="mps")
        out = backing[..., ::2]
    else:
        out = torch.empty_like(q)

    _clear_kernel_cache(_flash_attn_fwd)
    _flash_attn_fwd[(n // 32, 1)](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        z,
        h,
        n,
        32,
        32,
        d,
        False,
    )
    torch.mps.synchronize()
    assert maker_spy == expected_maker

    ref = torch.softmax((q * (1.0 / math.sqrt(d))) @ k.transpose(-2, -1), dim=-1) @ v
    _assert_only_first_row_nan(out, ref, f"dense route={route}")


@requires_mps
def test_mla_nan_q_row_is_row_local(cold_gpu_caches, fa_spy, maker_spy):
    torch.manual_seed(11502)
    mk = lambda width: torch.randn(MLA_Z, MLA_H, MLA_N, width, device="mps", dtype=torch.float16)
    qn, qr, kn, kr, v = mk(DN), mk(DR), mk(DN), mk(DR), mk(DV)
    qn[..., 0, :] = float("nan")
    out = torch.empty(MLA_Z, MLA_H, MLA_N, DV, device="mps", dtype=torch.float16)

    def st(t):
        return [t.stride(i) for i in range(4)]

    _clear_kernel_cache(_mla_value_path)
    _mla_value_path[(1, MLA_Z * MLA_H)](
        qn,
        qr,
        kn,
        kr,
        v,
        out,
        *st(qn),
        *st(qr),
        *st(kn),
        *st(kr),
        *st(v),
        *st(out),
        MLA_Z,
        MLA_H,
        MLA_N,
        32,
        32,
        DN,
        DR,
        DV,
        0,
    )
    torch.mps.synchronize()
    assert fa_spy == [True]
    assert maker_spy == ["simd"]

    scale = 1.0 / math.sqrt(DN + DR)
    scores = (qn.float() @ kn.float().transpose(-2, -1) + qr.float() @ kr.float().transpose(-2, -1)) * scale
    ref = torch.softmax(scores, dim=-1) @ v.float()
    _assert_only_first_row_nan(out, ref, "MLA simd")


@requires_mps
@pytest.mark.parametrize(
    "dtype,d",
    [(torch.float16, 64), (torch.float32, 128)],
    ids=["mma", "scalar"],
)
def test_varlen_nan_q_row_is_row_local(cold_gpu_caches, maker_spy, dtype, d):
    n = 64
    h = 2
    torch.manual_seed(11503 + d)
    q = torch.randn(n, h, d, device="mps", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q[0, ...] = float("nan")
    out = torch.empty_like(q)
    cu_q = torch.tensor([0, 32, n], device="mps", dtype=torch.int32)
    cu_k = cu_q.clone()
    scale = 1.0 / math.sqrt(d)

    _clear_kernel_cache(_varlen_fwd)
    _varlen_fwd[(1, 2 * h)](
        q,
        k,
        v,
        out,
        cu_q,
        cu_k,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        h,
        32,
        scale,
        32,
        32,
        d,
    )
    torch.mps.synchronize()
    assert maker_spy == (["varlen_mma"] if dtype == torch.float16 else ["varlen_scalar"])

    ref = torch.empty_like(out)
    for batch_start in (0, 32):
        for head in range(h):
            qb = q[batch_start : batch_start + 32, head].float()
            kb = k[batch_start : batch_start + 32, head].float()
            vb = v[batch_start : batch_start + 32, head].float()
            scores = (qb @ kb.transpose(-2, -1)) * scale
            ref[batch_start : batch_start + 32, head] = (torch.softmax(scores, dim=-1) @ vb).to(dtype)
    _assert_only_first_row_nan(out.permute(1, 0, 2), ref.permute(1, 0, 2), f"varlen {dtype} d={d}")


@requires_mps
@pytest.mark.parametrize(
    "dtype,d",
    [(torch.float16, 64), (torch.float32, 128)],
    ids=["mma", "scalar"],
)
def test_varlen_empty_k_sequence_preserves_zero_denominator_semantics(cold_gpu_caches, maker_spy, dtype, d):
    """A cross-attention batch item may have query rows but zero keys.  The
    source's all-masked tile produces NaN; the templates reach l==0 by taking
    no K iterations.  Both representations must store the same NaN rows without
    contaminating the following non-empty batch item.
    """
    h = 2
    torch.manual_seed(11603 + d)
    q = torch.randn(64, h, d, device="mps", dtype=dtype)
    k = torch.randn(32, h, d, device="mps", dtype=dtype)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_q = torch.tensor([0, 32, 64], device="mps", dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 32], device="mps", dtype=torch.int32)
    scale = 1.0 / math.sqrt(d)

    _clear_kernel_cache(_varlen_fwd)
    _varlen_fwd[(1, 2 * h)](
        q,
        k,
        v,
        out,
        cu_q,
        cu_k,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        h,
        32,
        scale,
        32,
        32,
        d,
    )
    torch.mps.synchronize()
    assert maker_spy == (["varlen_mma"] if dtype == torch.float16 else ["varlen_scalar"])
    assert torch.isnan(out[:32]).all()
    assert torch.isfinite(out[32:]).all()

    for head in range(h):
        scores = (q[32:, head].float() @ k[:, head].float().transpose(-2, -1)) * scale
        ref = torch.softmax(scores, dim=-1) @ v[:, head].float()
        assert (out[32:, head].float() - ref).abs().max().item() < 1e-3
