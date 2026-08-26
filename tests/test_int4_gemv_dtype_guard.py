"""Regression: INT4 GEMV must not reinterpret fp16 scale/zero buffers as float.

Authorized diagnostic only (2026-08-25).  This deliberately asserts the permanent
correct-or-refuse contract so it can become a regression test unchanged after a fix.
Run with private TRITON_CACHE_DIR and TRITON_MSL_CACHE_DIR values.
"""

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError


requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _int4_gemv_probe(
    x_ptr,
    w_ptr,
    o_ptr,
    s_ptr,
    z_ptr,
    N,
    K,
    ng,
    swn,
    ssn,
    BN: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
):
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + ok
        packed = tl.load(w_ptr + on[:, None] * swn + (kk // 2)[None, :])
        w4 = (packed >> ((kk % 2) * 4)[None, :]) & 0xF
        g = kk // G
        scale = tl.load(s_ptr + on[:, None] * ssn + g[None, :])
        zero = tl.load(z_ptr + on[:, None] * ssn + g[None, :])
        weight = (w4.to(tl.float32) - zero) * scale
        acc += tl.sum(tl.load(x_ptr + kk)[None, :] * weight, axis=1)
    tl.store(o_ptr + on, acc)


@requires_mps
@pytest.mark.parametrize("half_scale,half_zero", [(True, False), (False, True)])
def test_int4_gemv_fp16_scale_zero_correct_or_refuse(monkeypatch, half_scale, half_zero):
    import triton_msl.autotuning._quant_matmul_dispatch as quant_dispatch

    route = {"calls": 0}
    real = quant_dispatch._dispatch_int4_gemv

    def spy(*args, **kwargs):
        route["calls"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(quant_dispatch, "_dispatch_int4_gemv", spy)

    device = "mps"
    torch.manual_seed(8101 + int(half_scale) * 10 + int(half_zero))
    N, K, G = 128, 256, 128
    ng = K // G
    x = torch.randn(K, device=device, dtype=torch.float32)
    w4 = torch.randint(0, 16, (N, K), device=device, dtype=torch.int32)
    packed = (w4[:, 0::2] | (w4[:, 1::2] << 4)).to(torch.uint8).contiguous()
    scale_f32 = (torch.rand(N, ng, device=device) * 0.02 + 0.005).contiguous()
    zero_f32 = torch.randint(0, 16, (N, ng), device=device).float().contiguous()
    scale = scale_f32.half() if half_scale else scale_f32
    zero = zero_f32.half() if half_zero else zero_f32
    out = torch.full((N,), float("nan"), device=device)

    try:
        _int4_gemv_probe[(triton.cdiv(N, 32),)](
            x,
            packed,
            out,
            scale,
            zero,
            N,
            K,
            ng,
            packed.stride(0),
            scale.stride(0),
            BN=32,
            BK=64,
            G=G,
        )
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        print(f"INT4_GEMV_PROBE refused route_calls={route['calls']}")
        return

    group_index = torch.arange(K, device=device) // G
    ref = ((w4.float() - zero.float()[:, group_index]) * scale.float()[:, group_index]) @ x
    err = (out - ref).abs().max().item()
    print(
        "INT4_GEMV_PROBE "
        f"half_scale={half_scale} half_zero={half_zero} route_calls={route['calls']} err={err}"
    )
    assert err < 1e-2, (
        "INT4 GEMV violated correct-or-refuse for a non-f32 scale/zero buffer: "
        f"route_calls={route['calls']}, max_err={err}"
    )
