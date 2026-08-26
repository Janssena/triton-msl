"""Regression: symmetric int8 GEMM with dead M/N arguments swapped is correct-or-refuse."""

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
def _sym_int8_nmk_probe(
    a_ptr,
    w_ptr,
    c_ptr,
    s_ptr,
    N,
    M,
    K,
    sam,
    sak,
    swk,
    swn,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
    wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
    scale = tl.load(s_ptr + on)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        weight = tl.load(wp).to(tl.float32) * scale[None, :]
        acc += tl.dot(tl.load(ap), weight)
        ap += BK * sak
        wp += BK * swk
    tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)


@requires_mps
def test_symmetric_int8_swapped_mn_correct_or_refuse(monkeypatch):
    import triton_msl.autotuning._quant_matmul_dispatch as quant_dispatch

    route = {"calls": 0}
    real = quant_dispatch._dispatch_sym_int8

    def spy(*args, **kwargs):
        route["calls"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(quant_dispatch, "_dispatch_sym_int8", spy)

    device = "mps"
    torch.manual_seed(8201)
    M, N, K = 64, 32, 64
    a = torch.randn(M, K, device=device)
    weight = torch.randint(-127, 127, (K, N), device=device, dtype=torch.int8).contiguous()
    scale = (torch.rand(N, device=device) * 0.05 + 0.01).contiguous()
    out = torch.full((M, N), float("nan"), device=device)

    try:
        _sym_int8_nmk_probe[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
            a,
            weight,
            out,
            scale,
            N,
            M,
            K,
            a.stride(0),
            a.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            BM=32,
            BN=32,
            BK=32,
        )
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        print(f"SYM_INT8_MN_PROBE refused route_calls={route['calls']}")
        return

    ref = a @ (weight.float() * scale)
    err = (out - ref).abs().max().item()
    print(f"SYM_INT8_MN_PROBE route_calls={route['calls']} err={err}")
    assert err < 1e-2 * max(ref.abs().max().item(), 1.0), (
        "symmetric int8 GEMM violated correct-or-refuse for a swapped (N,M,K) declaration: "
        f"route_calls={route['calls']}, max_err={err}"
    )
