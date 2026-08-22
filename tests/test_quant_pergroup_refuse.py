"""Regression (audit #288, Finding 1): a GPTQ-style PER-GROUP int8 weight-only GEMM
(scale/zero loaded INSIDE the K-loop at k//G) must NOT route to the per-N fast kernel
(make_int8_matmul_fast), which indexes scale/zero as length-N and would silently drop
the per-K-group boundary. _maybe_quant_matmul_descriptor now refuses when the
scale/zero load is loop-variant (lives in the scf.for body). Correct-or-refuse: the
kernel either falls back (correct) or refuses — never silently-wrong GPU output.
"""
import pytest
import torch
import triton
import triton.language as tl

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _int8_pergroup_gemm(
    a_ptr, w_ptr, c_ptr, scale_ptr, zero_ptr, M, N, K,
    sam, sak, swk, swn, ssg, ssn, scm, scn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    w_ptrs = w_ptr + offs_k[:, None] * swk + offs_n[None, :] * swn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        g = k // G
        s = tl.load(scale_ptr + g * ssg + offs_n * ssn)  # per-GROUP, inside K-loop
        z = tl.load(zero_ptr + g * ssg + offs_n * ssn)
        a = tl.load(a_ptrs)
        w = (tl.load(w_ptrs).to(tl.float32) - z[None, :]) * s[None, :]
        acc += tl.dot(a, w)
        a_ptrs += BK * sak
        w_ptrs += BK * swk
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc)


@requires_mps
@pytest.mark.parametrize("M,N,K,G", [(64, 32, 64, 32), (32, 64, 128, 32)])
def test_pergroup_int8_gemm_not_silent_wrong(M, N, K, G):
    import warnings
    from triton_msl.errors import MetalNonRecoverableError

    dev = "mps"
    torch.manual_seed(0)
    ng = K // G
    a = torch.randn(M, K, device=dev, dtype=torch.float32)
    w_i8 = torch.randint(-127, 127, (K, N), device=dev, dtype=torch.int8)
    scale = (torch.rand(ng, N, device=dev) * 0.05 + 0.01).contiguous()
    zero = torch.randint(-8, 8, (ng, N), device=dev).float().contiguous()
    gidx = torch.arange(K, device=dev) // G
    ref = a @ ((w_i8.float() - zero[gidx]) * scale[gidx])  # true per-group dequant
    out = torch.zeros(M, N, device=dev, dtype=torch.float32)
    st = lambda t: t.stride()

    refused = False
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            _int8_pergroup_gemm[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
                a, w_i8, out, scale, zero, M, N, K,
                *st(a), *st(w_i8), *st(scale), *st(out), 32, 32, 32, G)
            torch.mps.synchronize()
        except MetalNonRecoverableError:
            refused = True
    refused = refused or any(
        "Refus" in str(w.message) or "fall back" in str(w.message) for w in wl)

    err = (out - ref).abs().max().item()
    # Never silently-wrong: EITHER it refused (fail-closed) OR the output is correct.
    assert refused or err < 1e-2 * max(ref.abs().max().item(), 1.0), (
        f"per-group int8 GEMM silently mis-computed: err {err:.3e}, no refuse")
