"""Per-group int8 weight-only GEMM (GPTQ-style: scale/zero loaded in the K-loop at
k//G). It was a silent-wrong (routed to the per-N fast kernel, dropping the group
boundary — 8e3526a made it refuse), then wired to a dedicated stride-generic scalar
template (make_int8_matmul_pergroup) so it now ROUTES and computes correctly for any
weight ([K,N]/[N,K]) and scale/zero ([n_groups,N]/[N,n_groups]) layout. This test
asserts routing + correctness (never silently-wrong) across layouts and group sizes.
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
    a_ptr,
    w_ptr,
    c_ptr,
    scale_ptr,
    zero_ptr,
    M,
    N,
    K,
    sam,
    sak,
    swk,
    swn,
    ssg,
    ssn,
    zsg,
    zsn,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
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
        s = tl.load(scale_ptr + g * ssg + offs_n * ssn)  # per-GROUP, in the K-loop
        z = tl.load(zero_ptr + g * zsg + offs_n * zsn)
        w = (tl.load(w_ptrs).to(tl.float32) - z[None, :]) * s[None, :]
        acc += tl.dot(tl.load(a_ptrs), w)
        a_ptrs += BK * sak
        w_ptrs += BK * swk
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc)


@requires_mps
@pytest.mark.parametrize("M,N,K,G", [(64, 32, 64, 32), (32, 64, 128, 32), (96, 48, 256, 64)])
@pytest.mark.parametrize("wl,sl", [("kn", "gn"), ("nk", "gn"), ("kn", "ng"), ("nk", "ng")])
def test_pergroup_int8_gemm_routes_and_computes(M, N, K, G, wl, sl):
    dev = "mps"
    torch.manual_seed(0)
    ng = K // G
    a = torch.randn(M, K, device=dev, dtype=torch.float32)
    wkn = torch.randint(-127, 127, (K, N), device=dev, dtype=torch.int8)
    s_ng = torch.rand(ng, N, device=dev) * 0.05 + 0.01  # DIFFERING per group
    z_ng = torch.randint(-8, 8, (ng, N), device=dev).float()
    gidx = torch.arange(K, device=dev) // G
    ref = a @ ((wkn.float() - z_ng[gidx]) * s_ng[gidx])

    # logical (k,n) weight strides + (g,n) scale/zero strides for each layout
    if wl == "kn":
        wbuf = wkn.contiguous()
        swk, swn = wbuf.stride(0), wbuf.stride(1)
    else:
        wbuf = wkn.t().contiguous()
        swk, swn = wbuf.stride(1), wbuf.stride(0)
    if sl == "gn":
        sbuf = s_ng.contiguous()
        zbuf = z_ng.contiguous()
        ssg, ssn, zsg, zsn = sbuf.stride(0), sbuf.stride(1), zbuf.stride(0), zbuf.stride(1)
    else:
        sbuf = s_ng.t().contiguous()
        zbuf = z_ng.t().contiguous()
        ssg, ssn, zsg, zsn = sbuf.stride(1), sbuf.stride(0), zbuf.stride(1), zbuf.stride(0)

    out = torch.zeros(M, N, device=dev, dtype=torch.float32)
    st = lambda t: t.stride()
    _int8_pergroup_gemm[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
        a,
        wbuf,
        out,
        sbuf,
        zbuf,
        M,
        N,
        K,
        st(a)[0],
        st(a)[1],
        swk,
        swn,
        ssg,
        ssn,
        zsg,
        zsn,
        st(out)[0],
        st(out)[1],
        32,
        32,
        32,
        G,
    )
    torch.mps.synchronize()
    assert (out - ref).abs().max().item() < 1e-2 * max(ref.abs().max().item(), 1.0)
