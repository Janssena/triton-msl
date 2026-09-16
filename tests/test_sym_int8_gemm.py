"""Symmetric weight-only int8 GEMM (no zero-point): out = a @ (w_i8.to(f32) * scale),
4 ptrs (a, w, c, scale). Previously refused (the canonical quant descriptor requires
a subf/zero); now routes to the stride-generic per-group scalar template with a
synthesized all-zero zeros buffer (ssg=0 -> per-N). Correct-or-refuse.
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
def _sym_int8(
    a_ptr,
    w_ptr,
    c_ptr,
    s_ptr,
    M,
    N,
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
        w = tl.load(wp).to(tl.float32) * scale[None, :]  # symmetric: no zero-point
        acc += tl.dot(tl.load(ap), w)
        ap += BK * sak
        wp += BK * swk
    tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)


@requires_mps
@pytest.mark.parametrize("M,N,K", [(64, 32, 128), (96, 48, 256), (32, 64, 64)])
@pytest.mark.parametrize("wl", ["kn", "nk"])
def test_symmetric_int8_gemm_routes_and_computes(M, N, K, wl):
    dev = "mps"
    torch.manual_seed(0)
    a = torch.randn(M, K, device=dev, dtype=torch.float32)
    wkn = torch.randint(-127, 127, (K, N), device=dev, dtype=torch.int8)
    scale = torch.rand(N, device=dev) * 0.05 + 0.01
    ref = a @ (wkn.float() * scale)
    if wl == "kn":
        wbuf = wkn.contiguous()
        swk, swn = wbuf.stride(0), wbuf.stride(1)
    else:
        wbuf = wkn.t().contiguous()
        swk, swn = wbuf.stride(1), wbuf.stride(0)
    out = torch.zeros(M, N, device=dev, dtype=torch.float32)
    st = lambda t: t.stride()
    _sym_int8[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
        a, wbuf, out, scale, M, N, K, st(a)[0], st(a)[1], swk, swn, st(out)[0], st(out)[1], 32, 32, 32
    )
    torch.mps.synchronize()
    assert (out - ref).abs().max().item() < 1e-2 * max(ref.abs().max().item(), 1.0)
