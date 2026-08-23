"""FAST simdgroup-MMA per-group int8 GEMM (~4-7x the scalar template). The per-group
descriptor carries the fast variant; the dispatch selects it ONLY when the runtime shape
meets its contract (contiguous row-major input [M,K] / weight [K,N] / output [M,N] and
M%(8*rr)==0, N%(8*rc)==0, K%bk==0, group%bk==0), else the stride-generic scalar kernel.
Both must be CORRECT — this pins the routing boundary at the fast edge and just past it.
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
def _pg_int8(a_ptr, w_ptr, c_ptr, scale_ptr, zero_ptr, M, N, K,
             sam, sak, swk, swn, ssg, ssn, zsg, zsn, scm, scn,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM); on = pid_n * BN + tl.arange(0, BN); ok = tl.arange(0, BK)
    ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
    wp = w_ptr + ok[:, None] * swk + on[None, :] * swn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        g = k // G
        s = tl.load(scale_ptr + g * ssg + on * ssn)
        z = tl.load(zero_ptr + g * zsg + on * zsn)
        w = (tl.load(wp).to(tl.float32) - z[None, :]) * s[None, :]
        acc += tl.dot(tl.load(ap), w)
        ap += BK * sak; wp += BK * swk
    tl.store(c_ptr + om[:, None] * scm + on[None, :] * scn, acc)


def _run(M, N, K, G):
    dev = "mps"; torch.manual_seed(0); ng = K // G
    a = torch.randn(M, K, device=dev)
    w = torch.randint(-8, 8, (K, N), device=dev, dtype=torch.int8).contiguous()   # [K,N] kn contiguous
    s = (torch.rand(ng, N, device=dev) * 0.03 + 0.01).contiguous()
    z = torch.randint(-4, 4, (ng, N), device=dev).float().contiguous()
    c = torch.zeros(M, N, device=dev)
    st = lambda t: t.stride()
    _pg_int8[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
        a, w, c, s, z, M, N, K, *st(a), *st(w), s.stride(0), s.stride(1),
        z.stride(0), z.stride(1), *st(c), BM=32, BN=32, BK=32, G=G)
    torch.mps.synchronize()
    gi = torch.arange(K, device=dev) // G
    ref = a @ ((w.float() - z[gi, :]) * s[gi, :])
    assert (c - ref).abs().max().item() < 3e-3, f"{M}x{N}x{K} G{G}: {(c-ref).abs().max().item():.3e}"


@requires_mps
@pytest.mark.parametrize("M,N,K,G", [
    (64, 32, 128, 32),    # M%32, N%16, K%32, G%32 all 0 + contiguous -> FAST path
    (128, 64, 256, 128),  # larger, fast
    (256, 512, 1024, 64),
])
def test_pergroup_int8_fast_path_correct(M, N, K, G):
    _run(M, N, K, G)


@requires_mps
@pytest.mark.parametrize("M,N,K,G", [
    (64, 24, 128, 32),    # N=24 not %16 -> scalar fallback
    (48, 32, 128, 32),    # M=48 not %32 -> scalar fallback
    (64, 32, 96, 32),     # K=96 not %32 -> scalar fallback
])
def test_pergroup_int8_scalar_fallback_correct(M, N, K, G):
    _run(M, N, K, G)
