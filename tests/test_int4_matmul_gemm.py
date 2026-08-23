"""Weight-only INT4 per-group GEMM (M>1), the GEMM companion to the int4 decode GEMV.
Packed uchar weight [K/2, N] (2 nibbles/byte, LOW=even k), per-group fp32 scale/zero,
fp32 in/out. Routes to make_int4_matmul_pergroup — correct-or-refuse via FULL loop-k
pinning: the byte index (k//2) and nibble (k%2) must share ONE unshifted loop k (a
tt.dot contracts the activation over the same k). A non-standard packing (high-nibble
-first `(k+1)%2` single-swap, or `(k+1)//2 + (k+1)%2` double-swap) REFUSES instead of
silently dequantizing the wrong nibble.
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
def _int4_gemm(
    a_ptr, w_ptr, c_ptr, s_ptr, z_ptr, M, N, K,
    sam, sak, wbk, wsn, scm, scn, ssg, ssn, zsg, zsn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + ok
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak)
        packed = tl.load(w_ptr + (kk // 2)[:, None] * wbk + rn[None, :] * wsn)
        w4 = (packed >> ((kk % 2) * 4)[:, None]) & 0xF
        g = k // G
        s = tl.load(s_ptr + g * ssg + rn * ssn)
        z = tl.load(z_ptr + g * zsg + rn * zsn)
        acc += tl.dot(a, (w4.to(tl.float32) - z[None, :]) * s[None, :])
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc)


@triton.jit
def _int4_gemm_highfirst(
    a_ptr, w_ptr, c_ptr, s_ptr, z_ptr, M, N, K,
    sam, sak, wbk, wsn, scm, scn, ssg, ssn, zsg, zsn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + ok
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak)
        packed = tl.load(w_ptr + (kk // 2)[:, None] * wbk + rn[None, :] * wsn)
        w4 = (packed >> (((kk + 1) % 2) * 4)[:, None]) & 0xF   # HIGH nibble for even k
        g = k // G
        s = tl.load(s_ptr + g * ssg + rn * ssn)
        z = tl.load(z_ptr + g * zsg + rn * zsn)
        acc += tl.dot(a, (w4.to(tl.float32) - z[None, :]) * s[None, :])
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc)


@triton.jit
def _int4_gemm_doubleswap(
    a_ptr, w_ptr, c_ptr, s_ptr, z_ptr, M, N, K,
    sam, sak, wbk, wsn, scm, scn, ssg, ssn, zsg, zsn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + ok
        kk2 = kk + 1                                           # byte AND nibble shifted
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak)
        packed = tl.load(w_ptr + (kk2 // 2)[:, None] * wbk + rn[None, :] * wsn)
        w4 = (packed >> ((kk2 % 2) * 4)[:, None]) & 0xF
        g = k // G
        s = tl.load(s_ptr + g * ssg + rn * ssn)
        z = tl.load(z_ptr + g * zsg + rn * zsn)
        acc += tl.dot(a, (w4.to(tl.float32) - z[None, :]) * s[None, :])
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc)


def _setup(M, N, K, G, wlayout, slayout):
    dev = "mps"; torch.manual_seed(0); ng = K // G
    a = torch.randn(M, K, device=dev)
    w4 = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)
    packed_kn = (w4[0::2, :] | (w4[1::2, :] << 4)).to(torch.uint8).contiguous()
    if wlayout == "kn":
        weight = packed_kn; wbk, wsn = weight.stride(0), weight.stride(1)
    else:
        weight = packed_kn.t().contiguous(); wbk, wsn = weight.stride(1), weight.stride(0)
    scale = (torch.rand(ng, N, device=dev) * 0.02 + 0.005)
    zero = torch.randint(0, 16, (ng, N), device=dev).float()
    if slayout == "gn":
        s_t, z_t = scale.contiguous(), zero.contiguous()
        ssg, ssn, zsg, zsn = s_t.stride(0), s_t.stride(1), z_t.stride(0), z_t.stride(1)
    else:
        s_t, z_t = scale.t().contiguous(), zero.t().contiguous()
        ssg, ssn, zsg, zsn = s_t.stride(1), s_t.stride(0), z_t.stride(1), z_t.stride(0)
    c = torch.zeros(M, N, device=dev)
    gi = torch.arange(K, device=dev) // G
    ref = a @ ((w4.float() - zero[gi, :]) * scale[gi, :])
    args = (a, weight, c, s_t, z_t, M, N, K, a.stride(0), a.stride(1), wbk, wsn,
            c.stride(0), c.stride(1), ssg, ssn, zsg, zsn)
    return args, c, ref


@requires_mps
@pytest.mark.parametrize("wl", ["kn", "nk"])
@pytest.mark.parametrize("sl", ["gn", "ng"])
@pytest.mark.parametrize("M,N,K,G", [(8, 16, 256, 128), (4, 32, 128, 64)])
def test_int4_gemm_routes_and_computes(M, N, K, G, wl, sl):
    args, c, ref = _setup(M, N, K, G, wl, sl)
    _int4_gemm[(triton.cdiv(M, 32), triton.cdiv(N, 16))](*args, BM=32, BN=16, BK=64, G=G)
    torch.mps.synchronize()
    assert (c - ref).abs().max().item() < 1e-2


@requires_mps
@pytest.mark.parametrize("kernel", [_int4_gemm_highfirst, _int4_gemm_doubleswap])
def test_int4_gemm_nonstandard_packing_refuses(kernel):
    args, _, _ = _setup(8, 16, 256, 128, "kn", "gn")
    with pytest.raises(MetalNonRecoverableError):
        kernel[(1, 1)](*args, BM=32, BN=16, BK=64, G=128)


@requires_mps
@pytest.mark.parametrize("M,N,K,G", [(32, 16, 256, 128), (64, 32, 512, 128), (128, 64, 1024, 64)])
def test_int4_gemm_fast_path_correct(M, N, K, G):
    # M%(8*rr)==0, N%(8*rc)==0, K%bk==0, G%bk==0 + contiguous kn -> FAST simdgroup-MMA
    # int4 path (int4_matmul_pergroup_fast, ~the int8 fast speed). Must stay exact.
    args, c, ref = _setup(M, N, K, G, "kn", "gn")
    _int4_gemm[(triton.cdiv(M, 32), triton.cdiv(N, 16))](*args, BM=32, BN=16, BK=64, G=G)
    torch.mps.synchronize()
    assert (c - ref).abs().max().item() < 1e-2
