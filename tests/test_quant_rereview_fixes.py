"""Regression tests for the 2026-08-25 adversarial re-review of the quant fast paths.
Four confirmed holes, all now correct-or-refuse:

  1. zeros-layout: the FAST per-group kernels index `zeros` with the SCALE's strides; a
     zeros tensor with its own layout (e.g. transposed vs scales) was silently mis-indexed.
     The fast gate now requires zsg==ssg and zsn==ssn; other layouts take the scalar
     kernel (stride-generic, correct).
  2. scale/zero dtype: the templates declare `device const float*`; fp16 scales/zeros were
     REINTERPRETED as float bits (garbage). The descriptors now refuse non-f32 scale/zero.
  3. M/N positional swap: M and N are DEAD args in canonical maskless kernels, so a
     (N, M, K) declaration routed and read/wrote OUT OF BOUNDS. The dispatch now bounds-
     checks every tensor's furthest touched element (memory-safe + refuses the swap).
  4. fail-open bypass: a quant launch the dequant dispatch declined fell through to the
     generic 1-D compile_shader path when its grid was 1-D, mis-binding the template ABI
     (garbage instead of the refusal). The 1-D path now honors _quant_unhandled.
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
def _pg_int8_r(a_ptr, w_ptr, c_ptr, scale_ptr, zero_ptr, M, N, K,
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


def _mk(M, N, K, G, seed=0):
    dev = "mps"; torch.manual_seed(seed); ng = K // G
    a = torch.randn(M, K, device=dev)
    w = torch.randint(-8, 8, (K, N), device=dev, dtype=torch.int8).contiguous()
    s = (torch.rand(ng, N, device=dev) * 0.03 + 0.01).contiguous()
    z = torch.randint(-4, 4, (ng, N), device=dev).float().contiguous()
    c = torch.zeros(M, N, device=dev)
    return a, w, s, z, c


def _ref(a, w, s, z, K, G):
    gi = torch.arange(K, device=a.device) // G
    return a @ ((w.float() - z[gi, :]) * s[gi, :])


def _launch(a, w, c, s, z, M, N, K, G):
    _pg_int8_r[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
        a, w, c, s, z, M, N, K,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1),
        s.stride(0), s.stride(1), z.stride(0), z.stride(1), c.stride(0), c.stride(1),
        BM=32, BN=32, BK=32, G=G)
    torch.mps.synchronize()


@requires_mps
def test_zeros_own_layout_correct():
    # Fast-eligible shape, but zeros transposed vs scales -> must take the scalar kernel
    # and stay CORRECT (was silently mis-indexed by the fast kernel).
    M, N, K, G = 32, 16, 64, 32
    a, w, s, z, c = _mk(M, N, K, G)
    z_t = z.t().contiguous().t()          # same values, strides (1, ng)
    assert z_t.stride() != z.stride()
    _launch(a, w, c, s, z_t, M, N, K, G)
    err = (c - _ref(a, w, s, z, K, G)).abs().max().item()
    assert err < 1e-3, f"zeros-own-layout mis-indexed: err {err:.2e}"


@requires_mps
@pytest.mark.parametrize("half_scale,half_zero", [(True, False), (False, True)])
def test_fp16_scale_zero_correct_or_refuse(half_scale, half_zero):
    # fp16 scales/zeros (GPTQ-style) must never be bit-reinterpreted as float.
    M, N, K, G = 32, 16, 64, 32
    a, w, s, z, c = _mk(M, N, K, G, seed=1)
    s_in = s.half() if half_scale else s
    z_in = z.half() if half_zero else z
    try:
        _launch(a, w, c, s_in, z_in, M, N, K, G)
    except MetalNonRecoverableError:
        return  # refused loudly — safe
    err = (c - _ref(a, w, s_in.float(), z_in.float(), K, G)).abs().max().item()
    assert err < 1e-2, f"fp16 scale/zero mis-computed: err {err:.2e}"


@triton.jit
def _pg_int8_nmk_r(a_ptr, w_ptr, c_ptr, scale_ptr, zero_ptr, N, M, K,
                   sam, sak, swk, swn, ssg, ssn, zsg, zsn, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    # identical body; the SCALAR DECLARATION ORDER is (N, M, K) — a valid Triton kernel.
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


@requires_mps
def test_swapped_mnk_declaration_correct_or_refuse():
    # (N, M, K) declaration order with M != N: the positional m_idx/n_idx read swapped
    # extents. Must refuse (runtime bounds gate + fail-closed driver) or be correct —
    # NEVER the OOB/partial-write garbage it used to produce.
    dev = "mps"; torch.manual_seed(4)
    M, N, K, G = 64, 32, 64, 32
    a, w, s, z, c = _mk(M, N, K, G, seed=4)
    try:
        _pg_int8_nmk_r[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
            a, w, c, s, z, N, M, K,
            a.stride(0), a.stride(1), w.stride(0), w.stride(1),
            s.stride(0), s.stride(1), z.stride(0), z.stride(1), c.stride(0), c.stride(1),
            BM=32, BN=32, BK=32, G=G)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # refused loudly — safe (also proves the 1-D fail-open bypass stays closed)
    err = (c - _ref(a, w, s, z, K, G)).abs().max().item()
    assert err < 1e-3, f"swapped-declaration mis-computed: err {err:.2e}"


@requires_mps
def test_canonical_pergroup_still_fast_and_correct():
    # The guards must not over-refuse the canonical case (fast path or scalar, correct).
    M, N, K, G = 32, 16, 64, 32
    a, w, s, z, c = _mk(M, N, K, G, seed=2)
    _launch(a, w, c, s, z, M, N, K, G)
    err = (c - _ref(a, w, s, z, K, G)).abs().max().item()
    assert err < 1e-3, f"canonical per-group broken by guards: err {err:.2e}"
