"""Regression tests for GitHub issue #4 (trifast -> Metal porting bugs).

Each test pins a reported silent-wrong or over-refusal to the integrity contract:
a kernel triton-msl accepts must be CORRECT, and one it cannot lower must be REFUSED
(``MetalNonRecoverableError``) -- never silently wrong.

The matmul kernels below rebuild addresses from base each iteration (no loop-carried
pointer) so they isolate issue #1 (reduction-extent arg name) from issue #2.
"""
import pytest

import triton
import triton.language as tl

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from triton_msl.errors import MetalNonRecoverableError

requires = pytest.mark.skipif(not HAS_TORCH, reason="torch needed")


@triton.jit
def _mm_K(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offk = k + tl.arange(0, BK)
        a = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
        b = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
        acc += tl.dot(tl.load(a), tl.load(b))
    tl.store(c_ptr + (offm[:, None] * scm + offn[None, :] * scn), acc)


@triton.jit
def _mm_DIM(a_ptr, b_ptr, c_ptr, M, N, DIM, sam, sak, sbk, sbn, scm, scn,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, DIM, BK):
        offk = k + tl.arange(0, BK)
        a = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
        b = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
        acc += tl.dot(tl.load(a), tl.load(b))
    tl.store(c_ptr + (offm[:, None] * scm + offn[None, :] * scn), acc)


@triton.jit
def _mm_depth(a_ptr, b_ptr, c_ptr, M, N, depth, sam, sak, sbk, sbn, scm, scn,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM); offn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, depth, BK):
        offk = k + tl.arange(0, BK)
        a = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
        b = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
        acc += tl.dot(tl.load(a), tl.load(b))
    tl.store(c_ptr + (offm[:, None] * scm + offn[None, :] * scn), acc)


@requires
@pytest.mark.parametrize("kernel", [_mm_K, _mm_DIM, _mm_depth],
                         ids=["K", "DIM", "depth"])
def test_reduction_extent_correct_or_refuse(kernel):
    """#1: the K-loop must reduce over the FULL extent regardless of the reduction
    arg's name, or refuse -- never silently drop the loop and return the first tile."""
    M = N = KK = 64
    torch.manual_seed(0)
    A = torch.randn(M, KK, device="cpu", dtype=torch.float32)
    B = torch.randn(KK, N, device="cpu", dtype=torch.float32)
    C = torch.empty(M, N, device="cpu", dtype=torch.float32)
    try:
        kernel[(M // 32, N // 32)](
            A, B, C, M, N, KK,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            BM=32, BN=32, BK=32,
        )
    except MetalNonRecoverableError:
        return  # refused loudly -- contract satisfied
    ref = A @ B
    err = (C - ref).abs().max().item()
    first_tile_err = (C - (A[:, :32] @ B[:32, :])).abs().max().item()
    assert err < 1e-3, (
        f"wrong result (err={err:.3f}); err vs first-tile-only={first_tile_err:.3f} "
        f"({'SILENT-WRONG: dropped the K-loop' if first_tile_err < 1e-3 else 'wrong'})"
    )


@requires
def test_over_31_buffer_args_refuses_clearly(tmp_path):
    """#7: a kernel with more arguments than Metal's 31 buffer slots must refuse with a
    clear, actionable message -- not the wall of cryptic "'buffer' attribute parameter is
    out of bounds" the Metal frontend emits (one line per overflowing arg, naming nothing).
    """
    import importlib.util

    n_scalars = 33  # 1 pointer + 33 scalars = 34 args, over the 31-slot limit
    names = [f"s{i}" for i in range(n_scalars)]
    body = " + ".join(names)
    src = (
        "import triton\nimport triton.language as tl\n\n"
        f"@triton.jit\ndef big_kernel(out_ptr, {', '.join(names)}):\n"
        f"    tl.store(out_ptr + tl.program_id(0), {body})\n"
    )
    modfile = tmp_path / "bigk.py"
    modfile.write_text(src)
    spec = importlib.util.spec_from_file_location("bigk", modfile)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = torch.zeros(1, device="cpu", dtype=torch.int32)
    with pytest.raises(MetalNonRecoverableError) as ei:
        mod.big_kernel[(1,)](out, *range(n_scalars))
    msg = str(ei.value).lower()
    assert "buffer" in msg and "31" in msg, f"refusal message not clear: {ei.value}"
