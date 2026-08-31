"""Dataflow pointer-role pins for non-matmul fast templates.

The flip, reshape-transpose, N-D transpose and row-sort detectors used to assign
input/output roles from pointer declaration order.  Output-first signatures therefore
made the template read the output sentinel and overwrite the real input.  Row sort had
an additional sibling: it resolved pointer names from stores, but still selected its
compute element type from ``ptr_args[0]``.

Every test asserts the distinctive template MSL was emitted, so a correct generic
fallback cannot masquerade as a template fix.
"""

import pytest

try:
    import Metal
    import torch
    import triton
    import triton.language as tl

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False


requires = pytest.mark.skipif(not HAS, reason="Metal + torch + triton needed")


def _msl(handle):
    asm = getattr(handle, "asm", {}) or {}
    return asm.get("msl", "") or asm.get("metal", "") or ""


if HAS:

    @triton.jit
    def _flip_output_first(OUT, INP, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, DIM: tl.constexpr):
        oi = tl.arange(0, M) * N * K
        oj = tl.arange(0, N) * K
        ok = tl.arange(0, K)
        off = oi[:, None, None] + oj[None, :, None] + ok[None, None, :]
        tl.store(OUT + off, tl.flip(tl.load(INP + off), DIM))


@requires
def test_flip_template_resolves_output_first_signature():
    # Pre-fix template: OUT was read as X and INP was written as Z; both max errors
    # were 162.0. The template marker makes this a detector pin, not just correctness.
    M, N, K, DIM = 4, 4, 4, 1
    original = torch.arange(M * N * K, device="mps", dtype=torch.float32).reshape(M, N, K)
    inp = original.clone()
    out = torch.full_like(inp, -99.0)
    handle = _flip_output_first[(1,)](out, inp, M=M, N=N, K=K, DIM=DIM, num_warps=4)
    torch.mps.synchronize()
    assert "uint _src" in _msl(handle), "flip fast template did not claim the pin"
    torch.testing.assert_close(out, original.flip(DIM), rtol=0, atol=0)
    torch.testing.assert_close(inp, original, rtol=0, atol=0)


if HAS:

    @triton.jit
    def _reshape_transpose_output_first(OUT, INP, M: tl.constexpr, N: tl.constexpr):
        block = tl.make_block_ptr(
            base=INP,
            shape=(M, N),
            strides=(N, 1),
            offsets=(0, 0),
            block_shape=(M, N),
            order=(1, 0),
        )
        x = tl.load(block)
        x = tl.reshape(x, (32, 4, 4, 2))
        x = tl.permute(x, (1, 2, 3, 0))
        tl.store(OUT + tl.arange(0, M * N), tl.reshape(x, (M * N,)))


@requires
def test_reshape_transpose_template_resolves_output_first_signature():
    # Pre-fix: output and input were swapped, each with max error/mutation 1122.
    M = N = 32
    original = torch.arange(M * N, device="mps", dtype=torch.int32).reshape(M, N)
    inp = original.clone()
    out = torch.full((M * N,), -99, device="mps", dtype=torch.int32)
    handle = _reshape_transpose_output_first[(1,)](out, inp, M=M, N=N, num_warps=4)
    torch.mps.synchronize()
    assert "uint _row = k %" in _msl(handle), "reshape-transpose fast template did not claim the pin"
    assert torch.equal(out.reshape(N, M), original.T)
    assert torch.equal(inp, original)


if HAS:

    @triton.jit
    def _nd_transpose_output_first(
        OUT,
        INP,
        s1: tl.constexpr,
        s2: tl.constexpr,
        s3: tl.constexpr,
        s4: tl.constexpr,
        o1: tl.constexpr,
        o2: tl.constexpr,
        o3: tl.constexpr,
        o4: tl.constexpr,
        t1: tl.constexpr,
        t2: tl.constexpr,
        t3: tl.constexpr,
        t4: tl.constexpr,
    ):
        in_desc = tl.make_tensor_descriptor(
            base=INP,
            shape=[s1, s2, s3, s4],
            strides=[s4 * s3 * s2, s4 * s3, s4, 1],
            block_shape=[s1, s2, s3, s4],
        )
        out_desc = tl.make_tensor_descriptor(
            base=OUT,
            shape=[o1 * o2 * o3 * o4],
            strides=[1],
            block_shape=[o1 * o2 * o3 * o4],
        )
        value = in_desc.load([0, 0, 0, 0]).permute((t1, t2, t3, t4))
        out_desc.store([0], value.reshape(out_desc.block_shape))


@requires
def test_nd_transpose_template_resolves_output_first_signature():
    # Pre-fix: the direct-copy template swapped pointers; output error and input
    # mutation were both 1122 on this exact permutation.
    shape = (4, 4, 4, 16)
    perm = (3, 1, 0, 2)
    total = 1
    for dim in shape:
        total *= dim
    original = torch.arange(total, device="mps", dtype=torch.int32).reshape(shape)
    inp = original.clone()
    out = torch.full((total,), -99, device="mps", dtype=torch.int32)
    out_shape = tuple(shape[i] for i in perm)
    handle = _nd_transpose_output_first[(1,)](out, inp, *shape, *out_shape, *perm, num_warps=8)
    torch.mps.synchronize()
    assert "for (uint k = lid" in _msl(handle), "N-D transpose fast template did not claim the pin"
    assert torch.equal(out, original.permute(perm).reshape(-1))
    assert torch.equal(inp, original)


if HAS:

    @triton.jit
    def _sort_output_first_mixed_dtype(OUT, INP, M: tl.constexpr, N: tl.constexpr):
        i = tl.arange(0, M)
        j = tl.arange(0, N)
        values = tl.load(INP + i[:, None] * N + j[None, :])
        tl.store(OUT + i[:, None] * N + j[None, :], tl.sort(values))


@requires
def test_row_sort_template_uses_traced_input_dtype():
    # The detector already traced pointer names, but derived compute_type from the
    # first DECLARED pointer. With output-first i8 and input i32 it sorted wrapped i8
    # values and produced 28/32 wrong stores instead of sorting i32 then casting.
    M, N = 2, 16
    base = torch.tensor(
        [130, -129, 256, -257, 5, 4, 3, 2, 1, 0, -1, -2, 127, -128, 500, -500],
        device="mps",
        dtype=torch.int32,
    )
    inp = torch.stack((base, base.flip(0)))
    original = inp.clone()
    out = torch.full((M, N), -99, device="mps", dtype=torch.int8)
    handle = _sort_output_first_mixed_dtype[(1,)](out, inp, M=M, N=N, num_warps=4)
    torch.mps.synchronize()
    assert "Bitonic sort of 16" in _msl(handle), "row-sort fast template did not claim the pin"
    assert torch.equal(out, original.sort(dim=1).values.to(torch.int8))
    assert torch.equal(inp, original)
