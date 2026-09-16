"""Kimi Delta Attention (KDA) / gated DeltaNet — a direct Metal op.

Linear/delta-rule attention: a per-key-dimension forget gate combined with the delta
rule's fast-weight correction. This is the attention the 2026 frontier models (Kimi,
DeltaNet family) are adopting, and it is *not* standard softmax attention. Its chunked
(parallel-prefill) form needs a UT-transform triangular solve, so it cannot be expressed
as a single ``@triton.jit`` kernel and does not route through the FlashAttention path.
It is dispatched directly through ``compile_shader``.

See ``triton_msl.codegen._msl_templates.make_kda_kernel`` for the algorithm and the
chunked derivation, and ``tests/test_kda.py`` for correctness against the recurrent form.
"""

from triton_msl.codegen._msl_templates import make_kda_decode_kernel, make_kda_kernel

_RT = None
_LIBS = {}  # fp16 bool -> compiled prefill library
_DEC_LIB = None


def _require_tensor_abi(name, tensor, *, shape, dtype, device):
    """Refuse a fixed-ABI mismatch before compiling or allocating Metal resources."""
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _require_mps(name, tensor):
    if tensor.device.type != "mps":
        raise ValueError(f"{name} must be on an mps device, got {tensor.device}")


def _state_layout_injective(state):
    """Prove positive-stride indices occupy distinct storage locations."""
    dimensions = sorted((stride, size) for size, stride in zip(state.shape, state.stride()) if size > 1)
    span = 1
    for stride, size in dimensions:
        if stride < span:
            return False
        span += (size - 1) * stride
    return True


def _kernel(fp16=False):
    """Lazily build + cache the compiled KDA prefill library (one per dtype)."""
    global _RT
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    if _RT is None:
        _RT = CompileShaderRuntime()
    if fp16 not in _LIBS:
        _LIBS[fp16] = _RT.get_library(make_kda_kernel(fp16=fp16))
    return _RT, _LIBS[fp16]


def kda_attention(q, k, v, a, beta):
    """Chunked-prefill gated-delta (KDA) attention.

    Args:
        q, k, v: ``[ZH, T, 64]`` float32 or float16 on the ``mps`` device (ZH = batch*heads).
        a:       ``[ZH, T, 64]`` per-key-dim forget gate, values in (0, 1).
        beta:    ``[ZH, T]`` delta-rule step (typically a sigmoid, in (0, 1)).

    Returns:
        ``[ZH, T, 64]`` output on ``mps``, same dtype as ``q``.

    Constraints: head dim is fixed at 64, ``T % 8 == 0`` (chunk size 8). One threadgroup
    per head. Accumulate and state are always fp32; fp16 inputs use a half-I/O kernel that
    casts on load and converts the MMA output on store (rel ~6e-4 vs an fp64 reference).
    A head with non-finite inputs or exceptional chunk intermediates is recomputed
    from zero by the scalar recurrence, within the same dispatch. This also covers
    underflowed cumulative gate products. Its finite values can round differently
    from the chunked path; neither path promises bitwise equality to the other.
    """
    import torch

    if q.dim() != 3 or q.shape[2] != 64:
        raise ValueError(f"kda_attention expects q of shape [ZH, T, 64], got {tuple(q.shape)}")
    if q.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"kda_attention supports float32/float16, got {q.dtype}")
    ZH, T, D = q.shape
    if ZH <= 0 or T <= 0:
        raise ValueError(f"kda_attention requires positive ZH and T, got ZH={ZH}, T={T}")
    if T % 8 != 0:
        raise ValueError(f"kda_attention requires T % 8 == 0 (chunk size 8), got T={T}")
    _require_mps("q", q)
    for name, tensor in (("k", k), ("v", v), ("a", a)):
        _require_tensor_abi(name, tensor, shape=(ZH, T, D), dtype=q.dtype, device=q.device)
    _require_tensor_abi("beta", beta, shape=(ZH, T), dtype=q.dtype, device=q.device)

    rt, lib = _kernel(fp16=q.dtype == torch.float16)
    out = torch.empty(ZH, T, D, device=q.device, dtype=q.dtype)
    args = [
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        beta.contiguous(),
        out,
        T,
    ]
    rt.dispatch(lib, "kda_prefill", args, threads=(ZH * 256, 1, 1), group_size=(256, 1, 1))
    return out


def _decode_kernel():
    """Lazily build + cache the KDA decode library."""
    global _RT, _DEC_LIB
    if _DEC_LIB is None:
        from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

        if _RT is None:
            _RT = CompileShaderRuntime()
        _DEC_LIB = _RT.get_library(make_kda_decode_kernel())
    return _RT, _DEC_LIB


def kda_decode_step(q, k, v, a, beta, S):
    """One autoregressive KDA decode step, updating the recurrent state in place.

    Args:
        q, k, v, a: ``[ZH, 64]`` float32 on ``mps`` (one token).
        beta:       ``[ZH]`` float32.
        S:          ``[ZH, 64, 64]`` float32 recurrent state, **updated in place**; pass the
                    same tensor across steps (start from ``torch.zeros``, or the prefill's
                    final state, to continue a sequence).

    Returns:
        ``[ZH, 64]`` float32 output for this token.
    """
    import torch

    if q.dim() != 2 or q.shape[1] != 64:
        raise ValueError(f"kda_decode_step expects q of shape [ZH, 64], got {tuple(q.shape)}")
    if q.dtype != torch.float32:
        raise ValueError(f"kda_decode_step supports float32, got {q.dtype}")
    ZH, D = q.shape
    if ZH <= 0:
        raise ValueError(f"kda_decode_step requires positive ZH, got ZH={ZH}")
    _require_mps("q", q)
    for name, tensor in (("k", k), ("v", v), ("a", a)):
        _require_tensor_abi(name, tensor, shape=(ZH, D), dtype=q.dtype, device=q.device)
    _require_tensor_abi("beta", beta, shape=(ZH,), dtype=q.dtype, device=q.device)
    _require_tensor_abi("S", S, shape=(ZH, D, D), dtype=q.dtype, device=q.device)
    if not _state_layout_injective(S):
        raise ValueError("S state layout cannot be proven non-overlapping")
    readonly = (q, k, v, a, beta)
    prepared = [
        tensor.clone(memory_format=torch.contiguous_format) if torch._C._overlaps(S, tensor) else tensor.contiguous()
        for tensor in readonly
    ]
    rt, lib = _decode_kernel()
    out = torch.empty(ZH, D, device=q.device, dtype=torch.float32)
    state = S if S.is_contiguous() else S.contiguous()
    args = [*prepared, state, out]
    rt.dispatch(lib, "kda_decode", args, threads=(ZH * 256, 1, 1), group_size=(256, 1, 1))
    if state is not S:
        S.copy_(state)
    return out
