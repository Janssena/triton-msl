"""Triton → MLX backend: zero-copy Metal dispatch via mx.fast.metal_kernel().

This dispatches triton-msl's OWN generated MSL through MLX's runtime — an
alternative launcher to torch.mps.compile_shader for code that lives on MLX
arrays. It does NOT call MLX's native ops (mx.matmul, MLX attention, ...), so it
borrows MLX's dispatch, not its speed; kernels run at the same throughput as the
compile_shader path. ("Not competitive with MLX in absolute terms" refers to
MLX's hand-tuned library kernels, which are a different thing from this launcher.)

Usage:
    import triton
    import triton.language as tl
    import triton_msl.mlx as tmlx
    import mlx.core as mx

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    x = mx.random.normal((1024,))
    y = mx.random.normal((1024,))
    out = mx.zeros((1024,))
    (result,) = tmlx.triton_call(
        add_kernel, x, y, out, 1024,
        grid=(4,), BLOCK=256,
    )
"""

from triton_msl.mlx.msl_extractor import extract_msl_for_mlx, MSLExtraction
from triton_msl.mlx.mlx_launcher import MLXLauncher

__all__ = ["triton_call", "mlx_available"]

# Cache: (fn_hash, sig_hash, constexpr_hash) → (MSLExtraction, metadata)
_compile_cache = {}

# packet 176: dispatch descriptors the compile_shader / torch.mps driver honours and this launcher
# does NOT — a route-only template ABI (packed scalar buffer, template-ordered arguments), a
# two-kernel split, a runtime-dispatch matmul descriptor, host-side address-bounds checks. Binding
# the Triton arguments positionally against such a kernel is silently wrong, so the route refuses.
_UNSUPPORTED_DESCRIPTORS = ("flash_attention", "mm_two_kernel", "fast_matmul", "quant_matmul", "batched_dot_bounds")


def mlx_available():
    """Check if MLX is available for metal_kernel dispatch."""
    try:
        import mlx.core as mx

        mx.fast.metal_kernel  # verify API exists
        return True
    except (ImportError, AttributeError):
        return False


def _mlx_dtype_to_triton_sig(dtype):
    """Map MLX dtype to Triton signature string."""
    import mlx.core as mx

    mapping = {
        mx.float32: "*fp32",
        mx.float16: "*fp16",
        mx.bfloat16: "*bf16",
        mx.int32: "*i32",
        mx.uint32: "*u32",
        mx.int16: "*i16",
        mx.uint16: "*u16",
        mx.int8: "*i8",
        mx.uint8: "*u8",
        mx.bool_: "*i1",
    }
    if dtype not in mapping:
        # packet 176: an unknown dtype (int64, uint64, float64, complex, ...) used to become a
        # *fp32 pointer silently — the kernel would then read the bytes as floats.
        from triton_msl.errors import MetalNonRecoverableError

        raise MetalNonRecoverableError(
            f"MLX route: array dtype {dtype} has no Triton signature mapping; refusing rather than "
            f"binding it as *fp32. Supported: float32/float16/bfloat16, int8/16/32, uint8/16/32, bool."
        )
    return mapping[dtype]


def _scalar_to_triton_sig(val):
    """Map a Python scalar to Triton signature string."""
    if isinstance(val, bool):
        return "i1"
    elif isinstance(val, int):
        if not (-(2 ** 31) <= val < 2 ** 31):
            # packet 176: the launcher passes Python ints as int32 scalars; a stride / element count
            # beyond int32 was truncated silently.
            from triton_msl.errors import MetalNonRecoverableError

            raise MetalNonRecoverableError(
                f"MLX route: integer argument {val} does not fit int32 (the launcher's scalar width); refusing."
            )
        return "i32"
    elif isinstance(val, float):
        return "fp32"
    return "i32"


def _build_signature(jit_fn, args, constexpr_kwargs):
    """Build Triton signature dict from @triton.jit fn and runtime args."""
    import mlx.core as mx

    arg_names = jit_fn.arg_names
    signature = {}
    constexprs = {}

    runtime_idx = 0
    for name in arg_names:
        if name in constexpr_kwargs:
            signature[name] = "constexpr"
            constexprs[name] = constexpr_kwargs[name]
        else:
            if runtime_idx >= len(args):
                raise ValueError(f"Not enough args: expected arg for '{name}' at position {runtime_idx}")
            arg = args[runtime_idx]
            if isinstance(arg, mx.array):
                signature[name] = _mlx_dtype_to_triton_sig(arg.dtype)
            elif arg is None:
                # Output placeholder — assume fp32 pointer
                signature[name] = "*fp32"
            else:
                signature[name] = _scalar_to_triton_sig(arg)
            runtime_idx += 1

    return signature, constexprs


def _compile_kernel(jit_fn, signature, constexprs):
    """Compile a @triton.jit function to MSL via Triton's pipeline."""
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend.compiler import MetalBackend

    target = GPUTarget("metal", "apple-m4", 32)

    src = ASTSource(fn=jit_fn, signature=signature, constexprs=constexprs)
    compiled = triton_compile(src, target=target, options={})

    msl_source = compiled.asm["msl"]
    metadata = compiled.metadata

    return msl_source, metadata, compiled


def _cache_key(jit_fn, signature, constexprs):
    """Build a hashable cache key."""
    sig_key = tuple(sorted(signature.items()))
    const_key = tuple(sorted((k, v) for k, v in constexprs.items()))
    fn_key = id(jit_fn)  # Use JITFunction identity
    return (fn_key, sig_key, const_key)


def triton_call(kernel_fn, *args, grid, num_warps=4, **constexpr_kwargs):
    """Call a @triton.jit kernel with MLX arrays via zero-copy Metal dispatch.

    Compiles the kernel to MSL, extracts the body for mx.fast.metal_kernel(),
    and dispatches with zero buffer copies.

    Args:
        kernel_fn: @triton.jit decorated function.
        *args: Kernel arguments in signature order (excluding constexpr).
            MLX arrays for pointer args, Python int/float for scalars.
            Output args should be MLX arrays (shape/dtype used for allocation).
        grid: Tuple of threadgroup counts, e.g. (4,) or (4, 2) or (4, 2, 1).
        num_warps: Accepted for API symmetry; the threadgroup size comes from the
            compiled kernel's metadata (block_size), not from this value.
        **constexpr_kwargs: Compile-time constants (e.g. BLOCK_SIZE=256).

    Returns:
        List of output MLX arrays.

    Example:
        x = mx.random.normal((1024,))
        y = mx.random.normal((1024,))
        out = mx.zeros((1024,))
        (result,) = triton_call(
            add_kernel, x, y, out, 1024,
            grid=(4,), BLOCK=256,
        )
    """
    # Keep JITFunction for arg_names; use .fn for compilation
    jit_fn = kernel_fn

    signature, constexprs = _build_signature(jit_fn, args, constexpr_kwargs)
    key = _cache_key(jit_fn, signature, constexprs)

    if key in _compile_cache:
        extraction, block_size, needs_2d_grid = _compile_cache[key]
    else:
        msl_source, metadata, compiled = _compile_kernel(jit_fn, signature, constexprs)

        # packet 176: fail closed on every dispatch descriptor this launcher does not honour.
        for _name in _UNSUPPORTED_DESCRIPTORS:
            if getattr(metadata, _name, None) is not None:
                from triton_msl.errors import MetalNonRecoverableError

                raise MetalNonRecoverableError(
                    f"MLX route: this kernel's lowering set the '{_name}' dispatch descriptor (a "
                    f"route-only template ABI, a two-kernel split, a runtime-dispatch matmul or "
                    f"host-side bounds checks), which mx.fast.metal_kernel dispatch does not implement; "
                    f"the positional binding would be silently wrong. Use the torch.mps path for this kernel."
                )

        block_size = getattr(metadata, "block_size", num_warps * 32)
        output_arg_indices = getattr(metadata, "output_arg_indices", None)
        needs_2d_grid = getattr(metadata, "needs_2d_grid", False)

        # packet 176: the parsed MSL signature must carry exactly the Triton signature's runtime
        # arguments — a packed or reordered ABI would otherwise be bound positionally.
        n_runtime = sum(1 for v in signature.values() if v != "constexpr")
        extraction = extract_msl_for_mlx(
            msl_source, output_arg_indices, expected_args=n_runtime,
            expected_signature=[(name, ty) for name, ty in signature.items() if ty != "constexpr"],
        )
        _compile_cache[key] = (extraction, block_size, needs_2d_grid)

    launcher = MLXLauncher(extraction, block_size=block_size, needs_2d_grid=needs_2d_grid)
    return launcher(grid, *args)
