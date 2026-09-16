"""Metal-specific benchmarking using GPU timestamps.

Defaults to synchronized wall time. Explicit GPU timing or labeled automatic
selection can use completed-command GPUStartTime/GPUEndTime.
"""

import math
import operator
import time


def metal_do_bench(
    fn, *, quantiles=None, warmup=25, rep=100, synchronize=None, clock="wall", return_metadata=False, **kwargs
):
    """Benchmark one completed workload per sample, with an explicit clock.

    Compatible with Triton's benchmarker interface (registered via
    MetalDriver.get_benchmarker()).

    Executes fn exactly once per warmup/sample, completing its work before
    reading timestamps. The default scalar/list result ALWAYS measures wall time.
    Automatic selection requires metadata and uses GPU time only if EVERY sample
    has valid timestamps, otherwise the whole wall-time series. Explicit GPU
    timing raises on unavailable timestamps; it never silently switches clocks.
    Never relaunches a callback to obtain a fallback sample.

    Args:
        fn: Callable to benchmark. Should dispatch Metal work.
            A returned waitable command buffer must cover completion of ALL its
            work, or synchronize must be provided to complete the remainder.
            GPU/auto timing has a STRONGER caller contract: the returned buffer's
            GPU interval must contain the entire workload. Returning just one of
            several dispatched buffers undercounts and violates that contract;
            use wall timing plus a completion callback for multi-buffer work.
            The helper cannot discover undisclosed queues or buffers.
        quantiles: List of quantiles to return (e.g. [0.5, 0.2, 0.8]).
            If None, returns the median time.
        warmup: Number of warmup iterations.
        rep: Number of timed iterations.
        synchronize: Optional completion callback for functions that do not
            return a waitable command buffer. An explicitly supplied callback is
            ALSO called after a returned buffer completes, so it can wait for
            other queues/buffers. Defaults to MPS synchronization
            when MPS is available, otherwise a no-op. Use a matching callback
            for another asynchronous runtime. Run on an otherwise idle queue;
            warmup=0 does not discard work queued before this invocation.
        clock: "wall" (default), "gpu" (strict), or "auto" (requires metadata).
        return_metadata: Return a dictionary containing value, clock, unit,
            boundary, requested clock, sample counts and any fallback reason.
            Measurement records must retain this dictionary, not strip its label.

    Returns:
        By default, a median or quantile list in milliseconds, preserving Triton's
        benchmarker interface with a fixed wall clock. With return_metadata=True,
        the same scalar/list is under "value" alongside the measurement boundary.
    """
    warmup, rep = operator.index(warmup), operator.index(rep)
    if warmup < 0 or rep <= 0:
        raise ValueError("warmup must be nonnegative and rep must be positive")
    if clock not in ("wall", "gpu", "auto"):
        raise ValueError("clock must be wall, gpu or auto")
    if clock == "auto" and not return_metadata:
        raise ValueError("automatic clock selection requires return_metadata=True")
    if quantiles is not None:
        quantiles = list(quantiles)
        if not quantiles or any(not math.isfinite(q) or not 0 <= q <= 1 for q in quantiles):
            raise ValueError("quantiles must be a nonempty sequence in [0, 1]")
    explicit_sync = synchronize is not None
    if not explicit_sync:
        try:
            import torch
        except ImportError:
            torch = None
        synchronize = torch.mps.synchronize if torch is not None and torch.backends.mps.is_available() else lambda: None
    if not callable(synchronize):
        raise TypeError("synchronize must be callable")

    def complete(result):
        wait = getattr(result, "waitUntilCompleted", None)
        if callable(wait):
            wait()
            error = getattr(result, "error", None)
            failure = error() if callable(error) else None
            if failure is not None:
                raise RuntimeError(f"Metal benchmark command failed: {failure}")
        if explicit_sync or not callable(wait):
            synchronize()

    for _ in range(warmup):
        complete(fn())

    wall_times, gpu_times = [], []
    for _ in range(rep):
        start = time.perf_counter()
        result = fn()
        complete(result)
        wall_times.append((time.perf_counter() - start) * 1000.0)
        if clock == "wall":
            continue
        gpu_start = getattr(result, "GPUStartTime", None)
        gpu_end = getattr(result, "GPUEndTime", None)
        if callable(gpu_start) and callable(gpu_end):
            a, b = gpu_start(), gpu_end()
            if math.isfinite(a) and math.isfinite(b) and 0 <= a < b:
                gpu_times.append((b - a) * 1000.0)

    gpu_valid = len(gpu_times) == rep
    if clock == "gpu" and not gpu_valid:
        raise ValueError("GPU clock requested but some timestamps are missing or invalid")
    used_clock = "gpu" if clock != "wall" and gpu_valid else "wall"
    times = sorted(gpu_times if used_clock == "gpu" else wall_times)
    if quantiles is None:
        value = times[len(times) // 2]
    else:
        value = [times[int(q * (len(times) - 1))] for q in quantiles]
    if not return_metadata:
        return value
    return {
        "value": value,
        "clock": used_clock,
        "requested_clock": clock,
        "unit": "ms",
        "boundary": (
            "entire-workload command buffer GPU start/end"
            if used_clock == "gpu"
            else "callback invocation through completion, including host overhead"
        ),
        "warmup": warmup,
        "samples": rep,
        "valid_gpu_timestamp_samples": len(gpu_times) if clock != "wall" else None,
        "fallback_reason": ("missing_or_invalid_gpu_timestamps" if clock == "auto" and not gpu_valid else None),
    }


class MetalBenchmark:
    """GPU-timed benchmark runner using Metal command buffer timestamps.

    This class directly owns its command buffers and measures GPU time using
    MTLCommandBuffer.GPUStartTime/GPUEndTime.
    """

    def __init__(self):
        import Metal

        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.queue = self.device.newCommandQueue()

    def time_kernel(self, pipeline, buffers, n_elements, block_size=256, warmup=10, rep=100):
        """Time a compute kernel dispatch using GPU timestamps.

        Args:
            pipeline: MTLComputePipelineState.
            buffers: List of MTLBuffer objects to bind.
            n_elements: Total elements (determines grid size).
            block_size: Threads per threadgroup.
            warmup: Warmup iterations.
            rep: Timed iterations.

        Returns:
            dict with keys: median_us, min_us, max_us, p10_us, p90_us,
            throughput_gb_s (if applicable), all_us.
        """
        import Metal

        n_groups = (n_elements + block_size - 1) // block_size

        # Warmup
        for _ in range(warmup):
            self._dispatch(pipeline, buffers, n_groups, block_size)

        # Timed runs
        gpu_times_us = []
        for _ in range(rep):
            cmd = self.queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipeline)
            for i, buf in enumerate(buffers):
                enc.setBuffer_offset_atIndex_(buf, 0, i)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_groups, 1, 1),
                Metal.MTLSizeMake(block_size, 1, 1),
            )
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            gpu_start = cmd.GPUStartTime()
            gpu_end = cmd.GPUEndTime()
            gpu_times_us.append((gpu_end - gpu_start) * 1e6)

        gpu_times_us.sort()
        n = len(gpu_times_us)

        return {
            "median_us": gpu_times_us[n // 2],
            "min_us": gpu_times_us[0],
            "max_us": gpu_times_us[-1],
            "p10_us": gpu_times_us[int(0.1 * (n - 1))],
            "p90_us": gpu_times_us[int(0.9 * (n - 1))],
            "all_us": gpu_times_us,
        }

    def _dispatch(self, pipeline, buffers, n_groups, block_size):
        """Dispatch a kernel (no timing)."""
        import Metal

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(n_groups, 1, 1),
            Metal.MTLSizeMake(block_size, 1, 1),
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()


def compute_throughput(n_bytes, time_us):
    """Compute throughput in GB/s from byte count and time in microseconds."""
    if time_us <= 0:
        return float("inf")
    return (n_bytes / 1e9) / (time_us / 1e6)


def compute_gflops(n_flops, time_us):
    """Compute GFLOP/s from flop count and time in microseconds."""
    if time_us <= 0:
        return float("inf")
    return (n_flops / 1e9) / (time_us / 1e6)


def format_benchmark_result(name, result, n_bytes=None, n_flops=None):
    """Format a benchmark result as a human-readable string."""
    lines = [f"  {name}:"]
    lines.append(
        f"    GPU time: {result['median_us']:.1f} us "
        f"(min={result['min_us']:.1f}, max={result['max_us']:.1f}, "
        f"p10={result['p10_us']:.1f}, p90={result['p90_us']:.1f})"
    )
    if n_bytes:
        bw = compute_throughput(n_bytes, result["median_us"])
        lines.append(f"    Bandwidth: {bw:.1f} GB/s")
    if n_flops:
        gf = compute_gflops(n_flops, result["median_us"])
        lines.append(f"    Compute: {gf:.1f} GFLOP/s")
    return "\n".join(lines)
