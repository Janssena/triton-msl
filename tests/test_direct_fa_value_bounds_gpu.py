"""Scheduled GPU qualification for corrected direct-maker value widths.

No GPU availability probe runs at collection; execute only in an assigned window.
"""
import math

import pytest
import torch

from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
from triton_msl.codegen.msl_emitter import make_flash_attention_kernel_simdgroup


@pytest.mark.parametrize("qk,v,dtype", [(128,32,torch.float16), (128,40,torch.float16),
                                      (128,96,torch.float16), (192,96,torch.float16),
                                      (128,32,torch.float32), (128,40,torch.float32),
                                      (128,96,torch.float32)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", [50, 128])
def test_direct_value_width_matches_reference_and_preserves_output_canaries(qk, v, dtype, causal, n):
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        pytest.skip("requires assigned MPS compile_shader window")
    generator = torch.Generator(device="cpu").manual_seed(814)
    q_cpu = torch.randn((1, 2, n, qk), generator=generator, dtype=dtype)
    k_cpu = torch.randn((1, 2, n, qk), generator=generator, dtype=dtype)
    v_cpu = torch.randn((1, 2, n, v), generator=generator, dtype=dtype)
    scores = (q_cpu.double() @ k_cpu.double().transpose(-2, -1)) / math.sqrt(qk)
    if causal:
        mask = torch.ones((n, n), dtype=torch.bool).triu(1)
        scores.masked_fill_(mask, -float("inf"))
    reference = torch.softmax(scores, dim=-1) @ v_cpu.double()

    q, k = q_cpu.to("mps"), k_cpu.to("mps")
    # Padded real allocations keep the regression witness inside allocated
    # memory even if an old maker addresses its surplus columns.
    value_backing = torch.full((1, 2, n, v + 128), 137.0, device="mps", dtype=dtype)
    values = value_backing[..., 64:64+v]
    values.copy_(v_cpu.to("mps"))
    backing = torch.full((1, 2, n + 2, v + 128), -8192.0, device="mps", dtype=dtype)
    output = backing[..., 1:n+1, 64:64+v]
    source = make_flash_attention_kernel_simdgroup(
        head_dim=qk, v_head_dim=v, out_dtype="fp16" if dtype == torch.float16 else "fp32",
        causal=causal, kernel_name="direct_value_bounds")
    runtime = CompileShaderRuntime()
    library = runtime.get_library(source)
    runtime.dispatch(library, "direct_value_bounds",
                     [q, k, values, output, *q.stride(), *k.stride(), *values.stride(), *output.stride(), 1, 2, n],
                     threads=(((n + 31) // 32) * 256, 2, 1), group_size=(256, 1, 1))
    torch.mps.synchronize()
    actual = output.cpu().double()
    # Existing direct-FA accuracy budgets; no tolerance relaxation.
    assert (actual - reference).abs().max().item() < (5e-3 if dtype == torch.float16 else 1e-3)
    retained = backing.cpu()
    untouched = torch.ones(retained.shape, dtype=torch.bool)
    untouched[..., 1:n+1, 64:64+v] = False
    assert torch.equal(retained[untouched], torch.full_like(retained[untouched], -8192.0))
