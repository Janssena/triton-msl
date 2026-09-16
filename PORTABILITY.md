# Portability — develop on Apple Silicon, run on NVIDIA or AMD

triton-msl is a **backend** for [Triton](https://github.com/triton-lang/triton),
not a separate language. Your `@triton.jit` source is standard Triton: the frontend
(Python → Triton IR) is shared with the CUDA and ROCm backends, and triton-msl only swaps
the final stage (Triton IR → Metal instead of → PTX / GCN). The tested sources below run
unchanged through Triton's NVIDIA/AMD backends with tensors on `cuda` rather than `mps`
(ROCm's torch also uses the `cuda` device). Other sources and configurations need their
own backend support and numerical checks; shared frontend syntax does not guarantee them.

## What's portable, and what isn't

| | Portable? | Why |
|---|---|---|
| **Kernel source** | ✅ Tested examples run unchanged | Each backend must still be checked for the source's semantics and supported envelope |
| **Performance** | ❌ No — and never will be | Block sizes, occupancy, fast-path routing are hardware-specific (different memory hierarchy + matrix units). Physics, not a gap. |
| **Numerics** | ⚠️ Measured per kernel | Bit-identical results were observed for the specific vector-add and IEEE matmul below; this is not a general fp32 guarantee |

The practical consequence: you can develop and correctness-debug a kernel's *logic* on the
Mac you already own. Validate the same source on the target backend before relying on
its results: frontend sharing alone does not establish backend correctness. Performance
tuning also needs the target GPU (see *Reproduce* below).

## Historical receipt — two examples bit-identical across three vendors

The three example kernels (`examples/local_triton_dev.py`), unmodified, run on all three
backends and checked against the **same** NumPy reference:

- **Mac:** Apple M4 Max (Metal) · triton-msl · Triton 3.7.0 · torch 2.12.1
- **NVIDIA:** A40 (CUDA) · Triton 3.0.0 · torch 2.4.1  *(rented, then terminated)*
- **AMD:** Instinct MI300X (ROCm 7.2.4 · gfx942) · Triton 3.6.0 · torch 2.10.0  *(rented, then terminated)*

| kernel | Mac vs NumPy | NVIDIA vs NumPy | AMD vs NumPy |
|---|---|---|---|
| `vector_add` (n = 98,432) | 0 | 0 | 0 |
| `fused_softmax` (128 × 781) | 7.45e-9 | 5.59e-9 | 7.45e-9 |
| `matmul` fp32 / `ieee` (256³) | 4.58e-5 | 4.58e-5 | 4.58e-5 |
| `matmul` `tf32` | **refused** in the recorded Metal run | 6.07e-2 | n/a in the published receipt |

**The strongest evidence: identical *bits* across all three vendors.** For `vector_add` and
the `ieee` matmul, the raw fp32 output isn't just close — it's the same SHA-256 on Apple
Metal, NVIDIA CUDA, *and* AMD ROCm:

| kernel | SHA-256 (first 12) | identical elements |
|---|---|---|
| `vector_add` | `0fee55ab3218` — Metal == CUDA == ROCm | 98,432 / 98,432 |
| `matmul` (ieee) | `b650d0b7667f` — Metal == CUDA == ROCm | 65,536 / 65,536 |

Softmax passed the reported tolerance comparison across all three (≤7.5e-9); that
observation alone does not identify the cause of every difference. The comparison crossed
**three Triton versions** (3.0.0 / 3.6.0 / 3.7.0) and three ISAs (Metal · PTX · CDNA3).

> **Scope (honest):** measured on these specific shapes and block sizes. Different tiling can
> introduce last-bit differences from accumulation order, so this is a *measured* result on a
> representative set, not a universal bit-equality or correctness guarantee. A divergence must
> be investigated; do not assume it is rounding merely because the source is shared.

## Precision policy: the tf32 comparison

The historical NVIDIA comparison used **tf32** (~10-bit mantissa) for one fp32-input
matmul arm and explicitly requested IEEE precision for the other. The same kernel:

- on NVIDIA with tf32 → **6.1e-2** error vs an fp64 reference (the tf32 mantissa loss);
- with `input_precision="ieee"` → **bit-identical** with the Mac.

The current local TF32 request errors in the tested Triton frontend before Metal execution,
so it earns no computation credit and is not a demonstrated backend refusal. Request
`input_precision="ieee"` explicitly for the byte-parity comparison. Do not generalize this
single example to every default precision policy, GPU generation or dot shape.

## Reproduce it yourself

Use a suitable CUDA or ROCm environment with `torch` and `triton`. Provision rented hardware
only after the exact validation source and inputs are prepared; setup time and pricing vary.
On ROCm, torch presents the GPU as the `cuda` device, so the same command works:

```bash
# on the NVIDIA (CUDA) or AMD (ROCm) box:
python3 benchmarks/cross_backend_verify.py cuda     # writes out_cuda.npz + prints vs-NumPy errors
# pull out_cuda.npz back to the Mac, then:
python3 benchmarks/cross_backend_verify.py mps      # writes out_mps.npz
# diff the two .npz element-wise -> the Mac <-> CUDA/ROCm deltas
```

`benchmarks/cross_backend_verify.py` defines the kernels once, runs them on `cuda` or
`mps`, saves the outputs, and (when both files exist) prints the cross-backend deltas.
This legacy convenience script alone is not a release receipt: retain classified outcomes,
full source/input/output hashes and process results, then run the final strict comparison.
A successful process exit or a matching short hash prefix alone is not numerical proof.

## Historical accepted567 local recheck

The retained accepted567 Mac run re-executed the same three kernel definitions
and frozen inputs. Vector-add and IEEE matmul match the previously published
12-character SHA-256 prefixes; their current full output hashes are:

- Vector-add: `0fee55ab3218214d32181d3c38bee0b21d8d658b9786d0666ce2f0d5ddd796b6`
- IEEE matmul: `b650d0b7667f107a44ed9fe7ad13984f017eea97b35af9485fc7fbce6943261c`

Softmax remains a tolerance comparison, not byte parity: its current maximum
absolute error against the frozen NumPy reference is 3.73e-9. IEEE matmul's
maximum error against the reference is 4.58e-5; byte equality across vendors is a
different question from equality to a higher-precision oracle.

Those observations alone did not renew the AMD/NVIDIA receipt. The later candidate-bound
checks below retain frozen sources and inputs, classified outcomes and raw outputs.

## Candidate-bound cross-vendor receipt — remote 2026-09-15; exact871 local rebind 2026-09-16

Packet 878 re-executed the frozen original and additive-IEEE contracts on exact871 in both native and pure Metal modes, then compared those current local outputs with the retained A40 and MI300X outputs. The remote jobs were not rerun. They used upstream Triton's CUDA/ROCm backends; **they do not validate the Metal compiler on another vendor**.
This is source-portability and numerical/output-byte evidence, not cross-vendor performance
qualification. Earlier hardware/software tables above remain historical.

| Arm | Actual software used for the explicit-IEEE companion |
|---|---|
| Apple M4 Max, native and pure Metal | Installed validation artifact bound to the named source receipt; exact871 local gates are separate |
| NVIDIA A40 | Python 3.11.11; PyTorch 2.8.0.dev20250319+cu128; Triton 3.3.0; driver 570.195.03 |
| AMD MI300X VF | Python 3.12.3; PyTorch 2.12.0+rocm7.14.0; Triton 3.7.1; HIP 7.14.60850 |

### Exact-byte checks

All **11 mandatory exact predicates** pass between every vendor pair, using packet 878's current exact871 native/pure Metal outputs and the retained remote outputs. Native and
pure Metal are compared separately. Local Metal passes 16/16 original required numerical rows;
each retained remote arm remains 14/16 because the two default-precision attention rows below fail. The two historically published exact-output examples
still hold; the additional predicates are scoped to their frozen source/input rows.

| Exact-output family | Required rows | Result |
|---|---:|---|
| Vector add and explicit-IEEE matmul | 2 | Byte-equal |
| Argmin/argmax | 4 | Byte-equal |
| Broadcast | 2 | Byte-equal |
| Scan | 1 | Byte-equal |
| Matmul epilogue | 2 | Byte-equal |

Output SHA-256 remains `0fee55ab3218214d32181d3c38bee0b21d8d658b9786d0666ce2f0d5ddd796b6`
for vector add and `b650d0b7667f107a44ed9fe7ad13984f017eea97b35af9485fc7fbce6943261c`
for IEEE matmul. Exact agreement among devices is distinct from agreement with the
higher-precision oracle. Other numerical rows are tolerance checks, not byte-parity claims.

### Default-precision failures and separate IEEE qualification

The original 17-row contract has 16 required rows and one optional TF32 diagnostic.
On both remote backends, **14/16 required rows pass** the retained source-oracle checks.
`attention_16x32_narrow0_causal0` and `attention_32x64_narrow0_causal0` fail at the
unchanged 3e-5 absolute/relative tolerances. Their default dot precision permits TF32/XF32,
as confirmed by retained compiler IR; Metal uses IEEE on those sources. The original
oracle did not account for that backend-specific precision, so these runs do not establish
default-precision cross-vendor accuracy at the stated threshold. **Their original overall
gate remains failed.** No tolerance was widened and no failed output was discarded.

Two additive contexts, suffixed `_ieee`, specify `input_precision='ieee'` at the dot
operations. Everything else—including inputs, masks, sentinel rows, oracle bytes and
tolerances—is preserved. These are new source contexts, not a repair to the old receipt.

| Explicit-IEEE context | Metal max absolute error | A40 max absolute error | MI300X max absolute error |
|---|---:|---:|---:|
| attention_16x32_narrow0_causal0_ieee | 3.58e-7 | 2.98e-7 | 2.38e-7 |
| attention_32x64_narrow0_causal0_ieee | 3.28e-7 | 2.98e-7 | 3.58e-7 |

Both contexts pass first/warm/final checks and all pairwise numerical comparisons in both
Metal modes. Largest cross-vendor absolute difference: **1.79e-7**. Sentinels remain exact.
No production/default change was necessary. The optional TF32 row computes remotely but
errors before execution on Metal; it is not credited as supported Metal computation.

Receipt bindings retained for independent review:

- Original portability runtime freeze: `0526046ee00a4165e976434e151bc4e3f2c0646a42876e14db84e8b0b92fcd0b`; exact871 source freeze: `f2ddc60329cedbdfe536142a8dea324e7c3000011851ff9c234a4c3a25d231db`.
- Additive companion contract: `cd37080ff91c9e35e8a218d92d10e061828444bfeae27690b07e630fcf3b364b`.
- Companion source: `b213ce5bf8865e268879f5930e5831f20ec8edeac5679088f1de96df2a367871`.
- A40 result archive: `f831938dc8ba0ff0eda54ce605adba1d3063b081290b5fe43f4ea15ab8c552f2`.
- MI300X result archive: `e56a4bca2c6c7c09e37b308c9a595c4791f89f0dec24ec71ebab809a32e061e8`.

Packets 800/802 retain the original source-default outcomes; 803/804 retain the additive
IEEE contexts and the precision investigation; 805 independently re-derives both comparisons.
The investigation's post-observation rounding model is corroboration, not an independent
hardware oracle. These bounded results do not certify other shapes, arbitrary kernels,
or the reporter's full AlphaFold workload. Independent M1 validation remains user-run
after merge, not implied by an M4 Max pass.

## The workflow this enables

**Home = the correctness / iteration loop** (free, fast, offline) — get the kernel's logic
right on your Mac. **A rented GPU = the target perf + numerical-validation pass**. You
don't need to *own* NVIDIA silicon for the inner loop; you need it occasionally for the
perf pass.

A kernel triton-msl **refuses** (see [`docs/SUPPORTED_OPS.md`](docs/SUPPORTED_OPS.md))
may lie outside the current lowering/proof envelope. Some limits are implementation
capabilities; others are the actual resource budget for a particular emitted kernel.
Neither a refusal nor a passing example certifies other layouts, shapes or operations.
