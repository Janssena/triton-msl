# triton-msl

> **0.3.0 — validation scope and limitations.** Fresh correctness gates, installed acceptance, four performance floors and a three-workload cold subset ran on the formatted 0.3.0 candidate at `73ba428` (packets 906–909). Final artifacts are rebuilt and rebound to those tested runtime payloads, with fresh and relocated CPU checks. The full warm comparisons and vendor receipts remain pre-format measurements, not reruns on the formatted artifacts. Historical throughput claims remain unqualified except for the explicitly scoped attention comparisons below; those comparisons clear none of the main-relative performance alerts. The reported AlphaFold/trifast workload is not validated end-to-end, no independent Apple Silicon validation has run (one M4 Max only), and the optional C++ route is unaudited/deferred and off by default—leaving it off or packaging its sources is not validation. See `CHANGELOG.md` and the release-facing limitations at https://github.com/bledden/triton-msl/blob/main/docs/RELEASE_CANDIDATE_LIMITATIONS.md.

> Fresh formatted-candidate evidence: 6,079 project passes / 16 skips; unchanged 9,342-node upstream map (5,780 passed / 3,562 skipped); installed acceptance 1,468 native / 252 pure with no failures/skips; four M4 Max floors. The pre-format native and pure warm comparisons retain 13 and 14 open >10% alerts respectively. KDA prefill is a separate pre-format 1.1016× alert (block range 1.0721–1.1313). Cold first-specialization costs remain substantial: the formatted subset observed native add at 9.48 s versus main's 0.63 s and compiled MLP at 10.02 s versus 0.86 s. These are single observations including compilation and live observation, not whole-app startup or evidence of improvement over the different full-matrix process history. The earlier packet 879 observations remain separately recorded in the limitations. These remain open performance regressions, not universal model slowdowns or proof that further improvement is impossible. Wrong/refusing main receives no ratio; varlen uses a changed retained route and is not a same-route speedup.

> Packet 878 re-executes the original portability contract locally on exact871: both native and pure Metal pass 16/16 required numerical rows. All 11 scoped exact-byte predicates pass against the retained A40/MI300X outputs. Only the retained remote default-precision arms remain 14/16: their two attention rows remain failed. Separate additive explicit-IEEE contexts pass 2/2 on every arm. This does not rewrite defaults, tolerances or failed receipts. M1 remains best-effort post-merge.

> **Validation observation boundary (option A):** with the native helper available, a recognized unchanged-state hit uses callback-free identity checks. It does not replay Python key/descriptor/metadata observation callbacks or the audit hooks those reads would trigger, including their side effects. Set `TRITON_MSL_IDENTITY_FAST_PATH=0` to run the complete evaluator on every invocation, including after an existing warm hit. Supported between-call configuration/provider changes remain detected. Both public-launch validations, argument/ABI checks, submission tracking and immediate assertion observation remain enabled. Public JIT and Inductor have different validator counts.

> **Half-accumulate warning:** the shipped maker docstring's historical “~4% faster”, “~1% max-abs error” and “~0.01%” figures are unvalidated for this release and are not universal error bounds. The mode is opt-in and off by default; validate actual inputs/shapes. Default-float rescale tests provide no credit for that opt-in mode. Formatting does not validate those historical claims.

Metal (Apple Silicon) backend for [Triton](https://github.com/triton-lang/triton) [\[1\]](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md)[\[2\]](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md). Write `@triton.jit` kernels and run them on your Mac's GPU.

```
@triton.jit → Triton TTIR → TTGIR → MSL → metallib → Apple GPU
```

**The tested `@triton.jit` sources also run on NVIDIA and AMD.** triton-msl is a Triton
*backend*, not a dialect. Source portability still depends on each backend's supported
operations and launch constraints; it is not a guarantee for every kernel. Historical runs
on a real NVIDIA A40 **and** an AMD Instinct MI300X found that two fp32 example kernels (vector-add,
`ieee` matmul) produced **byte-identical** output — the same SHA-256 across Apple Metal,
NVIDIA CUDA, and AMD ROCm — and softmax passed the reported tolerance comparison
([`PORTABILITY.md`](https://github.com/bledden/triton-msl/blob/main/PORTABILITY.md)). The 2026-09-15 candidate-bound runs confirm **11 scoped
exact-byte checks** across Metal, A40 and MI300X. Two default-precision attention rows fail
the retained accuracy criterion on both remote backends; separate explicit-`ieee` versions
pass the unchanged numerical tolerance on all three. This is not universal byte parity or
a retroactive pass for the default spelling. CUDA/ROCm runs use Triton's own backends,
not the Metal compiler. See the current receipt and precision limits in `PORTABILITY.md`.

## Status

**Alpha**: actively developed, not yet production-ready.

The historical counts below describe the earlier published tree. Fresh formatted-candidate results, separately attributed pre-format measurements and remaining limits are recorded in `docs/RELEASE_CANDIDATE_LIMITATIONS.md`.

- **0 failures** across the upstream Triton `test_core.py` suite: 5,560 kernels
  passed under upstream's assertions, 3,782 documented skips in that historical run.
  Those counts are not a proof that every skipped case is hardware-impossible or that
  every accepted source is correct. This campaign found and repaired additional
  lowering defects and capability gaps. The current per-node gate and the named
  capability/refusal register supersede the old absence-of-defects interpretation.
  Aligned with Triton [\[2\]](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md) release `3.7.0`.
  Measured by `scripts/run_upstream_tests.py` (the single source of truth for this
  count), which runs `--device cpu` (torch references compute on CPU while the Metal
  backend compiles and runs the kernels on the GPU, since upstream `test_core`
  otherwise assumes CUDA). Re-run it to reproduce; counts in this file and
  `CHANGELOG.md` are regenerated from it, not hand-maintained.
- **1,982 passed / 0 failed** in the project suite (codegen, GPU correctness,
  integration, FlashAttention, quantized int8/int4, KDA + FlashAttention-backward,
  MLX backend, fast-matmul / compile_shader zero-copy, `torch.compile`, and training).
  FlashAttention: causal + non-causal at **HEAD_DIM 32 / 64 / 128** (head_dim 64/128
  via the simdgroup-MMA kernel, dispatched zero-copy; see [\[4\]](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md) for the
  algorithm); **15 / 15** MLX backend tests; the project suite grew from
  434 → 603 → 716 → 877 → 1,971 → 1,982 since `0.1.0-alpha`.
  (A further ~20 C++-MLIR-backend tests skip unless that optional extension is
  built.)
- **`torch.compile` routes through triton-msl** on Python 3.10–3.14 (PyTorch
  Inductor [\[12\]](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md)), inference and training (AOTAutograd
  backward), static and `dynamic=True`; **32 / 32** `torch.compile` model tests
  (plus 1 inductor-config regression test = 33 total) and the training suite pass.
- Triton tutorials 01–03, 05 passing.
- Built against Triton's `TRITON_EXT_ENABLED=1` plugin architecture
  (upstream PR [#9783](https://github.com/triton-lang/triton/pull/9783)).
- **Integrity objective**: preserve the source semantics or *refuse*
  (`MetalNonRecoverableError`) when they cannot be established. This is the design
  standard, not a universal correctness guarantee; the update includes repairs for
  previously accepted sources that returned wrong values.
  See [`docs/SUPPORTED_OPS.md`](https://github.com/bledden/triton-msl/blob/main/docs/SUPPORTED_OPS.md) for the supported
  ops/dtypes matrix + the loud-refusal catalog, and
  [`docs/ARCHITECTURE.md`](https://github.com/bledden/triton-msl/blob/main/docs/ARCHITECTURE.md) "Lowering paths and the
  integrity model" for the lowering paths.

See [`REFERENCES.md`](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md) for citations and
[`docs/superpowers/specs/2026-05-30-triton-msl-roadmap.md`](https://github.com/bledden/triton-msl/blob/main/docs/superpowers/specs/2026-05-30-triton-msl-roadmap.md)
for the active pre-1.0 roadmap.

The optional `TRITON_MSL_USE_CPP=1` route is unaudited and off by default.
See [the C++ route contract](https://github.com/bledden/triton-msl/blob/main/docs/CPP_ROUTE_CONTRACT.md) for its explicit fallback
warnings, binary-production metadata, known-broken dot exclusion, and the
difference between a compiled binary and the route actually executed.

## Portability: develop on Apple Silicon, run on NVIDIA or AMD

triton-msl is a **backend** for Triton. The following three example sources,
unmodified, were historically run on an Apple M4 Max, a rented NVIDIA A40, **and** an AMD Instinct
MI300X (ROCm), each checked against the same NumPy reference:

| kernel | Mac vs NumPy | NVIDIA vs NumPy | AMD vs NumPy |
|---|---|---|---|
| `vector_add` | 0 | 0 | 0 |
| `fused_softmax` | 7.45e-9 | 5.59e-9 | 7.45e-9 |
| `matmul` (fp32 / `ieee`) | 4.58e-5 | 4.58e-5 | 4.58e-5 |
| `matmul` (`tf32`) | refused in the historical receipt | 6.07e-2 | n/a in that receipt |

**Measured source portability**: `vector_add` and the `ieee` matmul were **byte-identical**
across all three vendors — the same SHA-256 on Metal, CUDA, and ROCm — crossing Triton 3.0.0 /
3.6.0 / 3.7.0. **Performance is not** (block sizes and fast-path routing are hardware-specific).
Develop and debug kernel logic on the Mac, then validate the same frozen sources and inputs
on each target backend. Request `input_precision="ieee"` explicitly for the byte-parity
comparison; do not generalize these examples to every precision mode or kernel. The current
local TF32 request errors before backend execution and earns no computation credit.
The 2026-09-15 remote receipts are complete, with the two default-precision attention
failures preserved separately from the passing explicit-IEEE companion checks.
Historical and current scoped receipts and the reproduction procedure:
[`PORTABILITY.md`](https://github.com/bledden/triton-msl/blob/main/PORTABILITY.md).

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14 (Sonoma) or later — validated through macOS 26.6 (Tahoe)
- Xcode Command Line Tools: `xcode-select --install`
  - On **macOS 26 (Tahoe) / Xcode 26**, the Metal shader compiler ships as a
    separate on-demand component. If `xcrun metal --version` reports a missing
    Metal Toolchain, install it once with
    `sudo xcodebuild -downloadComponent MetalToolchain`.
- Python 3.10+
- PyTorch 2.12+ (2.5+ for the zero-copy MPS fast path; `torch.compile` is developed + tested against 2.12) and Triton 3.7.0

## Install

```bash
pip install triton-msl

# Triton is required but installed separately. There is no official macOS wheel,
# so build it from source (the primary, supported path, a one-time ~12 min build):
pip install git+https://github.com/triton-lang/triton.git
```

The release bundle includes a native `triton-msl` wheel for **GIL-enabled CPython
3.14, macOS 15+, arm64**, a pure Python wheel, and a source archive. The native
wheel contains three CPU host helpers: `_validation_native`, `_packed_native`, and
`_binder_native`; the pure wheel contains none. Missing helpers may fall back individually;
a discoverable broken helper fails loudly. These are not GPU kernel binaries or the optional
C++ compiler route. Native measurements do not describe Python-only installations.
Source builds support validation/packed helpers on GIL-enabled CPython 3.13/3.14;
the binder is 3.14-only. No runtime-qualified CPython 3.13 native wheel is included.
Earlier build/import checks do not establish Torch/Triton runtime coverage. Building helpers
requires Xcode Command Line Tools and does not replace Triton's separate requirements.

If you're on the exact platform tuple **Python 3.14 / macOS 15+ / Apple Silicon (M1–M5)**, an unofficial prebuilt Triton wheel is attached to the [GitHub releases](https://github.com/bledden/triton-msl/releases/tag/triton-wheel-3.7.0-cp314-macos-arm64) so you can skip the build:

```bash
pip install https://github.com/bledden/triton-msl/releases/download/triton-wheel-3.7.0-cp314-macos-arm64/triton-3.7.0+git4da2e268-cp314-cp314-macosx_15_0_arm64.whl
```

## Quick Start

### @triton.jit

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = torch.randn(n, device="cpu")
y = torch.randn(n, device="cpu")
out = torch.empty(n, device="cpu")
add_kernel[(n + 255) // 256,](x, y, out, n, BLOCK=256)
print(f"Max error: {(out - (x + y)).abs().max():.2e}")
```

Run a fuller, runnable version, vector-add, fused-softmax, and a tiled matmul, each
on a non-multiple shape (so every load/store boundary mask is exercised) and verified
against a NumPy reference, with [`examples/local_triton_dev.py`](https://github.com/bledden/triton-msl/blob/main/examples/local_triton_dev.py):

```bash
python examples/local_triton_dev.py
```

It's the same `@triton.jit` source you'd run on a CUDA GPU: develop and verify locally
on your Mac, then ship the identical kernels to NVIDIA.

### torch.compile

```python
import torch
import triton_msl.inductor
triton_msl.inductor.register_metal_triton_backend()

model = torch.nn.Sequential(
    torch.nn.Linear(256, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
).to("mps").eval()

# Registration selects Triton for Inductor's MPS device; there is no
# separate torch.compile backend named "metal".
compiled = torch.compile(model, backend="inductor")
x = torch.randn(32, 256, device="mps")
with torch.no_grad():
    out = compiled(x)
```

**Performance.** The compiled kernels dispatch zero-copy through the same `compile_shader`
fast path as the hand-written kernels. At small model sizes the compiled latency is roughly
on par with eager MPS, so the value here is coverage, not speed: `torch.compile` graphs
(inference and training) route through triton-msl and stay correct to eager within
floating-point tolerance, validated across 32 models spanning transformer blocks, a small
GPT, a ViT, ResNets, and LSTMs, forward and training backward.

**Coverage.** Transformer/attention, RNN, **CNN (incl. BatchNorm)**, normalization, and the
reductions (sum, product, mean, max/min incl. NaN-propagating, var/std, argmax/argmin, softmax,
logsumexp, cumsum/cumprod), including small under-filling reductions **and a 2-D reduction
fused with a 2-D scan in one kernel** (e.g. `x.sum(1) + x.cumprod(1)[:,-1]`), compile and match
eager. Anything genuinely beyond the hardware (a reduction/scan tile exceeding Metal's 1024
threads/threadgroup) is **refused loudly rather than mis-computed** (`MetalNonRecoverableError`,
never silent-wrong); the rest of the graph is unaffected.

### MLX

```python
import mlx.core as mx
import triton
import triton.language as tl
from triton_msl.mlx import triton_call

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = mx.random.normal((n,))
y = mx.random.normal((n,))
out = mx.zeros((n,))
results = triton_call(add_kernel, x, y, out, n, grid=(4,), BLOCK=256)
```

The MLX route re-launches the *same* emitted MSL through `mx.fast.metal_kernel`, so it inherits
every kernel-level guarantee of the torch path. Its own contract is narrower and enforced
(`MetalNonRecoverableError` before any dispatch): one kernel per launch with no template dispatch
descriptor (attention / matmul templates take the torch.mps path); outputs are **fresh** arrays, so a
kernel that reads or atomically updates an output pointer refuses; array dtypes must be in the
signature map and integer arguments must fit int32. One thing the route cannot check with MLX 0.32
(`mx.array` exposes neither strides nor contiguity): MLX hands the kernel a **row-contiguous copy** of
every input, so the stride arguments you pass must describe the row-contiguous layout of the array's
shape, not the strides of a transposed or sliced view.

### MPS tensors: zero-copy

The same `@triton.jit` kernel runs **zero-copy** on `torch` MPS tensors: the driver
dispatches the emitted Metal through `torch.mps.compile_shader`, skipping the host
round-trip (~10× faster on memory-bound kernels; on by default, no code change):

```python
x = torch.randn(n, device="mps")
y = torch.randn(n, device="mps")
out = torch.empty(n, device="mps")
add_kernel[(n + 255) // 256,](x, y, out, n, BLOCK=256)  # runs on the GPU, no copy
```

### Matmul (`tl.dot`)

fp16/fp32 matmuls (K%8) on MPS tensors take a direct simdgroup-matrix path, dispatched
zero-copy. A deterministic, occupancy-gated tile selector picks the coarsest blocking the
shape's M and N alignment allow: **M%32 + N%32** runs the `(4,4)` tile at ~11–12 TFLOP/s on
M4 Max; **M%8 or N%8** (not %32) runs a finer rescue tile (lower, but still far above the
generic path). **M%8 ≠ 0 or N%8 ≠ 0** (e.g. an odd vocab/hidden dim) falls to the ~2.4 TFLOP/s
generic path: a partial simdgroup strip can't be masked without writing past the matrix.

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    a = a_ptr + (offm[:, None] * sam + offk[None, :] * sak)
    b = b_ptr + (offk[:, None] * sbk + offn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(tl.load(a), tl.load(b))
        a += BK * sak; b += BK * sbk
    tl.store(c_ptr + (offm[:, None] * scm + offn[None, :] * scn), acc.to(tl.float16))

M = N = K = 2048
A = torch.randn(M, K, device="mps", dtype=torch.float16)
B = torch.randn(K, N, device="mps", dtype=torch.float16)
C = torch.empty(M, N, device="mps", dtype=torch.float16)
matmul_kernel[(M // 64, N // 64)](
    A, B, C, M, N, K,
    A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
    BM=64, BN=64, BK=32)
```

### Integrity contract: refused, never silently wrong

A kernel that triton-msl cannot lower correctly **raises `MetalNonRecoverableError`**
rather than returning garbage. For example, a pid-tiled matmul that bakes its M/N
dims as `constexpr` (so the true output strides can't be recovered) is refused:

```python
from triton_msl.errors import MetalNonRecoverableError

@triton.jit
def matmul_baked_dims(a_ptr, b_ptr, c_ptr, K,
                      M: tl.constexpr, N: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    a = a_ptr + (offm[:, None] * K + offk[None, :])
    b = b_ptr + (offk[:, None] * BN + offn[None, :])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _k in range(0, K, BK):
        acc += tl.dot(tl.load(a), tl.load(b)); a += BK; b += BK * BN
    tl.store(c_ptr + (offm[:, None] * BN + offn[None, :]), acc)

try:
    matmul_baked_dims[(2, 2)](A, B, C, K, M=64, N=64, BM=32, BN=32, BK=32)
except MetalNonRecoverableError as e:
    print("refused (not silent-wrong):", e)
```

See [`docs/SUPPORTED_OPS.md`](https://github.com/bledden/triton-msl/blob/main/docs/SUPPORTED_OPS.md) for the full op/dtype support
matrix and the loud-refusal catalog.

### FlashAttention

A FlashAttention v2 forward (causal + non-causal) runs through the standard `@triton.jit`
path for **head_dim 32, 64, and 128**. head_dim 64 and 128 route to an Apple
`simdgroup_matrix` MMA kernel (fp32 + fp16, any N_CTX); head_dim 32 uses the generic
lowering. See [`tests/test_flash_attention.py`](https://github.com/bledden/triton-msl/blob/main/tests/test_flash_attention.py) for the
kernel and launch.

The MMA kernel dispatches **zero-copy** through `compile_shader`. In the **historical,
August 2026 stack** cold A/B on M4 Max, that path was faster than PyTorch's
`scaled_dot_product_attention` in every measured case: **fp16 full 1.65–1.99×**, fp32 full
1.27–1.53×, and **causal up to ~4×** once
the kernel skips the all-masked upper-triangle KV blocks and spreads the online softmax
across all 256 threads (both exact — byte-identical to the fp32 reference). Against Apple's
own hand-tuned **MLX** FlashAttention it was about even at moderate sequence lengths and
**~0.88× at the largest** (full and causal alike); that gap traces to the kernel being
latency-bound on its device loads, with no async-copy engine available on Metal to hide
them. (An earlier "not competitive with MLX" reading was a dispatch bug — the kernel's 2-D
grid disqualified it from the zero-copy path, so every launch fell to the ~3× slower host
round-trip; the kernel itself was always fast.) These are historical observations,
not current MLX-parity or bottleneck guarantees.

**Current comparison (2026-09-16, packet 890):** the installed 0.3.0 native wheel
beats current PyTorch MPS SDPA on all 36 tested dense-attention medians, but takes
**23–44% longer than current MLX** (0.69–0.82× MLX speed, rounded). For batch 2,
8 heads, head dimension 128 and 1,024–4,096 tokens, native full attention is
**1.94–2.18× SDPA for fp16** and **1.36–1.41× for fp32**; fp16 causal is
**2.72–3.70×**. The pure wheel has separate results: 34/36 dense median wins
against SDPA, with some fp32 cases near parity and substantial block variation.
These are warmed synchronized API-call timings on the current stack, not an
exact August replay or an isolated GPU-kernel comparison. See
[the comparator table and attribution limits](https://github.com/bledden/triton-msl/blob/main/docs/RELEASE_CANDIDATE_LIMITATIONS.md#current-attention-comparators-890895).

A same-stack, block-alternated main/final follow-up (895) confirms the update
contributes to the gap: final native takes about **8–29% longer than checked
public main** on its 12 dense D128 cases, and **12–13% longer** on two MLA N2048
cases. Main also trails current MLX on every dense median. Two long-context
dense comparisons have paired blocks crossing parity; these ranges are measured
cases, not universal bounds or an attribution of the entire August-to-current gap.

**Latent attention (MLA).** A real DeepSeek/Kimi-style nope/rope kernel written in
`@triton.jit` **auto-routes**: the compiler recognizes the two chained query-key dot
products (a 128-wide "nope" plus a 64-wide "rope" part, a 192-wide contraction against a
128-wide output) and runs the asymmetric MMA kernel. The historical **1.19–1.59× SDPA**
range is not a current guarantee. Packet 890's native full MLA at 1,024/2,048 tokens
measures **1.08×/1.42× prejoined SDPA**, while its 512-token case loses. The 1.08×
median has blocks crossing parity. Split-input SDPA includes concatenation and
has a separate result; the current main-relative alert is yet another comparison.

Admission is source- and route-specific. This candidate also recovers bounded bf16,
8-/16-row decode and larger biased-attention tiles; there is no blanket rejection of all
bf16 or non-32 tiles. Unproved configurations refuse with `MetalNonRecoverableError`.
See `docs/CAPABILITY_AND_REFUSAL_CENSUS.md` for exact envelopes, rounding and mask limits.

### Frontier attention: linear/delta and a trainable backward

Two attention capabilities beyond the softmax-FA path ship as **direct Metal ops** (neither
is expressible as a single `@triton.jit` kernel), documented in
[`docs/attention_ops.md`](https://github.com/bledden/triton-msl/blob/main/docs/attention_ops.md):

- **KDA (Kimi Delta Attention / gated DeltaNet)** — linear/delta-rule attention with a
  per-key-dimension forget gate, the direction the newest models are taking. Chunked MMA
  prefill + recurrent decode, fp16/fp32, validated against a recurrent reference
  (`triton_msl.kda`).
- **A FlashAttention backward pass** — `triton_msl.fa_backward.flash_attention` is a
  `torch.autograd.Function` whose dQ/dK/dV run on Metal (tiled FA-2 backward, MMA, causal +
  full, fp16/fp32). The forward FA was inference-only; this makes attention **trainable** on
  the GPU.

### Quantized inference (int8 / int4)

Weight-only int8 and int4 matmuls **auto-route** to dedicated dequantizing kernels, in both
the natural `[K, N]` layout and the GPTQ-style `[N, K]`, dispatched zero-copy through
`compile_shader`. The path is **fail-closed**: a shape or dtype the fast kernel can't handle
is **refused** (`MetalNonRecoverableError`), never silent-wrong.

The historical decode comparison reported an int8 weight-only GEMV at about **3.7×**
fp32 decode while moving a quarter of the bytes; this speed ratio and roofline claim are
**unqualified on the current candidate**. Prefill (GEMM) is a memory-footprint
win rather than a speed one — fp32 MPS BLAS still runs the prefill matmul faster. int4 adds
per-group decode with zero points, and the skinny/deep matmul shapes that were
occupancy-starved gained a deterministic two-pass split-K.

### Tuning flags

All default-on; set to `0` to disable (an escape hatch for bisecting a regression):

| Flag | Effect when disabled |
|------|----------------------|
| `TRITON_MSL_COMPILE_SHADER=0` | Use the host-copy driver instead of the zero-copy `compile_shader` dispatch |
| `TRITON_MSL_FAST_MATMUL=0` | Use the generic matmul instead of the fast simdgroup-matrix path |
| `TRITON_MSL_MATMUL_AUTOTUNE=0` | Pin matmul tile selection to the fixed `(4,4)` blocking (M%32≠0 / N%32≠0 shapes drop to the generic path instead of the finer M%8/N%8 rescue tiles) |
| `TRITON_MSL_MEPT=0` | Disable the multi-element-per-thread register-array model |
| `TRITON_MSL_LEGACY=1` | Opt **in** to the heuristic legacy text parser (off by default, it can be silent-wrong) |
| `TRITON_MSL_FA_HALF_ACCUM=1` | Opt **in** to fp16 (half) MMA accumulators in FlashAttention. Off by default; fp16 kernels only, a no-op for fp32. Accuracy depends on the inputs, shape and accumulation order; no universal maximum-error bound is provided. Validate it for your workload before enabling it. The historically reported ~4% speedup has not been revalidated on this tree. |

## What Works

| Category | Operations |
|----------|-----------|
| **Elementwise** | add, sub, mul, div, exp, log, sqrt, abs, neg, SiLU, GELU, sigmoid, tanh, ReLU, leaky ReLU, clamp, FMA |
| **Reductions** | sum, max, min, argmax, argmin, xor_sum |
| **Dot product** | `tl.dot` with strided matmul and bounded source-replayed epilogues; not every arithmetic, dtype or tile combination. See the capability/refusal census. |
| **Attention** | FlashAttention [\[4\]](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md) forward, bounded dtype/tile/causal/decode and MLA routes; current speed comparisons remain unqualified. A Metal **backward** pass (`triton_msl.fa_backward`, trainable) and **KDA** linear/delta attention (`triton_msl.kda`) ship as direct ops ([`docs/attention_ops.md`](https://github.com/bledden/triton-msl/blob/main/docs/attention_ops.md)). Unsupported source contracts refuse explicitly. |
| **Quantized** | Weight-only **int8 / int4** matmul + decode GEMV, natural `[K, N]` and GPTQ `[N, K]` layouts, auto-routed; split-K for skinny/deep shapes. Historical speed and roofline figures are not current qualifications. |
| **Normalization** | Layer norm, RMS norm, batch norm |
| **Type casts** | FP32, FP16, BF16, INT8, INT16, INT32, bool |
| **Control flow** | `scf.for`, `scf.if`, while loops |
| **Atomics** | atomic_add, atomic_max, atomic_min, atomic_and, atomic_or, atomic_xor, CAS |
| **Tensor ops** | cat, join, split, interleave, reshape, permute, transpose, histogram, gather |
| **torch.compile** | 32 models including MLP, ResBlock, TransformerBlock, SmallGPT, MiniViT, LSTM |
| **MLX** | Zero-copy dispatch via `mx.fast.metal_kernel()` |

## What Doesn't Work

| Feature | Reason |
|---------|--------|
| FP64 | Metal has no FP64 support |
| FP8, TF32 | Not available on Apple GPUs |
| Multi-GPU | Apple Silicon is single-GPU |
| `tl.dot` with sizePerThread > 1 | Requires 2D cooperative execution model (addressed by the register-array spine, WS1) |
| Unstructured control flow (`cf.cond_br`) | Refused with `MetalNonRecoverableError` (never silent-wrong); a `cf`-dialect lowerer is WS2 |
| `tt.dot_scaled` (microscaling matmul) | No Apple microscaling hardware; refused |

## Performance (M4 Max [\[13\]](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md))

**Historical measurements, not requalified for this candidate.** Numbers use the zero-copy
`compile_shader` path (default-on); see
`reports/perf_baseline.json`. Hardware peak: 546 GB/s memory, 18.4 / 36.9 TFLOP/s
fp32 / fp16.

| Kernel | Size | Throughput | % of peak | vs host-copy path |
|--------|------|-----------|-----------|-------------------|
| Vector add | 16M | 347 GB/s | 64% | **13×** |
| Elementwise | 16M | 315 GB/s | 58% | 13.4× |
| Softmax | 8192×1024 | 232 GB/s | 42% | **17.8×** |
| Reduction | 16M | 235 GB/s | 43% | 8.2× |
| Matmul (fp32) | 2048³ | 11.2 TFLOP/s§ | 61% of fp32 peak | ~4× generic |
| Matmul (fp16 in / fp32 out) | 2048³ | 12.3 TFLOP/s | ≈ fp32 rate\* | ~4× generic |
| Matmul (fp16 in / fp16 out) | 2048³ | 12.2 TFLOP/s | ≈ fp32 rate\* | ~4× generic |
| Matmul (bf16 in / fp32 out)◊ | 2048³ | 12.0 TFLOP/s | ≈ fp32 rate\* | **~4.9× generic** |
| Matmul (bf16 in / bf16 out)◊ | 2048³ | 11.9 TFLOP/s | ≈ fp32 rate\* | **~4.9× generic** |
| FlashAttention (fp32 full, head_dim=128)‡ | Z=1,H=8,N=1024 | 5.1 TFLOP/s | ~28% of fp32 peak | **1.27–1.53× SDPA** |
| FlashAttention (fp16 full, head_dim=128)‡ | Z=1,H=8,N=1024 | 6.3 TFLOP/s | †| **1.65–1.99× SDPA** |

\* fp16 matmul uses fp16 inputs with a **float32 accumulator** (for precision). The
12.3 figure is the default **fp32-output** path; the true **fp16→fp16** path
(`out_dtype=fp16`) measures **12.2 TFLOP/s**, essentially identical, since both do
the same MACs and the output-cast cost is negligible. Either way it runs at roughly
the fp32 matrix-unit rate: Apple's simdgroup-matrix unit isn't faster for half
accumulation, so the 36.9 TFLOP/s fp16 figure is an unreachable vector-ALU peak. The ~58–64% memory-bound
and ~60% fp32-matmul numbers are **near the practical ceilings** for these kernel
classes on this hardware (the raw 546 / 18.4 / 36.9 spec peaks are not reachable by
compute), see the Phase-5 readiness audit (`docs/audits/`).

† FA fp16 accumulates in fp32. In the historical comparison, routed
zero-copy through `compile_shader` it was **faster than PyTorch SDPA in every measured case**
(fp16 full 1.65–1.99×, fp32 full 1.27–1.53×, causal up to ~4×) and **~0.88× Apple MLX** at
the largest sizes on the August stack. **Current native measurements are 0.69–0.82×
MLX across the tested dense panel**, while retaining 36/36 median wins versus
current SDPA; pure-wheel results differ. See the FlashAttention section above
and the limitations table; do not carry the historical MLX near-parity claim forward.

§ The fp32 matmul figure is the cold-machine peak (~11 TFLOP/s, verified 2026-06-24).
It is **thermally sensitive**: under sustained GPU load an M4 Max throttles the fp32
path to ~9 TFLOP/s, so `reports/perf_baseline.json`, re-measured after a long
benchmark session, currently records ~9.2; re-run `test_fast_matmul_perf` on an idle
machine to see the cold peak. fp16/fp16out throttle less (measured ~12.3/12.2 cold).
The 11.2 figure is the **dense row-major (contiguous-innermost) case only**.
**Non-contiguous / transposed / sliced operands** (e.g. `x @ w.t()`, a column-major
output, or a column slice `t[:, :K]` whose inner stride ≠ 1 or whose row stride ≠ the
matrix dim) fall **off** the simdgroup fast path; they are computed **correctly** by a
fully stride-aware scalar matmul (or refused when un-inferable; **never silently
wrong**), but that scalar path is **~15–23× slower** (below the generic floor).
`contiguous()` the operand before the kernel to stay on the fast path. Batched
rank-3 matmul has bounded support; it is not covered by these rank-2 throughput
figures. Unsupported batched address/layout contracts still refuse explicitly.

◊ bf16 matmul uses Apple's `simdgroup_bfloat8x8` matrix unit (float32 accumulate),
**verified on M4**; on a part without a bfloat matrix unit it falls back to the
(correct) generic float-compute path, never silently wrong.

‡ FA rows are **measured** on M4 Max (integration microbenchmark of the shipped
`make_flash_attention_kernel_simdgroup`: warmup + median over 50 iters). Absolute
throughput scales with sequence length (~6.8 / ~8.8 TFLOP/s fp32 / fp16 at N=2048), but the
meaningful comparison is now vs PyTorch SDPA and MLX (footnote † and the FlashAttention
section). Correctness is verified by the 16-case differential gate
(`tests/test_fa_simdgroup_diff.py`: simd == scalar oracle == torch, all pass).

**MPS tensors run zero-copy** via `torch.mps.compile_shader` (default-on); the prior
host-round-trip copy bottleneck is gone. CPU tensors and the MLX backend
[\[7\]](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md) (`mx.fast.metal_kernel`) also dispatch zero-copy.

## Architecture

```
@triton.jit kernel
    → Triton frontend (Python AST → TTIR)
    → Triton optimizer (TTIR → TTGIR)
    → mlir_walker.py: walk TTGIR module → IRGraph
    → generic_lowerer.py: IRGraph → MSL source
    → xcrun metal: MSL → AIR → metallib
    → driver.py: load metallib, dispatch on GPU
```

See [docs/ARCHITECTURE.md](https://github.com/bledden/triton-msl/blob/main/docs/ARCHITECTURE.md) for details.

## Contributing

See [CONTRIBUTING.md](https://github.com/bledden/triton-msl/blob/main/CONTRIBUTING.md).

## Citing

If you use `triton-msl` in research or technical work, see
[`CITING.md`](https://github.com/bledden/triton-msl/blob/main/CITING.md) for a suggested BibTeX entry. For citations of
the papers and projects this backend builds on (Triton, FlashAttention,
online softmax, MLX, Asahi/`applegpu`, the MSL specification, PyTorch
Inductor), see [`REFERENCES.md`](https://github.com/bledden/triton-msl/blob/main/REFERENCES.md).

## License

[MIT](https://github.com/bledden/triton-msl/blob/main/LICENSE)
