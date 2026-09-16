# Contributing to triton-msl

Bug reports, kernels, benchmarks, documentation and agent-assisted contributions
are welcome. You do not need a polished reproducer or a particular template to
start a conversation. Tell me what you tried, what you expected and what happened.

## Reporting a problem

Share what you already have. These are useful clues, not an admission checklist:

- A kernel or small runnable example, including the launch call. If the original
  is private, synthetic inputs with the same shapes and operations are great.
- Expected versus actual output, the error, or the performance change. Say what
  reference you used; numerical tolerances and precision modes help.
- Your Mac chip, macOS/Python versions, triton-msl version or fork commit, and
  relevant Torch/Triton/MLX versions. Is this Python JIT, `torch.compile`, MLX, or
  a separately loaded Metal library?
- For kernel problems, dtypes, shapes, strides, offsets/aliasing, grid, tile sizes
  and launch options when known. Keep masks and scalar arguments in the example:
  changing just those can change the compiler route.
- The smallest command that reproduces it and whether it is intermittent. A short
  error excerpt is usually more useful than an entire build log.

Plain prose, a code block, a linked public example or an existing benchmark are
all fine. Do not spend hours minimizing something before reporting it. Check the
[existing issues](https://github.com/bledden/triton-msl/issues) first; if the priority tracker already has your topic, a comment
there can link your reproduction or proposed PR. A separate issue is still welcome
when a new bug or substantial investigation needs its own discussion.

### Performance reports

If possible, compare both versions on the same machine, inputs and measurement
method. Separate installation/import, first compilation and warm calls. State
where you synchronize and what the timer includes; GPU time and time until a
Python call returns answer different questions. Repeated samples or a range help,
as does checking the output first. A wrong or refusing baseline is a capability
comparison, not a speedup.

One representative slow workflow is useful. You do not need the entire benchmark
suite or rented GPUs to open an issue.

### Safe evidence

Never include credentials, tokens, SSH keys, full environment dumps, private
inputs or model weights. Redact local paths and identifying data where appropriate.
Synthetic inputs are usually sufficient, including for protein-model workloads.
Share relevant configuration selectively.

If a bug may access memory out of bounds, do not rerun a known-unsafe shader to
prove it. Share the source, emitted code or error first so a bounded witness with
valid backing storage can be agreed on. Large allocations, long stress runs and
paid external machines are not prerequisites for a report.

## Pull requests and agent-assisted work

Agents are welcome. Please read the resulting diff and give a short account of:

1. The observed problem and baseline version/commit.
2. What changes, why it fixes that problem and which neighboring cases should stay
   unchanged.
3. What actually ran: commands or test names, hardware, results, skips and anything
   not tested.
4. Known limitations or behavior changes, including effects on other callers of
   shared helpers.

A regression test that fails for the old reason, a positive control that stays
working and a nearby negative case are especially helpful. CPU emission checks
are valuable; label them as such rather than claiming GPU execution. If the
hardware is unavailable, say so. Maintainers coordinate the broader correctness
and performance gates before merging; contributors are not expected to qualify
every supported device or run the full release campaign before opening a PR.

Do not make tests pass by weakening masks, assertions, tolerances, refusal
conditions or performance thresholds without clearly proposing and justifying
that behavior change. When optimizing, preserve the source computation and
describe the fallback as well as the fast path. Credit sources and preserve
applicable licences.

There is no requirement to disclose an agent's private conversation, prompts or
reasoning. A concise handoff separating measured facts, hypotheses and untested
work is enough; model/tool names are optional.

A diagnosis or reproducer may inform a different implementation when shared
compiler paths need a broader repair. I will distinguish that from adopting your
code, explain the disposition and credit the contribution. A fixed example does
not mean every shape or an entire application has been validated; reporter
follow-up is valuable.

## Development setup

Metal execution needs an Apple Silicon Mac and the matching compiler/runtime
stack. Follow the current [installation instructions](docs/USAGE.md#installation-details)
and [README](README.md#install) for macOS/Python requirements, PyTorch and the
qualified Triton source revision. The native helpers have narrower interpreter
and OS support than the Python fallback; do not infer runtime coverage from a
successful helper build alone.

For source development, use an isolated environment with those dependencies:

```bash
git clone https://github.com/bledden/triton-msl.git
cd triton-msl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
# Match the formatter used by hosted CI, after installing the dev dependencies.
python -m pip install "ruff==0.14.14"
```

Install PyTorch and Triton in that environment using the linked installation
instructions; the dev extra alone does not supply them. Install MLX only when
needed for the path being tested.

## Running tests

Start with the relevant tests and report their actual results. A focused pass is
useful development evidence, not a replacement for the maintainer's release gates.

```bash
python -m pytest tests/test_generic_lowerer.py -v
python -m pytest tests/test_gpu_correctness.py -v
python -m pytest tests/test_torch_compile.py -v
python -m pytest tests/test_mlx_backend.py -v
```

Coordinate GPU use before running the broader suites. For maintainers preparing
an integrated change or release, correctness and performance remain separate
required results:

```bash
# Both required local jobs, serialized in separate processes; use a NEW evidence path.
python scripts/run_project_tests.py --output /absolute/new/validation-evidence

# Correctness-only development pass: does NOT satisfy the performance obligation.
pytest tests/ --project-lane=correctness -v

# Dedicated performance-floor job (not amid randomized correctness).
pytest tests/ --project-lane=performance -p no:randomly -v

# Upstream conformance: regenerate the report rather than hand-editing counts.
# The runner's CPU reference device does not mean the Metal kernels run on CPU.
python scripts/run_upstream_tests.py --test-file test_core.py --timeout 1800
```

**Hosted PR CI checks lint and formatting, not Metal correctness or performance.**
GPU validation is performed separately on Apple Silicon with the required Triton
installation. Retain correctness and performance receipts separately. Bare
`pytest tests/` selects correctness only.

A passing floor job is not controlled performance qualification. The inherited
absolute floors target the campaign's M4 Max, not every M-series Mac. An M1
contributor is not expected to meet an M4 Max throughput floor; do not weaken it
or count a skip as a pass. Maintainers retain the obligation to run the qualified
gates and document device coverage. See [validation lanes](docs/VALIDATION_LANES.md)
for the full requirements.

## Running benchmarks

```bash
python benchmarks/bench_all.py             # All native kernel benchmarks
python benchmarks/bench_copy_overhead.py   # Buffer copy analysis
python benchmarks/mlx_vs_pyobjc.py         # MLX vs PyObjC comparison
```

## Code style

[Ruff](https://docs.astral.sh/ruff/) handles linting and formatting, and **both are
CI gates**. Use the version pinned in `.github/workflows/ci.yml`. Format only files
you changed while developing; avoid unrelated formatting churn. Check the full
CI selection before submitting:

```bash
ruff check triton_msl/ tests/
ruff format --check triton_msl/ tests/
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the compilation pipeline and design decisions.

Key directories:

- `triton_msl/codegen/`: TTGIR to MSL compilation (`generic_lowerer.py` is the primary path)
- `triton_msl/backend/`: Triton backend integration (`compiler.py`, `driver.py`)
- `triton_msl/inductor/`: `torch.compile` integration
- `triton_msl/mlx/`: MLX integration

Fork the repository and branch from `main`. Keep the change focused, include the
available evidence and open a PR. An early draft is welcome if you need help
with the design or access to validation hardware.
