# 0.3.0rc4 — validation scope

This candidate is **unpublished and under verification**. It supersedes the unpublished rc1–rc3
snapshots; their wheel hashes and installed-package approvals do not identify this candidate.

## Evidence and remaining acceptance work

The frozen rc4 code (493) passed 4,609 project tests with 12 skips and four performance tests
scheduled separately. Its upstream result is 5,780 passed / 3,562 skipped / zero failures;
all 9,342 node statuses match the baseline exactly. The unchanged four performance floors
passed, but remain unqualified under the controlled-measurement policy.

The new code deltas have independent reviews. The installed wheel passed 84 acceptance tests,
27 after relocation, and both portable validation scripts on the local M4 Max. Independent
assertion witnesses also passed. These results identify the frozen 493 code and its tested
archive, not every rc4-named build. This documentation-only update must be rebuilt and the
resulting archive rebound/verified before publication; final artifact acceptance and the
maintainer's outward authorization remain required. No independent M1 result is claimed.

## Reporter workload coverage

- Issue #11: fp32 32×32 matmul tiles with a K loop, runtime alpha/beta and column bias replay
  the source epilogue. Tests include partial output grids and nontrivial scales. Arbitrary
  tile sizes, dtypes and epilogues are not certified; the literal source uses unmasked input
  loads, so callers must allocate the complete loaded tiles.
- PR #7: independent-length, widened-P/V decode computes with 8-/16-row query tiles, including
  folded Q=1 and fp16 Q scale-rounding. Query and key lengths stay distinct. Causal alignment
  follows the source's top-left comparison, not an inferred bottom-right convention.
- Bias-without-LSE replay covers 32×32 tiles, head dimensions 32/64, fp16/fp32, independent
  lengths, non-unit bias strides and causal masks. Tests reconstruct this feature from the PR
  description, not the reporter's unavailable full biased workload. Smaller biased tiles and
  arbitrary narrowed probability recurrences are not covered by this admission rule.
- A runtime **attention score** scale after QK is not admitted merely because the separate
  matmul epilogue now computes. Moving a scale to Q changes rounding and is not universally
  equivalent.

Other limits include unprovable pointer roles, some multi-reduction layouts, narrow backward
delta recurrences and shapes over the threadgroup budget. See the actual diagnostic and
[capability register](CAPABILITY_AND_REFUSAL_CENSUS.md). Correct-or-refuse is the design
objective, not a proven universal absence of defects.

## Assertions and dispatch

Supported retained device assertions stop the workgroup before guarded accesses, record a
launch-local message and raise on the host before results return. Unsupported forms still
refuse. Assertion-bearing launches currently synchronize/read back a flag. Paired A/B against
earlier candidate 439 measured **+15.1% latency for a retained-assertion kernel** and **+11.9%
for GPT-2 small**, with the other nine tested workloads within ±1.2%. These are open alerts,
not merely unmeasured costs. Both runs used the same sources and valid inputs on one M4 Max,
10 blocks of 20 individually alternating pairs, without filtering samples. They do not
establish universal bounds or qualify the published PyTorch/MLX ratios.

The GPT-2 workload checks **2 of its 43** backend launches; one also changes threadgroup
width to preserve assertion execution. Readback waits include preceding GPU work, so their
duration is not an isolated sync penalty and the two costs have not been cleanly separated.
The earlier candidate silently omitted retained checks. Those checks will not be removed or
weakened to recover its timing.

A pre-invocation eligibility miss may take a slower supported route. Once invocation has been
attempted, an error raises without fallback replay, even when enqueue cannot be proven.

## What remains unqualified

- **Performance:** README ratios and absolute rates are historical until individually
  remeasured and tied to this artifact. Correctness does not qualify them. The four absolute
  floors (7.0 / 5.5 / 5.5 TFLOP/s and 250 GB/s) are M4 Max sentinels, not M1 budgets. An earlier
  in-suite failure remains unattributed; no threshold was lowered.
- **C++:** optional, unaudited and off by default. Enabling TRITON_MSL_USE_CPP=1 enters an
  unqualified surface; TRITON_MSL_FORCE_PYTHON=1 suppresses it. Default-off is not validation.
- **AlphaFold:** reporter-relevant classes were tested, not the full original workload.
- **Hardware:** local validation uses one M4 Max. Independent M1/M2/M3 acceptance is unrun.
  The M1 kit must name the final version and hashes before it is sent.
- **Exceptional values:** the [axis-to-test map](CAPABILITY_AND_REFUSAL_CENSUS.md) separates
  exact classification, finite controls and order-robust bounds. Welford m2 checks are
  admissibility checks, not an exact non-finite recurrence oracle or exhaustive coverage.

## Shutdown diagnostics

A historical installed-wheel run printed an upstream Triton finalizer AttributeError involving
kernel_unload_hook during teardown, after correct output with exit status zero. Later simpler
RC runs did not reproduce it. It remains disclosed, not suppressed or represented as fixed.
A similar-looking error during execution must not be dismissed as this shutdown issue.

## Reporting

Report minimal source, inputs, version, device and full diagnostics at
https://github.com/bledden/triton-msl/issues. Unexpected refusals and silent wrong values both
matter. Publication and reporter replies require the maintainer's authorization.
