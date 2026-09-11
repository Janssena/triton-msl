# 0.3.0rc4 — validation scope

This candidate is **unpublished and under verification**. It supersedes the unpublished rc1–rc3
snapshots; their wheel hashes and installed-package approvals do not identify this candidate.

## Evidence and remaining acceptance work

The frozen rc4 code (493) passed 4,609 project tests with 12 skips and four performance tests
scheduled separately. Its upstream result is 5,780 passed / 3,562 skipped / zero failures;
all 9,342 node statuses match the baseline exactly. The unchanged four performance floors
passed, but remain unqualified under the controlled-measurement policy.

The historical installed wheel passed 84 acceptance tests and 27 after relocation, plus local
portable scripts. These results identify the frozen 493 code and its archive, not the current
post-523/527/529 source. Final integrated correctness gates, public main comparison, cross-vendor
validation and fresh final-wheel/relocation acceptance are owed. The final artifact must be
rebuilt after metadata-affecting documentation changes. No independent M1 result is claimed.

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
  description, not the reporter's unavailable full biased workload. The later tile/precision and wide-row recoveries below have separate admission proofs;
  arbitrary combinations remain unqualified.
- A runtime **attention score** scale after QK is not admitted merely because the separate
  matmul epilogue now computes. Moving a scale to Q changes rounding and is not universally
  equivalent.

- Low-precision and rectangular K-loop epilogues (issue #11 family) replay for f16/bf16/f32
  storage and 16/32/64 tiles with f32 accumulation and f32 epilogue arithmetic. Activations,
  clamps, fused multiply-adds, row bias and non-tile-boundary masks refuse. Tiles needing more
  than 32 KiB of threadgroup memory after scratch reuse (64×64×64) refuse with a typed capacity
  error; 64×64×32, 64×32×64 and 32×64×64 sit exactly at the limit.
- Biased attention replay also covers 8/16-row query tiles, bf16 storage, explicit f16
  probabilities and D64 causal rows. Separate full-row replay computes tested fp16/D64
  32×64 noncausal (NQ=19/NK=97) and 64×32 causal (NQ=65/NK=97) cases. It distinguishes the native
  f32 -Inf sentinel from finite 64512.0 and preserves all program input reads before output
  stores, including the tested shared Bias/Out allocation. Arbitrary dtype/tile/recurrence
  combinations are not certified. Explicit bf16 dot operands, D128 and post-QK score scaling
  remain named limitations; single-key nk=1 specialization is a separately recorded refusal.
  The wide route performs 130 score evaluations per key block. Performance is unqualified.

Other limits include unprovable pointer roles, some multi-reduction layouts, narrow backward
delta recurrences and shapes over the threadgroup budget. See the actual diagnostic and
[capability register](CAPABILITY_AND_REFUSAL_CENSUS.md). Correct-or-refuse is the design
objective, not a proven universal absence of defects.

## Corrected silent-wrong: generic two-dimensional reductions (present since 0.2.0)

Before the allocator repair, threadgroup scratch reuse (`_alias_shared_memory`) considered
nonoverlapping textual first/last uses sufficient. Before this candidate the generic axis-1 reducer wrote its result on the line
after its own row loop, so the result array could be merged onto the input array; reducer `k`
then overwrote input slot `k` (row `k // N`) while a reducer in the first SIMD group could still
be reading that row. Measured on one M4 Max: wrong rows in about 2 of 100 launches of a 128×8
row-softmax tile (8,192 threadgroups per launch), on the 0.2.0 source and on the earlier rc4
candidate alike; 64×16 row-sum tiles showed the same family less often. Kernels whose reducers
fit one SIMD group (≤ 32 rows on axis 1, including the 32-row attention tiles) were not observed
to fail, and axis-0 reductions were structurally unaffected by this specific handoff. Generic
axis-1 reductions with more than 32 rows could be exposed when emitted with this alias pattern. The repair holds the result in registers, barriers,
then writes it under a lane guard on both axes; emission pins assert that ordering for both
axes. The 0.2.0 source contains the defect; users of that code reducing tiles with more than
32 rows through this generic alias pattern should treat those results as suspect until they upgrade.

The allocator now requires proved synchronization between reused lifetimes and checks loop
backedges/control flow (529). This closes the additional 516 handoff findings: reduce→scan
(public since 0.2.0) and join→split (local lineage; introduced by 6ab6db7, already in local main)
were witnessed numeric races. The argmin/argmax scratch handoff and two atomic handoff
families were structural concerns, not observed numeric failures. In the bounded 529 replay,
parent D/E cases were wrong 60/60 launches; repaired cases and the small control were wrong
0/80 launches. The 1,230 unique-emission audit found 0 remaining checker flags and no newly
exceeded 32 KiB capacity. Focused 529 tests passed 456 contracts; this is finite corpus evidence,
not a universal absence-of-races proof or final integrated gate.

The independent public output-offset defect C is repaired by 527: axis-1 argmin/argmax 1-D stores
retain program/runtime offsets and remap source pointer/mask coordinates to the result row.
Final focused 51 contracts and 16 adversarial live-launch GPU rows cover both reducers,
8×128/64×16 tiles, multiple programs, masks/strides/selects, negative offset views and
rank-1/axis-0/2-D-store controls with exact sentinels. A nonuniform tensor constant mask could
previously emit as always true; this store now refuses unparsed/nonuniform tensor constants
and unproved data-dependent address leaves. Scalar and true/false splat forms remain admitted.
An additional direct-store reproduction compiled an alternating dense boolean mask as `if (1)`
through the public native-IR compile entry point. The generic constant emitter now refuses
unparsed/nonuniform literals rather than coercing them to true or zero. It preserves decoded
numeric scalar/splats and typed special-float words; it does not implement general nonuniform
tensor constants or certify every constant consumer. The full integrated admission/status
ratchet is still required. Public 0.2.0 users should treat results from the affected 512/C/D
source patterns as suspect until using a validated corrected artifact; no advisory was sent.

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

- **Scratch aliasing:** the conservative 529 allocator and 527 store fixes have bounded focused
  evidence. Final combined gates and cross-vendor/artifact evidence remain separate obligations.

## Shutdown diagnostics

A historical installed-wheel run printed an upstream Triton finalizer AttributeError involving
kernel_unload_hook during teardown, after correct output with exit status zero. Later simpler
RC runs did not reproduce it. It remains disclosed, not suppressed or represented as fixed.
A similar-looking error during execution must not be dismissed as this shutdown issue.

## Reporting

Report minimal source, inputs, version, device and full diagnostics at
https://github.com/bledden/triton-msl/issues. Unexpected refusals and silent wrong values both
matter. Publication and reporter replies require the maintainer's authorization.
