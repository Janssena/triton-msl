# Retained device assertions

Inductor retains indirect-index bounds assertions by default. Debug mode is not
only an application author's choice, and disabling it is not a workaround for a
backend correctness defect.

The candidate supports native i1 predicates in generic kernels with proved
threadgroup-uniform control flow. A failing lane records the source message's
identifier in a fresh per-launch status buffer. All threads in that threadgroup
rendezvous, sample the failure state, rendezvous again, then return before any
subsequent source load, store or atomic. The second rendezvous prevents a later
check from racing another thread's observation of the previous verdict.

Both the compile-shader and host-roundtrip launchers inspect the status before
normal return; the host-roundtrip path checks before copying results back.
`MetalDeviceAssertionError` includes the failed source message and cannot trigger
fallback or replay. The status buffer is never reused between launches.
If several checks fail across threadgroups, the reported message is the
highest-numbered failing check in source traversal order, not necessarily the
first failure in time.

This is not transactional rollback: work performed before the failing assertion,
or by other threadgroups, may already have modified caller-owned storage. After
an exception those outputs are invalid. No guarded access after the failing
assertion runs in the stopped threadgroup. A source-masked condition such as
`predicate | ~mask` is evaluated as written, without inferring masks from messages.

The supported path uses one logical element per Metal thread where the tile fits,
or an exact-cover uniform wrapping loop. Multipass reductions, register-array
assertion predicates, unproved control flow, mixed 1-D tile ownership and device
callees remain explicit refusals. Specialized templates cannot discard a check;
assert-carrying kernels use the generic path. The experimental C++ compiler uses
the checked MSL path for these kernels, and the MLX adapter refuses the descriptor
because it does not implement the status-buffer/result-return contract.

Native scalar-true checks may be discharged when the source graph proves the
predicate true without loading device data. This does not discharge indirect-index
checks or turn unsupported assertion forms into silently ignored operations.

The descriptor is bound into version 2 of the packed launch contract. Old or
altered records must recompile; neither messages nor the hidden buffer binding
may be changed on a cached launcher. Kernels without retained assertions allocate
no status buffer and perform no new status readback. Assert-carrying kernels do
synchronize for the host check. A 2026-09-11 local paired comparison against candidate 439
measured +15.1% latency for a standalone asserted kernel and +11.9% for GPT-2 small.
The latter executes checks on 2 of 43 backend launches. These measured increases remain
open performance alerts; they are not universal bounds or qualified published claims.
The baseline omitted retained checks, and those checks must not be elided to improve timing.
See [RC limitations](RELEASE_CANDIDATE_LIMITATIONS.md) for the separate current
main-relative matrix and its 13 open alerts; the older 439 comparison is not that matrix.

If the host transfer/synchronization itself reports a separate native runtime error,
it can raise before the launch-local flag is inspected. The attempted-submission
boundary still prevents replay, but simultaneous native and assertion failures are
not guaranteed to retain assertion-message attribution. Do not infer the source of
that native error from its text or describe this diagnostics limit as repaired.
