# Atomic ordering contract

The generic lowerer preserves the source atomic's explicit `sem` and `scope`.
Omitting `sem` in Triton means `acq_rel`, not `relaxed`. Metadata is decoded
from the owning native IR operation; absent or unknown semantics, scope, or RMW
opcode refuse compilation rather than selecting a default.

## Mapping

The underlying Metal atomic remains relaxed. Ordered operations use a fence
before and/or after the **whole logical atomic**, including any compare-exchange
retry loop:

| Source semantic | Before | After |
|---|---|---|
| `relaxed` | none | none |
| `release` | fence | none |
| `acquire` | none | fence |
| `acq_rel` | fence | fence |

Each fence uses `memory_order_seq_cst`, with both `mem_device` and
`mem_threadgroup`. Both memory domains are included because later lowering stages
can introduce threadgroup scratch storage. Source `gpu` scope maps to
`thread_scope_device`; `cta` maps to `thread_scope_threadgroup`. `sys` refuses,
including for relaxed operations: device scope is not a substitute for system
scope.

Ordered atomics require a detected supported M-series device and a selected
Metal language version of at least 3.2, no newer than the device's reported
language capability. An explicit target does not bypass this check. Unknown
hardware or malformed/unsupported targets refuse. Relaxed atomics do not require
fences or this additional language check.

## Execution boundaries

Fences execute under the same scalar-owner, mask, or underfill guard as their
atomic. They are outside retry loops, not repeated per failed exchange. Existing
result-broadcast barriers remain unchanged. This is an atomic ordering mapping,
not a new cross-program barrier, a guarantee of program scheduling, or a liveness
guarantee for spin waits. Existing unsupported dtype and multi-element scatter
contracts remain in force.

The compiler's internal histogram atomics and the atomics used by the
permute-chained-reduce template are not source `tt.atomic_*` operations; this
mapping applies only to source atomic lowering. Those implementation-detail
atomics remain relaxed and are ordered within their threadgroup by the
templates' explicit barriers before and after accumulation. The direct-IR
16-bit exchange tests exercise the existing word-CAS emitter, not a claim that
Triton's Python frontend accepts 16-bit exchange.

## Evidence limits

Lowering-boundary tests check metadata, target refusals, fence scopes, and
placement around native and emulated atomic families. The relaxed shader corpus
must remain byte-identical; ordered rows may differ only by fence lines.
Bounded GPU message-passing tests check observed values without assuming every
reader is scheduled after a writer. Passing a finite litmus test is supporting
evidence, not proof of a memory model or of all scheduling interleavings.
