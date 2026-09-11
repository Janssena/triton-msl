# Correctness and performance are separate required results

`pytest tests` now selects the correctness lane. Four existing hardware floors
are marked `performance_sentinel` and **deselected**, not skipped or deleted:

| Workload | Unchanged floor |
|---|---:|
| 2048-cubed fp32 matmul | 7.0 TFLOP/s |
| 2048-cubed fp16-input/fp32-output matmul | 5.5 TFLOP/s |
| 2048-cubed fp16-input/fp16-output matmul | 5.5 TFLOP/s |
| 16M fp32 vector add | 250 GB/s |

These inherited absolute budgets target the campaign's M4 Max, not every Apple
Silicon device. They are not portable hardware qualification. No threshold,
timing statistic, workload or assertion in these four tests was changed by the
lane split. In particular, their inherited minimum-of-three timing is retained
for continuity of the floor diagnostic, **not** adopted as a claims estimator.
Compiler timeout contracts, timer unit tests, and real-model correctness remain
in correctness. The model tests' incidental timing does not qualify claims.

## Local jobs

Use the validation environment's Python:

```sh
python scripts/run_project_tests.py --output /absolute/new/evidence-directory
```

The default schedules both required jobs, serially in separate fresh processes:
randomized correctness (recorded seed) and fixed-order performance. Both run even
when one fails; a failure in either propagates to the command's exit status.
Skipped, missing, duplicate or unexecuted floor nodes cannot count as a pass.
Each worker asserts package identity in its collecting interpreter, verifies its
source freeze, isolates five cache directories, retains stdout/stderr and node
statuses, and saves/restores the generated performance report.
JUnit logging explicitly retains passing-test captured stdout/stderr as well as
failures; the outer raw file alone would omit successful pytest capture. Evidence directory
creation is exclusive. Timeout terminates only the worker's own process group;
partial raw output survives. There is no automatic retry.

`--mode correctness` and `--mode performance` run individual jobs; a correctness
pass leaves the performance obligation OPEN. `--collect-only` audits the partition
without executing tests and yields only `COLLECTION_ONLY`. Direct pytest usage:

```sh
pytest tests --project-lane=correctness --randomly-seed=423001
pytest tests --project-lane=performance -p no:randomly
```

A cooperative cross-worktree lock prevents overlap between invocations of this
runner. It does not control other GPU applications, raw pytest, or the compositor.
Coordinate the local GPU before execution. No hosted/self-hosted runner or remote
execution has been installed by this change.

## Qualification remains a distinct, open gate

The scheduler reports a passing floor job as `UNQUALIFIED_FLOOR_PASS`, never a
performance or release approval. It does not yet enforce the separately frozen
control protocol: calibration, promptly bracketing compute/bandwidth controls,
blind round inclusion, ten surviving paired rounds, source-faithful numerical and
executed-route controls, or hardware-budget qualification. Those remain required
before a result backs the claims matrix. An exit code is not proof of qualified
execution; read the classified receipt. No caller-supplied boolean can turn it green.

This is scheduling infrastructure, **not completion of the sentinel/claims work**.
The prior in-suite matmul failure remains a failed historical run even if either
new job passes. A floor failure stays a failure of the diagnostic; missing control
qualification cannot turn it into a pass or establish its cause. No claim is
cleared by moving tests.

## Bounded suite-context diagnostic

```sh
python scripts/run_project_tests.py --mode context --seed 423001 --output /absolute/new/context-evidence
```

This invokes the same four floors, the full randomized correctness lane, and the
four floors again in **one interpreter**, retaining all three outcomes. It is one
predeclared experiment, not a loop until a floor passes. Python/runtime caches and
module state persist; pytest starts a new session and private Inductor cache for
each phase, so this is not an exact replay of the original single-session failure.
Collection-only dry runs must precede execution. A pre/post difference associates
the change with elapsed suite context; it cannot distinguish thermal/power state,
memory/cache effects, fixture state, or time-varying external contention by itself.
The full-suite cost is real, not a cheap isolated throughput rerun. The diagnostic
earns no controlled performance-claim credit.
