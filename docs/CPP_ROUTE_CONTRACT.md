# Experimental C++ route

The default compiler is Python/MSL. `TRITON_MSL_USE_CPP=1` requests an **unaudited
development route** through C++ MLIR → LLVM → Metal. `TRITON_MSL_FORCE_PYTHON=1`
overrides it. A one-time warning accompanies the first C++ lowering attempt;
the Python/MSL campaign's correctness proofs do not certify C++-generated binaries.

Successful final compilation records `metadata.binary_route` as `cpp` or `msl`.
`metadata.cpp_fallback_reason` records why a requested C++ route was unavailable,
refused or failed, or is `None` for a successful C++ compilation/default MSL route.
Fallback emits a warning even without `TRITON_MSL_CPP_TRACE=1`; that debug flag
can additionally show the full exception traceback. Failed C++ compilation does
not overwrite the MSL block size before fallback.

These are **binary-production** records, not per-launch execution evidence. The
driver can execute separately emitted MSL through `torch.mps.compile_shader` or
a specialized runtime template, without using the compiled metallib. A benchmark
or numeric C++ claim must prove the executed route independently. This change
does not alter the driver's fast-path choice or claim any C++ speedup/recovery.

## Known-broken dot route

`tt.dot` is refused by the direct C++ lowering entry before native lowering, and
the ordinary compiler explicitly de-routes it to its separately generated MSL
kernel with a recorded reason/warning. This covers customary and quoted generic
MLIR spellings. It does not silently attempt C++ and hide its failure.

The old evidence covered single-tile grids; multi-tile/grid and loop-carried cases
are known broken. Compilation does not receive the eventual runtime grid, so a
small tile or earlier single-tile test does not prove future single-program use.
Until that launch/layout contract is implemented and audited, **no C++ dot
capability is claimed**, including single-tile. The MSL route's existing numeric
controls remain controls for MSL, not a way to label the C++ route green.

## Text-rewrite boundary

The C++ operation census and annotation/block-size rewrites currently support
custom MLIR assembly only. Generic quoted operation syntax is explicitly declined
before native lowering, even for an otherwise simple operation. This is a syntax
coverage restriction, not a claim that such kernels are mathematically unsafe.
Quoted module attribute names and values do not trigger this restriction.

Scalar `tt.load` also refuses at this shared boundary: its known native assertion
must be guarded for direct `make_llir` calls, not only ordinary compiler routing.
Both restrictions use the same disposition mechanism as the dot guard: the ordinary
compiler can use its separately generated MSL, while direct C++ entry refuses.
Recovering generic quoted syntax requires a structural operation/type/extent parser,
not another permissive regular expression. No new C++ capability is claimed here.

## What remains

This is not the full C++ audit. Correctness/coverage of the remaining experimental
families, source/build compatibility of optional native extensions, per-launch
instrumentation, full candidate gates and exact-release validation remain work.
No unmeasured performance assertion or default-on flip is made here.
