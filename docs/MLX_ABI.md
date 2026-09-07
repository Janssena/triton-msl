# MLX dispatch ABI

`triton_msl.mlx.triton_call` compiles this backend's MSL and launches it through
MLX. Runtime pointers and scalars may be interleaved in the Triton source
signature. The extractor retains each MSL buffer's source position, validates
its name, pointer/scalar kind and storage dtype, and supplies an explicit
pointer-then-scalar binding map to the launcher. Duplicate, missing or
out-of-range buffer positions refuse before dispatch. An argument count alone
does not establish ABI equivalence.

An output index refers to a **source runtime argument**, not to the ordinal
among pointers. It must identify a pointer. Output placeholders supply shapes
and dtypes; MLX allocates fresh outputs and does not copy their initial values.
Kernels that read or atomically update an output therefore refuse. The read
check follows supported pointer aliases and examines the expressions used to
form addresses, including alias offsets and store indices. Only a proved
address-base use is exempted; the rest of the expression remains subject to the
read check. Unknown pointer syntax refuses conservatively.

The route does not implement the torch driver's specialized dispatch
descriptors or packed/scalar-buffer template ABIs. Those kernels refuse rather
than being bound by argument count. In particular, a scalar represented by a
template as a `device int*` is not interchangeable with a source `i32` scalar.
An explicit adapter with its own source-role proof is needed to support it.

MLX supplies row-contiguous inputs here. Caller-provided stride arguments must
describe that layout, not an earlier noncontiguous view. This is a caller
contract, not a runtime stride check. Passing a placeholder does not imply
in-place updates or preservation of its unwritten elements.

These checks apply when a kernel is compiled/extracted. The broader cross-tree
and warm-cache invalidation work is separate and remains required before a
release: a new extractor cannot repair an already cached extraction or shader.
