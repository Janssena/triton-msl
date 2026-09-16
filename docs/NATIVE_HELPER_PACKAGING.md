# Optional host-helper packaging

The validation and packed-descriptor accelerators use the regular CPython ABI, not abi3. Automatic native builds require GIL-enabled CPython 3.13 or3.14 on macOS arm64. Other interpreters/platforms automatically retain the Python implementations. A requested native build on an unsupported interpreter/platform raises an error; compiler and installed-module import failures remain errors, with no silent Python substitution.

`TRITON_MSL_BUILD_VALIDATION_NATIVE` accepts `auto`, `0`, or `1` (default `auto`). `TRITON_MSL_BUILD_PACKED_NATIVE` accepts the same values and, when unset, inherits the validation setting. This preserves the existing `TRITON_MSL_BUILD_VALIDATION_NATIVE=0` fully Python build. The packed setting overrides inheritance only when explicitly supplied.

| Validation setting | Packed setting | Supported-platform artifact |
|---|---|---|
| unset/auto | unset | Both native helpers |
| 0 | unset | Fully Python |
| 1 | 0 | Native validation, Python packed copy |
| 0 | 1 | Python validation, native packed copy |
| 0 | auto | Python validation, native packed copy |
| 0 | 0 | Fully Python |

Both C sources are retained in source distributions. Scratch `.so`/`.dylib` files are excluded from package-data copying and source archives; only selected setuptools Extension products belong in native wheels. Build with fresh source/build directories when changing interpreter or flags. Wheel tags must match the interpreter's ordinary ABI and platform, or `py3-none-any` for the fully Python artifact.

The cache module selects both optional helpers before the first implementation/framework stamp. Actual loaded image bytes and shipped C source contents participate in identity. An absent extension retains Python; an installed broken extension does not count as absence. Relocated artifacts need fresh-process validation of selected files and both actual image hashes. This packaging contract does not claim GPU correctness, a new supported Python version, free-threaded support, or release qualification.

Native-helper presence is not a guarantee of saved-identity validator hits. The
shortcut requires ordinary, callback-free import metadata. Importing a namespace
package such as `mlx` retains its live namespace path and uses the full evaluator;
custom module/spec observers and nonstandard path containers do likewise. They
are not converted to cached lists, their callbacks must not be replayed by record
construction, and kernel admission is unchanged. Native fast-path latency results
must not be applied to these fallback states without measuring them. The packed
descriptor helper has its own eligibility and is not disabled by this condition.
