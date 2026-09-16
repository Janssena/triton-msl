"""Output-address proof: an address expression may itself read fresh MLX output.

These are extraction-boundary pins; no unsafe shader is launched. Positive cases
retain the emitter's vectorized store aliases and nested numeric indices.
"""

import pytest

pytest.importorskip("mlx.core")

from triton_msl.errors import MetalNonRecoverableError
from triton_msl.mlx.msl_extractor import extract_msl_for_mlx

_HEADER = """#include <metal_stdlib>
using namespace metal;
kernel void k(device int* x [[buffer(0)]], device int* o [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {
"""


@pytest.mark.parametrize(
    "body",
    [
        "device int* p = o + o[0]; p[0] = x[0];",
        "device int* p = (device int*)(o + o[0]); p[0] = x[0];",
        "device int* p = o; device int* q = p + p[0]; q[0] = x[0];",
        "device int* p = o; device int* q = p + o[0]; q[0] = x[0];",
        "o[(int)(*o)] = x[0];",
        "device int* p = o; p[(int)(*p)] = x[0];",
        "device int* p = (device int*)o[0]; p[0] = x[0];",
        "o[o[0]] = x[0];",
        "device int* p = o; p[0] += x[0];",
        "device int* p = o; p[0] = p[1] + x[0];",
    ],
)
def test_output_read_in_address_or_alias_refuses(body):
    with pytest.raises(MetalNonRecoverableError, match="output pointer 'o'"):
        extract_msl_for_mlx(_HEADER + body + "\n}", [1], expected_args=2)


@pytest.mark.parametrize(
    "body",
    [
        "device int* p = o + tid; p[0] = x[tid];",
        "device int* p = (device int*)(o + tid); device int* q = p + 1; q[0] = x[tid];",
        "device int4* p = (device int4*)(o + tid * 4); p[0] = int4(x[tid]);",
        "o[x[tid]] = 1;",
        "device int* p = o + x[tid]; p[0] = 1;",
        "/* o[0] is only a comment */ o[tid] = x[tid]; // read o[0] in comment\n",
    ],
)
def test_address_only_uses_and_input_reads_still_extract(body):
    ext = extract_msl_for_mlx(_HEADER + body + "\n}", [1], expected_args=2)
    assert ext.output_names == ["o"]
    assert ext.input_names == ["x"]
