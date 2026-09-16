"""A dense predicate is not Python truthiness, even outside reduction stores."""

import pytest

from tests.test_reduce_return_generic import _lower
from triton_msl.errors import MetalNonRecoverableError


def source(dtype, value, mask):
    return f"""#blocked = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32}} {{
 tt.func public @literal_store(%out: !tt.ptr<{dtype}>) {{
  %r = tt.make_range {{start = 0 : i32, end = 64 : i32}} : tensor<64xi32, #blocked>
  %ptr = tt.splat %out : !tt.ptr<{dtype}> -> tensor<64x!tt.ptr<{dtype}>, #blocked>
  %addr = tt.addptr %ptr, %r : tensor<64x!tt.ptr<{dtype}>, #blocked>, tensor<64xi32, #blocked>
  %value = arith.constant dense<{value}> : tensor<64x{dtype}, #blocked>
  %mask = arith.constant dense<{mask}> : tensor<64xi1, #blocked>
  tt.store %addr, %value, %mask : tensor<64x!tt.ptr<{dtype}>, #blocked>
  tt.return
 }}
}}"""


def test_direct_mixed_dense_mask_refuses_before_uniformization(tmp_path):
    mask = "[" + ", ".join(["true", "false"] * 32) + "]"
    with pytest.raises(MetalNonRecoverableError, match="nonuniform or unparsed tensor constant"):
        _lower(source("i32", "7", mask), tmp_path)


@pytest.mark.parametrize(
    "dtype,value,expected",
    [
        ("i32", "7", "7"),
        ("i64", "4294967296", "4294967296"),
        ("f32", "0xFF800000", "(-INFINITY)"),
        ("f32", "-0.000000e+00", "-0.0f"),
    ],
)
def test_known_splat_literals_still_lower(tmp_path, dtype, value, expected):
    msl = _lower(source(dtype, value, "true"), tmp_path)
    assert expected in msl
    assert "if (1)" in msl


def test_false_splat_mask_stays_false(tmp_path):
    msl = _lower(source("i32", "7", "false"), tmp_path)
    assert "if (0)" in msl
