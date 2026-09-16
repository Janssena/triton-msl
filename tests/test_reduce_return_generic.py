"""Packet 192 (188 §4) — the GENERIC reduce classifier honours the reduction's RETURNED value.

`classify_reduce_combine` classified a reduce by its region's LAST op; a legal TTGIR region
`%s = arith.addf %a, %b ; tt.reduce.return %a` (a projection — the reduce yields its first element,
the add is dead) was therefore lowered as a SUM. The shape is reachable through direct TTGIR
(IRSource), not the Python JIT (DCE removes the dead add); packet 153 found it for the FA proofs,
154 closed it there and 193 for normalization; the generic path was still open. Now the returned
id must be the classified op's, else the classifier returns None and the lowering refuses.
"""

import tempfile
from pathlib import Path

import pytest

import triton_msl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import IRSource
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.generic_lowerer import GenericLowerer
from triton_msl.codegen.mlir_walker import walk_ttgir
from triton_msl.errors import MetalNonRecoverableError

_HEADER = """#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @entry(%p: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %addr = tt.addptr %ps, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %addr : tensor<128x!tt.ptr<f32>, #blocked>
    %v = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %RET : f32
    }) : (tensor<128xf32, #blocked>) -> f32
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}
"""


def _lower(text, tmp_path):
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    path = tmp_path / "m.ttgir"
    path.write_text(text)
    ctx = ir.context()
    src = IRSource(str(path), ctx, backend)
    assert src.module.verify()
    g = walk_ttgir(src.module, backend.parse_options({}))
    return GenericLowerer(g, backend.parse_options({})).lower()


def test_plain_sum_still_lowers_as_a_sum(tmp_path):
    msl = _lower(_HEADER.replace("%RET", "%s"), tmp_path)
    assert "simd_sum" in msl


@pytest.mark.parametrize("ret", ["%a", "%b"])
def test_projection_return_refuses_on_the_generic_path(tmp_path, ret):
    """Pre-192: lowered as a sum (`simd_sum` emitted) although the IR returns a block argument."""
    with pytest.raises(MetalNonRecoverableError):
        _lower(_HEADER.replace("%RET", ret), tmp_path)


def test_missing_return_metadata_refuses(tmp_path):
    """A walker that could not record the returned id must not imply a sum (154's rule, now on the generic path)."""
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    path = tmp_path / "m.ttgir"
    path.write_text(_HEADER.replace("%RET", "%s"))
    ctx = ir.context()
    src = IRSource(str(path), ctx, backend)
    g = walk_ttgir(src.module, backend.parse_options({}))
    red = next(o for o in g.ops if o.op == "tt.reduce")
    red.attrs.pop("return_ids", None)
    with pytest.raises(MetalNonRecoverableError):
        GenericLowerer(g, backend.parse_options({})).lower()
