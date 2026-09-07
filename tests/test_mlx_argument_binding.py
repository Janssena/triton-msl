"""The MLX ABI must preserve source positions, kinds, names and storage widths."""
from types import SimpleNamespace
import pytest

mx = pytest.importorskip("mlx.core")
import triton
import triton.language as tl
import triton_msl.mlx as tmlx
from triton_msl.errors import MetalNonRecoverableError
from triton_msl.mlx.mlx_launcher import MLXLauncher


@triton.jit
def _output_first(O, n, X, BLOCK: tl.constexpr):
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(O + c, tl.load(X + c, c < n, other=0) + 3, c < n)


@triton.jit
def _output_last(X, n, O, BLOCK: tl.constexpr):
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(O + c, tl.load(X + c, c < n, other=0) + 3, c < n)


@pytest.fixture(autouse=True)
def cold(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    tmlx._compile_cache.clear()


@pytest.mark.parametrize("fn", [_output_first, _output_last])
def test_interleaved_binding_at_launch_boundary(fn, monkeypatch):
    x, out = mx.arange(64, dtype=mx.float32), mx.zeros((64,))
    bound = []
    def build(self):
        def capture(**kw):
            bound.append((self.ext, kw))
            return []
        self._kernel = capture
    monkeypatch.setattr(MLXLauncher, "_build_kernel", build)
    args = (out, 64, x) if fn is _output_first else (x, 64, out)
    assert tmlx.triton_call(fn, *args, grid=(2,), BLOCK=32) == []
    assert len(bound) == 1
    ext, kw = bound[0]
    assert ext.input_names == ["X"] and ext.scalar_names == ["n"] and ext.output_names == ["O"]
    assert kw["inputs"][0] is x and kw["inputs"][1] == 64
    assert kw["output_shapes"] == [(64,)] and kw["output_dtypes"] == [mx.float32]


@pytest.mark.parametrize("fn", [_output_first, _output_last])
def test_interleaved_binding_gpu(fn):
    x, out = mx.arange(64, dtype=mx.float32), mx.zeros((64,))
    args = (out, 64, x) if fn is _output_first else (x, 64, out)
    (result,) = tmlx.triton_call(fn, *args, grid=(2,), BLOCK=32)
    mx.eval(result)
    assert mx.array_equal(result, x + 3).item()


@pytest.mark.parametrize("decls,outputs", [
    ("device float* X [[buffer(0)]], constant int& n [[buffer(1)]], device float* O [[buffer(2)]]", [0]),
    ("device float* O [[buffer(0)]], device int* n [[buffer(1)]], device float* X [[buffer(2)]]", [0]),
    ("device int* O [[buffer(0)]], constant int& n [[buffer(1)]], device float* X [[buffer(2)]]", [0]),
    ("device float* O [[buffer(0)]], constant int& n [[buffer(0)]], device float* X [[buffer(2)]]", [0]),
    ("device float* O [[buffer(0)]], constant int& n [[buffer(1)]], device float* X [[buffer(3)]]", [0]),
    ("device float* O [[buffer(0)]], constant int& n [[buffer(1)]], device float* X [[buffer(2)]]", [1]),
])
def test_same_count_wrong_abi_refuses_before_launcher(decls, outputs, monkeypatch):
    msl = "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(" + decls + ") { O[0] = X[0]; }"
    metadata = SimpleNamespace(block_size=1, output_arg_indices=outputs, needs_2d_grid=False)
    monkeypatch.setattr(tmlx, "_compile_kernel", lambda *a: (msl, metadata, None))
    def no_launch(*a, **kw):
        raise AssertionError("unproved ABI reached launcher")
    monkeypatch.setattr(MLXLauncher, "__call__", no_launch)
    with pytest.raises(MetalNonRecoverableError, match="MLX route"):
        tmlx.triton_call(_output_first, mx.zeros((1,)), 1, mx.ones((1,)), grid=(1,), BLOCK=1)
