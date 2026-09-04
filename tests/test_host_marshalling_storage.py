"""P0 pins for storage-faithful host-roundtrip tensor marshalling.

The host launcher must preserve pointer semantics, not merely copy each tensor
view's logical ``numel`` into an independent buffer.  Triton kernels may form
in-storage addresses before/after a view base, and distinct arguments may alias
one backing storage.  These rows force the host path and assert the source
program's memory semantics with canaries around every nontrivial output view.
"""

import os
from unittest import mock

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError


requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _copy_runtime_strides_2d(X, Y, sx, sy, BLOCK: tl.constexpr):
    # program_id(1) deliberately selects the host path even though the logical
    # data is 1-D.  Two y-grid programs cover 64 elements when BLOCK=32.
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i * sy, tl.load(X + i * sx))


@triton.jit
def _copy_runtime_stride_1d(X, Y, sx, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(Y + i, tl.load(X + i * sx))


@triton.jit
def _copy_2d_view(X, Y, s0, s1, M: tl.constexpr, N: tl.constexpr):
    r = tl.program_id(0) * M + tl.arange(0, M)
    c = tl.program_id(1) * N + tl.arange(0, N)
    value = tl.load(X + r[:, None] * s0 + c[None, :] * s1)
    tl.store(Y + r[:, None] * N + c[None, :], value)


@triton.jit
def _store_then_load_alias(A, B, O, BLOCK: tl.constexpr):
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    before = tl.load(A + i)
    tl.store(A + i, before + 10.0)
    # A and B may be aliases.  The source order requires this load to see the
    # preceding store from the same lane.
    after = tl.load(B + i)
    tl.store(O + i, after)


@triton.jit
def _two_alias_outputs_store_o2_then_o1(O1, O2, BLOCK: tl.constexpr):
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(O2 + i, i + 200.0)
    # When O1 and O2 alias, this second store wins.  Per-argument mirrors whose
    # copy-back follows argument order produce the opposite result.
    tl.store(O1 + i, i + 100.0)


_KERNELS = (
    _copy_runtime_strides_2d,
    _copy_runtime_stride_1d,
    _copy_2d_view,
    _store_then_load_alias,
    _two_alias_outputs_store_o2_then_o1,
)


@pytest.fixture()
def forced_host_fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    os.makedirs(tmp_path / "msl", exist_ok=True)
    for kernel in _KERNELS:
        cache = getattr(kernel, "device_caches", None)
        assert cache is not None
        cache.clear()


def _host_launch_spy(monkeypatch):
    import triton_msl.backend.driver as driver

    launches = []
    utils = driver._get_utils()
    original = utils.launch

    def spy(*args, **kwargs):
        launches.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(utils, "launch", spy)
    return launches


def _launch_copy(x, y, sx, sy):
    _copy_runtime_strides_2d[(1, 2)](x, y, sx, sy, BLOCK=32, num_warps=4)
    torch.mps.synchronize()


@requires_mps
def test_zero_copy_negative_stride_path_is_unchanged(monkeypatch, tmp_path):
    """The 1-D compile-shader path already sees the real backing allocation."""
    monkeypatch.delenv("TRITON_MSL_COMPILE_SHADER", raising=False)
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    _copy_runtime_stride_1d.device_caches.clear()
    host_launches = _host_launch_spy(monkeypatch)
    storage = torch.arange(64, dtype=torch.float32, device="mps")
    out = torch.full((64,), float("nan"), device="mps")
    _copy_runtime_stride_1d[(1,)](storage[63:], out, -1, N=64, num_warps=4)
    torch.mps.synchronize()
    assert host_launches == []
    torch.testing.assert_close(out, torch.flip(storage, (0,)), rtol=0, atol=0)


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_host_path_negative_input_stride_reads_before_view_base(monkeypatch, forced_host_fresh, device):
    launches = _host_launch_spy(monkeypatch)
    storage = torch.arange(64, dtype=torch.float32, device=device)
    x = storage[63:]
    out = torch.full((64,), float("nan"), device=device)
    _launch_copy(x, out, -1, 1)
    assert launches == [True]
    torch.testing.assert_close(out, torch.flip(storage, (0,)), rtol=0, atol=0)


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_host_path_positive_stride_reads_past_view_inside_storage(monkeypatch, forced_host_fresh, device):
    launches = _host_launch_spy(monkeypatch)
    storage = torch.arange(136, dtype=torch.float32, device=device)
    x = storage[8:40]
    out = torch.full((64,), float("nan"), device=device)
    _launch_copy(x, out, 2, 1)
    assert launches == [True]
    torch.testing.assert_close(out, storage[8:136:2], rtol=0, atol=0)


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_host_path_positive_offset_strided_view_stays_correct(monkeypatch, forced_host_fresh, device):
    launches = _host_launch_spy(monkeypatch)
    storage = torch.arange(40, dtype=torch.float32, device=device).reshape(4, 10)
    x = storage[:, 1:9:2]
    out = torch.full((4, 4), float("nan"), device=device)
    _copy_2d_view[(1, 1)](x, out, *x.stride(), M=4, N=4, num_warps=4)
    torch.mps.synchronize()
    assert launches == [True]
    torch.testing.assert_close(out, x, rtol=0, atol=0)


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_host_path_shared_storage_distinct_view_offsets(monkeypatch, forced_host_fresh, device):
    """One shared mirror must bind each aliased argument at its own base."""
    launches = _host_launch_spy(monkeypatch)
    sentinel = -12345.0
    storage = torch.full((130,), sentinel, dtype=torch.float32, device=device)
    storage[1:65] = torch.arange(64, dtype=torch.float32, device=device)
    source = storage[1:65]
    output = storage[65:129]
    expected = source.clone()
    _launch_copy(source, output, 1, 1)
    assert launches == [True]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(source, expected, rtol=0, atol=0)
    assert storage[0].item() == sentinel
    assert storage[129].item() == sentinel


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_host_path_negative_output_stride_writes_before_view_base(monkeypatch, forced_host_fresh, device):
    launches = _host_launch_spy(monkeypatch)
    x = torch.arange(64, dtype=torch.float32, device=device)
    sentinel = -12345.0
    storage = torch.full((66,), sentinel, device=device)
    y = storage[64:65]
    _launch_copy(x, y, 1, -1)
    assert launches == [True]
    torch.testing.assert_close(storage[1:65], torch.flip(x, (0,)), rtol=0, atol=0)
    assert storage[0].item() == sentinel
    assert storage[65].item() == sentinel


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_host_path_preserves_input_output_alias_store_then_load(monkeypatch, forced_host_fresh, device):
    launches = _host_launch_spy(monkeypatch)
    sentinel = -12345.0
    storage = torch.full((66,), sentinel, dtype=torch.float32, device=device)
    storage[1:65] = torch.arange(64, dtype=torch.float32, device=device)
    a = storage[1:65]
    b = storage[1:65]  # distinct view object, same storage and offset
    before = a.clone()
    observed = torch.full_like(a, float("nan"))
    _store_then_load_alias[(1, 2)](a, b, observed, BLOCK=32, num_warps=4)
    torch.mps.synchronize()
    assert launches == [True]
    torch.testing.assert_close(a, before + 10.0, rtol=0, atol=0)
    torch.testing.assert_close(observed, before + 10.0, rtol=0, atol=0)
    assert storage[0].item() == sentinel
    assert storage[65].item() == sentinel


@requires_mps
@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_host_path_multiple_alias_outputs_follow_kernel_store_order(monkeypatch, forced_host_fresh, device):
    launches = _host_launch_spy(monkeypatch)
    sentinel = -12345.0
    storage = torch.full((66,), sentinel, device=device)
    out1 = storage[1:65]
    out2 = storage[1:65]  # distinct view object, same storage and offset
    _two_alias_outputs_store_o2_then_o1[(1, 2)](out1, out2, BLOCK=32, num_warps=4)
    torch.mps.synchronize()
    assert launches == [True]
    expected = torch.arange(64, dtype=torch.float32, device=device) + 100.0
    torch.testing.assert_close(out1, expected, rtol=0, atol=0)
    assert storage[0].item() == sentinel
    assert storage[65].item() == sentinel


@requires_mps
def test_host_path_tensorwrapper_negative_input_and_output(monkeypatch, forced_host_fresh):
    launches = _host_launch_spy(monkeypatch)
    x_storage = torch.arange(64, dtype=torch.int8, device="mps")
    x = triton.reinterpret(x_storage[63:], tl.uint8)
    sentinel = -1
    y_storage = torch.full((66,), sentinel, dtype=torch.int8, device="mps")
    y = triton.reinterpret(y_storage[64:65], tl.uint8)
    _launch_copy(x, y, -1, -1)
    assert launches == [True]
    # Both pointers run backward, so their effects cancel in backing-storage
    # order: storage[1] receives input 0 and storage[64] receives input 63.
    expected = torch.arange(64, dtype=torch.int8, device="mps")
    torch.testing.assert_close(y_storage[1:65], expected, rtol=0, atol=0)
    assert y_storage[0].item() == sentinel
    assert y_storage[65].item() == sentinel


@requires_mps
def test_host_path_float64_partial_storage_refuses(monkeypatch, forced_host_fresh):
    """The width-converting fp64 path cannot faithfully mirror a partial view."""
    launches = _host_launch_spy(monkeypatch)
    storage = torch.arange(64, dtype=torch.float64)
    x = storage[63:]
    out = torch.full((64,), float("nan"), dtype=torch.float64)
    with pytest.raises(MetalNonRecoverableError, match="float64.*complete contiguous storage"):
        _launch_copy(x, out, -1, 1)
    assert launches == []


@requires_mps
def test_host_path_float64_complete_contiguous_still_works(monkeypatch, forced_host_fresh):
    """Storage hardening must preserve the established dense fp64 conversion."""
    launches = _host_launch_spy(monkeypatch)
    x = torch.arange(64, dtype=torch.float64)
    out = torch.full((64,), float("nan"), dtype=torch.float64)
    _launch_copy(x, out, 1, 1)
    assert launches == [True]
    torch.testing.assert_close(out, x, rtol=0, atol=0)


@requires_mps
def test_host_path_float64_alias_refuses(monkeypatch, forced_host_fresh):
    """Separate fp32 conversions cannot preserve fp64 storage identity."""
    launches = _host_launch_spy(monkeypatch)
    storage = torch.arange(64, dtype=torch.float64)
    alias = storage.view_as(storage)
    with pytest.raises(MetalNonRecoverableError, match="aliased float64 arguments"):
        _launch_copy(storage, alias, 1, 1)
    assert launches == []


@requires_mps
def test_host_path_full_storage_safety_cap_refuses(monkeypatch, forced_host_fresh):
    """An oversized backing allocation refuses before copying or dispatch."""
    launches = _host_launch_spy(monkeypatch)
    x = torch.arange(64, dtype=torch.float32, device="mps")
    out = torch.empty_like(x)
    original_storage = torch.Tensor.untyped_storage

    class StorageProxy:
        def __init__(self, storage):
            self._storage = storage

        def nbytes(self):
            return (1 << 30) + 4

        def data_ptr(self):
            return self._storage.data_ptr()

    def oversized_storage(tensor):
        return StorageProxy(original_storage(tensor))

    with mock.patch.object(torch.Tensor, "untyped_storage", oversized_storage):
        with pytest.raises(MetalNonRecoverableError, match="exceeding the 1073741824-byte safety limit"):
            _launch_copy(x, out, 1, 1)
    assert launches == []


@requires_mps
def test_host_path_complete_contiguous_storage_is_exempt_from_cap(monkeypatch, forced_host_fresh):
    """A whole-storage mirror above the cap adds no bytes over the old path."""
    import triton_msl.backend.driver as driver

    monkeypatch.setattr(driver, "_HOST_MIRROR_MAX_BYTES", 128, raising=False)
    launches = _host_launch_spy(monkeypatch)
    x = torch.arange(64, dtype=torch.float32, device="mps")
    out = torch.full_like(x, float("nan"))
    _launch_copy(x, out, 1, 1)
    assert launches == [True]
    torch.testing.assert_close(out, x, rtol=0, atol=0)


@requires_mps
def test_host_path_offset_storage_above_cap_still_refuses(monkeypatch, forced_host_fresh):
    """The exemption must not admit an offset view that expands the mirror."""
    import triton_msl.backend.driver as driver

    monkeypatch.setattr(driver, "_HOST_MIRROR_MAX_BYTES", 128, raising=False)
    launches = _host_launch_spy(monkeypatch)
    storage = torch.arange(65, dtype=torch.float32, device="mps")
    x = storage[1:]
    out = torch.full((64,), float("nan"), device="mps")
    with pytest.raises(MetalNonRecoverableError, match="exceeding the 128-byte safety limit"):
        _launch_copy(x, out, 1, 1)
    assert launches == []
