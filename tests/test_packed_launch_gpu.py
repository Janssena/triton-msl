"""GPU controls for packed records: refuse before writes; snapshot before hooks."""

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal GPU")


@triton.jit
def _packed_add(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) + 1)


@pytest.fixture
def compiled(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    _packed_add.device_caches.clear()
    x = torch.arange(32, dtype=torch.float32, device="mps")
    y = torch.full_like(x, -99)
    kernel = _packed_add[(1,)](x, y, 32)
    torch.mps.synchronize()
    torch.testing.assert_close(y, x + 1, atol=0, rtol=0)
    y.fill_(-99)
    torch.mps.synchronize()
    return kernel, x, y


@pytest.mark.parametrize("field,value", [(1, True), (3, 256), (4, []), (9, ["mla"])])
def test_changed_live_packed_record_refuses_without_hook_or_write(compiled, field, value):
    kernel, x, y = compiled
    packed = list(kernel.packed_metadata)
    packed[field] = value
    events = []
    with pytest.raises(MetalNonRecoverableError, match="packed launch contract"):
        kernel.run(1, 1, 1, None, kernel.function, packed, None, lambda _: events.append("hook"), None, x, y, 32)
    torch.mps.synchronize()
    assert events == []
    torch.testing.assert_close(y, torch.full_like(y, -99), atol=0, rtol=0)


def test_hook_cannot_change_the_already_checked_dispatch(compiled):
    kernel, x, y = compiled
    packed = list(kernel.packed_metadata)
    events = []

    def hook(_):
        events.append("hook")
        packed[9] = ["mla"]  # malformed route would refuse if read after the hook

    kernel.run(1, 1, 1, None, kernel.function, packed, None, hook, None, x, y, 32)
    torch.mps.synchronize()
    assert events == ["hook"] and packed[9] == ["mla"]
    torch.testing.assert_close(y, x + 1, atol=0, rtol=0)
