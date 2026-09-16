"""Real resident-JIT policy transitions; no cross-version upgrade claim."""

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.backend import compiler, driver
from triton_msl.errors import MetalNonRecoverableError

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal GPU")


@triton.jit
def _resident_add(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(Y + offsets, tl.load(X + offsets) + 1)


@triton.jit
def _warmup_add(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(Y + offsets, tl.load(X + offsets) + 2)


@pytest.fixture
def cold(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "1")
    # Isolation ONCE at test entry. Never clear between the paired transitions.
    _resident_add.device_caches.clear()
    _warmup_add.device_caches.clear()
    x = torch.arange(32, device="mps", dtype=torch.float32)
    y = torch.full_like(x, -99)
    return x, y


def test_live_policy_change_recompiles_only_own_jit_and_preserves_user_hook(cold, monkeypatch):
    x, y = cold
    calls, events = [], []
    real = compiler.MetalBackend.make_msl

    def emit(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(compiler.MetalBackend, "make_msl", staticmethod(emit))
    user_hook = lambda *a, **kw: events.append("user")
    _resident_add.add_pre_run_hook(user_hook)
    try:
        first = _resident_add[(1,)](x, y, BLOCK=32)
        torch.mps.synchronize()
        torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
        warm = _resident_add[(1,)](x, y, BLOCK=32)
        assert warm is first and len(calls) == 1
        monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "0")
        changed = _resident_add[(1,)](x, y, BLOCK=32)
        torch.mps.synchronize()
        torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
        assert changed is not first and len(calls) == 2
        assert changed.metadata.execution_contract != first.metadata.execution_contract
        assert events == ["user", "user", "user"]
    finally:
        _resident_add.pre_run_hooks.remove(user_hook)


def test_direct_old_resident_handle_refuses_before_launch_hook_or_write(cold, monkeypatch):
    x, y = cold
    first = _resident_add[(1,)](x, y, BLOCK=32)
    torch.mps.synchronize()
    y.fill_(-99)
    torch.mps.synchronize()
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "0")
    events = []

    def forbidden_runtime():
        pytest.fail("stale direct handle reached runtime access")

    monkeypatch.setattr(driver, "_get_utils", forbidden_runtime)
    with pytest.raises(MetalNonRecoverableError, match="execution contract"):
        first.run(
            1,
            1,
            1,
            None,
            first.function,
            first.packed_metadata,
            None,
            lambda _: events.append("launch-enter"),
            None,
            x,
            y,
            32,
        )
    torch.mps.synchronize()
    assert events == []
    torch.testing.assert_close(y, torch.full_like(y, -99), rtol=0, atol=0)


def test_warmup_only_handle_refuses_a_policy_change_before_first_pipeline(cold, monkeypatch):
    x, y = cold
    warm = _warmup_add.warmup(x, y, BLOCK=32, grid=(1,))
    assert warm.module is None and warm.function is None
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "0")
    with pytest.raises(MetalNonRecoverableError, match="execution contract"):
        _warmup_add[(1,)](x, y, BLOCK=32)
    torch.mps.synchronize()
    torch.testing.assert_close(y, torch.full_like(y, -99), rtol=0, atol=0)
