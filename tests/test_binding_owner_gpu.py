"""Real temporary-input ownership on both existing Metal launch routes."""
import gc
import weakref

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _temporary_copy(x, output, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(output + offsets, tl.load(x + offsets) + 3.0)


@pytest.mark.parametrize("route", ["compile_shader", "host"])
def test_temporary_input_storage_survives_real_execution(monkeypatch, tmp_path, route):
    if not torch.backends.mps.is_available():
        pytest.skip("requires Metal")
    from triton_msl.backend import driver
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1" if route == "compile_shader" else "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "metal"))
    _temporary_copy.device_caches.clear()
    device = "mps" if route == "compile_shader" else "cpu"
    expected = torch.arange(128, dtype=torch.float32) + 3
    output = torch.full((128,), -99.0, dtype=torch.float32, device=device)
    refs, dispatches = [], []
    if route == "compile_shader":
        runtime = driver._get_compile_shader_runtime()
        original = runtime.dispatch
        def observed(*args, **kwargs):
            dispatches.append("compile_shader")
            return original(*args, **kwargs)
        monkeypatch.setattr(runtime, "dispatch", observed)
    else:
        utils = driver._get_utils()
        original = utils.launch
        def observed(*args, **kwargs):
            dispatches.append("host")
            assert kwargs.get("sync", True)
            return original(*args, **kwargs)
        monkeypatch.setattr(utils, "launch", observed)

    def temporary():
        value = torch.arange(128, dtype=torch.float32, device=device)
        refs.append(weakref.ref(value))
        return value

    for _ in range(3):
        output.fill_(-99)
        _temporary_copy[(1,)](temporary(), output, BLOCK=128)
        # Release unreachable Python objects before explicit MPS synchronization,
        # then exercise normal allocator reuse while work may still be pending.
        gc.collect()
        pressure = [torch.full((128,), -777.0, device=device) for _ in range(8)]
        torch.mps.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        del pressure
    assert dispatches == [route] * 3
    gc.collect()
    assert all(ref() is None for ref in refs)
