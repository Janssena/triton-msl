"""Caller ownership through the real launcher; CPU endpoints never submit work."""
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch
import triton.backends

from triton_msl.backend import _cache_contract, _launch_contract, _launch_signature, driver
from triton_msl.errors import MetalNonRecoverableError, PostSubmitError


@pytest.mark.parametrize("mode", ["success", "binding_refusal", "enter_failure", "dispatch_failure", "exit_failure"])
def test_temporary_argument_owners_survive_required_launch_boundaries(monkeypatch, mode):
    events, references = [], {}

    class Owner:
        def __del__(self):
            events.append("owner-released")

    class Tensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("mps")

    def pointer():
        value = torch.empty(4, device="cpu").as_subclass(Tensor)
        value.storage_owner = Owner()
        references["tensor"] = weakref.ref(value)
        references["owner"] = weakref.ref(value.storage_owner)
        return value

    def alive(boundary):
        # Collection cannot take away resources still needed by this launch.
        gc.collect()
        assert references["tensor"]() is not None
        assert references["owner"]() is not None
        events.append(boundary)

    class Runtime:
        def available(self):
            alive("available")
            return True

        def is_unsupported(self, source):
            alive("eligibility")
            return False

        def get_library(self, source):
            alive("library")
            return object()

        def dispatch(self, library, kernel, args, **kwargs):
            alive("dispatch")
            assert args[0] is references["tensor"]()
            if mode == "dispatch_failure":
                raise RuntimeError("controlled invocation failure")
            # Synchronous CPU stand-in: all work is completed before returning.
            alive("completion")

        def mark_unsupported(self, source):
            pytest.fail("attempted invocation cannot enter fallback")

    def enter(_):
        alive("enter")
        if mode == "enter_failure":
            raise RuntimeError("controlled enter failure")

    def leave(_):
        alive("exit")
        if mode == "exit_failure":
            raise RuntimeError("controlled exit failure")

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    monkeypatch.setattr(_cache_contract, "execution_contract", lambda: "controlled-stamp")
    monkeypatch.setattr(driver, "_get_utils", lambda: object())
    runtime = Runtime()
    monkeypatch.setattr(driver, "_get_compile_shader_runtime", lambda: runtime)
    launcher = driver.MetalLauncher.__new__(driver.MetalLauncher)
    launcher.arg_names = ["pointer", "value"]
    launcher.signature = dict(pointer="*fp32", value="i8" if mode == "binding_refusal" else "i32")
    launcher.constexpr_indices = set()
    launcher._binding_plan = _launch_signature.make_binding_plan(launcher.arg_names, launcher.signature)
    launcher._execution_contract = "controlled-stamp"
    packed = (4, 1, 0, 32, None, False, None, None, None, None, None, None)
    launcher._packed_contract = _launch_contract._canonical(packed)
    launcher._msl, launcher._msl_block_size, launcher.kernel_name = "source", 32, "kernel"

    error = None
    try:
        launcher(1, 1, 1, None, None, packed, None, enter, leave, pointer(), 128)
    except (MetalNonRecoverableError, RuntimeError) as caught:
        error = type(caught)
    if mode == "binding_refusal":
        assert error is MetalNonRecoverableError
        assert not any(e in events for e in ("enter", "dispatch", "completion"))
    elif mode == "enter_failure":
        assert error is RuntimeError
        assert "enter" in events and "dispatch" not in events
    elif mode in ("dispatch_failure", "exit_failure"):
        assert error is PostSubmitError
        assert events.count("dispatch") == 1
        assert ("completion" in events) == (mode == "exit_failure")
    else:
        assert error is None
        assert events.count("dispatch") == events.count("completion") == events.count("exit") == 1
    # No retained owner is needed after this completed call/no-submit failure.
    # Do not require an accidental cycle or prescribe exact finalizer timing.
    gc.collect()
    assert references["tensor"]() is None
    assert references["owner"]() is None
    assert events.count("owner-released") == 1
