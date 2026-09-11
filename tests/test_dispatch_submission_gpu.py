"""Actual driver hooks and runtime return failures may not apply atomics twice."""

import sys

import pytest
import torch
import triton
import triton.language as tl

from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
from triton_msl.errors import PostSubmitError


@triton.jit
def _accumulate(X, O, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.atomic_add(O + i % 8, tl.load(X + i))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("failure", ["hook", "runtime_return", "compile_miss"])
def test_real_driver_does_not_repeat_submitted_atomic(monkeypatch, failure):
    from triton.knobs import runtime
    from triton_msl.backend.driver import _get_compile_shader_runtime

    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    x = torch.ones(256, device="mps")
    out = torch.zeros(8, device="mps")
    # Compile and establish the source's one-application value before injection.
    _accumulate[(1,)](x, out, B=256)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.full((8,), 32.0))
    out.zero_()
    rt = _get_compile_shader_runtime()
    calls = []
    original_dispatch = rt.dispatch

    def observed(*a, **kw):
        calls.append("invoked")
        return original_dispatch(*a, **kw)

    monkeypatch.setattr(rt, "dispatch", observed)
    assert _get_compile_shader_runtime() is rt
    assert _get_compile_shader_runtime().dispatch is observed
    hook_calls = []

    def hook(metadata):
        hook_calls.append("hook")
        raise RuntimeError("actual Triton launch-exit hook")

    if failure == "hook":
        monkeypatch.setattr(runtime, "launch_exit_hook", hook)
    elif failure == "compile_miss":

        def miss(source):
            raise RuntimeError("known pre-invocation library miss")

        monkeypatch.setattr(rt, "get_library", miss)

    def trace(frame, event, arg):
        if frame.f_code is CompileShaderRuntime.dispatch.__code__ and event == "return":
            raise RuntimeError("after the actual library invocation returned")
        return trace

    old = sys.gettrace()
    try:
        if failure == "runtime_return":
            sys.settrace(trace)
        if failure == "compile_miss":
            _accumulate[(1,)](x, out, B=256)
        else:
            with pytest.raises(PostSubmitError) as caught:
                _accumulate[(1,)](x, out, B=256)
            assert isinstance(caught.value.__cause__, RuntimeError)
    finally:
        sys.settrace(old)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.full((8,), 32.0)), "exactly once, never 64"
    assert calls == ([] if failure == "compile_miss" else ["invoked"])
    assert hook_calls == (["hook"] if failure == "hook" else [])
