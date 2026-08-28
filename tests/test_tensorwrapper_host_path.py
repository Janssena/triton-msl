"""Regression: a ``TensorWrapper`` (``triton.reinterpret``) must work through the HOST
dispatch path, not just the zero-copy ``compile_shader`` one.

Background (2026-08-27, surfaced by GPT re-review 012). Triton's ``TensorWrapper`` —
what ``to_triton`` produces for unsigned dtypes, wrapping e.g. an int8 tensor as uint8 —
exposes ``device`` (so ``is_mps`` routes it into the host strided-staging branch) but
NOT ``is_contiguous``/``as_strided``/``storage_offset``. The staging branch called
``arg.is_contiguous()`` unconditionally, so ANY wrapper reaching the host path crashed
with an AttributeError. The upstream suite masked this: those kernels normally route
through zero-copy ``compile_shader`` and never touch host staging — the trigger is
ROUTING, not test order. (An isolated run of the upstream case can land either way
depending on dispatch, which is why 012's isolated repro crashed while another passed.)

The fix answers layout questions through the wrapper's ``base`` (same storage,
byte-for-byte; only the dtype label differs), and passes the base — not the wrapper —
to the faithful-strided copy-back, whose scratch allocation
(``_torch.empty(dtype=tensor.dtype)``) would choke on a wrapper's triton dtype.

This test forces the host path via ``TRITON_MSL_COMPILE_SHADER=0`` (read per launch, so
a monkeypatched env var takes effect) and pins numeric correctness, not just
no-crash.
"""

import pytest
import torch
import triton
import triton.language as tl

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _u8_add_one(x, o, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o + i, tl.load(x + i) + 1)


def _run_wrapped(N=64):
    dev = "mps"
    xb = torch.arange(N, device=dev, dtype=torch.int8)
    ob = torch.zeros(N, device=dev, dtype=torch.int8)
    xw = triton.reinterpret(xb, tl.uint8)
    ow = triton.reinterpret(ob, tl.uint8)
    _u8_add_one[(1,)](xw, ow, N=N)
    torch.mps.synchronize()
    got = ob.to(torch.int16).cpu()
    exp = torch.arange(N, dtype=torch.int16) + 1
    assert (got == exp).all(), f"wrapped uint8 kernel wrong: got[:8]={got[:8].tolist()}"


@requires_mps
def test_tensorwrapper_via_host_path(monkeypatch):
    # Pre-fix: AttributeError ('TensorWrapper' object has no attribute
    # 'is_contiguous') at the strided-staging contiguity probe.
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    _run_wrapped()


@requires_mps
def test_tensorwrapper_via_default_path():
    # The zero-copy route always worked; pinned so the two dispatch paths cannot
    # silently diverge for wrapped tensors.
    _run_wrapped()
