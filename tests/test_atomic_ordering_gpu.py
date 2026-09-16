"""Hardware supporting evidence, not a proof of ordering or scheduler progress."""

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU required")


@triton.jit
def _message(Data, Flag, Out, ROUNDS: tl.constexpr):
    pid = tl.program_id(0)
    pair = pid // 2
    if pid % 2 == 0:
        tl.store(Data + pair, pair * 17 + 123)
        tl.atomic_xchg(Flag + pair, 1, sem="release", scope="gpu")
    else:
        observed = -1
        for _ in range(ROUNDS):
            flag = tl.atomic_add(Flag + pair, 0, sem="acquire", scope="gpu")
            if flag == 1:
                observed = tl.load(Data + pair)
        tl.store(Out + pair, observed)


@pytest.mark.parametrize("repetition", range(4))
def test_bounded_release_acquire_message_passing(repetition, tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    calls = []
    real = GenericLowerer._atomic_ordering_fences

    def spy(self, ssa):
        result = real(self, ssa)
        calls.append((ssa.attrs["sem"], ssa.attrs["scope"], result))
        return result

    monkeypatch.setattr(GenericLowerer, "_atomic_ordering_fences", spy)
    count = 128
    data = torch.zeros(count, dtype=torch.int32, device="mps")
    flags = torch.zeros_like(data)
    out = torch.full_like(data, -999)
    _message.device_caches.clear()
    _message[(count * 2,)](data, flags, out, 16)
    torch.mps.synchronize()
    result = out.cpu()
    expected = torch.arange(count, dtype=torch.int32) * 17 + 123
    observed = result != -1
    assert bool(observed.any()), "no communication observed: the test provides no ordering evidence"
    assert torch.equal(result[observed], expected[observed])
    assert torch.equal(data.cpu(), expected)
    assert bool((flags.cpu() == 1).all())
    assert {sem for sem, _, _ in calls} == {"release", "acquire"}
    assert all(scope == "gpu" for _, scope, _ in calls)
    print("MESSAGE_PASSING", repetition, "OBSERVED", int(observed.sum()), "OF", count, flush=True)
