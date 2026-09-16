"""CPU-only pins for the fixed direct-KDA Metal ABI."""

from types import SimpleNamespace

import pytest
import torch

import triton_msl.kda as kda


class _Tensor:
    def __init__(self, shape, dtype=torch.float32, device="mps:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)

    def dim(self):
        return len(self.shape)


def test_fixed_abi_helper_checks_shape_dtype_and_device():
    expected = _Tensor((2, 8, 64))
    kda._require_tensor_abi("k", expected, shape=(2, 8, 64), dtype=torch.float32, device=torch.device("mps:0"))
    for tensor, match in (
        (_Tensor((1, 8, 64)), "shape"),
        (_Tensor((2, 8, 64), torch.float16), "dtype"),
        (_Tensor((2, 8, 64), device="cpu"), "must be on"),
    ):
        with pytest.raises(ValueError, match=match):
            kda._require_tensor_abi("k", tensor, shape=(2, 8, 64), dtype=torch.float32, device=torch.device("mps:0"))


@pytest.mark.parametrize("peer,shape", [("k", (1, 8, 64)), ("v", (2, 7, 64)), ("a", (2, 8, 32)), ("beta", (1, 8))])
def test_prefill_peer_shape_refuses_before_compilation(monkeypatch, peer, shape):
    tensors = {name: _Tensor((2, 8, 64)) for name in ("q", "k", "v", "a")}
    tensors["beta"] = _Tensor((2, 8))
    tensors[peer] = _Tensor(shape)
    monkeypatch.setattr(kda, "_kernel", lambda **kw: pytest.fail("compiled before validation"))
    with pytest.raises(ValueError, match=peer):
        kda.kda_attention(tensors["q"], tensors["k"], tensors["v"], tensors["a"], tensors["beta"])


def test_prefill_peer_dtype_and_q_device_refuse_before_compilation(monkeypatch):
    monkeypatch.setattr(kda, "_kernel", lambda **kw: pytest.fail("compiled before validation"))
    q = _Tensor((2, 8, 64), device="cpu")
    with pytest.raises(ValueError, match="q must be on an mps"):
        kda.kda_attention(q, q, q, q, _Tensor((2, 8), device="cpu"))

    q = _Tensor((2, 8, 64))
    with pytest.raises(ValueError, match="k must have dtype"):
        kda.kda_attention(q, _Tensor((2, 8, 64), torch.float16), q, q, _Tensor((2, 8)))


def test_empty_launches_refuse_before_compilation(monkeypatch):
    monkeypatch.setattr(kda, "_kernel", lambda **kw: pytest.fail("compiled before validation"))
    empty = _Tensor((0, 8, 64))
    with pytest.raises(ValueError, match="positive ZH and T"):
        kda.kda_attention(empty, empty, empty, empty, _Tensor((0, 8)))

    monkeypatch.setattr(kda, "_decode_kernel", lambda: pytest.fail("compiled before validation"))
    empty = _Tensor((0, 64))
    with pytest.raises(ValueError, match="positive ZH"):
        kda.kda_decode_step(empty, empty, empty, empty, _Tensor((0,)), _Tensor((0, 64, 64)))


def test_decode_q_dtype_and_peer_abi_refuse_before_compilation(monkeypatch):
    monkeypatch.setattr(kda, "_decode_kernel", lambda: pytest.fail("compiled before validation"))
    half = _Tensor((2, 64), torch.float16)
    with pytest.raises(ValueError, match="supports float32"):
        kda.kda_decode_step(half, half, half, half, _Tensor((2,), torch.float16), _Tensor((2, 64, 64), torch.float16))

    q = _Tensor((2, 64))
    with pytest.raises(ValueError, match="S must have shape"):
        kda.kda_decode_step(q, q, q, q, _Tensor((2,)), _Tensor((1, 64, 64)))


def test_decode_noncontiguous_state_stages_and_copies_back(monkeypatch):
    q = torch.zeros((2, 64), device="meta")
    state = torch.zeros((2, 64, 64), device="meta").transpose(1, 2)
    beta = torch.zeros((2,), device="meta")
    monkeypatch.setattr(kda, "_require_mps", lambda *args: None)
    monkeypatch.setattr(torch._C, "_overlaps", lambda *args: False)
    seen = SimpleNamespace(state=None)

    class _RT:
        def dispatch(self, lib, name, args, **kwargs):
            seen.state = args[5]

    monkeypatch.setattr(kda, "_decode_kernel", lambda: (_RT(), object()))
    result = kda.kda_decode_step(q, q, q, q, beta, state)
    assert seen.state.is_contiguous() and seen.state is not state
    assert result.shape == (2, 64)


def test_state_layout_injectivity_uses_geometry_not_debug_heuristic():
    dense = torch.empty((2, 64, 64)).transpose(1, 2)
    padded = torch.empty((2, 64, 80))[:, :, :64]
    expanded = torch.empty((2, 1, 64)).expand(2, 64, 64)
    assert kda._state_layout_injective(dense)
    assert kda._state_layout_injective(padded)
    assert not kda._state_layout_injective(expanded)


def test_decode_shared_storage_snapshots_readonly_input(monkeypatch):
    base = torch.zeros((2, 65, 64), device="meta")
    q = base[:, 0, :]
    state = base[:, 1:, :]
    other = torch.zeros((2, 64), device="meta")
    beta = torch.zeros((2,), device="meta")
    monkeypatch.setattr(kda, "_require_mps", lambda *args: None)
    seen = SimpleNamespace(q=None)

    class _RT:
        def dispatch(self, lib, name, args, **kwargs):
            seen.q = args[0]

    monkeypatch.setattr(kda, "_decode_kernel", lambda: (_RT(), object()))
    kda.kda_decode_step(q, other, other, other, beta, state)
    assert seen.q is not q


def test_decode_overlapping_state_layout_refuses_before_compilation(monkeypatch):
    q = torch.zeros((2, 64), device="meta")
    state = torch.zeros((2, 1, 64), device="meta").expand(2, 64, 64)
    beta = torch.zeros((2,), device="meta")
    monkeypatch.setattr(kda, "_require_mps", lambda *args: None)
    monkeypatch.setattr(kda, "_decode_kernel", lambda: pytest.fail("compiled before validation"))
    with pytest.raises(ValueError, match="cannot be proven non-overlapping"):
        kda.kda_decode_step(q, q, q, q, beta, state)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("layout", ["transpose", "padded"])
def test_decode_gpu_noncontiguous_state_copyback_matches_contiguous(layout):
    torch.manual_seed(745)
    q, k, v = (torch.randn((1, 64), device="mps") for _ in range(3))
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn((1, 64), device="mps"))
    beta = torch.sigmoid(torch.randn((1,), device="mps"))
    initial = torch.randn((1, 64, 64), device="mps")
    reference = initial.clone()
    if layout == "transpose":
        state = torch.empty((1, 64, 64), device="mps").transpose(1, 2)
        padding = None
    else:
        backing = torch.full((1, 64, 80), 314.0, device="mps")
        state, padding = backing[:, :, :64], backing[:, :, 64:]
    state.copy_(initial)
    expected = kda.kda_decode_step(q, k, v, a, beta, reference)
    actual = kda.kda_decode_step(q, k, v, a, beta, state)
    torch.mps.synchronize()
    assert torch.equal(actual, expected)
    assert torch.equal(state, reference)
    if padding is not None:
        assert torch.equal(padding, torch.full_like(padding, 314.0))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_decode_gpu_disjoint_shared_storage_q_uses_prelaunch_snapshot():
    torch.manual_seed(746)
    q_initial = torch.randn((1, 64), device="mps")
    state_initial = torch.randn((1, 64, 64), device="mps")
    k, v = (torch.randn((1, 64), device="mps") for _ in range(2))
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn((1, 64), device="mps"))
    beta = torch.sigmoid(torch.randn((1,), device="mps"))

    reference = state_initial.clone()
    expected = kda.kda_decode_step(q_initial.clone(), k, v, a, beta, reference)

    backing = torch.empty((1, 65, 64), device="mps")
    alias_q, alias_state = backing[:, 0, :], backing[:, 1:, :]
    alias_q.copy_(q_initial)
    alias_state.copy_(state_initial)
    actual = kda.kda_decode_step(alias_q, k, v, a, beta, alias_state)
    torch.mps.synchronize()
    assert torch.equal(actual, expected)
    assert torch.equal(alias_state, reference)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_decode_gpu_overlapping_q_state_uses_prelaunch_snapshot():
    torch.manual_seed(747)
    initial = torch.randn((1, 64, 64), device="mps")
    q_snapshot = initial[:, 0, :].clone()
    state_reference = initial.clone()
    k, v = (torch.randn((1, 64), device="mps") for _ in range(2))
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn((1, 64), device="mps"))
    beta = torch.sigmoid(torch.randn((1,), device="mps"))

    # Both oracle inputs are detached before either launch mutates state.
    expected = kda.kda_decode_step(q_snapshot, k, v, a, beta, state_reference)
    alias_state = initial.clone()
    alias_q = alias_state[:, 0, :]
    actual = kda.kda_decode_step(alias_q, k, v, a, beta, alias_state)
    torch.mps.synchronize()
    assert torch.equal(actual, expected)
    assert torch.equal(alias_state, state_reference)
