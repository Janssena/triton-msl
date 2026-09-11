"""Loader generations may skip a name scan only after a complete stable proof."""
from pathlib import Path
import pytest


@pytest.fixture
def state(tmp_path):
    from triton_msl.backend._framework_contract import _LoadedNativeGuard
    first, second = tmp_path / "first.so", tmp_path / "second.so"
    first.touch()
    second.touch()
    state = {"names": [str(first).encode()], "generation": (10, 4096, 1), "reads": 0}
    guard = object.__new__(_LoadedNativeGuard)
    guard.providers = {first.resolve(), second.resolve()}
    guard.approved_names = set()
    guard.checked_names = None
    guard.checked_count = None
    guard._generation_checked = None
    guard._generation_reader = lambda: state["generation"]
    guard.count = lambda: len(state["names"])
    def name(index):
        state["reads"] += 1
        return state["names"][index]
    guard.name = name
    return guard, state, first, second


def test_unchanged_generation_reuses_only_a_completed_scan(state):
    guard, state, _, _ = state
    guard.verify()
    assert state["reads"] == 1
    guard.verify()
    assert state["reads"] == 1


def test_same_count_new_provider_does_not_reuse_generation(state, tmp_path):
    guard, state, _, _ = state
    guard.verify()
    foreign = tmp_path / "foreign.so"
    foreign.touch()
    state.update(names=[str(foreign).encode()], generation=(11, 4096, 1))
    with pytest.raises(RuntimeError, match="outside.*proved"):
        guard.verify()


def test_known_provider_reordering_is_still_rejected(state):
    guard, state, _, second = state
    state["names"].append(str(second).encode())
    state["generation"] = (11, 4096, 2)
    guard.verify()
    state["names"].reverse()
    state["generation"] = (12, 4096, 2)
    with pytest.raises(RuntimeError, match="selection changed"):
        guard.verify()


def test_generation_change_during_scan_is_not_published(state):
    guard, state, _, _ = state
    values = iter(((10, 4096, 1), (11, 4096, 1)))
    guard._generation_reader = lambda: next(values)
    with pytest.raises(RuntimeError, match="changed during"):
        guard.verify()
    assert guard._generation_checked is None and guard.checked_count is None


def test_unreadable_generation_is_not_a_cached_approval(state):
    guard, state, _, _ = state
    guard.verify()
    def unreadable():
        raise RuntimeError("unreadable native generation")
    guard._generation_reader = unreadable
    with pytest.raises(RuntimeError, match="unreadable"):
        guard.verify()


def test_unsupported_generation_reader_retains_full_name_scan(state):
    guard, state, _, _ = state
    guard._generation_reader = None
    guard.verify()
    guard.verify()
    assert state["reads"] == 2


def test_new_generation_with_same_names_requires_a_new_scan(state):
    guard, state, _, _ = state
    guard.verify()
    state["generation"] = (11, 4096, 1)
    guard.verify()
    assert state["reads"] == 2
    assert Path(state["names"][0].decode()).resolve() in guard.providers
