"""Hot-path reuse requires equal contents, not equal mapping identity or size."""
import os
import importlib.util
import sys
from types import FunctionType, MappingProxyType, MethodType

import pytest

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _environment_snapshot as environment
from triton_msl.backend import _toolchain_contract as toolchain
from triton_msl.errors import MetalNonRecoverableError


def test_snapshot_tracks_same_size_edits_new_keys_and_is_unaliased(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_SNAPSHOT_WITNESS", "a")
    first = environment.environment_snapshot()
    owned = environment.is_snapshot(first)
    if owned:
        assert type(first) is MappingProxyType
        assert environment.environment_snapshot() is first
        with pytest.raises(TypeError):
            first["TRITON_MSL_SNAPSHOT_WITNESS"] = "broken"
    else:
        assert first is os.environ
    monkeypatch.setenv("TRITON_MSL_SNAPSHOT_WITNESS", "b")
    second = environment.environment_snapshot()
    assert second["TRITON_MSL_SNAPSHOT_WITNESS"] == "b"
    assert first["TRITON_MSL_SNAPSHOT_WITNESS"] == ("a" if owned else "b")
    monkeypatch.setenv("DYLD_FUTURE_UNKNOWN_SNAPSHOT_TEST", "injected")
    selection = tuple(os.environ.get(key) for key in toolchain._SELECTION)
    monkeypatch.setattr(toolchain, "_snapshot", (selection, "previous"))
    with pytest.raises(MetalNonRecoverableError, match="DYLD_FUTURE_UNKNOWN_SNAPSHOT_TEST"):
        toolchain.toolchain_identity()
    monkeypatch.setenv("DYLD_FUTURE_UNKNOWN_SNAPSHOT_TEST", "")
    assert toolchain.toolchain_identity() == "previous"
    monkeypatch.delenv("DYLD_FUTURE_UNKNOWN_SNAPSHOT_TEST")
    assert "DYLD_FUTURE_UNKNOWN_SNAPSHOT_TEST" not in environment.environment_snapshot()


def test_custom_mapping_and_getter_keep_live_semantics(monkeypatch):
    original = os.environ
    custom = {"TRITON_MSL_FA_FAST": "0"}
    monkeypatch.setattr(os, "environ", custom)
    assert environment.environment_snapshot() is custom
    assert cache.effective_policy()["FA_FAST"] is False
    custom["TRITON_MSL_FA_FAST"] = "1"
    assert cache.effective_policy()["FA_FAST"] is True
    # Read-only access is not ownership: another holder can mutate this backing.
    proxy = MappingProxyType(custom)
    monkeypatch.setattr(os, "environ", proxy)
    assert environment.environment_snapshot() is proxy
    assert not environment.is_snapshot(proxy)
    assert cache.effective_policy()["FA_FAST"] is True
    custom["TRITON_MSL_FA_FAST"] = "0"
    assert cache.effective_policy()["FA_FAST"] is False
    selection = tuple(proxy.get(key) for key in toolchain._SELECTION)
    monkeypatch.setattr(toolchain, "_snapshot", (selection, "foreign-proxy-control"))
    assert toolchain.toolchain_identity() == "foreign-proxy-control"
    custom["DYLD_FUTURE_FOREIGN_PROXY"] = "injected"
    with pytest.raises(MetalNonRecoverableError, match="DYLD_FUTURE_FOREIGN_PROXY"):
        toolchain.toolchain_identity()
    monkeypatch.setattr(os, "environ", original)
    prior = original.get
    owned_get = "get" in vars(original)
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TRITON_MSL_FA_FAST", "1")
            patch.setattr(original, "get", lambda key, default=None:
                          "0" if key == "TRITON_MSL_FA_FAST" else prior(key, default))
            assert environment.environment_snapshot() is original
            assert cache.effective_policy()["FA_FAST"] is False
            patch.setattr(original, "get", MethodType(environment._get, {"TRITON_MSL_FA_FAST": "0"}))
            assert environment.environment_snapshot() is original
            assert cache.effective_policy()["FA_FAST"] is False
            class PretendMethod:
                __func__ = environment._get
                __self__ = original
                def __call__(self, key, default=None):
                    return "0" if key == "TRITON_MSL_FA_FAST" else prior(key, default)
            patch.setattr(original, "get", PretendMethod())
            assert environment.environment_snapshot() is original
            assert cache.effective_policy()["FA_FAST"] is False
    finally:
        # monkeypatch restores an inherited bound method as an instance attr.
        # Restore absence too, or the next class-level witness sees nothing.
        if not owned_get:
            delattr(original, "get")


def test_policy_copy_and_same_mapping_changes_cannot_stale(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", "0")
    first = cache.effective_policy()
    first["FA_HALF_ACCUM"] = True
    assert cache.effective_policy()["FA_HALF_ACCUM"] is False
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", "1")
    assert cache.effective_policy()["FA_HALF_ACCUM"] is True
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", "0")
    assert cache.effective_policy()["FA_HALF_ACCUM"] is False


def test_preimport_overrides_are_not_certified_as_standard(monkeypatch):
    monkeypatch.setenv("SNAPSHOT_PREIMPORT_WITNESS", "snapshot-original")
    for mode in ("get", "getitem", "codec", "codec_closure"):
        with monkeypatch.context() as patch:
            expected = "snapshot-altered"
            if mode == "get":
                original = type(os.environ).get
                if "get" in vars(os.environ):
                    patch.delattr(os.environ, "get")
                patch.setattr(type(os.environ), "get", lambda self, key, default=None:
                              "snapshot-altered" if key == "SNAPSHOT_PREIMPORT_WITNESS"
                              else original(self, key, default))
            elif mode == "getitem":
                original = type(os.environ).__getitem__
                patch.setattr(type(os.environ), "__getitem__", lambda self, key:
                              "snapshot-altered" if key == "SNAPSHOT_PREIMPORT_WITNESS"
                              else original(self, key))
            elif mode == "codec":
                original = os.environ.decodevalue
                patch.setattr(os.environ, "decodevalue", lambda value:
                              "snapshot-altered" if value == b"snapshot-original" else original(value))
            else:
                original = os.environ.encodekey
                if not hasattr(original, "__code__") or not original.__closure__:
                    # No recognized codec optimization exists in this context.
                    assert environment.environment_snapshot() is os.environ
                    continue
                def cell(value):
                    return (lambda: value).__closure__[0]
                borrowed = FunctionType(original.__code__, original.__globals__,
                                        closure=(cell("utf-16"),))
                patch.setattr(os.environ, "encodekey", borrowed)
                expected = None  # The altered key codec cannot find the UTF-8 key.
            spec = importlib.util.spec_from_file_location("snapshot_preimport_probe", environment.__file__)
            assert os.environ.get("SNAPSHOT_PREIMPORT_WITNESS") == expected, mode
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            snapshot = module.environment_snapshot()
            assert snapshot is os.environ, mode
            assert not module.is_snapshot(snapshot), mode
            assert snapshot.get("SNAPSHOT_PREIMPORT_WITNESS") == expected, mode


def test_decode_uses_private_copy_even_when_live_values_change_and_return(monkeypatch):
    if not environment._standard:
        assert environment.environment_snapshot() is os.environ
        return  # No snapshot optimization exists in this interpreter context.
    monkeypatch.setenv("TRITON_MSL_FA_FAST", "0")
    monkeypatch.setattr(environment, "_cached", None)
    events = []
    def trace(frame, event, arg):
        if frame.f_code is environment.environment_snapshot.__code__ and event == "line":
            if "raw" in frame.f_locals and "values" not in frame.f_locals and not events:
                os.environ["TRITON_MSL_FA_FAST"] = "1"
                events.append("changed-after-copy")
            elif "values" in frame.f_locals and len(events) == 1:
                os.environ["TRITON_MSL_FA_FAST"] = "0"
                events.append("restored-after-decode")
        return trace
    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        snapshot = environment.environment_snapshot()
    finally:
        sys.settrace(previous_trace)
    assert events == ["changed-after-copy", "restored-after-decode"]
    assert os.environ["TRITON_MSL_FA_FAST"] == snapshot["TRITON_MSL_FA_FAST"] == "0"
    assert cache.effective_policy()["FA_FAST"] is False
