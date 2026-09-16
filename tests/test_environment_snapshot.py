"""Hot-path reuse requires equal contents, not equal mapping identity or size."""

import os
import importlib.util
import sys
from types import FunctionType, MappingProxyType, MethodType
from collections.abc import Mapping

import pytest

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _environment_snapshot as environment
from triton_msl.backend import _toolchain_contract as toolchain
from triton_msl.errors import MetalNonRecoverableError


def _isolated_snapshot(monkeypatch, foreign=None):
    """Clone functions so code-mutation witnesses never alter installed code."""

    def clone(fn, namespace=None):
        return FunctionType(
            fn.__code__,
            fn.__globals__ if namespace is None else namespace,
            fn.__name__,
            fn.__defaults__,
            fn.__closure__,
        )

    functions = {
        "get": clone(Mapping.get),
        "getitem": clone(os._Environ.__getitem__),
        "iter": clone(os._Environ.__iter__),
        "copy": clone(os._Environ.copy),
        "encode": clone(os.environ.encodekey),
        "decode": clone(os.environ.decodekey),
    }
    if foreign is not None:
        name, overrides = foreign
        fn = functions[name]
        functions[name] = clone(fn, dict(fn.__globals__, **overrides))
    monkeypatch.setattr(Mapping, "get", functions["get"])
    for name, attribute in (("getitem", "__getitem__"), ("iter", "__iter__"), ("copy", "copy")):
        monkeypatch.setattr(os._Environ, attribute, functions[name])
    private = os._Environ(
        {b"TRITON_MSL_FA_HALF_ACCUM": b"0", b"ALT": b"1"},
        functions["encode"],
        functions["decode"],
        functions["encode"],
        functions["decode"],
    )
    monkeypatch.setattr(os, "environ", private)
    spec = importlib.util.spec_from_file_location("isolated_environment_recognition", environment.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, private, functions


def _changed_codec(encoding, *, encode=False):
    if encode:

        def changed(value):
            return ("ALT" if value == "TRITON_MSL_FA_HALF_ACCUM" else value).encode(encoding)
    else:

        def changed(value):
            return "1" if value == b"0" else value.decode(encoding)

    return changed


@pytest.mark.parametrize("name", ("get", "getitem", "iter", "copy", "encode", "decode"))
def test_same_function_code_change_after_admission_stays_live(monkeypatch, name):
    with monkeypatch.context() as patch:
        module, private, functions = _isolated_snapshot(patch)
        before = module.environment_snapshot()
        assert module.is_snapshot(before)
        assert before["TRITON_MSL_FA_HALF_ACCUM"] == "0"
        assert module.environment_snapshot() is before  # Positive unchanged reuse.
        fn = functions[name]
        raw = dict(private._data)

        def changed_iter(self):
            yield "ALT"

        replacements = {
            "get": lambda self, key, default=None: "1",
            "getitem": lambda self, key: "1",
            "iter": changed_iter,
            "copy": lambda self: {"changed": "1"},
        }
        changed = (
            _changed_codec(fn.__closure__[0].cell_contents, encode=name == "encode")
            if name in ("encode", "decode")
            else replacements[name]
        )
        original_code = fn.__code__
        fn.__code__ = changed.__code__
        after = module.environment_snapshot()
        assert after is private
        assert not module.is_snapshot(after)
        assert private._data == raw
        if name == "iter":
            assert tuple(after) == ("ALT",)
        elif name == "copy":
            assert after.copy() == {"changed": "1"}
        else:
            assert after.get("TRITON_MSL_FA_HALF_ACCUM") == "1"
        fn.__code__ = original_code
        assert module.environment_snapshot() is before


def test_same_getter_defaults_change_after_admission_stays_live(monkeypatch):
    with monkeypatch.context() as patch:
        module, private, functions = _isolated_snapshot(patch)
        before = module.environment_snapshot()
        assert module.is_snapshot(before)
        assert before.get("absent") is None
        functions["get"].__defaults__ = ("changed-default",)
        after = module.environment_snapshot()
        assert after is private
        assert after.get("absent") == "changed-default"


@pytest.mark.parametrize(
    "name,overrides",
    (
        ("get", {"KeyError": ValueError}),
        ("getitem", {"KeyError": ValueError}),
        ("iter", {"list": lambda value: [b"ALT"]}),
        ("copy", {"dict": lambda value: {"changed": "1"}}),
        ("encode", {"str": bytes}),
    ),
)
def test_equal_frozen_code_with_foreign_globals_is_not_admitted(monkeypatch, name, overrides):
    with monkeypatch.context() as patch:
        module, private, _ = _isolated_snapshot(patch, (name, overrides))
        after = module.environment_snapshot()
        assert after is private
        assert not module.is_snapshot(after)
        if name == "get":
            with pytest.raises(KeyError):
                after.get("absent")
        elif name == "getitem":
            with pytest.raises(KeyError) as caught:
                after["absent"]
            assert caught.value.args == (b"absent",)
        elif name == "iter":
            assert tuple(after) == ("ALT",)
        elif name == "copy":
            assert after.copy() == {"changed": "1"}
        else:
            with pytest.raises(TypeError):
                after.get("TRITON_MSL_FA_HALF_ACCUM")


def test_live_standard_global_dependency_change_stays_live(monkeypatch):
    with monkeypatch.context() as patch:
        module, private, functions = _isolated_snapshot(patch)
        before = module.environment_snapshot()
        assert module.is_snapshot(before)
        patch.setitem(functions["iter"].__globals__, "list", lambda value: [b"ALT"])
        after = module.environment_snapshot()
        assert after is private
        assert tuple(after) == ("ALT",)


def test_snapshot_tracks_same_size_edits_new_keys_and_is_unaliased(monkeypatch):
    monkeypatch.setenv("TRITON_MSL_SNAPSHOT_WITNESS", "a")
    first = environment.environment_snapshot()
    owned = environment.is_snapshot(first)
    if owned:
        assert type(first) is MappingProxyType
        assert environment.environment_snapshot() is first
        with pytest.raises(TypeError):
            first["TRITON_MSL_SNAPSHOT_WITNESS"] = "broken"
        # Equal payloads with different identities and insertion order still
        # mean the same snapshot; neither may accidentally bypass a later edit.
        key = b"TRITON_MSL_SNAPSHOT_WITNESS"
        raw_value = os.environ._data.pop(key)
        os.environ._data[key] = bytes(bytearray(raw_value))
        assert environment.environment_snapshot() is first
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
            patch.setattr(
                original, "get", lambda key, default=None: "0" if key == "TRITON_MSL_FA_FAST" else prior(key, default)
            )
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
    if hasattr(os, "environb"):
        monkeypatch.setenv("TRITON_MSL_FA_FAST", "0")
        assert cache.effective_policy()["FA_FAST"] is False  # Prime ordinary bytes.
        state = ["0"]

        class PretendBytes(type):
            def __hash__(cls):
                return hash(bytes)

            def __eq__(cls, other):
                return other is bytes or type.__eq__(cls, other)

        class EncodedValue(bytes, metaclass=PretendBytes):
            def decode(self, *args, **kwargs):
                return state[0]

        monkeypatch.setitem(os.environb, b"TRITON_MSL_FA_FAST", EncodedValue(b"0"))
        assert environment.environment_snapshot() is os.environ
        assert cache.effective_policy()["FA_FAST"] is False
        state[0] = "1"
        assert os.environ.get("TRITON_MSL_FA_FAST") == "1"
        assert cache.effective_policy()["FA_FAST"] is True
        # A byte-equal foreign KEY also must not inherit an owned bytes proof.
        # Replace through the encoded dict: dict assignment to an equal key
        # would otherwise retain the original key object.
        raw_key = b"TRITON_MSL_FA_FAST"
        os.environ._data.pop(raw_key)
        os.environ._data[EncodedValue(raw_key)] = b"0"
        try:
            assert environment.environment_snapshot() is os.environ
        finally:
            os.environ._data.pop(raw_key)
            os.environ._data[raw_key] = b"0"


def test_preimport_overrides_are_not_certified_as_standard(monkeypatch):
    monkeypatch.setenv("SNAPSHOT_PREIMPORT_WITNESS", "snapshot-original")
    for mode in ("get", "getitem", "codec", "codec_closure"):
        with monkeypatch.context() as patch:
            expected = "snapshot-altered"
            if mode == "get":
                original = type(os.environ).get
                if "get" in vars(os.environ):
                    patch.delattr(os.environ, "get")
                patch.setattr(
                    type(os.environ),
                    "get",
                    lambda self, key, default=None: "snapshot-altered"
                    if key == "SNAPSHOT_PREIMPORT_WITNESS"
                    else original(self, key, default),
                )
            elif mode == "getitem":
                original = type(os.environ).__getitem__
                patch.setattr(
                    type(os.environ),
                    "__getitem__",
                    lambda self, key: "snapshot-altered"
                    if key == "SNAPSHOT_PREIMPORT_WITNESS"
                    else original(self, key),
                )
            elif mode == "codec":
                original = os.environ.decodevalue
                patch.setattr(
                    os.environ,
                    "decodevalue",
                    lambda value: "snapshot-altered" if value == b"snapshot-original" else original(value),
                )
            else:
                original = os.environ.encodekey
                if not hasattr(original, "__code__") or not original.__closure__:
                    # No recognized codec optimization exists in this context.
                    assert environment.environment_snapshot() is os.environ
                    continue

                def cell(value):
                    return (lambda: value).__closure__[0]

                borrowed = FunctionType(original.__code__, original.__globals__, closure=(cell("utf-16"),))
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


def test_rebound_identity_predicate_cannot_hide_changed_environment(monkeypatch):
    import operator

    for mode in ("live", "preimport"):
        with monkeypatch.context() as patch:
            patch.setenv("TRITON_MSL_FA_FAST", "0")
            if mode == "preimport":
                patch.setattr(operator, "is_", lambda *args: True)
                spec = importlib.util.spec_from_file_location("snapshot_identity_preimport", environment.__file__)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                module = environment
            first = module.environment_snapshot()
            assert first["TRITON_MSL_FA_FAST"] == "0"
            if mode == "live":
                patch.setattr(module, "is_", lambda *args: True)
            patch.setenv("TRITON_MSL_FA_FAST", "1")
            second = module.environment_snapshot()
            assert second["TRITON_MSL_FA_FAST"] == "1", mode
            if module.is_snapshot(second):
                assert second is not first, mode
