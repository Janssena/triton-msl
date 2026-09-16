"""Cold admission and warm decline must not replay import metadata callbacks."""

from importlib._bootstrap_external import _NamespacePath
from importlib.machinery import ModuleSpec
import os
import sys
import types

import pytest

from identity_framework_fixture import ordinary_framework
from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _framework_contract as fw
from triton_msl.backend import _identity_fast_path as fast
from triton_msl.errors import MetalNonRecoverableError
from test_identity_live_owners import _warm_contract


def _outcome(fn):
    try:
        return ("stamp", fn())
    except Exception as exc:
        return ("exception", type(exc).__name__, str(exc), type(exc.__cause__).__name__ if exc.__cause__ else None)


def test_real_namespace_fallback_keeps_single_full_iteration(monkeypatch):
    """No ordinary_framework fixture: preserve all other real loaded namespaces."""
    module = sys.modules["numpy"]
    spec = module.__spec__
    locations = tuple(spec.submodule_search_locations)
    namespace = _NamespacePath("numpy", locations, lambda *args: None)
    original_iter = _NamespacePath.__iter__
    results = []
    with monkeypatch.context() as m:
        m.setattr(spec, "submodule_search_locations", namespace)
        for mode in ("full", "fast"):
            fast._record = None
            cache.execution_contract()  # Initialize inventory outside the callback trace.
            fast._record = None
            calls = []

            def observed(self):
                if self is namespace:
                    calls.append("iter")
                return original_iter(self)

            with monkeypatch.context() as current:
                current.setattr(_NamespacePath, "__iter__", observed)
                if mode == "full":
                    current.setattr(fast, "build", lambda stamp: None)
                hits = fast._stats["hits"]
                outcome = _outcome(cache.execution_contract)
                assert fast._stats["hits"] == hits
            results.append((outcome, calls))
        assert results[0] == results[1]
        assert results[0][0][0] == "stamp" and results[0][1] == ["iter"]
        assert fast._record is None, "a real namespace cannot produce a usable native record"
    fast._record = None


@pytest.mark.usefixtures("ordinary_framework")
@pytest.mark.parametrize("stage", ["cold", "warm"])
@pytest.mark.parametrize(
    "kind",
    [
        "paths",
        "list-subclass",
        "tuple-subclass",
        "truth-error",
        "iteration-error",
        "policy-side-effect",
        "module-getter",
        "module-getattr",
        "spec-getter",
        "spec-descriptor",
    ],
)
def test_metadata_callback_trace_matches_complete_evaluator(monkeypatch, stage, kind):
    module = sys.modules["numpy"]
    spec = module.__spec__
    paths = tuple(spec.submodule_search_locations)
    results = []
    for mode in ("full", "fast"):
        _warm_contract()
        if stage == "cold" or mode == "full":
            fast._record = None
        calls = []
        with monkeypatch.context() as m:

            class Paths:
                def __bool__(self):
                    calls.append("bool")
                    if kind == "truth-error":
                        raise RuntimeError("locations truth witness")
                    if kind == "policy-side-effect":
                        m.setenv("TRITON_MSL_FA_FAST", "0" if calls.count("bool") == 1 else "1")
                    return True

                def __iter__(self):
                    calls.append("iter")
                    if kind == "iteration-error":
                        raise RuntimeError("locations iteration witness")
                    return iter(paths)

                def __len__(self):
                    calls.append("len")
                    return len(paths)

            class ListPaths(list):
                __bool__ = Paths.__bool__
                __iter__ = Paths.__iter__
                __len__ = Paths.__len__

            class TuplePaths(tuple):
                __bool__ = Paths.__bool__
                __iter__ = Paths.__iter__
                __len__ = Paths.__len__

            if kind in ("paths", "truth-error", "iteration-error", "policy-side-effect"):
                m.setattr(spec, "submodule_search_locations", Paths())
            elif kind in ("list-subclass", "tuple-subclass"):
                m.setattr(
                    spec,
                    "submodule_search_locations",
                    ListPaths(paths) if kind == "list-subclass" else TuplePaths(paths),
                )
            elif kind == "module-getter":

                class Module(types.ModuleType):
                    def __getattribute__(self, name):
                        if name == "__spec__":
                            calls.append("module.spec")
                        return super().__getattribute__(name)

                m.setattr(module, "__class__", Module)
            elif kind == "module-getattr":

                def missing(name):
                    if name == "__spec__":
                        calls.append("module.missing-spec")
                        return spec
                    raise AttributeError(name)

                m.delattr(module, "__spec__")
                m.setattr(module, "__getattr__", missing, raising=False)
            elif kind == "spec-getter":

                class Spec(ModuleSpec):
                    def __getattribute__(self, name):
                        if name in ("origin", "submodule_search_locations"):
                            calls.append("spec." + name)
                        return super().__getattribute__(name)

                m.setattr(spec, "__class__", Spec)
            else:

                def locations(self):
                    if self is spec:
                        calls.append("spec.locations-descriptor")
                    return self.__dict__["submodule_search_locations"]

                m.setattr(ModuleSpec, "submodule_search_locations", property(locations), raising=False)
            if mode == "full":
                m.setattr(fast, "build", lambda stamp: None)
            hits = fast._stats["hits"]
            actual = _outcome(cache.execution_contract)
            results.append((actual, list(calls), os.environ.get("TRITON_MSL_FA_FAST")))
            assert fast._stats["hits"] == hits
            assert fast._record is None, "unsupported owners/locations must not be recorded"
        fast._record = None
    assert results[0] == results[1], (stage, kind, results)
    assert results[0][1], "positive observer control did not execute"


@pytest.mark.usefixtures("ordinary_framework")
def test_live_replaced_spec_dictionary_cannot_certify_old_paths(monkeypatch, tmp_path):
    stamp = _warm_contract()
    spec = sys.modules["numpy"].__spec__
    replacement = dict(spec.__dict__)
    replacement["submodule_search_locations"] = [str(tmp_path)]
    with monkeypatch.context() as m:
        m.setattr(spec, "__dict__", replacement)
        hits = fast._stats["hits"]
        with pytest.raises(MetalNonRecoverableError, match="framework selection changed"):
            cache.validate_execution_contract(stamp)
        assert fast._stats["hits"] == hits
    fast._record = None


@pytest.mark.usefixtures("ordinary_framework")
def test_cold_spec_collision_key_is_not_observed_twice(monkeypatch):
    spec = sys.modules["numpy"].__spec__
    original = spec.__dict__
    outcomes = []
    for mode in ("full", "fast"):
        _warm_contract()
        calls = []

        class Key(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                calls.append(other)
                return str.__eq__(self, other)

        replacement = {k: v for k, v in original.items() if k != "origin"}
        replacement[Key("origin")] = original["origin"]
        with monkeypatch.context() as m:
            m.setattr(spec, "__dict__", replacement)
            fast._record = None
            if mode == "full":
                m.setattr(fast, "build", lambda stamp: None)
            outcome = _outcome(cache.execution_contract)
            outcomes.append((outcome, list(calls)))
            assert fast._record is None
        fast._record = None
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0][0] == "stamp"
    assert outcomes[0][1] == ["origin"], "exactly one complete-evaluator observation"
