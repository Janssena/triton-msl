"""Derived stamp reuse never replaces live checks or crosses invocation boundaries."""
import json
import os
import sys

import pytest

from triton_msl.backend import _cache_contract as cache
from triton_msl.backend import _environment_snapshot as env
from triton_msl.backend import _toolchain_contract as tc
from triton_msl.errors import MetalNonRecoverableError


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(cache, "implementation_identity", lambda: "controlled-package")
    monkeypatch.setattr(cache, "framework_identity", lambda: "controlled-framework")
    # The parent has no derived-stamp cache; allow the mechanism pin to reach
    # its actual duplicate-capture witness there instead of failing at setup.
    if hasattr(cache, "_execution_snapshot"):
        monkeypatch.setattr(cache, "_execution_snapshot", None)
    monkeypatch.setattr(tc, "_snapshot", (tuple(os.environ.get(k) for k in tc._SELECTION), "controlled-toolchain"))


def literal():
    return json.dumps({"schema": 1, "source": cache.source_contract(),
                       "toolchain": cache.toolchain_identity()}, sort_keys=True, separators=(",", ":"))


def test_single_capture_reuses_only_after_live_checks(inputs, monkeypatch):
    events = []
    original = env.environment_snapshot
    def framework():
        events.append("framework/native")
        return "controlled-framework"
    monkeypatch.setattr(cache, "framework_identity", framework)
    owned = env.is_snapshot(original())
    def observe(frame, event, arg):
        if event == "call" and frame.f_code is original.__code__:
            events.append("environment")
    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        first = cache.execution_contract()
        events.clear()
        assert cache.execution_contract() == first
    finally:
        sys.setprofile(previous)
    assert events == (["framework/native", "environment"] if owned else
                      ["framework/native", "environment", "environment"])
    assert first == literal()
    if owned:
        assert cache._execution_snapshot[6] is first
        # Warm hits do not re-encode the stamp; live checks/policy still run.
        def forbidden(*args):
            pytest.fail("unchanged owned inputs rebuilt the derived stamp")
        monkeypatch.setattr(cache, "_encode_execution", forbidden)
        assert cache.execution_contract() is first
        monkeypatch.setenv("DYLD_UNRECOGNIZED_EXECUTION_TEST", "new-input")
        with pytest.raises(MetalNonRecoverableError, match="DYLD_UNRECOGNIZED_EXECUTION_TEST"):
            cache.execution_contract()


def test_changes_type_aliases_and_failed_checks_stay_live(inputs, monkeypatch):
    import triton_msl
    first = cache.execution_contract()
    for name in cache._STAMP_FLAGS:
        key = "TRITON_MSL_" + name
        with monkeypatch.context() as patch:
            patch.setenv(key, "0")
            zero = cache.execution_contract()
            assert zero == literal()
            patch.setenv(key, "1")
            one = cache.execution_contract()
            assert one == literal() and one != zero
    for target, key, value in ((cache, "SOURCE_SCHEMA", True),
                               (cache, "SOURCE_SCHEMA", 3.0),
                               (triton_msl, "CODEGEN_VERSION", "alternate-label")):
        with monkeypatch.context() as patch:
            patch.setattr(target, key, value)
            assert cache.execution_contract() == literal()
            assert cache.execution_contract() != first
    with monkeypatch.context() as patch:
        patch.setenv("TRITON_MSL_CPP_SKIP", "changed")
        assert cache.execution_contract() == literal() != first
    for provider in ("implementation_identity", "framework_identity"):
        with monkeypatch.context() as patch:
            patch.setattr(cache, provider, lambda: "changed-input")
            assert cache.execution_contract() == literal() != first
    with monkeypatch.context() as patch:
        patch.setattr(tc, "_snapshot", (tc._snapshot[0], "changed-compiler"))
        assert cache.execution_contract() == literal() != first
    # A provider may return mutable/custom JSON input. Identity alone is not
    # enough; these values must always take the original encoder path.
    with monkeypatch.context() as patch:
        mutable = ["before"]
        patch.setattr(cache, "implementation_identity", lambda: mutable)
        before = cache.execution_contract()
        mutable[0] = "after"
        assert cache.execution_contract() == literal() != before
    def failed():
        raise MetalNonRecoverableError("live native check failed")
    monkeypatch.setattr(cache, "framework_identity", failed)
    with pytest.raises(MetalNonRecoverableError, match="live native check failed"):
        cache.execution_contract()


def test_framework_side_effect_precedes_capture_and_foreign_mapping_stays_live(inputs, monkeypatch):
    def framework():
        os.environ["TRITON_MSL_FA_FAST"] = "0"
        return "controlled-framework"
    monkeypatch.setenv("TRITON_MSL_FA_FAST", "1")
    monkeypatch.setattr(cache, "framework_identity", framework)
    value = json.loads(cache.execution_contract())
    assert value["source"]["policy"]["FA_FAST"] is False
    assert cache.execution_contract() == literal()
    monkeypatch.setattr(cache, "framework_identity", lambda: "controlled-framework")
    foreign = dict(os.environ)
    monkeypatch.setattr(os, "environ", foreign)
    first = cache.execution_contract()
    foreign["TRITON_MSL_FA_FAST"] = "1"
    assert cache.execution_contract() == literal() != first
    foreign["DYLD_FOREIGN_EXECUTION_TEST"] = "injection"
    with pytest.raises(MetalNonRecoverableError, match="DYLD_FOREIGN_EXECUTION_TEST"):
        cache.execution_contract()
    del foreign["DYLD_FOREIGN_EXECUTION_TEST"]
    class StatefulGetter(dict):
        calls = 0
        @property
        def get(self):
            self.calls += 1
            count = self.calls
            return lambda key, default=None: ('1' if count % 2 else '0') if key == 'TRITON_MSL_FA_FAST' else dict.get(self, key, default)
    # Compare against the literal original call sequence on independent maps.
    # An extra ownership probe invokes this getter and changes a real flag.
    reference = StatefulGetter(foreign)
    monkeypatch.setattr(os, "environ", reference)
    expected = literal()
    candidate = StatefulGetter(foreign)
    monkeypatch.setattr(os, "environ", candidate)
    assert cache.execution_contract() == expected
    assert candidate.calls == reference.calls


def test_second_invocation_and_selector_are_not_waived(inputs, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", "0")
    first = cache.execution_contract()
    assert cache.validate_execution_contract(first) == first
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", "1")
    with pytest.raises(MetalNonRecoverableError, match="execution contract"):
        cache.validate_execution_contract(first)
    monkeypatch.setenv("DEVELOPER_DIR", "/different-selector")
    with pytest.raises(MetalNonRecoverableError, match="toolchain selection changed"):
        cache.execution_contract()


def test_callbacks_select_remaining_providers_without_replaying_prefix(inputs, monkeypatch):
    original_policy = cache.effective_policy
    original_compiler = cache.toolchain_identity
    original_native_compiler = tc.toolchain_identity
    original_capture = env.environment_snapshot
    for mode in ("framework_policy", "framework_compiler", "framework_native_compiler",
                 "policy_compiler", "capture_compiler", "capture_changes_environment"):
        results = []
        for run in (literal, cache.execution_contract):
            with monkeypatch.context() as patch:
                events = []
                patch.setattr(cache, "effective_policy", original_policy)
                patch.setattr(cache, "toolchain_identity", original_compiler)
                patch.setattr(tc, "toolchain_identity", original_native_compiler)
                patch.setattr(env, "environment_snapshot", original_capture)
                patch.setattr(cache, "_execution_snapshot", None, raising=False)
                patch.delenv("DYLD_CALLBACK_EXECUTION_TEST", raising=False)
                def compiler():
                    events.append("changed_compiler")
                    return "callback-compiler"
                def policy():
                    events.append("changed_policy")
                    if mode == "policy_compiler":
                        patch.setattr(cache, "toolchain_identity", compiler)
                    return {"callback-selected-policy": True}
                def framework():
                    events.append("framework")
                    if mode in ("framework_policy", "policy_compiler"):
                        patch.setattr(cache, "effective_policy", policy)
                    elif mode == "framework_compiler":
                        patch.setattr(cache, "toolchain_identity", compiler)
                    elif mode == "framework_native_compiler":
                        patch.setattr(tc, "toolchain_identity", compiler)
                    return "controlled-framework"
                def capture():
                    events.append("capture")
                    value = original_capture()
                    if mode == "capture_compiler":
                        patch.setattr(cache, "toolchain_identity", compiler)
                    else:
                        patch.setenv("DYLD_CALLBACK_EXECUTION_TEST", "injected-after-capture")
                    return value
                patch.setattr(cache, "framework_identity", framework)
                if mode.startswith("capture_"):
                    patch.setattr(env, "environment_snapshot", capture)
                try:
                    outcome = ("stamp", run())
                except MetalNonRecoverableError as exc:
                    outcome = ("refusal", str(exc))
                results.append((outcome, events))
        assert results[0] == results[1], (mode, results)
        assert results[1][1].count("framework") == 1
