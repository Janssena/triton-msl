"""Observable comparison behavior shared by Python and optional native copies.

The release/helper receipt must also state which implementation these executed;
passing these on a Python-only installation is not native execution evidence.
"""

import math

import pytest

from triton_msl.backend import _launch_contract as c
from triton_msl.errors import MetalNonRecoverableError


@pytest.mark.parametrize("mode", ["same-nan", "same-custom", "comparison-error", "truth-error"])
def test_copysign_result_comparison_cannot_use_identity_shortcut(monkeypatch, mode):
    packed = [4, 1, 0, 32, [3], False, None, None, None, ["flash_attention", "x" * 1500, -0.0], None, None]
    expected = c._canonical(packed)
    assert c._packed_snapshot_plan(expected) is not None
    events = []
    sentinel = ValueError("comparison callback witness")

    class Truth:
        def __bool__(self):
            events.append("truth")
            if mode == "truth-error":
                raise sentinel
            return True

    class Different:
        def __ne__(self, other):
            assert other is self
            events.append("comparison")
            if mode == "comparison-error":
                raise sentinel
            return Truth()

    shared = float("nan") if mode == "same-nan" else Different()

    def copysign(a, b):
        events.append("copysign")
        return shared

    monkeypatch.setattr(math, "copysign", copysign)
    error_type = ValueError if mode.endswith("error") else MetalNonRecoverableError
    with pytest.raises(error_type) as raised:
        c.validate_packed_launch(packed, expected)
    if mode.endswith("error"):
        assert raised.value is sentinel
    assert events[:2] == ["copysign", "copysign"]
    if mode != "same-nan":
        assert events[2] == "comparison"
    if mode in ("same-custom", "truth-error"):
        assert events[3] == "truth"


@pytest.mark.parametrize(
    "mode",
    [
        "true",
        "false",
        "comparison-error",
        "truth-error",
        "truth-reads-finalization",
    ],
)
def test_comparison_temporary_lifetime_matches_original_python(monkeypatch, mode):
    """Use this interpreter's original order, not a CPython-version assumption."""
    outcomes = []
    for copy in (c._checked_packed_copy_python, c._checked_packed_copy):
        events = []
        sentinel = ValueError("lifetime callback witness")

        class Verdict:
            def __bool__(self):
                events.append("truth")
                if mode == "truth-error":
                    raise sentinel
                if mode == "truth-reads-finalization":
                    return "left.finalize" in events and "right.finalize" in events
                return mode != "false"

            def __del__(self):
                events.append("verdict.finalize")

        class Operand:
            def __init__(self, label):
                self.label = label

            def __ne__(self, other):
                events.append(self.label + ".compare." + other.label)
                if mode == "comparison-error":
                    raise sentinel
                return Verdict()

            def __del__(self):
                events.append(self.label + ".finalize")

        count = 0

        def copysign(a, b):
            nonlocal count
            label = "left" if count == 0 else "right"
            count += 1
            events.append(label + ".provider")
            return Operand(label)

        monkeypatch.setattr(math, "copysign", copysign)
        try:
            result = copy(-0.0, (float, -0.0))
            outcome = ("return", result)
        except Exception as error:
            outcome = (type(error), str(error))
            if mode.endswith("error"):
                assert error is sentinel
                # Do not retain callback frames through the witness exception.
                sentinel.__traceback__ = None
        outcomes.append((outcome, events.copy()))
        assert count == 2
        assert "left.compare.right" in events
        if mode == "truth-reads-finalization":
            assert outcome[0] is MetalNonRecoverableError
            assert events.index("left.finalize") < events.index("truth")
            assert events.index("right.finalize") < events.index("truth")
    assert outcomes[0] == outcomes[1]
