"""New composition regressions; compare exact retained Python recursion semantics."""

import types
import pytest
from triton_msl.backend import _launch_contract as c


def namespace(arm):
    ns = dict(vars(c))
    original = types.FunctionType(c._checked_packed_copy_python.__code__, ns)
    ns["_checked_packed_copy_python"] = original
    if arm == "python":
        fn = original
    else:
        # Match the selected wrapper's globals, without a closure or hidden root.
        fn = types.FunctionType(c._checked_packed_copy.__code__, ns)
    ns["_checked_packed_copy"] = fn
    return ns, fn


@pytest.mark.parametrize("container", ["list", "dict"])
@pytest.mark.parametrize("change", ["replace", "code", "remove", "raise", "fallback"])
@pytest.mark.parametrize("when", [1, 2])
def test_recursive_helper_is_selected_after_each_callback(container, change, when):
    if c.PACKED_COPY_IMPLEMENTATION != "native":
        pytest.skip("native-specific differential")
    outcomes = []
    for arm in ("python", "native"):
        ns, fn = namespace(arm)
        events = []
        sentinel = RuntimeError("chosen callback error")

        def replacement(value, plan):
            events.append(("replacement", value))
            if change == "raise":
                raise sentinel
            if change == "fallback":
                raise c._UsePackedJSON
            return 99

        # Code replacement must have the original function's zero closure size.
        ns["_replacement_provider"] = replacement
        exec("def code_replacement(value, plan):\n    return _replacement_provider(value, plan)\n", ns)

        def copysign(a, b):
            events.append(("copysign", b))
            if len(events) == when:
                if change == "code":
                    fn.__code__ = ns["code_replacement"].__code__
                elif change == "remove":
                    del ns["_checked_packed_copy"]
                else:
                    ns["_checked_packed_copy"] = replacement
            return -1.0

        ns["math"] = types.SimpleNamespace(copysign=copysign)
        if container == "list":
            value, plan = [-0.0, 7], (list, ((float, -0.0), (int, 7)))
        else:
            value, plan = {"a": -0.0, "b": 7}, (dict, (("a", (float, -0.0)), ("b", (int, 7))))
        try:
            result = fn(value, plan)
            outcome = ("return", result)
        except Exception as exc:
            outcome = ("error", type(exc), str(exc), exc is sentinel)
            exc.__traceback__ = None
        outcomes.append((outcome, events))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize(
    "name,value,plan",
    [
        ("_UsePackedJSON", object(), (str, "expected")),
        ("_refuse", 2, (int, 1)),
        ("_checked_packed_copy", [1], (list, ((int, 1),))),
    ],
)
def test_provider_lookup_preserves_exact_collision_exception(name, value, plan):
    if c.PACKED_COPY_IMPLEMENTATION != "native":
        pytest.skip("native-specific differential")
    outcomes = []
    for arm in ("python", "native"):
        ns, fn = namespace(arm)
        sentinel = RuntimeError("chosen live lookup error")
        events = []

        class Collision:
            def __hash__(self):
                return hash(name)

            def __eq__(self, other):
                events.append(other)
                raise sentinel

        del ns[name]
        ns[Collision()] = object()
        with pytest.raises(RuntimeError) as exc:
            fn(value, plan)
        assert exc.value is sentinel
        outcomes.append(events)
    assert outcomes[0] == outcomes[1] == [name]


def test_native_only_python_leaf_dependency_preserves_lookup_error():
    if c.PACKED_COPY_IMPLEMENTATION != "native":
        pytest.skip("native-specific control")
    ns, fn = namespace("native")
    name = "_checked_packed_copy_python"
    sentinel = RuntimeError("chosen native dependency error")

    class Collision:
        def __hash__(self):
            return hash(name)

        def __eq__(self, other):
            raise sentinel

    del ns[name]
    ns[Collision()] = object()
    with pytest.raises(RuntimeError) as exc:
        fn(-0.0, (float, -0.0))
    assert exc.value is sentinel
