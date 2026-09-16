"""The cheaper native call protocol keeps live lookup and allocation semantics."""

import types
import sys

import pytest

from triton_msl.backend import _launch_contract as c


pytestmark = pytest.mark.skipif(c.PACKED_COPY_IMPLEMENTATION != "native", reason="requires the optional native copier")


@pytest.mark.parametrize("nargs", [0, 1, 2, 4, 8])
def test_native_arity_error_keeps_original_parser_wording(nargs):
    with pytest.raises(TypeError) as error:
        c._packed_native.copy(*([None] * nargs))
    assert str(error.value) == f"copy() takes exactly 3 arguments ({nargs} given)"


def test_key_reuse_does_not_cache_the_provider_or_its_exception():
    ns = dict(vars(c))
    wrapper = types.FunctionType(c._checked_packed_copy.__code__, ns)
    ns["_checked_packed_copy"] = wrapper
    value, plan = [7], (list, ((int, 7),))
    assert wrapper(value, plan) == [7]
    calls = []
    sentinel = RuntimeError("live provider failure after a warm hit")

    def replacement(value, plan):
        calls.append((value, plan))
        raise sentinel

    ns["_checked_packed_copy"] = replacement
    with pytest.raises(RuntimeError) as error:
        wrapper(value, plan)
    assert error.value is sentinel
    assert calls == [(7, (int, 7))]
    ns["_checked_packed_copy"] = wrapper
    assert wrapper(value, plan) == [7]


def test_repeated_copies_own_every_nested_container():
    value = [[7], {"stride": [1, 32]}]
    plan = (list, ((list, ((int, 7),)), (dict, (("stride", (list, ((int, 1), (int, 32)))),))))
    first = c._checked_packed_copy(value, plan)
    second = c._checked_packed_copy(value, plan)
    assert first == second == value
    for root in (value, first):
        assert second is not root
        assert second[0] is not root[0]
        assert second[1] is not root[1]
        assert second[1]["stride"] is not root[1]["stride"]
    first[1]["stride"][0] = -99
    value[0][0] = -88
    assert second == [[7], {"stride": [1, 32]}]


@pytest.mark.parametrize(
    "name,value,plan,provider",
    [
        ("_checked_packed_copy", [7], (list, ((int, 7),)), lambda value, plan: value),
        ("_UsePackedJSON", object(), (str, "expected"), c._UsePackedJSON),
        ("_refuse", 2, (int, 1), lambda message: "replacement"),
        ("_checked_packed_copy_python", -0.0, (float, -0.0), lambda value, plan: value),
    ],
)
def test_lookup_keeps_fresh_noninterned_callback_argument(name, value, plan, provider):
    interned = sys.intern(name)
    observed = []

    class Key:
        def __hash__(self):
            return hash(interned)

        def __eq__(self, other):
            observed.append(other)
            if other is interned:
                raise RuntimeError("interned key observed")
            return other == interned

    namespace = {Key(): provider}
    for _ in range(2):
        try:
            c._packed_native.copy(value, plan, namespace)
        except c._UsePackedJSON:
            assert name == "_UsePackedJSON"
    assert observed == [name, name]
    assert observed[0] is not observed[1]
    assert all(key is not interned for key in observed)
