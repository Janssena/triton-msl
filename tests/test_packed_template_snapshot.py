"""Large template records preserve the JSON boundary and per-call ownership."""

import json

import pytest

from triton_msl.backend._launch_contract import validate_packed_launch
from triton_msl.errors import MetalNonRecoverableError


def _record(value):
    return [4, 1, 0, 128, [1], False, None, None, None, ["template", "shader text\n" * 128, value], None, None]


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@pytest.mark.parametrize(
    "wanted,changed",
    [
        (1, True),
        (1, 1.0),
        (0.0, -0.0),
        (-0.0, 0.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        ({"size": [16]}, {"size": [17]}),
        ("same-size-a", "same-size-b"),
    ],
)
def test_large_record_rejects_json_distinct_aliases(wanted, changed):
    packed = _record(wanted)
    expected = _encode(packed)
    packed[9][2] = changed
    with pytest.raises(MetalNonRecoverableError, match="packed launch contract"):
        validate_packed_launch(packed, expected)


def test_large_record_owns_each_return_before_caller_hook_mutation():
    packed = _record({"bounds": [0, 64], "nested": [[1]]})
    expected = _encode(packed)
    first = validate_packed_launch(packed, expected)
    second = validate_packed_launch(packed, expected)
    # A hook holds the live caller record; another user can hold an old return.
    packed[9][2]["bounds"][1] = 65
    first[9][2]["nested"][0][0] = 9
    assert _encode(second) == expected
    assert second[9] is not first[9] and second[9][2] is not first[9][2]
    with pytest.raises(MetalNonRecoverableError):
        validate_packed_launch(packed, expected)


def test_large_record_preserves_tuple_list_and_dictionary_key_semantics():
    packed = _record({"1": (True, 1, 1.0, -0.0)})
    expected = _encode(packed)
    snapshot = validate_packed_launch(tuple(packed), expected)
    assert type(snapshot[9][2]["1"]) is list and _encode(snapshot) == expected
    # JSON stringifies numeric object keys. Keep its original path for these.
    packed[9][2] = {1: [True, 1, 1.0, -0.0]}
    assert _encode(validate_packed_launch(packed, expected)) == expected


def test_large_custom_values_keep_original_json_encoder_semantics():
    class ReorderedList(list):
        def __iter__(self):
            return iter([1, 2])

    class Integer(int):
        pass

    packed = _record([ReorderedList([9, 8]), Integer(3)])
    expected = _encode(packed)
    assert _encode(validate_packed_launch(packed, expected)) == expected


@pytest.mark.parametrize("value", ["\ud83d\ude00", {"\ud83d\ude00": 1}])
def test_large_record_preserves_json_surrogate_pair_equivalence(value):
    packed = _record(value)
    expected = _encode(packed)
    assert _encode(validate_packed_launch(packed, expected)) == expected


@pytest.mark.parametrize("change", ["whitespace", "float-spelling", "duplicate-key"])
def test_large_record_never_admits_noncanonical_expected_strings(change):
    packed = _record({"a": 1.0})
    expected = _encode(packed)
    if change == "whitespace":
        expected = " " + expected
    elif change == "float-spelling":
        expected = expected.replace('"a":1.0', '"a":1e0')
    else:
        expected = expected.replace('"a":1.0', '"a":0,"a":1.0')
    with pytest.raises(MetalNonRecoverableError):
        validate_packed_launch(packed, expected)


def test_deep_record_keeps_original_json_support():
    value = 1
    for _ in range(80):
        value = [value]
    packed = _record(value)
    expected = _encode(packed)
    assert _encode(validate_packed_launch(packed, expected)) == expected
