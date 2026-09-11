"""Packed descriptor contents cannot change between compilation and dispatch.

Independent literal record builder permits running the same pins before the
producer/consumer contract exists. These are CPU boundary witnesses, not GPU math.
"""
import json
from types import SimpleNamespace

import pytest
from triton.backends.compiler import GPUTarget

from triton_msl.backend import _cache_contract as contract, compiler, driver
from triton_msl.errors import MetalNonRecoverableError

FIELDS = ("num_warps", "num_ctas", "shared", "block_size", "output_arg_indices",
          "needs_2d_grid", "mm_two_kernel", "fast_matmul", "quant_matmul",
          "flash_attention", "batched_dot_bounds", "device_assert")


@pytest.fixture(autouse=True)
def controlled_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(contract, "toolchain_identity", lambda: "controlled-toolchain")
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))
    # Real compilation constructs the backend (and imports its module map)
    # before stamping a product. Do the same before synthesizing metadata:
    # otherwise the first constructor legitimately invalidates the old stamp
    # by loading native providers, masking the packed-field witness below.
    compiler.MetalBackend(GPUTarget("metal", "apple-m4", 32))


def _stamp():
    return json.dumps({"schema": 1, "source": contract.source_contract(),
                       "toolchain": contract.toolchain_identity()},
                      sort_keys=True, separators=(",", ":"))


def _metadata():
    data = dict(zip(FIELDS, (4, 1, 0, 128, [1], False, None, None, None, None, None, None)))
    data.update(name="packed_contract", execution_contract=_stamp())
    data["launch_contract"] = json.dumps(
        {"schema": 2, "kind": "metal-packed-launch", "name": data["name"],
         "execution_contract": data["execution_contract"], "fields": {f: data[f] for f in FIELDS}},
        sort_keys=True, separators=(",", ":"), allow_nan=False)
    return SimpleNamespace(**data)


def _launcher(metadata):
    src = SimpleNamespace(constants={}, signature={"X": "*fp32", "O": "*fp32"},
                          fn=SimpleNamespace(arg_names=["X", "O"]))
    return driver.MetalLauncher(src, metadata)


@pytest.mark.parametrize("field", FIELDS)
def test_restored_metadata_missing_any_packed_field_refuses(field):
    metadata = _metadata()
    delattr(metadata, field)
    with pytest.raises(MetalNonRecoverableError, match="packed launch contract"):
        compiler.MetalBackend(GPUTarget("metal", "apple-m4", 32)).pack_metadata(metadata)


@pytest.mark.parametrize("field,value", [
    ("name", "other_kernel"), ("block_size", 256), ("output_arg_indices", []),
    ("needs_2d_grid", True), ("num_ctas", True), ("shared", 4096),
    ("flash_attention", ["flash_attention", "wrong shader", 1024]),
    ("batched_dot_bounds", {"new": "unproved bounds"}),
    ("device_assert", {'schema': 1, 'messages': ['altered'], 'buffer_index': 2}),
])
def test_restored_descriptor_change_refuses_before_stash(field, value, monkeypatch):
    metadata = _metadata()
    setattr(metadata, field, value)
    events = []
    monkeypatch.setattr(compiler, "_load_stashed_msl", lambda *a: events.append("stash"))
    with pytest.raises(MetalNonRecoverableError, match="packed launch contract"):
        _launcher(metadata)
    assert not events


@pytest.mark.parametrize("damage", ["missing", "short", "long", "bool_alias", "output", "route"])
def test_live_tuple_change_refuses_before_hooks_and_runtime(damage, monkeypatch):
    metadata = _metadata()
    launcher = _launcher(metadata)
    packed = [getattr(metadata, f) for f in FIELDS]
    if damage == "missing":
        packed = None
    elif damage == "short":
        packed.pop()
    elif damage == "long":
        packed.append(None)
    elif damage == "bool_alias":
        packed[1] = True
    elif damage == "output":
        packed[4].append(0)
    else:
        packed[9] = ["flash_attention", "other shader", 1024]
    events = []
    monkeypatch.setattr(driver, "_get_utils", lambda: events.append("runtime"))
    with pytest.raises(MetalNonRecoverableError, match="packed launch contract"):
        launcher(1, 1, 1, None, None, packed, None, lambda _: events.append("hook"), None)
    assert not events


def test_current_packed_descriptor_reaches_runtime(monkeypatch):
    metadata = _metadata()
    launcher = _launcher(metadata)
    packed = compiler.MetalBackend(GPUTarget("metal", "apple-m4", 32)).pack_metadata(metadata)
    class ReachedRuntime(Exception):
        pass
    def runtime():
        raise ReachedRuntime
    monkeypatch.setattr(driver, "_get_utils", runtime)
    with pytest.raises(ReachedRuntime):
        pointer = SimpleNamespace(data_ptr=lambda: 0)
        launcher(1, 1, 1, None, None, packed, None, None, None, pointer, pointer)


def test_checked_snapshot_cannot_be_changed_through_caller_aliases():
    from triton_msl.backend import _launch_contract
    metadata = _metadata()
    expected = _launch_contract.validate_launch_metadata(metadata)
    packed = [getattr(metadata, f) for f in FIELDS]
    snapshot = _launch_contract.validate_packed_launch(packed, expected)
    packed[4].append(0)
    packed[9] = ["flash_attention", "another shader", 1024]
    assert snapshot[4] == [1] and snapshot[9] is None


@pytest.mark.parametrize("container", [None, (), "legacy", 1])
def test_unknown_metadata_container_refuses(container):
    from triton_msl.backend import _launch_contract
    with pytest.raises(MetalNonRecoverableError, match="packed launch contract"):
        _launch_contract.validate_launch_metadata(container)
