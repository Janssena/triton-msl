"""Already-loaded providers are hashed inputs, never library-name exemptions."""
from pathlib import Path
import struct

import pytest


def image(*commands, payload=b""):
    body = b"".join(commands)
    return struct.pack("<8I", 0xFEEDFACF, 0x0100000C, 0, 6, len(commands), len(body), 0, 0) + body + payload


def dependency(name):
    text = name.encode() + b"\0"
    size = (24 + len(text) + 7) & ~7
    return (struct.pack("<6I", 0xC, size, 24, 0, 0, 0) + text).ljust(size, b"\0")


@pytest.fixture
def ambient(tmp_path, monkeypatch):
    from triton_msl.backend import _framework_contract as module

    executable = tmp_path / "Python"
    executable.write_bytes(image(payload=b"interpreter"))
    first, second = tmp_path / "unlisted_one.so", tmp_path / "unlisted_two.so"
    first.write_bytes(image(payload=b"one"))
    second.write_bytes(image(payload=b"two"))
    names = [str(executable).encode(), str(first).encode()]
    roots = {"python-executable": (executable,)}
    original_guard = module._LoadedNativeGuard

    def guard(providers):
        result = object.__new__(original_guard)
        result.count = lambda: len(names)
        result.name = lambda index: names[index]
        result.providers = providers
        result.approved_names = set()
        result.checked_count = None
        return result

    monkeypatch.setattr(module, "_LoadedNativeGuard", guard)
    monkeypatch.setattr(module, "_loaded_native_images", lambda: tuple(names), raising=False)
    monkeypatch.setattr(module, "_native_guard", None)
    return module, roots, names, first, second


def test_preloaded_unlisted_provider_is_inventoried(ambient):
    module, roots, _, first, _ = ambient
    result = module._native_dependencies(roots)
    assert result
    assert first.resolve() in module._native_guard.providers
    module._native_guard.verify()


def test_preloaded_provider_bytes_change_identity(ambient):
    module, roots, _, first, _ = ambient
    before = module._native_dependencies(roots)
    first.write_bytes(image(payload=b"replacement"))
    assert module._native_dependencies(roots) != before


def test_loaded_provider_order_participates_in_identity(ambient):
    module, roots, names, _, second = ambient
    names.append(str(second).encode())
    before = module._native_dependencies(roots)
    names[1:] = reversed(names[1:])
    assert module._native_dependencies(roots) != before


def test_late_uninventoried_provider_still_refuses(ambient):
    module, roots, names, _, second = ambient
    module._native_dependencies(roots)
    names.append(str(second).encode())
    with pytest.raises(RuntimeError, match="outside.*proved"):
        module._native_guard.verify()
    assert str(second).encode() not in module._native_guard.approved_names


@pytest.mark.parametrize("damage", ["malformed", "missing_dependency", "missing_file"])
def test_ambient_provider_requires_a_complete_native_proof(ambient, damage):
    module, roots, _, first, _ = ambient
    if damage == "malformed":
        first.write_bytes(b"not Mach-O")
    elif damage == "missing_dependency":
        first.write_bytes(image(dependency("@loader_path/absent.dylib")))
    else:
        first.unlink()
    # The old graph also safely refused these inputs, at its coverage check.
    # A different private exception class must not manufacture a pre-fix bite.
    with pytest.raises((ValueError, OSError, RuntimeError), match="native|Mach-O|dependency|No such file"):
        module._native_dependencies(roots)


def test_native_load_during_inventory_is_not_silently_admitted(ambient, monkeypatch):
    from triton_msl.backend import _native_dependencies as native
    module, roots, names, _, second = ambient
    original = native.dependency_graph
    def graph(*args, **kwargs):
        result = original(*args, **kwargs)
        names.append(str(second).encode())
        return result
    monkeypatch.setattr(native, "dependency_graph", graph)
    with pytest.raises(RuntimeError, match="changed during|outside.*proved"):
        module._native_dependencies(roots)


def test_loaded_dependency_outside_package_is_content_bound(ambient):
    module, roots, names, first, second = ambient
    first.write_bytes(image(dependency(str(second))))
    names.append(str(second).encode())
    before = module._native_dependencies(roots)
    second.write_bytes(image(payload=b"new external dependency"))
    assert module._native_dependencies(roots) != before


def test_loaded_install_name_is_a_provider_not_a_guessed_search_directory(ambient):
    from test_native_dependency_contract import _command
    module, roots, names, first, second = ambient
    # A wheel can carry an obsolete build-time RPATH yet legitimately bind an
    # already-loaded dylib by its full LC_ID_DYLIB name (e.g. torchvision).
    first.write_bytes(image(dependency("@rpath/exact/provider.dylib"),
                            _command(0x8000001C, "/nonexistent/build/lib")))
    second.write_bytes(image(_command(0xD, "@rpath/exact/provider.dylib"), payload=b"v1"))
    names.append(str(second).encode())
    before = module._native_dependencies(roots)
    assert {first.resolve(), second.resolve()} <= module._native_guard.providers
    second.write_bytes(image(_command(0xD, "@rpath/exact/provider.dylib"), payload=b"v2"))
    assert module._native_dependencies(roots) != before


@pytest.mark.parametrize("damage", ["unloaded", "different_full_name", "duplicate_install_name"])
def test_loaded_install_name_near_misses_do_not_supply_a_provider(ambient, damage):
    from test_native_dependency_contract import _command
    module, roots, names, first, second = ambient
    first.write_bytes(image(dependency("@rpath/exact/provider.dylib")))
    name = "@rpath/other/provider.dylib" if damage == "different_full_name" else "@rpath/exact/provider.dylib"
    second.write_bytes(image(_command(0xD, name)))
    if damage != "unloaded":
        names.append(str(second).encode())
    if damage == "duplicate_install_name":
        other = second.with_name("third.dylib")
        other.write_bytes(image(_command(0xD, name), payload=b"other implementation"))
        names.append(str(other).encode())
    with pytest.raises(ValueError, match="unproved|missing|ambiguous"):
        module._native_dependencies(roots)


def test_same_initial_native_graph_is_repeatable(ambient):
    module, roots, _, _, _ = ambient
    assert module._native_dependencies(roots) == module._native_dependencies(roots)


def test_initial_order_is_bound_even_for_already_known_providers(ambient):
    module, roots, names, first, second = ambient
    roots.update({"first": (first,), "second": (second,)})
    names.append(str(second).encode())
    before = module._native_dependencies(roots)
    names[1:] = reversed(names[1:])
    assert module._native_dependencies(roots) != before


def test_real_regex_import_before_first_snapshot_is_supported():
    """Exercise the actual extension which broke the randomized project gate."""
    import importlib.util
    import subprocess
    import sys
    import triton_msl

    if sys.platform != "darwin" or importlib.util.find_spec("regex") is None:
        pytest.skip("requires Darwin and the optional regex package")
    root = Path(triton_msl.__file__).resolve().parent.parent
    program = """
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import triton_msl
assert Path(triton_msl.__file__).resolve().parent == root / 'triton_msl'
import regex
from triton_msl.backend import _framework_contract as contract
before = contract.framework_identity()
assert Path(regex._regex.__file__).resolve() in contract._native_guard.providers
assert before == contract.framework_identity()
print('REGEX_NATIVE_BOUND_IDENTITY_VERIFIED', triton_msl.__file__)
"""
    result = subprocess.run([sys.executable, "-c", program, str(root)],
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REGEX_NATIVE_BOUND_IDENTITY_VERIFIED" in result.stdout


def test_real_torchvision_import_rekeys_without_losing_torch_providers():
    """Real wheel/import sequence which the 301 focused slice did not cover."""
    import importlib.util
    import subprocess
    import sys
    import triton_msl

    if sys.platform != "darwin" or importlib.util.find_spec("torchvision") is None:
        pytest.skip("requires Darwin and the optional torchvision package")
    root = Path(triton_msl.__file__).resolve().parent.parent
    program = """
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import triton
import torch
import triton_msl
assert Path(triton_msl.__file__).resolve().parent == root / 'triton_msl'
from triton_msl.backend import _framework_contract as contract
before = contract.framework_identity()
import torchvision
after = contract.framework_identity()
assert after != before
assert after == contract.framework_identity()
providers = contract._native_guard.providers
assert (Path(torchvision.__file__).parent / '_C.so').resolve() in providers
assert (Path(torch.__file__).parent / 'lib/libc10.dylib').resolve() in providers
print('TORCHVISION_NATIVE_BOUND_IDENTITY_VERIFIED', triton_msl.__file__)
"""
    result = subprocess.run([sys.executable, "-c", program, str(root)],
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TORCHVISION_NATIVE_BOUND_IDENTITY_VERIFIED" in result.stdout
