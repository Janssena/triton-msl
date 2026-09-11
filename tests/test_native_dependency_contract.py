"""Checked Mach-O dependency closure; fixtures are data, never executed."""
from pathlib import Path
import struct

import pytest

ARM64 = 0x0100000C
X86_64 = 0x01000007


def _command(cmd, text):
    fixed = 12 if cmd == 0x8000001C else 24
    body = text.encode() + b"\0"
    size = (fixed + len(body) + 7) & ~7
    return (struct.pack("<III", cmd, size, fixed) + bytes(fixed - 12) + body).ljust(size, b"\0")


def _image(*commands, cpu=ARM64):
    body = b"".join(commands)
    return struct.pack("<8I", 0xFEEDFACF, cpu, 0, 6, len(commands), len(body), 0, 0) + body


def test_parser_keeps_dependency_kinds_and_rpaths(tmp_path):
    from triton_msl.backend._native_dependencies import parse_image
    path = tmp_path / "image.dylib"
    path.write_bytes(_image(_command(0xC, "@rpath/one.dylib"),
                            _command(0x80000018, "/usr/lib/two.dylib"),
                            _command(0x8000001F, "@loader_path/three.dylib"),
                            _command(0x8000001C, "@loader_path/lib"),
                            _command(0xD, "@rpath/this.dylib")))
    image = parse_image(path, ARM64)
    assert image.dependencies == (("@rpath/one.dylib", False), ("/usr/lib/two.dylib", True),
                                  ("@loader_path/three.dylib", False))
    assert image.rpaths == ("@loader_path/lib",)
    assert image.install_name == "@rpath/this.dylib"
    # There is no well-defined full-name binding with two LC_ID_DYLIBs.
    path.write_bytes(_image(_command(0xD, "@rpath/one.dylib"), _command(0xD, "@rpath/two.dylib")))
    with pytest.raises(ValueError, match="duplicate.*install name"):
        parse_image(path, ARM64)
    data = bytearray(_image(_command(0xD, "@rpath/not-a-dylib")))
    struct.pack_into("<I", data, 12, 8)  # MH_BUNDLE is not MH_DYLIB
    path.write_bytes(data)
    with pytest.raises(ValueError, match="invalid.*install name"):
        parse_image(path, ARM64)


def test_fat_parser_selects_the_actual_cpu_slice(tmp_path):
    from triton_msl.backend._native_dependencies import parse_image
    a, b = _image(_command(0xC, "/usr/lib/arm.dylib")), _image(_command(0xC, "/usr/lib/x86.dylib"), cpu=X86_64)
    start = 8 + 40
    data = struct.pack(">II", 0xCAFEBABE, 2)
    data += struct.pack(">IIIII", X86_64, 0, start, len(b), 0)
    data += struct.pack(">IIIII", ARM64, 0, start + len(b), len(a), 0)
    path = tmp_path / "fat.so"
    path.write_bytes(data + b + a)
    assert parse_image(path, ARM64).dependencies == (("/usr/lib/arm.dylib", False),)


@pytest.mark.parametrize("damage", ["short_header", "short_commands", "zero_command", "bad_string_offset", "no_nul", "dyld_environment", "lazy_info"])
def test_unproved_native_load_commands_refuse(tmp_path, damage):
    from triton_msl.backend._native_dependencies import parse_image
    command = _command(0xC, "/usr/lib/lib.dylib")
    data = _image(command)
    if damage == "short_header": data = data[:12]
    elif damage == "short_commands": data = data[:-1]
    elif damage == "zero_command": data = data[:36] + bytes(4) + data[40:]
    elif damage == "bad_string_offset": data = data[:40] + struct.pack("<I", 10000) + data[44:]
    elif damage == "no_nul": data = data[:56] + b"x" * (len(data) - 56)
    elif damage == "dyld_environment": data = _image(_command(0x27, "DYLD_LIBRARY_PATH=/foreign"))
    else: data = _image(struct.pack("<4I", 0x3A, 16, 0, 0))
    path = tmp_path / "bad.so"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="native|Mach-O|load|command|string"):
        parse_image(path, ARM64)


def test_ambiguous_rpath_is_not_a_basename_guess(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_manifest
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "lib.dylib").write_bytes(_image() + name.encode())
    root = tmp_path / "root.so"
    root.write_bytes(_image(_command(0xC, "@rpath/lib.dylib"),
                           _command(0x8000001C, "@loader_path/a"),
                           _command(0x8000001C, "@loader_path/b")))
    with pytest.raises(ValueError, match="ambiguous"):
        dependency_manifest([root], cpu_type=ARM64, executable=root)


def test_absolute_external_provider_is_content_bound(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_manifest
    external = tmp_path / "external.dylib"
    external.write_bytes(_image() + b"first implementation")
    root = tmp_path / "root.so"
    root.write_bytes(_image(_command(0xC, str(external))))
    first = dependency_manifest([root], cpu_type=ARM64, executable=root)
    external.write_bytes(_image() + b"second implementation")
    assert dependency_manifest([root], cpu_type=ARM64, executable=root) != first


def test_loader_relative_dependency_and_cycle_are_bounded(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_manifest
    a, b = tmp_path / "a.so", tmp_path / "b.dylib"
    a.write_bytes(_image(_command(0xC, "@loader_path/b.dylib")))
    b.write_bytes(_image(_command(0xC, "@loader_path/a.so")))
    rows = dependency_manifest([a], cpu_type=ARM64, executable=a)
    assert len(rows) == 2


@pytest.mark.parametrize("reference", ["relative.dylib", "@unknown/lib.dylib", "@rpath/missing.dylib", "@loader_path/missing.dylib"])
def test_unknown_or_missing_strong_provider_refuses(tmp_path, reference):
    from triton_msl.backend._native_dependencies import dependency_manifest
    root = tmp_path / "root.so"
    root.write_bytes(_image(_command(0xC, reference)))
    with pytest.raises((ValueError, FileNotFoundError), match="native|dependency|rpath|provider"):
        dependency_manifest([root], cpu_type=ARM64, executable=root)


def test_system_framework_uses_the_separate_os_build_contract(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_manifest
    root = tmp_path / "root.so"
    root.write_bytes(_image(_command(0xC, "/System/Library/Frameworks/Metal.framework/Metal")))
    rows = dependency_manifest([root], cpu_type=ARM64, executable=root)
    assert rows[0][1] == (("/System/Library/Frameworks/Metal.framework/Metal", "system-build"),)


def test_framework_identity_includes_external_native_bytes(tmp_path, monkeypatch):
    from triton_msl.backend import _framework_contract as framework
    root = tmp_path / "package"
    root.mkdir()
    (root / "__init__.py").write_text("VERSION = 'unchanged'\n")
    external = tmp_path / "external.dylib"
    external.write_bytes(_image() + b"first")
    image = root / "plugin.so"
    image.write_bytes(_image(_command(0xC, str(external))))
    roots = {"triton": (root,)}
    monkeypatch.setattr(framework, "_snapshot", None)
    monkeypatch.setattr(framework, "_native_guard", None)
    monkeypatch.setattr(framework, "_discover_selection", lambda: (roots, {}))
    def native(_roots):
        from triton_msl.backend._native_dependencies import dependency_manifest
        return dependency_manifest([image], cpu_type=ARM64, executable=image)
    monkeypatch.setattr(framework, "_native_dependencies", native, raising=False)
    first = framework.framework_identity()
    external.write_bytes(_image() + b"second")
    monkeypatch.setattr(framework, "_snapshot", None)  # model the required restart
    assert framework.framework_identity() != first


def test_extension_supplies_inherited_rpath_to_its_dylib(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_manifest
    libs = tmp_path / "lib"
    libs.mkdir()
    entry, first, second = tmp_path / "core.so", libs / "first.dylib", libs / "second.dylib"
    entry.write_bytes(_image(_command(0xC, "@rpath/first.dylib"), _command(0x8000001C, "@loader_path/lib")))
    first.write_bytes(_image(_command(0xC, "@rpath/second.dylib")))
    second.write_bytes(_image())
    rows = dependency_manifest([entry], cpu_type=ARM64, executable=entry)
    assert len(rows) == 3
    assert rows[0][1] == (("@rpath/first.dylib", ("provider", 1)),)
    assert rows[1][1] == (("@rpath/second.dylib", ("provider", 2)),)
    # In isolation the same dylib is NOT given a guessed adjacent-directory path.
    with pytest.raises(ValueError, match="rpath"):
        dependency_manifest([first], cpu_type=ARM64, executable=entry)


def test_new_loaded_external_native_image_refuses_before_reusing_snapshot(tmp_path):
    from triton_msl.backend._framework_contract import _LoadedNativeGuard
    known, foreign = tmp_path / "known.so", tmp_path / "foreign.so"
    known.write_bytes(_image())
    foreign.write_bytes(_image() + b"different")
    names = [str(known).encode()]
    guard = object.__new__(_LoadedNativeGuard)
    guard.count = lambda: len(names)
    guard.name = lambda i: names[i]
    guard.providers = frozenset((known.resolve(),))
    guard.checked_count = None
    guard.approved_names = set()
    guard.verify()
    names.append(str(foreign).encode())
    with pytest.raises(RuntimeError, match="outside.*proved"):
        guard.verify()
    assert guard.checked_count == 1  # failure never certifies the new selection


def test_same_count_native_replacement_is_not_a_warm_identity_hit(tmp_path):
    from triton_msl.backend._framework_contract import _LoadedNativeGuard
    known, foreign = tmp_path / "known.so", tmp_path / "foreign.so"
    known.write_bytes(_image())
    foreign.write_bytes(_image() + b"other")
    names = [str(known).encode()]
    guard = object.__new__(_LoadedNativeGuard)
    guard.count, guard.name = lambda: len(names), lambda i: names[i]
    guard.providers = frozenset((known.resolve(),))
    guard.checked_count, guard.approved_names = None, set()
    guard.verify()
    names[0] = str(foreign).encode()
    with pytest.raises(RuntimeError, match="outside.*proved"):
        guard.verify()
    assert names[0] not in guard.approved_names


def test_python_entrypoints_are_not_all_packaged_dylibs_or_linux_resources(tmp_path):
    from triton_msl.backend._framework_contract import _native_entrypoints
    executable, extension = tmp_path / "Python", tmp_path / "core.so"
    executable.write_bytes(_image())
    extension.write_bytes(_image())
    (tmp_path / "dependency.dylib").write_bytes(_image())
    (tmp_path / "linux.so").write_bytes(b"\x7fELF" + b"Linux-only packaged resource")
    roots = {"python-executable": (executable,), "mlx": (tmp_path,)}
    assert set(_native_entrypoints(roots)) == {executable, extension}
    (tmp_path / "unknown.so").write_bytes(b"unproved native entry point")
    with pytest.raises(RuntimeError, match="unproved.*entry"):
        _native_entrypoints(roots)


def test_torch_explicit_ctypes_global_dependency_is_a_named_entrypoint(tmp_path):
    from triton_msl.backend._framework_contract import _native_entrypoints
    executable = tmp_path / "Python"
    executable.write_bytes(_image())
    package = tmp_path / "torch"
    (package / "lib").mkdir(parents=True)
    library = package / "lib/libtorch_global_deps.dylib"
    library.write_bytes(_image())
    (package / "lib/not_an_entry.dylib").write_bytes(_image())
    roots = {"python-executable": (executable,), "torch": (package,)}
    assert set(_native_entrypoints(roots)) == {executable, library}
