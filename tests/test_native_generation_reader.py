"""Own-process ABI, fail-closed states, and real same-count dlclose/dlopen."""

import ctypes as C
import json
import os
import platform
import subprocess
import sys

import pytest


def _info():
    from triton_msl.backend._native_generation import _ImageInfosPrefix

    info = _ImageInfosPrefix()
    info.version = 17
    info.infoArray = 4096
    info.infoArrayCount = 2
    info.infoArrayChangeTimestamp = 100
    info.dyldAllImageInfosAddress = C.addressof(info)
    return info


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 99),
        ("dyldAllImageInfosAddress", 1),
        ("infoArray", 0),
        ("infoArrayCount", 0),
        ("infoArrayCount", 1_000_001),
        ("infoArrayChangeTimestamp", 0),
    ],
)
def test_invalid_generation_state_cannot_be_read_as_stable(field, value):
    from triton_msl.backend._native_generation import _GenerationReader

    info = _info()
    reader = _GenerationReader(info, C.addressof(info))
    setattr(info, field, value)
    with pytest.raises(RuntimeError):
        reader()


@pytest.mark.parametrize("version", [0, 1, 14, 19, 2**32 - 1])
def test_unknown_prefix_version_is_not_certified(version):
    from triton_msl.backend._native_generation import _GenerationReader

    info = _info()
    info.version = version
    with pytest.raises(RuntimeError, match="version"):
        _GenerationReader(info, C.addressof(info))


def test_generation_token_tracks_timestamp_pointer_and_count():
    from triton_msl.backend._native_generation import _GenerationReader

    info = _info()
    reader = _GenerationReader(info, C.addressof(info))
    assert reader() == (100, 4096, 2)
    info.infoArrayChangeTimestamp = 101
    assert reader() == (101, 4096, 2)
    info.infoArray = 8192
    info.infoArrayCount = 3
    assert reader() == (101, 8192, 3)


@pytest.mark.parametrize(
    "result", [(0, 368, 1), (-8, 368, 1), (1 << 64, 368, 1), (4097, 368, 1), (4096, 191, 1), (4096, 368, 0)]
)
def test_unproved_task_info_retains_full_scan(monkeypatch, result):
    from triton_msl.backend import _native_generation as module

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module, "_task_image_info", lambda: result)
    assert module.create_generation_reader() is None


def test_unavailable_task_info_retains_full_scan(monkeypatch):
    from triton_msl.backend import _native_generation as module

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")

    def unavailable():
        raise RuntimeError("unavailable own-process query")

    monkeypatch.setattr(module, "_task_image_info", unavailable)
    assert module.create_generation_reader() is None


@pytest.mark.skipif(sys.platform != "darwin" or platform.machine() != "arm64", reason="Apple arm64 ABI")
def test_prefix_offsets_match_the_selected_sdk(tmp_path):
    from triton_msl.backend._native_generation import _ImageInfosPrefix

    source, executable = tmp_path / "offsets.c", tmp_path / "offsets"
    source.write_text("""#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <mach/task_info.h>
#include <mach-o/dyld_images.h>
int main(void) {
 printf("{\\\"task_size\\\":%zu,\\\"task_count\\\":%zu,\\\"self\\\":%zu,\\\"timestamp\\\":%zu}\\n",
  sizeof(task_dyld_info_data_t), (size_t)TASK_DYLD_INFO_COUNT,
  offsetof(struct dyld_all_image_infos, dyldAllImageInfosAddress),
  offsetof(struct dyld_all_image_infos, infoArrayChangeTimestamp));
 return 0;
}
""")
    subprocess.run(["xcrun", "clang", str(source), "-o", str(executable)], check=True, capture_output=True)
    result = json.loads(subprocess.check_output([str(executable)], text=True))
    assert result == {
        "task_size": 20,
        "task_count": 5,
        "self": _ImageInfosPrefix.dyldAllImageInfosAddress.offset,
        "timestamp": _ImageInfosPrefix.infoArrayChangeTimestamp.offset,
    }


@pytest.mark.skipif(sys.platform != "darwin" or platform.machine() != "arm64", reason="Apple arm64 dyld")
def test_real_same_count_library_replacement_changes_generation(tmp_path):
    from triton_msl.backend._native_generation import create_generation_reader

    libraries = []
    for value in (1, 2):
        source, library = tmp_path / f"image{value}.c", tmp_path / f"image{value}.dylib"
        source.write_text(f"int generation_probe(void) {{ return {value}; }}\n")
        subprocess.run(
            ["xcrun", "clang", "-dynamiclib", str(source), "-o", str(library)], check=True, capture_output=True
        )
        libraries.append(library)
    reader = create_generation_reader()
    assert reader is not None, "the supported host must prove the fast reader, not silently fall back"
    lib = C.CDLL(None)
    lib.dlopen.argtypes, lib.dlopen.restype = [C.c_char_p, C.c_int], C.c_void_p
    lib.dlclose.argtypes, lib.dlclose.restype = [C.c_void_p], C.c_int
    lib._dyld_image_count.restype = C.c_uint32
    before = reader()
    first = second = None
    try:
        first = lib.dlopen(os.fsencode(libraries[0]), 2)
        assert first
        loaded = reader()
        count = lib._dyld_image_count()
        assert loaded[0] > before[0]
        closing, first = first, None
        assert lib.dlclose(closing) == 0
        removed = reader()
        assert removed[0] > loaded[0]
        second = lib.dlopen(os.fsencode(libraries[1]), 2)
        assert second
        replaced = reader()
        assert lib._dyld_image_count() == count
        assert replaced[0] > removed[0] and replaced != loaded
    finally:
        for handle in (first, second):
            if handle:
                assert lib.dlclose(handle) == 0
