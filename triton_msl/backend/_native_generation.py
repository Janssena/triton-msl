"""Own-process dyld change token; an optimization of, never a substitute for, proof.

ABI: active SDK mach/task_info.h and mach-o/dyld_images.h (version15 prefix).
Apple dyld ExternallyViewableState updates infoArrayChangeTimestamp on both add
and remove, monotonically even within one clock tick. No private getter or
Python callback under the loader lock is used. Unknown ABI keeps full scanning.
Concurrent loader/installation mutation during compilation remains unsupported.
"""

import ctypes as C
import platform
import sys


class _ImageInfosPrefix(C.Structure):
    _fields_ = [
        ("version", C.c_uint32),
        ("infoArrayCount", C.c_uint32),
        ("infoArray", C.c_void_p),
        ("notification", C.c_void_p),
        ("processDetachedFromSharedRegion", C.c_bool),
        ("libSystemInitialized", C.c_bool),
        ("dyldImageLoadAddress", C.c_void_p),
        ("jitInfo", C.c_void_p),
        ("dyldVersion", C.c_void_p),
        ("errorMessage", C.c_void_p),
        ("terminationFlags", C.c_uint64),
        ("coreSymbolicationShmPage", C.c_void_p),
        ("systemOrderFlag", C.c_uint64),
        ("uuidArrayCount", C.c_uint64),
        ("uuidArray", C.c_void_p),
        ("dyldAllImageInfosAddress", C.c_void_p),
        ("initialImageCount", C.c_uint64),
        ("errorKind", C.c_uint64),
        ("errorClientOfDylibPath", C.c_void_p),
        ("errorTargetDylibPath", C.c_void_p),
        ("errorSymbol", C.c_void_p),
        ("sharedCacheSlide", C.c_uint64),
        ("sharedCacheUUID", C.c_uint8 * 16),
        ("sharedCacheBaseAddress", C.c_uint64),
        ("infoArrayChangeTimestamp", C.c_uint64),
    ]


class _GenerationReader:
    def __init__(self, info, address):
        self.info = info
        self.address = address
        self.version = info.version
        if self.version not in (15, 16, 17, 18):
            raise RuntimeError("unproved dyld image-info version")
        self()

    def __call__(self):
        info = self.info
        before = info.infoArrayChangeTimestamp
        pointer, count = info.infoArray, info.infoArrayCount
        if info.version != self.version or info.dyldAllImageInfosAddress != self.address:
            raise RuntimeError("native loader generation ABI changed")
        after = info.infoArrayChangeTimestamp
        # NULL denotes an in-progress dyld update, not an empty/stable inventory.
        if not pointer or not before or before != after or not 0 < count <= 1_000_000:
            raise RuntimeError("native image selection changed during generation inspection")
        return after, pointer, count


def _task_image_info():
    """Public TASK_DYLD_INFO for mach_task_self ONLY, packed as five natural_t's."""
    lib = C.CDLL(None)
    task = C.c_uint32.in_dll(lib, "mach_task_self_").value
    query = lib.task_info
    query.argtypes = [C.c_uint32, C.c_int, C.c_void_p, C.POINTER(C.c_uint32)]
    query.restype = C.c_int
    words, count = (C.c_uint32 * 5)(), C.c_uint32(5)
    if query(task, 17, words, C.byref(count)) != 0 or count.value != 5:
        raise RuntimeError("own-process TASK_DYLD_INFO unavailable")
    # The SDK packs this struct at4: uint64 address, uint64 size, int32 format.
    # Natural-word decoding avoids ctypes' deprecated implicit MSVC packing.
    return words[0] | words[1] << 32, words[2] | words[3] << 32, words[4]


def create_generation_reader():
    """None means use the existing full name scan, never an unknown identity."""
    if (
        sys.platform != "darwin"
        or platform.machine() != "arm64"
        or sys.byteorder != "little"
        or C.sizeof(C.c_void_p) != 8
    ):
        return None
    try:
        if (
            C.sizeof(_ImageInfosPrefix) != 192
            or _ImageInfosPrefix.dyldAllImageInfosAddress.offset != 104
            or _ImageInfosPrefix.infoArrayChangeTimestamp.offset != 184
        ):
            return None
        address, size, layout = _task_image_info()
        if layout != 1 or not 0 < address < 1 << 64 or address % 8 or size < C.sizeof(_ImageInfosPrefix):
            return None
        info = _ImageInfosPrefix.from_address(address)
        return _GenerationReader(info, address)
    except (AttributeError, OSError, ValueError, RuntimeError):
        return None
