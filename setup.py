"""Optional host accelerators; both Python implementations remain available.

This helper uses the regular CPython ABI, not abi3. A failed requested/native
build is an error, never a quietly substituted pure-Python artifact.
"""
import os
import platform
import sys
import sysconfig

from setuptools import Extension, setup


def validation_extension_enabled(mode, implementation, version, system, machine, gil_disabled):
    if mode not in ("auto", "0", "1"):
        raise ValueError("TRITON_MSL_BUILD_VALIDATION_NATIVE must be auto, 0 or 1")
    supported = (implementation == "cpython" and (3, 13) <= tuple(version) < (3, 15)
                 and system == "darwin" and machine == "arm64" and not gil_disabled)
    if mode == "1" and not supported:
        raise RuntimeError("native validation build requires GIL-enabled CPython 3.13/3.14 on macOS arm64")
    return supported and mode != "0"



def packed_extension_enabled(mode, implementation, version, system, machine, gil_disabled):
    if mode not in ("auto", "0", "1"):
        raise ValueError("TRITON_MSL_BUILD_PACKED_NATIVE must be auto, 0 or 1")
    supported = (implementation == "cpython" and (3, 13) <= tuple(version) < (3, 15)
                 and system == "darwin" and machine == "arm64" and not gil_disabled)
    if mode == "1" and not supported:
        raise RuntimeError("native packed-copy build requires GIL-enabled CPython 3.13/3.14 on macOS arm64")
    return supported and mode != "0"


def binder_extension_enabled(mode, implementation, version, system, machine, gil_disabled):
    if mode not in ("auto", "0", "1"):
        raise ValueError("TRITON_MSL_BUILD_BINDER_NATIVE must be auto, 0 or 1")
    supported = (implementation == "cpython" and (3, 14) <= tuple(version) < (3, 15)
                 and system == "darwin" and machine == "arm64" and not gil_disabled)
    if mode == "1" and not supported:
        raise RuntimeError("native binder build requires GIL-enabled CPython 3.14 on macOS arm64")
    return supported and mode != "0"


def extension_default_mode(name, validation_mode):
    # A pure build stays pure. A requested 3.13 validation/packed build must not
    # turn into a forced, unsupported 3.14-only binder build. Explicit binder
    # requests are still validated independently by binder_extension_enabled.
    if name == "_binder_native":
        return "0" if validation_mode == "0" else "auto"
    return validation_mode


platform_args = (
    sys.implementation.name, sys.version_info[:2], sys.platform, platform.machine(),
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
)
validation_mode = os.environ.get("TRITON_MSL_BUILD_VALIDATION_NATIVE", "auto")
extensions = []
for name, variable, select in (
    ("_validation_native", "TRITON_MSL_BUILD_VALIDATION_NATIVE", validation_extension_enabled),
    ("_packed_native", "TRITON_MSL_BUILD_PACKED_NATIVE", packed_extension_enabled),
    ("_binder_native", "TRITON_MSL_BUILD_BINDER_NATIVE", binder_extension_enabled),
):
    default_mode = extension_default_mode(name, validation_mode)
    if select(os.environ.get(variable, default_mode), *platform_args):
        extensions.append(Extension(
            "triton_msl.backend." + name,
            sources=["triton_msl/backend/" + name + ".c"],
            # Inherited -g records absolute temporary build paths in Mach-O
            # debug maps. Keep symbols, without path-dependent debug maps.
            extra_compile_args=["-O3", "-Wall", "-Wextra", "-g0"],
        ))
setup(ext_modules=extensions)
