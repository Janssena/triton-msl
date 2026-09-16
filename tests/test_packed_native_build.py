"""Build selection cannot silently change artifact type or leak scratch binaries."""

import ast
from pathlib import Path

import pytest


@pytest.fixture
def selection():
    source = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "packed_extension_enabled")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize("version", [(3, 13), (3, 14)])
@pytest.mark.parametrize("mode,expected", [("auto", True), ("1", True), ("0", False)])
def test_native_selection(selection, version, mode, expected):
    assert selection(mode, "cpython", version, "darwin", "arm64", False) is expected


@pytest.mark.parametrize(
    "implementation,version,system,machine,gil",
    [
        ("cpython", (3, 12), "darwin", "arm64", False),
        ("cpython", (3, 15), "darwin", "arm64", False),
        ("cpython", (3, 14), "darwin", "arm64", True),
        ("cpython", (3, 14), "linux", "aarch64", False),
        ("cpython", (3, 14), "darwin", "x86_64", False),
        ("pypy", (3, 13), "darwin", "arm64", False),
    ],
)
def test_unsupported_auto_is_python_but_explicit_request_raises(
    selection, implementation, version, system, machine, gil
):
    assert selection("auto", implementation, version, system, machine, gil) is False
    with pytest.raises(RuntimeError, match="requires GIL-enabled"):
        selection("1", implementation, version, system, machine, gil)


def test_unknown_build_mode_is_not_silently_pure(selection):
    with pytest.raises(ValueError, match="auto, 0 or 1"):
        selection("typo", "cpython", (3, 14), "darwin", "arm64", False)


@pytest.mark.parametrize("validation", [None, "auto", "0", "1"])
@pytest.mark.parametrize("packed", [None, "auto", "0", "1"])
def test_combined_build_modes_preserve_legacy_opt_out(validation, packed):
    from types import SimpleNamespace

    tree = ast.parse((Path(__file__).resolve().parents[1] / "setup.py").read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    env = {}
    if validation is not None:
        env["TRITON_MSL_BUILD_VALIDATION_NATIVE"] = validation
    if packed is not None:
        env["TRITON_MSL_BUILD_PACKED_NATIVE"] = packed
    outputs = []
    namespace = dict(
        os=SimpleNamespace(environ=env),
        sys=SimpleNamespace(implementation=SimpleNamespace(name="cpython"), version_info=(3, 14), platform="darwin"),
        platform=SimpleNamespace(machine=lambda: "arm64"),
        sysconfig=SimpleNamespace(get_config_var=lambda key: False),
        Extension=lambda name, **kwargs: name,
        setup=lambda **kwargs: outputs.extend(kwargs["ext_modules"]),
    )
    exec(compile(tree, "setup.py", "exec"), namespace)
    effective_validation = validation if validation is not None else "auto"
    effective_packed = packed if packed is not None else effective_validation
    expected = []
    if effective_validation != "0":
        expected.append("triton_msl.backend._validation_native")
    if effective_packed != "0":
        expected.append("triton_msl.backend._packed_native")
    if effective_validation != "0":
        expected.append("triton_msl.backend._binder_native")
    assert outputs == expected
