"""Build selection cannot silently change artifact type or leak scratch binaries."""

import ast
from pathlib import Path

import pytest


@pytest.fixture
def selection():
    source = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validation_extension_enabled")
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
