"""Independent optional-binder build policy; archive inventories remain owed."""

import ast
from pathlib import Path

import pytest


@pytest.fixture
def policy():
    path = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(path.read_text())
    names = {
        "extension_default_mode",
        "binder_extension_enabled",
        "validation_extension_enabled",
        "packed_extension_enabled",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("mode", ["auto", "0", "1"])
@pytest.mark.parametrize("minor", [12, 13, 14, 15])
def test_binder_default_does_not_break_other_python_native_builds(policy, mode, minor):
    default = policy["extension_default_mode"]("_binder_native", mode)
    assert default == ("0" if mode == "0" else "auto")
    assert policy["binder_extension_enabled"](default, "cpython", (3, minor), "darwin", "arm64", False) is (
        mode != "0" and minor == 14
    )


@pytest.mark.parametrize("name", ["_validation_native", "_packed_native"])
@pytest.mark.parametrize("mode", ["auto", "0", "1"])
def test_existing_helper_default_modes_are_unchanged(policy, name, mode):
    assert policy["extension_default_mode"](name, mode) == mode


def test_explicit_unsupported_binder_request_stays_loud(policy):
    with pytest.raises(RuntimeError, match="CPython 3.14"):
        policy["binder_extension_enabled"]("1", "cpython", (3, 13), "darwin", "arm64", False)
    for name in ("validation_extension_enabled", "packed_extension_enabled"):
        assert policy[name]("1", "cpython", (3, 13), "darwin", "arm64", False)


def test_binder_source_and_binary_packaging_rules_are_explicit():
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text().splitlines()
    assert "include triton_msl/backend/_binder_native.c" in manifest
    for suffix in ("so", "dylib"):
        assert f"exclude triton_msl/backend/_binder_native*.{suffix}" in manifest
    text = (root / "pyproject.toml").read_text()
    section = text.split("[tool.setuptools.exclude-package-data]", 1)[1].split("\n[", 1)[0]
    for suffix in ("so", "dylib"):
        assert f'"_binder_native*.{suffix}"' in section
