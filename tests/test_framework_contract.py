"""Framework bytes, including shader headers, bind source and resident products.

Synthetic restart/selection tests are cache-contract evidence, not GPU wrongs.
"""

from pathlib import Path

import pytest
import triton

from triton_msl.backend import _cache_contract as cache
from triton_msl.errors import MetalNonRecoverableError


@pytest.mark.parametrize("boundary", ["source", "resident", "binary", "backend"])
def test_framework_change_moves_every_product_boundary(monkeypatch, boundary):
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend.compiler import MetalBackend

    monkeypatch.setattr(cache, "toolchain_identity", lambda: "controlled-toolchain")
    monkeypatch.setattr(cache, "framework_identity", lambda: "framework-one", raising=False)
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    read = {
        "source": lambda: cache.source_key("same-ir", "same-options"),
        "resident": cache.execution_contract,
        "binary": lambda: cache.binary_key("same-msl", "same-options", "msl", ["-std=metal3.2"]),
        "backend": backend.hash,
    }[boundary]
    first = read()
    monkeypatch.setattr(cache, "framework_identity", lambda: "framework-two", raising=False)
    assert read() != first


def test_old_resident_contract_does_not_survive_framework_upgrade(monkeypatch):
    monkeypatch.setattr(cache, "toolchain_identity", lambda: "controlled-toolchain")
    monkeypatch.setattr(cache, "framework_identity", lambda: "old", raising=False)
    old = cache.execution_contract()
    monkeypatch.setattr(cache, "framework_identity", lambda: "new", raising=False)
    with pytest.raises(MetalNonRecoverableError, match="stale|incompatible"):
        cache.validate_execution_contract(old)


@pytest.fixture
def framework(tmp_path, monkeypatch):
    from triton_msl.backend import _framework_contract as framework

    root = tmp_path / "torch"
    root.mkdir()
    for name, content in [
        ("__init__.py", b"VERSION = 'unchanged'"),
        ("runtime.so", b"native implementation"),
        ("shader.h", b"float shader_function(float x) { return x; }"),
        ("resource.json", b'{"runtime": true}'),
    ]:
        (root / name).write_bytes(content)
    roots = {"torch": (root,), "triton": (root,), "mlx": ()}
    metadata = {"python_abi": "controlled", "python_version": "3.x", "byteorder": "little"}
    monkeypatch.setattr(framework, "_snapshot", None)
    monkeypatch.setattr(framework, "_discover_selection", lambda: (roots, metadata))
    # This fixture deliberately contains non-executable placeholder native data.
    # Real Mach-O closure is tested separately, including its framework binding.
    monkeypatch.setattr(framework, "_native_dependencies", lambda roots: ())
    monkeypatch.setattr(framework, "_native_guard", None)
    return framework, roots, metadata


@pytest.mark.parametrize("name", ["__init__.py", "runtime.so", "shader.h", "resource.json"])
def test_same_version_changed_runtime_bytes_rekey_after_restart(framework, monkeypatch, name):
    module, roots, _ = framework
    first = module.framework_identity()
    (roots["torch"][0] / name).write_bytes(b"different runtime contents, same version")
    monkeypatch.setattr(module, "_snapshot", None)
    assert module.framework_identity() != first


def test_bytecode_build_artifacts_are_not_relocation_identity(framework, monkeypatch):
    module, roots, _ = framework
    first = module.framework_identity()
    pycache = roots["torch"][0] / "__pycache__"
    pycache.mkdir()
    (pycache / "module.cpython.pyc").write_bytes(b"contains an absolute build path")
    monkeypatch.setattr(module, "_snapshot", None)
    assert module.framework_identity() == first


def test_empty_venv_platstdlib_is_inventoried_not_required_to_supply_files(framework, monkeypatch, tmp_path):
    module, roots, _ = framework
    stdlib = tmp_path / "base-stdlib"
    stdlib.mkdir()
    (stdlib / "os.py").write_text("base implementation")
    platstdlib = tmp_path / "venv-stdlib"
    platstdlib.mkdir()
    packages = platstdlib / "site-packages"
    packages.mkdir()
    (packages / "unrelated.py").write_text("inventoried through package selection instead")
    roots.update({"python-stdlib": (stdlib,), "python-platstdlib": (platstdlib,)})
    first = module.framework_identity()
    # Empty supplemental stdlib is a real venv layout, not unknown identity.
    # It must remain inventoried: a provider added after restart changes the key.
    (platstdlib / "provider.py").write_text("new platform implementation")
    monkeypatch.setattr(module, "_snapshot", None)
    assert module.framework_identity() != first


def test_empty_primary_stdlib_is_not_certified_by_supplemental_root_memo(framework, tmp_path):
    module, roots, _ = framework
    empty = tmp_path / "empty"
    empty.mkdir()
    # python-platstdlib sorts first. Its empty allowance must not leak through
    # physical-root deduplication into the required primary stdlib role.
    roots.update({"python-stdlib": (empty,), "python-platstdlib": (empty,)})
    with pytest.raises(MetalNonRecoverableError, match="empty toolchain input tree"):
        module.framework_identity()
    with pytest.raises(RuntimeError, match="empty toolchain input tree"):
        module._tree_manifest(empty)


def test_framework_selection_change_refuses_instead_of_hot_reloading(framework, monkeypatch, tmp_path):
    module, roots, _ = framework
    module.framework_identity()
    roots["torch"] = (tmp_path / "a different installation",)
    with pytest.raises(MetalNonRecoverableError, match="selection changed.*restart"):
        module.framework_identity()


def test_optional_framework_installation_changes_process_selection(framework, tmp_path):
    module, roots, _ = framework
    module.framework_identity()
    roots["mlx"] = (tmp_path / "new mlx",)
    with pytest.raises(MetalNonRecoverableError, match="selection changed.*restart"):
        module.framework_identity()


def test_framework_snapshot_does_not_rehash_every_launch(framework, monkeypatch):
    module, _, _ = framework
    calls = []
    original = module._tree_manifest

    def read(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_tree_manifest", read)
    first = module.framework_identity()
    assert module.framework_identity() == first
    assert len(calls) == 1  # physical-root dedup, both logical module roles retained


def test_incomplete_framework_tree_refuses(framework):
    module, roots, _ = framework
    (roots["torch"][0] / "broken-native.so").symlink_to("missing")
    with pytest.raises(MetalNonRecoverableError, match="framework.*identity"):
        module.framework_identity()


def test_unknown_required_framework_is_not_an_absent_identity(monkeypatch):
    from triton_msl.backend import _framework_contract as module

    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="required.*triton"):
        module._discover_selection()


def test_identical_relocated_framework_retains_identity(framework, monkeypatch, tmp_path):
    import shutil

    module, roots, _ = framework
    first = module.framework_identity()
    relocated = tmp_path / "relocated"
    shutil.copytree(roots["torch"][0], relocated)
    roots["torch"] = roots["triton"] = (relocated,)
    monkeypatch.setattr(module, "_snapshot", None)
    assert module.framework_identity() == first


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from triton_msl.backend import _framework_contract as module

    root = tmp_path / "installed"
    root.mkdir()
    (root / "__init__.py").write_text("pass\n")
    # Isolate import-state experiments from pytest and the real Torch module.
    fake_sys = SimpleNamespace(
        **{
            name: getattr(module.sys, name)
            for name in ("platform", "version", "implementation", "byteorder", "meta_path", "path_hooks")
        }
    )
    fake_sys.path = list(module.sys.path)
    fake_sys.modules = {}
    calls = []

    def find(name):
        calls.append(name)
        loaded = fake_sys.modules.get(name)
        return (
            loaded.__spec__
            if loaded
            else SimpleNamespace(origin=str(root / "__init__.py"), submodule_search_locations=[str(root)])
        )

    monkeypatch.setattr(module, "sys", fake_sys)
    monkeypatch.setattr(module.importlib.util, "find_spec", find)
    monkeypatch.setattr(module, "_snapshot", None)
    monkeypatch.setattr(module, "_discovery_snapshot", None)
    monkeypatch.setattr(module, "_owned_selection", None)
    monkeypatch.setattr(module, "_python_roots", lambda: {})
    monkeypatch.setattr(module, "_native_dependencies", lambda roots: ())
    monkeypatch.setattr(module, "_native_guard", None)
    return module, fake_sys, root, calls


def test_warm_selection_needs_no_repeated_filesystem_resolution(discovery, monkeypatch):
    module, _, _, calls = discovery
    first = module.framework_identity()
    count = len(calls)

    def forbidden(*args, **kwargs):
        pytest.fail("unchanged import selection must not resolve the filesystem per launch")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(module, "_format_selection", forbidden)
    assert module.framework_identity() == first
    assert len(calls) == count


def test_only_owned_deeply_immutable_selection_reuses_serialization(discovery, monkeypatch):
    from types import MappingProxyType

    module, _, root, _ = discovery
    roots, metadata = module._discover_selection()
    expected = module._format_selection(dict(roots), dict(metadata))
    assert module._selection_identity(roots, metadata) == expected
    assert type(module._json_metadata(metadata)) is dict
    assert module._json_metadata(metadata) == dict(metadata)
    with pytest.raises(TypeError):
        roots["torch"] = ()
    with pytest.raises(TypeError):
        metadata["byteorder"] = "changed"
    # Wrapper type alone proves nothing about ownership or nested values.
    foreign_roots = {"torch": [root]}
    foreign_metadata = {"nested": ["old"]}
    assert module._json_metadata(foreign_metadata) is foreign_metadata

    class ForeignMetadata(dict):
        def items(self):
            return [("provider", "custom-items-provider")]

    custom = ForeignMetadata(foreign_metadata)
    assert module._json_metadata(custom) is custom
    assert module.json.dumps(custom) != module.json.dumps(dict(custom))
    assert module.json.dumps(module._json_metadata(custom)) == module.json.dumps(custom)
    foreign = MappingProxyType(foreign_roots), foreign_metadata
    first = module._selection_identity(*foreign)
    foreign_roots["torch"].append(root / "different")
    foreign_metadata["nested"][0] = "new"
    assert module._selection_identity(*foreign) != first
    with pytest.raises(TypeError):  # Original JSON encoder does not admit this.
        module._selection_identity(foreign[0], MappingProxyType(foreign_metadata))
    # Nonstandard environment values cannot be frozen merely by wrapping them.
    monkeypatch.setattr(module, "_discovery_snapshot", None)
    mutable = ["old"]
    monkeypatch.setattr(module.os, "environ", {"PYTORCH_MPS_FAST_MATH": mutable})
    roots, metadata = module._discover_selection()
    first = module._selection_identity(roots, metadata)
    mutable[0] = "new"
    assert module._selection_identity(roots, metadata) != first


def test_new_loaded_module_cannot_hide_behind_discovery_memo(discovery, tmp_path):
    from types import SimpleNamespace

    module, fake_sys, _, _ = discovery
    module.framework_identity()
    other = tmp_path / "different"
    other.mkdir()
    spec = SimpleNamespace(origin=str(other / "__init__.py"), submodule_search_locations=[str(other)])
    fake_sys.modules["torch"] = SimpleNamespace(__spec__=spec)
    with pytest.raises(MetalNonRecoverableError, match="selection changed.*restart"):
        module.framework_identity()


def test_same_loaded_module_mutated_spec_is_rechecked(discovery, tmp_path):
    from types import SimpleNamespace

    module, fake_sys, root, _ = discovery
    spec = SimpleNamespace(origin=str(root / "__init__.py"), submodule_search_locations=[str(root)])
    fake_sys.modules["torch"] = SimpleNamespace(__spec__=spec)
    module.framework_identity()
    other = tmp_path / "different"
    other.mkdir()
    spec.origin = str(other / "__init__.py")
    spec.submodule_search_locations[:] = [str(other)]
    with pytest.raises(MetalNonRecoverableError, match="selection changed.*restart"):
        module.framework_identity()


def test_unrelated_search_path_change_rechecks_but_preserves_proved_selection(discovery):
    module, fake_sys, _, calls = discovery
    first = module.framework_identity()
    count = len(calls)
    fake_sys.path.append("an unrelated kernel-test directory")
    assert module.framework_identity() == first
    assert len(calls) > count


@pytest.mark.parametrize("name", ["PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK"])
def test_process_initialized_torch_policy_cannot_change_under_old_handle(discovery, monkeypatch, name):
    module, _, _, _ = discovery
    monkeypatch.delenv(name, raising=False)
    module.framework_identity()
    monkeypatch.setenv(name, "1")
    with pytest.raises(MetalNonRecoverableError, match="selection changed.*restart"):
        module.framework_identity()


@pytest.mark.parametrize(
    "name",
    [
        "DYLD_FRAMEWORK_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_FALLBACK_FRAMEWORK_PATH",
        "DYLD_ROOT_PATH",
        "DYLD_IMAGE_SUFFIX",
        "DYLD_VERSIONED_LIBRARY_PATH",
        "DYLD_VERSIONED_FRAMEWORK_PATH",
    ],
)
def test_untracked_native_search_override_refuses_even_after_snapshot(monkeypatch, name):
    from triton_msl.backend import _toolchain_contract as toolchain

    for key in toolchain._EXTERNAL_SEARCH:
        monkeypatch.delenv(key, raising=False)
    selection = tuple(toolchain.os.environ.get(key) for key in toolchain._SELECTION)
    monkeypatch.setattr(toolchain, "_snapshot", (selection, "previous-valid-snapshot"))
    monkeypatch.setenv(name, "/untracked/native/provider")
    with pytest.raises(MetalNonRecoverableError, match="untracked compiler search"):
        toolchain.toolchain_identity()


def _selected_cpp(discovery, tmp_path):
    from types import SimpleNamespace

    module, fake_sys, _, _ = discovery
    outside = tmp_path / "editable-native-provider.so"
    outside.write_bytes(b"selected native bytes")
    spec = SimpleNamespace(origin=str(outside), submodule_search_locations=None)
    fake_sys.modules["triton_msl._triton_msl_cpp"] = SimpleNamespace(__spec__=spec)
    return module, outside, spec


def test_selected_native_child_is_not_inferred_from_top_level_package(discovery, tmp_path):
    module, outside, _ = _selected_cpp(discovery, tmp_path)
    roots, _ = module._discover_selection()
    assert roots["triton_msl._triton_msl_cpp"] == (outside,)


def test_selected_native_child_bytes_rekey_after_restart(discovery, tmp_path, monkeypatch):
    module, outside, _ = _selected_cpp(discovery, tmp_path)
    first = module.framework_identity()
    outside.write_bytes(b"replacement selected native bytes")
    monkeypatch.setattr(module, "_snapshot", None)
    assert module.framework_identity() != first


def test_selected_native_child_change_refuses_in_same_process(discovery, tmp_path):
    module, _, spec = _selected_cpp(discovery, tmp_path)
    module.framework_identity()
    other = tmp_path / "new-provider.so"
    other.write_bytes(b"new native provider")
    spec.origin = str(other)
    with pytest.raises(MetalNonRecoverableError, match="selection changed.*restart"):
        module.framework_identity()


def test_optional_native_child_absence_is_explicit(discovery, monkeypatch):
    module, _, _, _ = discovery
    original = module.importlib.util.find_spec
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: None if name == "triton_msl._triton_msl_cpp" else original(name),
    )
    roots, _ = module._discover_selection()
    assert roots["triton_msl._triton_msl_cpp"] == ()
