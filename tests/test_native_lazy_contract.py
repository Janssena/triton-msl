"""Late native imports rekey products; no stale resident stamp is re-certified."""

from pathlib import Path
import importlib.util

import pytest

# Load the fixture from this test file's actual sibling, not whichever foreign
# worktree's tests directory a cross-tree baseline runner happens to put first.
_spec = importlib.util.spec_from_file_location(
    "native_ambient_fixture", Path(__file__).with_name("test_native_ambient_contract.py")
)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
ambient = _fixtures.ambient


@pytest.fixture
def runtime(ambient, monkeypatch):
    module, roots, names, first, second = ambient
    monkeypatch.setattr(module, "_snapshot", None)
    monkeypatch.setattr(module, "_discover_selection", lambda: (roots, {}))
    return module, roots, names, first, second


def test_late_provider_builds_a_new_verified_identity(runtime):
    module, _, names, _, second = runtime
    before = module.framework_identity()
    names.append(str(second).encode())
    after = module.framework_identity()
    assert after != before
    assert second.resolve() in module._native_guard.providers
    assert module.framework_identity() == after


@pytest.mark.parametrize("change", ["known_addition", "removal", "reorder"])
def test_known_native_selection_changes_also_rekey(runtime, change):
    module, roots, names, first, second = runtime
    roots.update({"first": (first,), "second": (second,)})
    if change != "known_addition":
        names.append(str(second).encode())
    before = module.framework_identity()
    if change == "known_addition":
        names.append(str(second).encode())
    elif change == "removal":
        names.pop()
    else:
        names[1:] = reversed(names[1:])
    after = module.framework_identity()
    assert after != before
    assert module.framework_identity() == after


@pytest.mark.parametrize("boundary", ["source", "binary", "resident", "backend"])
def test_late_provider_rekeys_every_product_boundary(runtime, monkeypatch, boundary):
    from triton_msl.backend import _cache_contract as cache
    from triton_msl.backend.compiler import MetalBackend
    from triton.backends.compiler import GPUTarget

    module, _, names, _, second = runtime
    monkeypatch.setattr(cache, "framework_identity", module.framework_identity)
    monkeypatch.setattr(cache, "toolchain_identity", lambda: "controlled-toolchain")
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    read = {
        "source": lambda: cache.source_key("same-ir", "same-options"),
        "binary": lambda: cache.binary_key("same-msl", "same-options", "msl", ["-std=metal3.2"]),
        "resident": cache.execution_contract,
        "backend": backend.hash,
    }[boundary]
    before = read()
    names.append(str(second).encode())
    assert read() != before


def test_old_resident_stamp_is_rejected_after_lazy_import(runtime, monkeypatch):
    from triton_msl.backend import _cache_contract as cache
    from triton_msl.errors import MetalNonRecoverableError

    module, _, names, _, second = runtime
    monkeypatch.setattr(cache, "framework_identity", module.framework_identity)
    monkeypatch.setattr(cache, "toolchain_identity", lambda: "controlled-toolchain")
    old = cache.execution_contract()
    names.append(str(second).encode())
    new = cache.execution_contract()
    assert new != old
    with pytest.raises(MetalNonRecoverableError, match="stale|incompatible"):
        cache.validate_execution_contract(old)
    cache.validate_execution_contract(new)


def test_failed_lazy_inventory_never_certifies_the_old_snapshot(runtime):
    from triton_msl.errors import MetalNonRecoverableError

    module, _, names, _, second = runtime
    old = module.framework_identity()
    guard = module._native_guard
    second.write_bytes(b"malformed late native image")
    names.append(str(second).encode())
    for _ in range(2):
        with pytest.raises(MetalNonRecoverableError):
            module.framework_identity()
        assert module._snapshot[1] == old
        assert module._native_guard is guard


def test_post_inventory_failure_rolls_back_the_native_guard(runtime, monkeypatch):
    from triton_msl.errors import MetalNonRecoverableError

    module, roots, names, _, second = runtime
    old = module.framework_identity()
    guard = module._native_guard
    names.append(str(second).encode())
    calls = []

    def unstable():
        calls.append(True)
        return roots, ({"changed": True} if len(calls) == 2 else {})

    monkeypatch.setattr(module, "_discover_selection", unstable)
    with pytest.raises(MetalNonRecoverableError):
        module.framework_identity()
    assert module._snapshot[1] == old
    assert module._native_guard is guard
    monkeypatch.setattr(module, "_discover_selection", lambda: (roots, {}))
    assert module.framework_identity() != old


def test_interrupted_inventory_rolls_back_digest_and_guard(runtime, monkeypatch):
    module, _, names, _, second = runtime
    old = module.framework_identity()
    guard = module._native_guard
    names.append(str(second).encode())
    original = module.json.dumps

    def interrupted(value, *args, **kwargs):
        if isinstance(value, dict) and "packages" in value:
            raise KeyboardInterrupt("controlled post-inventory interruption")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(module.json, "dumps", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.framework_identity()
    assert module._snapshot[1] == old
    assert module._native_guard is guard
    monkeypatch.setattr(module.json, "dumps", original)
    assert module.framework_identity() != old


def test_jit_owner_cache_is_cleared_under_the_new_native_identity(runtime, monkeypatch):
    from triton_msl.backend import _cache_contract as cache

    module, _, names, _, second = runtime
    monkeypatch.setattr(cache, "framework_identity", module.framework_identity)
    monkeypatch.setattr(cache, "toolchain_identity", lambda: "controlled-toolchain")

    class Owner:
        def __init__(self):
            self.device_caches = {0: "old executable"}
            self.hooks = []

        def add_pre_run_hook(self, hook):
            self.hooks.append(hook)

    owner, unrelated = Owner(), Owner()
    assert cache.install_jit_policy_guard(owner, cache.execution_contract())
    owner.hooks[0]()
    assert owner.device_caches
    names.append(str(second).encode())
    owner.hooks[0]()
    assert owner.device_caches == {}
    assert unrelated.device_caches == {0: "old executable"}


def test_real_z3_lazy_import_rekeys_without_certifying_old_identity():
    import importlib.util
    import subprocess
    import sys
    import triton_msl

    if sys.platform != "darwin" or importlib.util.find_spec("z3") is None:
        pytest.skip("requires Darwin and the optional Z3 package")
    root = Path(triton_msl.__file__).resolve().parent.parent
    program = """
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import triton_msl
assert Path(triton_msl.__file__).resolve().parent == root / 'triton_msl'
from triton_msl.backend import _framework_contract as contract
assert 'z3' not in sys.modules, 'probe must import Z3 AFTER the cold snapshot'
before = contract.framework_identity()
import z3
assert z3.simplify(z3.IntVal(1) + z3.IntVal(2)).as_long() == 3
after = contract.framework_identity()
assert after != before
assert contract.framework_identity() == after
print('LAZY_Z3_REKEY_VERIFIED', triton_msl.__file__, before, after)
"""
    result = subprocess.run([sys.executable, "-c", program, str(root)], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LAZY_Z3_REKEY_VERIFIED" in result.stdout
