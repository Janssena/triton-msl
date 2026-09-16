"""Cold native selection must precede the first recorded execution identity."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import textwrap

import pytest
import triton_msl

ROOT = Path(triton_msl.__file__).resolve().parent.parent
IMAGE = ROOT / "triton_msl/backend" / ("_binder_native" + sysconfig.get_config_var("EXT_SUFFIX"))


def _run(tmp_path, source, root=ROOT, *args):
    script = tmp_path / "cold.py"
    script.write_text(textwrap.dedent(source))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(script), str(root), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "stdout.log").write_text(result.stdout)
    (tmp_path / "stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COLD_INITIALIZATION_PASS" in result.stdout


@pytest.mark.skipif(not IMAGE.is_file(), reason="requires optional native binder image")
@pytest.mark.parametrize("order", ["cache", "signature", "raw-native"])
def test_native_initialization_precedes_stamp_and_later_pointer_only_plan(tmp_path, order):
    _run(
        tmp_path,
        """
        import importlib
        from pathlib import Path
        import sys
        root = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(root))
        import torch
        import triton
        import triton_msl
        assert Path(triton_msl.__file__).resolve().parent.parent == root
        assert 'triton_msl.backend._launch_signature' not in sys.modules
        if sys.argv[2] == 'signature':
            importlib.import_module('triton_msl.backend._launch_signature')
        elif sys.argv[2] == 'raw-native':
            importlib.import_module('triton_msl.backend._binder_native')
        from triton_msl.backend import _cache_contract as cache
        assert 'triton_msl.backend._launch_signature' in sys.modules
        binding = sys.modules['triton_msl.backend._launch_signature']
        native = binding._binder_native
        assert native is not None and binding.bind_arguments_with_plan is native.bind_with_plan
        assert Path(native.__file__).resolve().parent == root / 'triton_msl/backend'
        from triton_msl.backend import _framework_contract as framework
        before = cache.execution_contract()
        images = framework._loaded_native_images()
        assert importlib.import_module('triton_msl.backend._launch_signature') is binding
        names, signature = ['x', 'BLOCK'], {'x': '*fp32', 'BLOCK': 'constexpr'}
        plan = binding.make_binding_plan(names, signature)
        assert plan is None
        args = (None, 128)
        assert binding.bind_arguments_with_plan(args, names, signature, plan) == binding.bind_arguments(args, names, signature, _plan=plan)
        assert framework._loaded_native_images() == images
        assert cache.execution_contract() == before
        assert cache.validate_execution_contract(before) == before
        print('COLD_INITIALIZATION_PASS')
    """,
        ROOT,
        order,
    )


def test_actual_pure_package_initializes_binding_without_native_helpers(tmp_path):
    pure = tmp_path / "pure"
    package = pure / "triton_msl"
    for source in (ROOT / "triton_msl").rglob("*"):
        if source.is_file() and source.suffix in (".py", ".c") and "__pycache__" not in source.parts:
            target = package / source.relative_to(ROOT / "triton_msl")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    assert not list(package.rglob("*.so"))
    _run(
        tmp_path,
        """
        from pathlib import Path
        import sys
        root = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(root))
        import triton_msl
        assert Path(triton_msl.__file__).resolve().parent.parent == root
        from triton_msl.backend import _cache_contract as cache
        assert 'triton_msl.backend._launch_signature' in sys.modules
        binding = sys.modules['triton_msl.backend._launch_signature']
        assert cache._validation_native is None and binding._binder_native is None
        assert cache._packed_launch_contract._packed_native is None
        names, signature = ['n'], {'n': 'i32'}
        plan = binding.make_binding_plan(names, signature)
        assert binding.bind_arguments_with_plan((17,), names, signature, plan)[3] == [b'\\x11\\x00\\x00\\x00']
        print('COLD_INITIALIZATION_PASS')
    """,
        pure,
    )


@pytest.mark.parametrize("error_type", ["ImportError", "RuntimeError"])
def test_discoverable_broken_binder_is_not_hidden_by_cold_cache_import(tmp_path, error_type):
    _run(
        tmp_path,
        """
        import importlib.abc
        import importlib.util
        import sys
        sys.path.insert(0, sys.argv[1])
        error = {'ImportError': ImportError, 'RuntimeError': RuntimeError}[sys.argv[2]]('broken binder sentinel')
        class Broken(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'triton_msl.backend._binder_native':
                    return importlib.util.spec_from_loader(fullname, self)
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise error
        sys.meta_path.insert(0, Broken())
        try:
            import triton_msl.backend._cache_contract
        except BaseException as caught:
            assert caught is error
        else:
            raise AssertionError('cold cache import hid broken binder')
        assert 'triton_msl.backend._cache_contract' not in sys.modules
        print('COLD_INITIALIZATION_PASS')
    """,
        ROOT,
        error_type,
    )
