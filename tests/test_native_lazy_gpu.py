"""Fresh-process, warm-JIT native-identity transition on the actual GPU."""
from pathlib import Path
import os
import subprocess
import sys

if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])

import pytest
import torch
import triton
import triton.language as tl
import triton_msl


@triton.jit
def _copy_plus_one(x, output, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(output + offsets, tl.load(x + offsets) + 1.0)


def _probe(root, cache_dir):
    from triton_msl.backend import compiler, driver
    from triton_msl.errors import MetalNonRecoverableError
    assert Path(triton_msl.__file__).resolve().parent == root / "triton_msl"
    assert "z3" not in sys.modules, "Z3 must be imported after the warm control"
    for key, leaf in (("TRITON_CACHE_DIR", "triton"), ("TRITON_MSL_CACHE_DIR", "msl"),
                      ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TORCH_EXTENSIONS_DIR", "extensions")):
        os.environ[key] = str(cache_dir / leaf)
    os.environ["TRITON_MSL_COMPILE_SHADER"] = "0"
    os.environ["TRITON_MSL_USE_CPP"] = "0"
    os.environ.pop("TRITON_ALWAYS_COMPILE", None)  # a genuine warm-cache witness
    calls, launches = [], []
    original_msl = compiler.MetalBackend.make_msl
    def emit(*args, **kwargs):
        calls.append(True)
        return original_msl(*args, **kwargs)
    compiler.MetalBackend.make_msl = staticmethod(emit)
    utils = driver._get_utils()
    original_launch = utils.launch
    def launch(*args, **kwargs):
        launches.append(True)
        return original_launch(*args, **kwargs)
    utils.launch = launch
    x = torch.arange(128, dtype=torch.float32)
    output = torch.full_like(x, -99.0)
    first = _copy_plus_one[(1,)](x, output, BLOCK=128)
    torch.testing.assert_close(output, x + 1, rtol=0, atol=0)
    warm = _copy_plus_one[(1,)](x, output, BLOCK=128)
    assert warm is first and len(calls) == 1 and len(launches) == 2
    assert "z3" not in sys.modules
    print("WARM_CONTROL_VERIFIED; EMISSIONS=1; METALLIB_LAUNCHES=2", flush=True)
    import z3
    assert z3.simplify(z3.IntVal(1) + z3.IntVal(2)).as_long() == 3
    try:
        first[(1, 1, 1)](x, output)
    except MetalNonRecoverableError:
        pass
    else:
        raise AssertionError("old direct handle was re-certified after a native transition")
    assert len(launches) == 2
    print("OLD_HANDLE_REFUSED_BEFORE_DISPATCH; METALLIB_LAUNCHES=2", flush=True)
    output.fill_(-99)
    fresh = _copy_plus_one[(1,)](x, output, BLOCK=128)
    assert fresh is not first and len(calls) == 2 and len(launches) == 3
    assert fresh.metadata.execution_contract != first.metadata.execution_contract
    torch.testing.assert_close(output, x + 1, rtol=0, atol=0)
    print("WARM_JIT_RECOMPILED; OLD_HANDLE_REFUSED_BEFORE_DISPATCH; METALLIB_LAUNCHES=3", flush=True)


def test_lazy_z3_recompiles_warm_jit_and_rejects_old_handle_gpu(tmp_path):
    import importlib.util
    if not torch.backends.mps.is_available() or importlib.util.find_spec("z3") is None:
        pytest.skip("requires Metal and optional Z3")
    root = Path(triton_msl.__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(root), str(tmp_path)],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARM_JIT_RECOMPILED; OLD_HANDLE_REFUSED_BEFORE_DISPATCH; METALLIB_LAUNCHES=3" in result.stdout
    print(result.stdout, end="")


if __name__ == "__main__":
    _probe(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
