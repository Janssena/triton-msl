"""Real AST/TTGIR compiler-to-metadata boundary; deliberately no GPU dispatch."""
import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.backend.compiler import MetalBackend
from test_metallib_concurrent import requires_metal_compiler
from test_resident_execution_contract import _stamp


@triton.jit
def _contract_copy(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(Y + offsets, tl.load(X + offsets))


def test_actual_ttgir_source_product_preserves_native_context(tmp_path, monkeypatch):
    """No linker/GPU: prove a real TTGIR emission publishes and then restores."""
    import json
    from triton._C.libtriton import ir
    import triton_msl.codegen.msl_emitter as emitter

    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    source = ASTSource(_contract_copy, {"X": "*fp32", "Y": "*fp32"}, {"BLOCK": 32})
    context = ir.context()
    ir.load_dialects(context)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    metadata = {"hash": "test-outer-hash", "target": target, **vars(options)}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    initial = dict(metadata)
    calls = []
    original = emitter.emit_msl
    def emit(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(emitter, "emit_msl", emit)
    first = backend.make_msl(module, metadata, options)
    records = list(tmp_path.glob("*.meta.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["kind"] == "msl-source-product"
    second_metadata = dict(initial)
    assert backend.make_msl(module, second_metadata, options) == first
    assert calls == [True]
    assert type(second_metadata["target"]) is GPUTarget
    assert second_metadata["target"] == target
    assert json.dumps(second_metadata, default=vars, sort_keys=True) == json.dumps(metadata, default=vars, sort_keys=True)


@requires_metal_compiler
@pytest.mark.parametrize("source_kind", ["ast", "ttgir"])
def test_actual_compile_stamps_and_restores_without_loading_a_pipeline(tmp_path, monkeypatch, source_kind):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    source = ASTSource(_contract_copy, {"X": "*fp32", "Y": "*fp32"}, {"BLOCK": 32})
    target = GPUTarget("metal", "apple-m4", 32)
    options = {"num_warps": 4, "target_metal_version": "3.2"}
    compiled = triton.compile(source, target=target, options=options)
    if source_kind == "ttgir":
        path = tmp_path / "contract.ttgir"
        path.write_text(compiled.asm["ttgir"])
        source = str(path)
        compiled = triton.compile(source, target=target, options=options)
    assert getattr(compiled.metadata, "execution_contract", None) == _stamp()
    assert compiled.module is None and compiled.function is None  # never _init_handles/run
    assert compiled.packed_metadata[3] == compiled.metadata.block_size
    assert compiled.kernel

    def forbidden(*args, **kwargs):
        pytest.fail("unchanged valid product missed the outer persistent cache")
    monkeypatch.setattr(MetalBackend, "add_stages", forbidden)
    restored = triton.compile(source, target=target, options=options)
    assert restored.metadata.execution_contract == _stamp()
    assert restored.kernel == compiled.kernel
    assert restored.module is None and restored.function is None


@requires_metal_compiler
@pytest.mark.parametrize("use_cpp", [False, True])
def test_cold_direct_compile_and_outer_restore_initialize_before_identity(tmp_path, use_cpp):
    """The child starts cold: no pytest collection, device pre-warm or compile retry."""
    import importlib.util
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap
    import triton_msl

    if use_cpp and importlib.util.find_spec("triton_msl._triton_msl_cpp") is None:
        pytest.skip("optional C++ extension unavailable")
    root = Path(triton_msl.__file__).resolve().parent.parent
    worker = tmp_path / "cold_direct.py"
    worker.write_text(textwrap.dedent('''\
        import os
        from pathlib import Path
        import sys
        sys.path.insert(0, sys.argv[1])
        import torch
        import triton
        import triton.language as tl
        import triton_msl
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import ASTSource
        assert Path(triton_msl.__file__).resolve().parent.parent == Path(sys.argv[1])
        assert "Metal" not in sys.modules, "the public compile call must start cold"

        @triton.jit
        def add(X, O, N: tl.constexpr):
            i = tl.arange(0, N)
            tl.store(O + i, tl.load(X + i) + 1.0)

        x = torch.arange(128, dtype=torch.float32)
        output = torch.full_like(x, -99)
        source = ASTSource(fn=add, signature={"X": "*fp32", "O": "*fp32"}, constexprs={"N": 128})
        target = GPUTarget("metal", "apple-m4", 32)
        compiled = triton.compile(source, target=target)
        assert compiled.module is None and compiled.function is None
        expected = "cpp" if os.environ["TRITON_MSL_USE_CPP"] == "1" else "msl"
        assert compiled.metadata.binary_route == expected
        from triton_msl.backend.compiler import MetalBackend
        def forbidden(*args, **kwargs):
            raise AssertionError("warm outer lookup missed after cold initialization")
        MetalBackend.add_stages = forbidden
        restored = triton.compile(source, target=target)
        assert restored.hash == compiled.hash and restored.kernel == compiled.kernel
        assert restored.metadata.execution_contract == compiled.metadata.execution_contract
        assert restored.module is None and restored.function is None
        restored[(1, 1, 1)](x, output)
        assert torch.equal(output, x + 1)
        print("COMPUTED_EXACT; COLD_DIRECT_AND_OUTER_HIT_VERIFIED; ROUTE=" + expected, flush=True)
    '''))
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("TRITON_MSL_"):
            env.pop(key)
    env.update(TRITON_DEFAULT_BACKEND="metal", TRITON_ALWAYS_COMPILE="0",
               TRITON_MSL_USE_CPP="1" if use_cpp else "0", TRITON_MSL_COMPILE_SHADER="0",
               PYTHONPATH=str(root))
    for key in ("TRITON_CACHE_DIR", "TRITON_MSL_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR", "CLANG_MODULE_CACHE_PATH", "TMPDIR"):
        directory = tmp_path / key
        directory.mkdir()
        env[key] = str(directory)
    result = subprocess.run([sys.executable, str(worker), str(root)], env=env,
                            capture_output=True, text=True, timeout=180)
    (tmp_path / "raw.txt").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPUTED_EXACT; COLD_DIRECT_AND_OUTER_HIT_VERIFIED; ROUTE=" in result.stdout


@pytest.mark.parametrize("boundary", ["hash", "add_stages"])
def test_runtime_initialization_precedes_key_and_producer_capture(monkeypatch, boundary):
    """Controlled CPU ordering witness for both public and stage-only callers."""
    from triton_msl.backend import _cache_contract as contract, driver
    events = []
    class Runtime:
        @property
        def device(self):
            events.append("initialize")
            return object()
    monkeypatch.setattr(driver, "_get_utils", lambda: Runtime())
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    def source():
        assert events and events[0] == "initialize", events
        events.append("identity")
        return {"controlled": "initialized"}
    monkeypatch.setattr(contract, "source_contract", source)
    monkeypatch.setattr(contract, "toolchain_identity", lambda: "controlled-toolchain")
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    if boundary == "hash":
        assert backend.hash().startswith("metal-")
    else:
        backend.add_stages({}, backend.parse_options({}))
    assert events == ["initialize", "identity"]
