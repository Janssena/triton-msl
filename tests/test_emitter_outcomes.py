"""Lowering outcomes are facts, not words in a kernel's name or source."""

from types import SimpleNamespace

import pytest

from cache_helpers import patch_live_singleton_method

import triton_msl.codegen.generic_lowerer as generic
import triton_msl.codegen.msl_emitter as emitter
from triton_msl.errors import MetalNonRecoverableError


def _module(tmp_path, name):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import IRSource
    from triton_msl.backend.compiler import MetalBackend

    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    path = tmp_path / "source.ttgir"
    path.write_text(
        """module attributes {"ttg.num-ctas" = 1 : i32,
      "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func public @NAME(%out: !tt.ptr<f32>) {
        %c = arith.constant 1.0 : f32
        tt.store %out, %c : !tt.ptr<f32>
        tt.return
      }
    }""".replace("NAME", name)
    )
    ctx = ir.context()
    source = IRSource(str(path), ctx, backend)
    assert source.module.verify()
    return ctx, source, backend.parse_options({})


@pytest.mark.parametrize("name", ["plain_source", "UNKNOWN_source", "UNSUPPORTED_source"])
def test_kernel_name_does_not_decide_lowering_outcome(tmp_path, monkeypatch, name):
    monkeypatch.delenv("TRITON_MSL_LEGACY", raising=False)
    ctx, source, options = _module(tmp_path, name)
    metadata = {}
    msl = emitter.emit_msl(source.module, metadata, options)
    assert metadata["name"] == name
    assert f"kernel void {name}(" in msl
    assert metadata["output_arg_indices"] == [0]
    # The name is the only semantic difference from the canonical control.
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    control_ctx, control, control_options = _module(control_dir, "plain_source")
    assert msl.replace(name, "plain_source") == emitter.emit_msl(control.module, {}, control_options)


@pytest.mark.parametrize("consumer", ["kernel", "register_value", "callee"])
def test_unresolved_ssa_refuses_at_its_consumer(consumer):
    if consumer == "callee":
        lowerer = generic._DeviceFuncLowerer(SimpleNamespace())
        lookup = lowerer._lookup
    else:
        lowerer = generic.GenericLowerer.__new__(generic.GenericLowerer)
        lowerer.env = {}
        lowerer.env_types = {}
        lowerer.env_n_elems = {}
        lookup = lowerer._lookup if consumer == "kernel" else lowerer._lookup_regval
    with pytest.raises(MetalNonRecoverableError, match="unresolved SSA value.*123"):
        lookup(123)


@pytest.mark.parametrize("location", ["kernel", "callee"])
def test_unsupported_outcome_survives_diagnostic_comment_redaction(tmp_path, monkeypatch, location):
    """Synthetic unhandled op: no consumer, so only the explicit outcome can refuse."""
    from triton_msl.codegen.mlir_walker import CalledFunc, SSAValue

    monkeypatch.delenv("TRITON_MSL_LEGACY", raising=False)
    injections, redactions = [], []

    def inject(_class, original_register):
        def register_with_unhandled_op(lowerer):
            original_register(lowerer)
            injections.append(location)
            op = SSAValue(123456789, "unhandled", "test.unhandled", [], {}, "f32", "f32", False)
            if location == "kernel":
                lowerer._lower_op_dispatch(op)
            else:
                callee = CalledFunc("unhandled_callee", [], [op], [])
                lowerer.kb._device_functions.append(lowerer._lower_one_called_func(callee))

        return register_with_unhandled_op

    def redact_kernel(_class, _original_comment):
        def comment(kb, text):
            if "test.unhandled" in text:
                redactions.append("kernel")
            kb._emit("// diagnostic")

        return comment

    def redact_device(_class, original_emit):
        def emit(lowerer, line):
            if line.startswith("//") and "test.unhandled" in line:
                redactions.append("callee")
            original_emit(lowerer, "// diagnostic" if line.startswith("//") else line)

        return emit

    # MEPT tests reload these modules after collection. Patch the current class
    # objects consumed by production, not class aliases retained during collection.
    patch_live_singleton_method(monkeypatch, lambda: generic.GenericLowerer, "_register_args", inject)
    patch_live_singleton_method(monkeypatch, lambda: generic.KernelBuilder, "comment", redact_kernel)
    patch_live_singleton_method(monkeypatch, lambda: generic._DeviceFuncLowerer, "_emit", redact_device)
    ctx, source, options = _module(tmp_path, "plain_source")
    with pytest.raises(MetalNonRecoverableError, match="generic lowerer could not lower"):
        emitter.emit_msl(source.module, {}, options)
    assert injections == [location], "the intended unsupported operation was not injected exactly once"
    assert redactions == [location], "the injected operation's diagnostic was not redacted exactly once"
