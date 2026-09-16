"""Every admitted spelling must pass the C++ boundary, including direct entry.

CPU-only: never invoke native lowering on the known scalar-load assertion path.
"""

from types import SimpleNamespace
import sys

import pytest
import triton  # noqa: F401

from triton_msl.backend.compiler import MetalBackend
from triton_msl.backend.cpp_families import cpp_refusal_reason, cpp_safe_text
from triton_msl.errors import MetalNonRecoverableError


UNPROVED = [
    '%x = "tt.load"(%p) : (!tt.ptr<f32>) -> f32',
    '%x = "arith.constant"() <{value = 1.0 : f32}> : () -> f32',
    '"unregistered.effect"() : () -> ()',
    "%x = tt.load %p : !tt.ptr<f32>",
]


@pytest.mark.parametrize("text", UNPROVED)
def test_unproved_spelling_has_an_explicit_cpp_disposition(monkeypatch, text):
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    reason = cpp_refusal_reason(text)
    assert reason is not None
    assert MetalBackend._has_complex_ops(text)


@pytest.mark.parametrize("text", UNPROVED)
def test_direct_entry_refuses_before_native_or_annotation_rewrite(monkeypatch, text):
    monkeypatch.setitem(sys.modules, "triton_msl._triton_msl_cpp", SimpleNamespace())

    def forbidden(*args):
        raise AssertionError("unproved spelling reached annotation rewrite/native lowering")

    monkeypatch.setattr(MetalBackend, "_strip_ttg_annotations", staticmethod(forbidden))
    with pytest.raises(MetalNonRecoverableError, match=r"C\+\+ route"):
        MetalBackend.make_llir(text, {"name": "probe"}, SimpleNamespace())


@pytest.mark.parametrize(
    "text",
    [
        "%x = tt.load %p : tensor<128x!tt.ptr<f32>, #blocked>",
        'module attributes {"ttg.target" = "metal:32", "ttg.num-warps" = 4 : i32} {\n'
        "  %x = arith.constant 1.0 : f32\n}",
        '#loc8 = loc("x_ptr"(#loc))\n%x = arith.constant 1.0 : f32 loc(#loc8)',
        'tt.func public @k(%p: !tt.ptr<f32> loc("p"(#loc))) {\n tt.return\n}',
    ],
)
def test_custom_spelling_and_quoted_module_attributes_remain_admitted(monkeypatch, text):
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    assert cpp_refusal_reason(text) is None
    assert not MetalBackend._has_complex_ops(text)


def test_dtype_gate_ignores_source_path_and_comment_text(monkeypatch):
    """A checkout name is not semantic dtype evidence for C++ routing."""
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "1")
    text = (
        '#loc = loc("/tmp/fix-bf16-path/kernel.py":1:1)\n'
        "%x = arith.addf %a, %b : tensor<128xf32> // mentions f16 and i8\n"
    )
    assert cpp_safe_text(text)
    assert not MetalBackend._has_complex_ops(text)


@pytest.mark.parametrize("dtype", ["f16", "bf16", "i8", "i16", "i1"])
def test_dtype_gate_still_rejects_real_unsafe_element_types(dtype):
    assert not cpp_safe_text(f"%x = tt.load %p : tensor<128x{dtype}>")
