"""AST/IRSource signatures, rather than Python value guesses, determine binding.

CPU mapping pins precede the fix. Actual IRSource GPU rows require typed scalar
bytes on the host route and compare against independently rounded scalar values.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

from triton_msl.backend import _cache_contract as contract, driver
from triton_msl.errors import MetalNonRecoverableError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_packed_launch_contract import _metadata


@pytest.fixture
def controlled(monkeypatch, tmp_path):
    monkeypatch.setattr(contract, "toolchain_identity", lambda: "controlled-toolchain")
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))


@pytest.mark.parametrize(
    "source_sig,expected",
    [
        ({0: "*f16", 1: "f16"}, {0: "*fp16", 1: "fp16"}),
        ({0: "*bf16", 1: "bf16"}, {0: "*bf16", 1: "bf16"}),
        ({0: "*f32", 1: "f32"}, {0: "*fp32", 1: "fp32"}),
        ({0: "*i64", 1: "i64"}, {0: "*i64", 1: "i64"}),
    ],
)
def test_irsource_dense_positions_and_aliases_are_preserved(controlled, source_sig, expected):
    launcher = driver.MetalLauncher(SimpleNamespace(signature=source_sig), _metadata())
    assert launcher.arg_names == [0, 1]
    assert launcher.signature == expected


@pytest.mark.parametrize(
    "signature,expected",
    [
        ({"X": "*fp32", "BLOCK": "constexpr", "v": "fp16", "O": "*fp32"}, ["X", "BLOCK", "v", "O"]),
        ({"X": "*fp32", "v": "fp16", "O": "*fp32"}, ["X", "v", "O"]),
        ({"O": "*fp32", "v": "fp16", "X": "*fp32"}, ["X", "v", "O"]),
    ],
)
def test_ast_partial_signature_does_not_insert_omitted_constexpr(controlled, signature, expected):
    src = SimpleNamespace(
        signature=signature, constants={(1,): 32}, fn=SimpleNamespace(arg_names=["X", "BLOCK", "v", "O"])
    )
    launcher = driver.MetalLauncher(src, _metadata())
    assert launcher.arg_names == expected
    assert launcher.constexpr_indices == ({1} if "BLOCK" in signature else set())


@pytest.mark.parametrize("signature", [{1: "fp16"}, {0: "fp16", 2: "fp16"}, {True: "fp16"}, {"value": "fp16"}])
def test_unbound_irsource_positions_refuse(controlled, signature):
    with pytest.raises(MetalNonRecoverableError, match="source signature"):
        driver.MetalLauncher(SimpleNamespace(signature=signature), _metadata())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize(
    "dtype,torch_dtype",
    [
        ("f16", torch.float16),
        ("bf16", torch.bfloat16),
        ("f32", torch.float32),
        ("i8", torch.int8),
        ("i16", torch.int16),
        ("i32", torch.int32),
        ("i64", torch.int64),
    ],
)
@pytest.mark.parametrize("value", [1.5, -2.75])
def test_irsource_narrow_scalar_host_binding(tmp_path, monkeypatch, dtype, torch_dtype, value):
    _run_irsource_scalar(tmp_path, monkeypatch, dtype, torch_dtype, value, allow_fast=False)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
@pytest.mark.parametrize("value", [2, True])
def test_irsource_numeric_cast_host_binding(tmp_path, monkeypatch, value):
    # Even with the fast route enabled, int/bool→declared-fp32 needs typed host
    # packing. Passing raw Python integer bits to constant float& is not a cast.
    _run_irsource_scalar(tmp_path, monkeypatch, "f32", torch.float32, value, allow_fast=True)


def _run_irsource_scalar(tmp_path, monkeypatch, dtype, torch_dtype, value, *, allow_fast):
    if dtype.startswith("i"):
        value = int(value)  # integer controls are actually passed as Python ints
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1" if allow_fast else "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    path = tmp_path / "scalar.ttgir"
    path.write_text(f"""module attributes {{"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @scalar(%p: !tt.ptr<{dtype}>, %v: {dtype}) {{
    tt.store %p, %v : !tt.ptr<{dtype}>
    tt.return
  }}
}}""")
    compiled = triton.compile(
        str(path), target=GPUTarget("metal", "apple-m4", 32), options={"num_warps": 4, "target_metal_version": "3.2"}
    )
    print("IR_SIGNATURE", compiled.src.signature, flush=True)
    out = torch.full((1,), -99, device="mps", dtype=torch_dtype)
    utils = driver._get_utils()
    calls = []
    real = utils.launch

    def spy(*args, **kwargs):
        calls.append(1)
        scalar_buffer, _ = args[3][1]
        required = torch.empty((), dtype=torch_dtype).element_size()
        # Stop before GPU dispatch if the host allocated too few scalar bytes.
        # Such a baseline failure is an ABI violation, not a measured GPU wrong.
        assert scalar_buffer.length() >= required, (dtype, scalar_buffer.length(), required)
        return real(*args, **kwargs)

    monkeypatch.setattr(utils, "launch", spy)
    compiled[(1, 1, 1)](out, value)
    torch.mps.synchronize()
    print("IR_SCALAR", dtype, value, "GOT", out.item(), "HOST_HITS", len(calls), flush=True)
    assert len(calls) == 1
    torch.testing.assert_close(out.cpu(), torch.tensor([value], dtype=torch_dtype), atol=0, rtol=0)


@pytest.mark.parametrize(
    "dtype,value,hex_bytes",
    [
        ("i1", False, "00"),
        ("i1", True, "01"),
        ("u1", 1, "01"),
        ("i8", -2, "fe"),
        ("u8", 254, "fe"),
        ("i16", -2, "feff"),
        ("u16", 65534, "feff"),
        ("i32", -2, "feffffff"),
        ("u32", 4294967294, "feffffff"),
        ("i64", 2, "0200000000000000"),
        ("i64", -2, "feffffffffffffff"),
        ("u64", (1 << 63) + 3, "0300000000000080"),
        ("f16", 1.5, "003e"),
        ("fp16", 1.00048828125, "003c"),
        ("fp16", 1.00146484375, "023c"),
        ("bf16", 1.5, "c03f"),
        ("bf16", 1.00390625, "803f"),
        ("bf16", 1.01171875, "823f"),
        ("f32", 1.5, "0000c03f"),
        ("fp32", -2, "000000c0"),
    ],
)
def test_scalar_bytes_follow_declared_width(dtype, value, hex_bytes):
    from triton_msl.backend._launch_signature import scalar_bytes

    assert scalar_bytes(value, dtype) == bytes.fromhex(hex_bytes)


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("i8", 128),
        ("u8", -1),
        ("i64", 1 << 63),
        ("u64", 1 << 64),
        ("i32", 1.5),
        ("i1", 2),
        ("fp64", 1.5),
        (None, 1),
        ("unknown", 1),
    ],
)
def test_unproved_scalar_conversion_refuses(dtype, value):
    from triton_msl.backend._launch_signature import scalar_bytes

    with pytest.raises(MetalNonRecoverableError, match="source signature"):
        scalar_bytes(value, dtype)


@pytest.mark.parametrize(
    "dtype,value,safe",
    [
        ("fp32", 1.5, True),
        ("fp32", 2, False),
        ("fp32", True, False),
        ("i32", 1.5, False),
        ("i64", 2, True),
        ("i8", 128, False),
        ("u8", -1, False),
    ],
)
def test_fast_scalar_route_checks_python_representation(dtype, value, safe):
    launcher = SimpleNamespace(arg_names=["o", "v"], constexpr_indices=set(), signature={"o": "*fp32", "v": dtype})
    pointer = SimpleNamespace(data_ptr=lambda: 0)
    assert driver._compile_shader_scalars_ok(launcher, [pointer, value]) is safe


def test_nested_constexpr_does_not_shift_output_copyback_positions():
    from triton_msl.backend._launch_signature import bind_arguments

    x, out = SimpleNamespace(data_ptr=lambda: 1), SimpleNamespace(data_ptr=lambda: 2)
    leaves, types, origins, payloads = bind_arguments(
        ((x, 2), out), ["pair", "out"], {"pair": ("*fp32", "constexpr"), "out": "*fp32"}
    )
    assert leaves == [x, out] and types == ["*fp32", "*fp32"]
    assert origins == [0, 1] and payloads == [None, None]


@pytest.mark.parametrize("args", [(), (1,), ((1, 2), 3), ((1,), 3)])
def test_argument_shape_must_match_source(args):
    from triton_msl.backend._launch_signature import bind_arguments

    with pytest.raises(MetalNonRecoverableError, match="source signature"):
        bind_arguments(args, ["p", "v"], {"p": "*fp32", "v": "fp32"})


@triton.jit
def _tuple_const_copy(pair, out):
    i = tl.arange(0, 32)
    tl.store(out + i, tl.load(pair[0] + i) + pair[1])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")
def test_tuple_constexpr_host_binding_copies_the_actual_output(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "0")
    monkeypatch.setenv("TRITON_MSL_USE_CPP", "0")
    _tuple_const_copy.device_caches.clear()
    x = torch.arange(32, device="mps", dtype=torch.float32)
    out = torch.full_like(x, -99)
    calls = []
    utils = driver._get_utils()
    real = utils.launch

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(utils, "launch", spy)
    compiled = _tuple_const_copy[(1,)]((x, tl.constexpr(2)), out)
    torch.mps.synchronize()
    print(
        "TUPLE_SIGNATURE",
        compiled.src.signature,
        "OUTPUT_INDICES",
        compiled.packed_metadata[4],
        "ERR",
        (out - (x + 2)).abs().max().item(),
        flush=True,
    )
    assert len(calls) == 1
    torch.testing.assert_close(out, x + 2, atol=0, rtol=0)
