"""Wide-scan contract pins for packet 325.

This is intentionally a mechanism matrix, not the 101-row Cartesian matrix from
the packet-184 prototype.  The upstream scan census supplies broad shape/dtype
coverage once its >1024 skip is removed; these rows pin the distinct lowering
contracts that are easy to regress without bloating the project suite.
"""

from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import triton_msl

from triton_msl.errors import MetalNonRecoverableError


requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")


@triton.jit
def _cumsum_2d(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr, REVERSE: tl.constexpr):
    row = tl.arange(0, M)[:, None]
    col = tl.arange(0, N)[None, :]
    offset = row * N + col
    value = tl.load(x_ptr + offset)
    result = tl.cumsum(value, axis=AXIS, reverse=REVERSE)
    tl.store(out_ptr + offset, result)


@triton.jit
def _cumprod_2d(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)[:, None]
    col = tl.arange(0, N)[None, :]
    offset = row * N + col
    value = tl.load(x_ptr + offset)
    tl.store(out_ptr + offset, tl.cumprod(value, axis=1))


@triton.jit
def _cummax_combine(left_v, left_i, right_v, right_i):
    take_left = left_v > right_v
    return tl.where(take_left, left_v, right_v), tl.where(take_left, left_i, right_i)


@triton.jit
def _cummax_2d(x_ptr, out_ptr, index_ptr, M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)[:, None]
    col = tl.arange(0, N)[None, :]
    offset = row * N + col
    value = tl.load(x_ptr + offset)
    index = tl.broadcast_to(col.to(tl.int64), (M, N))
    out, out_index = tl.associative_scan((value, index), 1, _cummax_combine)
    tl.store(out_ptr + offset, out)
    tl.store(index_ptr + offset, out_index)


@triton.jit
def _cumsum_1d(x_ptr, out_ptr, N: tl.constexpr):
    offset = tl.arange(0, N)
    value = tl.load(x_ptr + offset)
    tl.store(out_ptr + offset, tl.cumsum(value, axis=0))


@triton.jit
def _get_first(left, right):
    return left


@triton.jit
def _first_2d(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, REVERSE: tl.constexpr):
    row = tl.arange(0, M)[:, None]
    col = tl.arange(0, N)[None, :]
    offset = row * N + col
    value = tl.load(x_ptr + offset)
    result = tl.associative_scan(value, axis=1, combine_fn=_get_first, reverse=REVERSE)
    tl.store(out_ptr + offset, result)


@triton.jit
def _two_cumsums_1d(x_ptr, out_a_ptr, out_b_ptr, N: tl.constexpr):
    offset = tl.arange(0, N)
    value = tl.load(x_ptr + offset)
    tl.store(out_a_ptr + offset, tl.cumsum(value, axis=0))
    tl.store(out_b_ptr + offset, tl.cumsum(value + 1, axis=0))


@triton.jit
def _masked_strided_cumsum_2d(
    x_ptr,
    out_ptr,
    valid_n,
    row_stride,
    col_stride,
    M: tl.constexpr,
    N: tl.constexpr,
):
    row = tl.arange(0, M)[:, None]
    col = tl.arange(0, N)[None, :]
    address = row * row_stride + col * col_stride
    mask = col < valid_n
    value = tl.load(x_ptr + address, mask=mask, other=0.0)
    result = tl.cumsum(value, axis=1)
    tl.store(out_ptr + address, result, mask=mask)


def _reference_scan(x, axis, reverse):
    source = x.flip(axis) if reverse else x
    out = torch.cumsum(source, axis)
    return out.flip(axis) if reverse else out


def _scan_lowerer(fn=_cumsum_2d):
    from tests.test_fa_bwd_routing import _build_lowerer

    if expected_root := os.environ.get("TRITON_MSL_EXPECTED_ROOT"):
        assert Path(triton_msl.__file__).resolve().is_relative_to(Path(expected_root).resolve())
    if fn is _cummax_2d:
        signature = {"x_ptr": "*fp32", "out_ptr": "*fp32", "index_ptr": "*i64"}
        constants = {"M": 2, "N": 1024}
    else:
        signature = {"x_ptr": "*fp32", "out_ptr": "*fp32"}
        constants = {"M": 2, "N": 1024, "AXIS": 1, "REVERSE": False}
    return _build_lowerer(fn, signature, constants)


def test_wide_scan_contract_uses_native_metadata_not_legacy_strings():
    """The capability proof must survive unusable first-result spellings."""
    lowerer = _scan_lowerer()
    scan = next(op for op in lowerer.graph.ops if op.op == "tt.scan")

    pending = list(lowerer.graph.ops)
    while pending:
        op = pending.pop()
        op.type_str = "unusable legacy type"
        pending.extend(op.region_ops or [])
        pending.extend(op.else_ops or [])
    actual = lowerer._prove_scan_native_contract(scan)

    assert actual["shape"] == (2, 1024)
    assert actual["slot_dtypes"] == ["fp32"]


def test_wide_scan_refuses_missing_or_contradictory_native_metadata():
    """One node covers four distinct per-result ownership/type boundaries."""
    for damage in ("missing_operand", "wrong_result_index", "wrong_block", "wrong_return_width"):
        lowerer = _scan_lowerer(_cummax_2d)
        scan = next(op for op in lowerer.graph.ops if op.op == "tt.scan")
        result_ids = list(scan.result_ids)
        block_ids = list(scan.attrs["block_arg_ids"])
        return_op = next(op for op in scan.region_ops if op.op == "tt.scan.return")

        if damage == "missing_operand":
            lowerer.graph.result_meta.pop(scan.operand_ids[0])
        elif damage == "wrong_result_index":
            meta = lowerer.graph.result_meta[result_ids[1]]
            lowerer.graph.result_meta[result_ids[1]] = replace(meta, result_index=0)
        elif damage == "wrong_block":
            meta = lowerer.graph.result_meta[block_ids[-1]]
            lowerer.graph.result_meta[block_ids[-1]] = replace(meta, owner_id=meta.owner_id + 1)
        else:
            value_id = return_op.operand_ids[1]
            meta = lowerer.graph.result_meta[value_id]
            lowerer.graph.result_meta[value_id] = replace(meta, type=replace(meta.type, width=32))

        with pytest.raises(MetalNonRecoverableError, match="native per-result metadata"):
            lowerer.lower()


@requires_mps
@pytest.mark.parametrize(
    "shape,axis,reverse,dtype,num_warps",
    [
        ((2, 1024), 1, False, torch.float32, 4),
        ((1024, 2), 0, True, torch.int32, 16),
        ((4, 512), 1, True, torch.bfloat16, 4),
    ],
    ids=["row-fp32-w4", "column-reverse-i32-w16", "row-reverse-bf16-w4"],
)
def test_wide_cumsum_projects_flat_ownership(cold_gpu_caches, shape, axis, reverse, dtype, num_warps):
    torch.manual_seed(325)
    if dtype == torch.int32:
        x = torch.randint(-3, 4, shape, device="mps", dtype=dtype)
    else:
        x = torch.randn(shape, device="mps", dtype=dtype)
    out = torch.empty_like(x)
    _cumsum_2d[(1,)](
        x,
        out,
        M=shape[0],
        N=shape[1],
        AXIS=axis,
        REVERSE=reverse,
        num_warps=num_warps,
    )
    torch.mps.synchronize()
    reference = _reference_scan(x.float() if dtype == torch.bfloat16 else x, axis, reverse)
    if dtype == torch.int32:
        assert torch.equal(out.cpu(), reference.cpu())
    else:
        tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-4
        assert torch.allclose(out.float().cpu(), reference.cpu(), rtol=tolerance, atol=8 * tolerance)


@requires_mps
def test_wide_cumprod_replays_the_combiner(cold_gpu_caches):
    torch.manual_seed(326)
    x = (1 + torch.randn((2, 1024), device="mps") * 1e-3).float()
    out = torch.empty_like(x)
    _cumprod_2d[(1,)](x, out, M=2, N=1024, num_warps=4)
    torch.mps.synchronize()
    assert torch.allclose(out.cpu(), torch.cumprod(x.cpu(), 1), rtol=2e-4, atol=2e-4)


@requires_mps
def test_wide_multivalue_scan_preserves_per_slot_types(cold_gpu_caches):
    torch.manual_seed(327)
    x = torch.randn((2, 1024), device="mps")
    out = torch.empty_like(x)
    index = torch.empty((2, 1024), device="mps", dtype=torch.int64)
    _cummax_2d[(1,)](x, out, index, M=2, N=1024, num_warps=4)
    torch.mps.synchronize()
    reference, reference_index = torch.cummax(x.cpu(), 1)
    assert torch.equal(out.cpu(), reference)
    assert torch.equal(index.cpu(), reference_index.to(torch.int64))


@requires_mps
def test_wide_reverse_scan_preserves_noncommutative_operand_order(cold_gpu_caches):
    x = torch.arange(2048, device="mps", dtype=torch.int32).reshape(2, 1024)
    for reverse in (False, True):
        out = torch.empty_like(x)
        _first_2d[(1,)](x, out, M=2, N=1024, REVERSE=reverse, num_warps=4)
        torch.mps.synchronize()
        boundary = x[:, -1:] if reverse else x[:, :1]
        assert torch.equal(out.cpu(), boundary.expand_as(x).cpu())


@requires_mps
def test_wide_scan_handles_unaligned_storage(cold_gpu_caches):
    torch.manual_seed(328)
    backing = torch.randn(2049, device="mps")
    x = backing[1:].reshape(2, 1024)
    out_backing = torch.empty(2049, device="mps")
    out = out_backing[1:].reshape(2, 1024)
    _cumsum_2d[(1,)](x, out, M=2, N=1024, AXIS=1, REVERSE=False, num_warps=4)
    torch.mps.synchronize()
    assert torch.allclose(out.cpu(), torch.cumsum(x.cpu(), 1), rtol=1e-4, atol=8e-4)


@requires_mps
def test_wide_scan_proves_runtime_strides_and_tail_masks(cold_gpu_caches):
    """Address and mask arrays must share the exact flat row/column projection."""
    torch.manual_seed(329)
    m, n, valid_n = 2, 1024, 777
    row_stride, col_stride = 2053, 2
    storage = row_stride * m
    x = torch.zeros(storage, device="mps")
    out = torch.full((storage,), -123.0, device="mps")
    logical = torch.randn((m, valid_n), device="mps")
    rows = torch.arange(m, device="mps")[:, None]
    cols = torch.arange(valid_n, device="mps")[None, :]
    addresses = rows * row_stride + cols * col_stride
    x[addresses] = logical
    _masked_strided_cumsum_2d[(1,)](
        x,
        out,
        valid_n,
        row_stride,
        col_stride,
        M=m,
        N=n,
        num_warps=4,
    )
    torch.mps.synchronize()
    assert torch.allclose(out[addresses].cpu(), torch.cumsum(logical.cpu(), 1), rtol=1e-4, atol=8e-4)
    tail_addresses = rows * row_stride + torch.arange(valid_n, n, device="mps")[None, :] * col_stride
    assert torch.equal(out[tail_addresses].cpu(), torch.full((m, n - valid_n), -123.0))


@requires_mps
def test_wide_1d_scan_uses_register_array(cold_gpu_caches):
    x = torch.randint(-5, 6, (4096,), device="mps", dtype=torch.int32)
    out = torch.empty_like(x)
    kernel = _cumsum_1d[(1,)](x, out, N=4096, num_warps=4)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), torch.cumsum(x.cpu(), 0))
    assert "scan_d <<= 1u" in kernel.asm["msl"]


@requires_mps
@pytest.mark.parametrize("n", [8192, 16384], ids=["equal-32k", "above-32k"])
def test_scan_at_or_above_threadgroup_budget_refuses(cold_gpu_caches, n):
    """A single i32 slot at or above 32 KiB must refuse with its budget."""
    x = torch.zeros(n, device="mps", dtype=torch.int32)
    out = torch.empty_like(x)
    with pytest.raises(MetalNonRecoverableError, match="32 KiB"):
        _cumsum_1d[(1,)](x, out, N=n, num_warps=4)


@requires_mps
def test_scan_budget_includes_an_earlier_live_scan_buffer(cold_gpu_caches):
    """Two 4096-element i32 scan buffers coexist and exactly fill 32 KiB."""
    x = torch.zeros(4096, device="mps", dtype=torch.int32)
    out_a = torch.empty_like(x)
    out_b = torch.empty_like(x)
    with pytest.raises(MetalNonRecoverableError, match="all live static.*32768"):
        _two_cumsums_1d[(1,)](x, out_a, out_b, N=4096, num_warps=4)
