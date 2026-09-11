"""Emission contracts for aliased two-dimensional reduce scratch."""

import re

import pytest
import triton
import triton.language as tl

from tests.test_template_scalar_abi import _lower


@triton.jit
def _reduce_axis_1(x, y, M: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    values = tl.load(x + rows[:, None] * N + cols[None, :])
    tl.store(y + rows, tl.sum(values, 1))


@triton.jit
def _reduce_axis_0(x, y, M: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    values = tl.load(x + rows[:, None] * N + cols[None, :])
    tl.store(y + cols, tl.sum(values, 0))


@pytest.mark.parametrize(
    "kernel,axis,extent,loop_var",
    [(_reduce_axis_1, 1, 64, "j"), (_reduce_axis_0, 0, 16, "i")],
)
def test_aliased_reduce_waits_for_all_reads_before_result_write(kernel, axis, extent, loop_var):
    compiled = _lower(kernel, {"x": "*fp32", "y": "*fp32"}, {"M": 64, "N": 16})
    msl = compiled.lower()

    # The allocator deliberately pools the full input and the small result.
    declarations = re.findall(r"threadgroup float (shared_\d+)\[(\d+)\];", msl)
    assert declarations == [("shared_0", "1024")]

    # Every reducer first commits to a register. No pooled result slot may be
    # overwritten until all simdgroups have completed their input-array reads.
    pattern = rf"""
        for\s*\(uint\s+{loop_var}\s*=\s*0;[^{{]+\{{.*?
        reduced_\d+\s*=\s*acc;\s*
        \}}\s*
        threadgroup_barrier\(mem_flags::mem_threadgroup\);\s*
        if\s*\(lid\s*<\s*{extent}u\)\s+shared_0\[lid\]\s*=\s*reduced_\d+;
    """
    assert re.search(pattern, msl, re.DOTALL | re.VERBOSE), f"axis {axis} result write lacks its pre-write barrier"
