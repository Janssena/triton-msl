"""CPU checks of emitted memory-order and direct-maker column coverage."""

from collections import Counter
import re

import pytest

from triton_msl.codegen.msl_emitter import make_flash_attention_kernel_simdgroup
from triton_msl.codegen._msl_templates import make_kda_decode_kernel


def test_kda_decode_orders_cross_thread_state_reads_and_writes():
    source = make_kda_decode_kernel()
    first_read = source.index("acc += k[hd+l]*a[hd+l]*S[")
    write = source.index("S[hS+idx] =")
    last_read = source.index("acc += q[hd+l]*S[")
    # State (l=1,j=0) is read by thread 0 but updated by thread 64.
    assert (1 * 64 + 0) % 256 != 0
    for boundary in (source[first_read:write], source[write:last_read]):
        (flags,) = re.findall(r"threadgroup_barrier\(([^)]+)\)", boundary)
        assert "mem_flags::mem_threadgroup" in flags
        assert "mem_flags::mem_device" in flags


@pytest.mark.parametrize(
    "qk,v",
    [
        (128, 32),
        (128, 40),
        (192, 96),
        (32, 96),
        (96, 96),
        (192, 72),
        (128, 120),
        (192, 136),
        (128, 248),
        (8, 8),
        (32, 32),
        (128, 64),
        (192, 128),
        (192, 192),
    ],
)
@pytest.mark.parametrize("dtype,half_accumulate", [("fp32", False), ("fp16", False), ("fp16", True)])
def test_direct_maker_covers_each_value_column_once_without_out_of_bounds(qk, v, dtype, half_accumulate):
    if dtype == "fp32" and qk == 192:
        # This combination already exceeds the scratch budget. Keep its
        # refusal; column-coverage repair must not widen the resource envelope.
        with pytest.raises(ValueError, match="threadgroup memory.*32KB"):
            make_flash_attention_kernel_simdgroup(head_dim=qk, v_head_dim=v, out_dtype=dtype)
        return
    source = make_flash_attention_kernel_simdgroup(
        head_dim=qk, v_head_dim=v, out_dtype=dtype, half_accumulate=half_accumulate
    )
    rounds = int(re.search(r"TPG = (\d+)u", source).group(1))
    store_guard = re.search(r"if \(qr2 < N_CTX && \(dc2 < (D|\d+u)\)\)", source)
    bound = (
        qk if store_guard and store_guard.group(1) == "D" else (int(store_guard.group(1)[:-1]) if store_guard else None)
    )
    written = Counter()
    for group in range(8):
        for tile in range(rounds):
            column_tile = group + tile * 8
            for column in range(column_tile * 8, column_tile * 8 + 8):
                if bound is None or column < bound:
                    written[column] += 1
    assert written == Counter(range(v))
    # Full V loads are 8-wide. Even surplus groups must load a valid tile;
    # the tail staging predicate must use the same value-column boundary.
    full_guard = re.search(r"\(ct\*8u < (D|\d+u) \? ct\*8u : 0u\)", source)
    if store_guard:
        assert full_guard and full_guard.group(1) == store_guard.group(1)
        assert f"ct*8u + cc < {store_guard.group(1)}" in source
    for group in range(8):
        for tile in range(rounds):
            start = (group + tile * 8) * 8
            if full_guard and start >= bound:
                start = 0
            assert 0 <= start <= v - 8


@pytest.mark.parametrize("qk,v", [(0, 64), (64, 0), (-8, 64), (64, -8), (63, 64), (64, 63)])
def test_direct_maker_refuses_nonpositive_or_partial_fragment_width(qk, v):
    with pytest.raises(ValueError, match="head_dim"):
        make_flash_attention_kernel_simdgroup(head_dim=qk, v_head_dim=v)


@pytest.mark.parametrize("width", [8, 16, 32, 40, 64, 128, 192])
def test_symmetric_explicit_width_keeps_default_emission(width):
    assert make_flash_attention_kernel_simdgroup(
        head_dim=width, out_dtype="fp16"
    ) == make_flash_attention_kernel_simdgroup(head_dim=width, v_head_dim=width, out_dtype="fp16")
