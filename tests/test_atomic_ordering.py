"""Closed atomic metadata and fence placement at the lowering boundary (190/201)."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl

from test_fa_bwd_routing import _build_lowerer
from triton_msl.backend.device_detect import _parse_chip
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _atomic(P, O, V, SEM: tl.constexpr, SCOPE: tl.constexpr, KIND: tl.constexpr, N: tl.constexpr):
    x = tl.arange(0, N)
    if KIND == "cas":
        cmp = tl.full((N,), 0, P.dtype.element_ty)
        val = tl.full((N,), V, P.dtype.element_ty)
        old = tl.atomic_cas(P + x, cmp, val, sem=SEM, scope=SCOPE)
    elif KIND == "exchange":
        old = tl.atomic_xchg(P + x, V, sem=SEM, scope=SCOPE)
    elif KIND == "and":
        old = tl.atomic_and(P + x, V, mask=x % 2 == 0, sem=SEM, scope=SCOPE)
    elif KIND == "or":
        old = tl.atomic_or(P + x, V, mask=x % 2 == 0, sem=SEM, scope=SCOPE)
    elif KIND == "xor":
        old = tl.atomic_xor(P + x, V, mask=x % 2 == 0, sem=SEM, scope=SCOPE)
    elif KIND == "max":
        old = tl.atomic_max(P + x, V, mask=x % 2 == 0, sem=SEM, scope=SCOPE)
    elif KIND == "min":
        old = tl.atomic_min(P + x, V, mask=x % 2 == 0, sem=SEM, scope=SCOPE)
    else:
        old = tl.atomic_add(P + x, V, mask=x % 2 == 0, sem=SEM, scope=SCOPE)
    tl.store(O + x, old)


@triton.jit
def _default(P, O):
    old = tl.atomic_add(P, 1)
    tl.store(O, old)


@triton.jit
def _mixed_regions(P, O, COUNT):
    x = tl.arange(0, 32)
    old = tl.atomic_add(P + x, 1, sem="release", scope="cta")
    for i in range(COUNT):
        old += tl.atomic_xchg(P + x, i, sem="acquire", scope="gpu")
    tl.store(O + x, old)


FAMILIES = [
    ("add", "i32"),
    ("add", "u32"),
    ("add", "fp32"),
    ("exchange", "fp32"),
    ("add", "fp16"),
    ("exchange", "fp16"),
    ("add", "bf16"),
    ("exchange", "bf16"),
    ("cas", "i32"),
    ("cas", "fp32"),
    ("and", "i32"),
    ("or", "i32"),
    ("xor", "i32"),
    ("max", "i32"),
    ("min", "i32"),
    ("max", "u32"),
    ("min", "u32"),
]
SEMS = ["relaxed", "acquire", "release", "acq_rel"]


@pytest.fixture(autouse=True)
def known_cpu_target(monkeypatch):
    # This is a CPU codegen contract test, not evidence about the machine's GPU.
    monkeypatch.setattr(
        "triton_msl.backend.device_detect.get_device_info",
        lambda: SimpleNamespace(chip_family="M4", metal_version="3.2", metal_std_flag="-std=metal3.2"),
    )


def _lowerer(kind="add", dt="i32", sem="acq_rel", scope="gpu", n=32, version="3.2"):
    # The public frontend disallows 16-bit exchange, but the lowerer has a word-CAS
    # exchange branch. Exercise that branch with an explicit typed graph mutation,
    # not a claim that the unsupported Python spelling currently compiles.
    word_exchange = kind == "exchange" and dt in ("fp16", "bf16")
    low = _build_lowerer(
        _atomic,
        {"P": "*" + dt, "O": "*" + dt, "V": "fp32" if "f" in dt else dt},
        {"SEM": sem, "SCOPE": scope, "KIND": "add" if word_exchange else kind, "N": n},
    )
    if word_exchange:
        (atomic,) = [op for op in low.graph.ops if op.op == "tt.atomic_rmw"]
        atomic.attrs["rmw_op"] = "exch"
    low.options = replace(low.options, target_metal_version=version)
    return low


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("sem", SEMS)
@pytest.mark.parametrize("scope", ["gpu", "cta"])
def test_each_family_brackets_one_logical_atomic(family, sem, scope):
    low = _lowerer(*family, sem, scope)
    (atomic,) = [op for op in low.graph.ops if op.op.startswith("tt.atomic_")]
    assert atomic.attrs["sem"] == sem and atomic.attrs["scope"] == scope
    if family[0] != "cas":
        expected_op = {"exchange": "exch"}.get(family[0], family[0])
        if family[1] == "u32" and family[0] in ("max", "min"):
            expected_op = "u" + family[0]
        if family[0] == "add" and "f" in family[1]:
            expected_op = "fadd"
        assert atomic.attrs["rmw_op"] == expected_op
    msl = low.lower()
    fence = (
        "atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, "
        "memory_order_seq_cst, thread_scope_" + ("device" if scope == "gpu" else "threadgroup") + ");"
    )
    expected = int(sem in ("release", "acq_rel")) + int(sem in ("acquire", "acq_rel"))
    assert msl.count("atomic_thread_fence(") == expected
    assert msl.count(fence) == expected
    first_atomic = min(
        msl.index(s)
        for s in ("atomic_load_explicit(", "atomic_fetch_", "atomic_exchange_", "atomic_compare_exchange_")
        if s in msl
    )
    last_atomic = max(
        msl.rindex(s)
        for s in ("atomic_load_explicit(", "atomic_fetch_", "atomic_exchange_", "atomic_compare_exchange_")
        if s in msl
    )
    if sem in ("release", "acq_rel"):
        assert msl.index(fence) < first_atomic
    if sem in ("acquire", "acq_rel"):
        assert msl.rindex(fence) > last_atomic
    # Fences stay outside a retry loop, and inside the same mask guard as the atomic.
    depth = 0
    depths = []
    for line in msl.splitlines():
        if "atomic_thread_fence(" in line:
            depths.append(depth)
        if "while (" in line:
            loop_depth = depth
        depth += line.count("{") - line.count("}")
    if "while (" in msl:
        assert all(d == loop_depth for d in depths)


# gpu/cta metadata is already asserted for every sem in the family matrix above.
# Keep the separate decoder check only for sys, which cannot reach emission.
@pytest.mark.parametrize("scope", ["sys"])
@pytest.mark.parametrize("sem", SEMS)
@pytest.mark.parametrize("kind", ["add", "cas"])
def test_native_walker_retains_semantics_and_scope(scope, sem, kind):
    low = _lowerer(kind=kind, sem=sem, scope=scope)
    (atomic,) = [op for op in low.graph.ops if op.op.startswith("tt.atomic_")]
    assert atomic.attrs["sem"] == sem
    assert atomic.attrs["scope"] == scope


@pytest.mark.parametrize("sem", SEMS)
def test_system_scope_refuses_even_when_relaxed(sem):
    with pytest.raises(MetalNonRecoverableError, match="scope"):
        _lowerer(sem=sem, scope="sys").lower()


@pytest.mark.parametrize(
    "field,bad",
    [("sem", None), ("sem", "unknown"), ("scope", None), ("scope", "unknown"), ("rmw_op", None), ("rmw_op", "unknown")],
)
def test_missing_or_unknown_metadata_never_defaults(field, bad):
    low = _lowerer()
    (atomic,) = [op for op in low.graph.ops if op.op.startswith("tt.atomic_")]
    if bad is None:
        atomic.attrs.pop(field, None)
    else:
        atomic.attrs[field] = bad
    with pytest.raises(MetalNonRecoverableError, match="atomic"):
        low.lower()


@pytest.mark.parametrize("version", ["3.0", "3.1", "invalid", "3.2-extra", "4.1"])
def test_ordered_atomic_requires_a_proved_language_target(version):
    with pytest.raises(MetalNonRecoverableError, match="Metal"):
        _lowerer(version=version).lower()


def test_auto_target_is_checked_not_assumed(monkeypatch):
    monkeypatch.setattr(
        "triton_msl.backend.device_detect.get_device_info",
        lambda: SimpleNamespace(chip_family="M1", metal_version="3.1"),
    )
    with pytest.raises(MetalNonRecoverableError, match="Metal"):
        _lowerer(version="auto").lower()


def test_explicit_language_does_not_override_device_capability(monkeypatch):
    monkeypatch.setattr(
        "triton_msl.backend.device_detect.get_device_info",
        lambda: SimpleNamespace(chip_family="M1", metal_version="3.1"),
    )
    with pytest.raises(MetalNonRecoverableError, match="Metal"):
        _lowerer(version="3.2").lower()


def test_detected_chip_family_normalization_reaches_ordered_atomic_gate(monkeypatch):
    # Pin the producer/consumer coupling: device detection must keep the chip
    # variant separate because the ordered-atomic gate accepts the normalized
    # family token, not a display name such as "M4 Max".
    family, variant = _parse_chip("Apple M4 Max")
    assert (family, variant) == ("M4", "Max")
    monkeypatch.setattr(
        "triton_msl.backend.device_detect.get_device_info",
        lambda: SimpleNamespace(chip_family=family, metal_version="3.2", metal_std_flag="-std=metal3.2"),
    )
    msl = _lowerer(version="3.2").lower()
    assert msl.count("atomic_thread_fence(") == 2


def test_relaxed_old_target_needs_no_fence():
    assert "atomic_thread_fence(" not in _lowerer(sem="relaxed", version="3.0").lower()


def test_source_default_is_acq_rel_not_relaxed():
    low = _build_lowerer(_default, {"P": "*i32", "O": "*i32"}, {})
    low.options = replace(low.options, target_metal_version="3.2")
    (atomic,) = [op for op in low.graph.ops if op.op.startswith("tt.atomic_")]
    assert atomic.attrs["sem"] == "acq_rel" and atomic.attrs["scope"] == "gpu"
    assert low.lower().count("atomic_thread_fence(") == 2


def test_native_metadata_stays_with_its_operation_across_regions():
    low = _build_lowerer(_mixed_regions, {"P": "*i32", "O": "*i32", "COUNT": "i32"}, {})
    low.options = replace(low.options, target_metal_version="3.2")

    def walk(ops):
        for op in ops:
            yield op
            yield from walk(op.region_ops or [])
            yield from walk(op.else_ops or [])

    atomics = [op for op in walk(low.graph.ops) if op.op == "tt.atomic_rmw"]
    assert len(atomics) == 2
    assert {(op.attrs["rmw_op"], op.attrs["sem"], op.attrs["scope"]) for op in atomics} == {
        ("add", "release", "cta"),
        ("exch", "acquire", "gpu"),
    }
    msl = low.lower()
    cta = msl.index("thread_scope_threadgroup")
    gpu = msl.index("thread_scope_device")
    assert cta < msl.index("atomic_fetch_add_explicit")
    assert msl.index("atomic_exchange_explicit") < gpu
    assert msl.count("atomic_thread_fence(") == 2


@pytest.mark.parametrize("n", [1, 8, 256, 1024, 2048])
@pytest.mark.parametrize("dt", ["i32", "fp32", "fp16", "bf16"])
def test_mask_and_underfill_guard_enclose_both_fences(n, dt):
    low = _lowerer(dt=dt, n=n)
    msl = low.lower()
    if n == 2048:
        assert low._needs_wrapping and "for (" in msl and "_loop_e" in msl
    # Retain the actual enclosing braces, not merely equal indentation: both
    # fences and the first atomic load/fetch must be inside the SAME mask/owner
    # block. Retry-loop inner branches are deliberately not compared here.
    stack = []
    boundaries = []
    first_atomic = None
    for index, line in enumerate(msl.splitlines()):
        for _ in range(line.count("}")):
            stack.pop()
        if "atomic_thread_fence(" in line:
            boundaries.append(tuple(stack))
        if first_atomic is None and ("atomic_load_explicit(" in line or "atomic_fetch_add_explicit(" in line):
            first_atomic = tuple(stack)
        for _ in range(line.count("{")):
            stack.append(index)
    assert len(boundaries) == 2
    assert boundaries[0] == first_atomic == boundaries[1]
    assert len(first_atomic) >= 2  # kernel plus active-lane/mask guard
