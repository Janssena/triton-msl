"""CPU-only declaration/value and post-hook proofs; no device or pipeline setup."""

import struct
from types import SimpleNamespace

import pytest
import torch
import triton.backends

from triton_msl.backend import _launch_signature as signature, driver
from triton_msl.errors import MetalNonRecoverableError, PostSubmitError


@pytest.mark.parametrize(
    "ty,values",
    [
        ("i1", [False, True, 0, 1, 2]),
        ("u8", [0, 255, -1, 256]),
        ("i32", [1, -3, 1 << 31]),
        ("u64", [0, (1 << 63) + 3, 1 << 64]),
        ("fp16", [1.00048828125, 1.00146484375, -0.0, 1e20]),
        ("fp32", [1.5, 2, True, -0.0, float("inf"), float("nan"), 1e100]),
    ],
)
def test_planned_binding_observes_every_live_value_and_declared_width(ty, values):
    names = ["pointer", "value", "s0", "s1", "s2", "BLOCK"]
    declared = dict(pointer="*fp32", value=ty, s0="i32", s1="i32", s2="i32", BLOCK="constexpr")
    plan = signature.make_binding_plan(names, declared)
    assert plan is not None
    pointer = torch.empty(1, device="cpu")
    for value in values:
        args = (pointer, value, 1, 2, 3, object())
        try:
            original = signature.bind_arguments(args, names, declared)
        except MetalNonRecoverableError:
            with pytest.raises(MetalNonRecoverableError):
                signature.bind_arguments_with_plan(args, names, declared, plan)
        else:
            planned = signature.bind_arguments_with_plan(args, names, declared, plan)
            assert all(a is b for a, b in zip(original[0], planned[0]))
            assert planned[1:] == original[1:]
            planned[0].clear()
            assert signature.bind_arguments_with_plan(args, names, declared, plan)[0]


def test_mutated_public_declarations_use_the_live_binder():
    names = ["pointer", "value", "s0", "s1", "s2"]
    declared = dict(pointer="*fp32", value="i32", s0="i32", s1="i32", s2="i32")
    plan = signature.make_binding_plan(names, declared)
    assert plan is not None
    args = (torch.empty(1, device="cpu"), 128, 1, 2, 3)
    assert signature.bind_arguments_with_plan(args, names, declared, plan)[3][1] == struct.pack("<i", 128)
    declared["value"] = "i8"
    with pytest.raises(MetalNonRecoverableError):
        signature.bind_arguments_with_plan(args, names, declared, plan)
    declared["value"] = "fp32"
    assert signature.bind_arguments_with_plan(args, names, declared, plan)[3][1] == struct.pack("<f", 128)
    names.reverse()
    with pytest.raises(MetalNonRecoverableError):
        signature.bind_arguments_with_plan(args, names, declared, plan)


def test_custom_values_and_nested_declarations_preserve_original_behavior():
    class Integer(int):
        pass

    pointer = SimpleNamespace(data_ptr=lambda: 1)
    for names, declared, args in [
        (["p", "v"], {"p": "*fp32", "v": "i32"}, (pointer, Integer(2))),
        (["pair", "out"], {"pair": ("*fp32", "constexpr"), "out": "*fp32"}, ((pointer, 2), pointer)),
    ]:
        plan = signature.make_binding_plan(names, declared)
        assert signature.bind_arguments_with_plan(args, names, declared, plan) == signature.bind_arguments(
            args, names, declared
        )


def test_exact_tensor_getter_cannot_hide_a_changed_next_declaration(monkeypatch):
    pointer = torch.empty(1, device="cpu")
    for planned in (False, True):
        names, declared = ["p", "v"], dict(p="*fp32", v="i32")
        plan = signature.make_binding_plan(names, declared)
        calls = []

        def changed(self):
            calls.append("getter")
            declared["v"] = "i8"
            return lambda: 0

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "data_ptr", property(changed))
            with pytest.raises(MetalNonRecoverableError):
                if planned:
                    signature.bind_arguments_with_plan((pointer, 128), names, declared, plan)
                else:
                    signature.bind_arguments((pointer, 128), names, declared)
        assert calls == ["getter"]


def test_custom_scalar_fallback_never_replays_callbackful_prefix(monkeypatch):
    pointer = torch.empty(1, device="cpu")
    runs = []
    for planned in (False, True):
        calls = []

        class Integer(int):
            def __eq__(self, other):
                calls.append(("scalar_eq", other))
                return super().__eq__(other)

        def getter(self):
            calls.append(("pointer_getter",))
            return lambda: 0

        names, declared = ["p", "v"], dict(p="*fp32", v="i1")
        plan = signature.make_binding_plan(names, declared)
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "data_ptr", property(getter))
            args = (pointer, Integer(1))
            result = (
                signature.bind_arguments_with_plan(args, names, declared, plan)
                if planned
                else signature.bind_arguments(args, names, declared)
            )
        assert result[3] == [None, b"\x01"]
        runs.append(calls)
    assert runs[0] == runs[1]
    assert runs[1].count(("pointer_getter",)) == 1
    assert any(event[0] == "scalar_eq" for event in runs[1])


def test_eligibility_reuses_only_the_same_scalar_position_type_and_value(monkeypatch):
    pointer = torch.empty(1, device="cpu")
    names = ["p", "v", "BLOCK"]
    declared = dict(p="*fp32", v="i32", BLOCK="constexpr")
    launcher = SimpleNamespace(arg_names=names, signature=declared, constexpr_indices={2})
    args = (pointer, 128, 32)
    checked = signature.bind_arguments(args, names, declared)
    real = signature.scalar_bytes
    calls = []

    def pack(value, ty):
        calls.append((value, ty))
        return real(value, ty)

    monkeypatch.setattr(signature, "scalar_bytes", pack)
    assert driver._compile_shader_scalars_ok(launcher, list(args[:2]), checked=checked)
    assert calls == []
    assert not driver._compile_shader_scalars_ok(launcher, [pointer, 1 << 31], checked=checked)
    assert calls[-1] == (1 << 31, "i32")
    declared["v"] = "i8"  # A hook/runtime callback changed the live declaration.
    assert not driver._compile_shader_scalars_ok(launcher, list(args[:2]), checked=checked)
    assert calls[-1] == (128, "i8")
    declared["v"] = "fp32"
    assert not driver._compile_shader_scalars_ok(launcher, list(args[:2]), checked=checked)
    declared["v"] = "i32"
    launcher.constexpr_indices = {1}
    assert not driver._compile_shader_scalars_ok(launcher, [pointer, 32], checked=checked)


@pytest.mark.parametrize("mode", ["clean", "hook_changes_type", "dispatch_error", "exit_hook_error"])
def test_generic_dispatch_keeps_post_hook_and_submission_boundaries(monkeypatch, mode):
    from triton_msl.backend import _cache_contract, _launch_contract

    class CPUOnlyTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("mps")

    pointer = torch.empty(1, device="cpu").as_subclass(CPUOnlyTensor)

    class HostFallback(Exception):
        pass

    class Utils:
        @property
        def buffer_pool(self):
            raise HostFallback

    class Runtime:
        def available(self):
            return True

        def is_unsupported(self, _):
            return False

        def get_library(self, _):
            return object()

        def mark_unsupported(self, _):
            events.append("blacklist")

        def dispatch(self, *args, **kwargs):
            events.append("dispatch")  # Recorder only: never calls a GPU runtime.
            if mode == "dispatch_error":
                raise RuntimeError("attempted invocation")

    events = []
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", "1")
    monkeypatch.setattr(_cache_contract, "execution_contract", lambda: "controlled-current-stamp")
    monkeypatch.setattr(driver, "_get_utils", lambda: Utils())
    monkeypatch.setattr(driver, "_get_compile_shader_runtime", lambda: Runtime())
    launcher = driver.MetalLauncher.__new__(driver.MetalLauncher)
    launcher.arg_names = ["p", "v"]
    launcher.signature = dict(p="*fp32", v="i32")
    launcher.constexpr_indices = set()
    launcher._binding_plan = signature.make_binding_plan(launcher.arg_names, launcher.signature)
    launcher._execution_contract = "controlled-current-stamp"
    packed = (4, 1, 0, 32, None, False, None, None, None, None, None, None)
    launcher._packed_contract = _launch_contract._canonical(packed)
    launcher._msl, launcher._msl_block_size, launcher.kernel_name = "controlled-source", 32, "record_only"

    def enter(_):
        events.append("enter")
        if mode == "hook_changes_type":
            launcher.signature["v"] = "i8"

    def leave(_):
        events.append("exit")
        if mode == "exit_hook_error":
            raise RuntimeError("after invocation")

    def call():
        launcher(1, 1, 1, None, None, packed, None, enter, leave, pointer, 128)

    if mode == "hook_changes_type":
        with pytest.raises(HostFallback):
            call()
        assert events == ["enter"]
    elif mode in ("dispatch_error", "exit_hook_error"):
        with pytest.raises(PostSubmitError):
            call()
        assert events.count("dispatch") == 1 and "blacklist" not in events
    else:
        call()
        assert events == ["enter", "dispatch", "exit"]
