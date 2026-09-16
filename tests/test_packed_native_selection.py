"""The optional optimization must not turn broken imports into a quiet fallback."""

from types import SimpleNamespace

import pytest

from triton_msl.backend import _launch_contract as c


def test_missing_optional_module_keeps_python():
    calls = []

    def missing(name):
        calls.append(name)
        raise AssertionError("proved absence must not invoke a loader")

    def absent(name):
        calls.append(name)
        return None

    assert c._select_checked_copy(missing, absent, {}) is None
    assert calls == ["triton_msl.backend._packed_native"]


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("missing dependency", name="required_dependency"),
        ModuleNotFoundError("installed initializer names itself", name="triton_msl.backend._packed_native"),
        ImportError("invalid extension"),
        RuntimeError("initializer failed"),
        OSError("binary cannot load"),
    ],
)
def test_actual_import_failure_stays_loud(error):
    def broken(name):
        raise error

    with pytest.raises(type(error)) as raised:
        c._select_checked_copy(broken, lambda name: object(), {})
    assert raised.value is error


def test_extension_entry_point_required():
    with pytest.raises(ImportError, match="no callable copy"):
        c._select_checked_copy(lambda name: SimpleNamespace(copy=None), lambda name: object(), {})


def test_present_extension_selected_by_identity():
    extension = SimpleNamespace(copy=lambda *args: None)
    assert c._select_checked_copy(lambda name: extension, lambda name: object(), {}) is extension


@pytest.mark.parametrize("provider", [None, SimpleNamespace(copy=lambda *args: None)])
def test_current_module_selection_is_not_replaced_by_spec_lookup(provider):
    name = "triton_msl.backend._packed_native"

    def unexpected_spec(_):
        raise AssertionError("sys.modules selection is already authoritative")

    def selected(_):
        if provider is None:
            raise ModuleNotFoundError("import blocked by None", name=name)
        return provider

    if provider is None:
        with pytest.raises(ModuleNotFoundError, match="blocked by None"):
            c._select_checked_copy(selected, unexpected_spec, {name: None})
    else:
        assert c._select_checked_copy(selected, unexpected_spec, {name: provider}) is provider


def test_c_source_participates_in_implementation_identity(tmp_path):
    from triton_msl.backend._cache_contract import _package_content

    (tmp_path / "__init__.py").write_text("")
    helper = tmp_path / "helper.c"
    helper.write_text("first implementation")
    first = _package_content(tmp_path)
    helper.write_text("second implementation")
    second = _package_content(tmp_path)
    assert first != second
    assert {name for name, _ in first} == {"__init__.py", "helper.c"}
