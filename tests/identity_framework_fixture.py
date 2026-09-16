"""An explicit admitted import-metadata baseline for fast-path mechanism tests.

Only tests requesting this fixture get normalization. Modules/providers remain
loaded; the original standard namespace path object is restored after each test.
This fixture must never be used by workload/performance or namespace controls.
"""
from importlib._bootstrap_external import _NamespacePath
from importlib.machinery import ModuleSpec
import sys
from types import ModuleType

import pytest

from triton_msl.backend import _framework_contract as fw
from triton_msl.backend import _identity_fast_path as fast


@pytest.fixture
def ordinary_framework():
    saved = []
    fast._record = None
    try:
        for name in fw._SELECTION_NAMES:
            module = sys.modules.get(name)
            if type(module) is not ModuleType:
                continue
            spec = module.__dict__.get('__spec__')
            if type(spec) is not ModuleSpec:
                continue
            locations = spec.__dict__.get('submodule_search_locations')
            if type(locations) is _NamespacePath:
                paths = list(locations)
                assert all(type(path) is str for path in paths)
                saved.append((spec, locations))
                spec.submodule_search_locations = paths
        yield
    finally:
        for spec, locations in reversed(saved):
            spec.submodule_search_locations = locations
        fast._record = None
