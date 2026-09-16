"""Fresh-process native error-notification control; no backend/GPU imports."""

import importlib.util
import json
from pathlib import Path
import sys

if sys.flags.optimize:
    raise RuntimeError("normal interpreter required")
image, helper, mode = sys.argv[1:]
spec = importlib.util.spec_from_file_location("_validation_native", image)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
if Path(native.__file__).resolve() != Path(image).resolve():
    raise RuntimeError("wrong image")
spec = importlib.util.spec_from_file_location("notification_error", helper)
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)
value, changed, absent = object(), object(), object()
mapping, other = {"value": value}, {}
calls = []


class Foreign(str):
    def __eq__(self, other):
        calls.append("equality")
        return super().__eq__(other)

    __hash__ = str.__hash__


if mode == "dep_primary":
    probes = ((5, mapping, other, "value", value),)
elif mode == "dep_fallback":
    probes = ((5, other, mapping, "value", value),)
else:
    probes = ((4, mapping, "value", absent, value),)
    if mode == "new_foreign_key":
        probes += ((4, mapping, "extra", absent, absent),)
assert native.identity_probes(probes) is True
assert native.identity_probes(probes) is True
observations = []
original_hook = sys.unraisablehook


def hook(error):
    observations.append((type(error.exc_value).__name__, native.identity_probes(probes), mapping["value"] is value))


sys.unraisablehook = hook
try:
    watcher.install(mapping)
    if mode == "new_foreign_key":
        mapping[Foreign("extra")] = changed
    else:
        mapping["value"] = changed
    result = native.identity_probes(probes)
finally:
    events = watcher.remove()
    sys.unraisablehook = original_hook
assert events == 1 and observations == [("RuntimeError", True, True)], observations
assert result is False, "post-write validation reused pre-write state"
assert calls == [], "foreign-key census invoked equality"
print(json.dumps(dict(mode=mode, result=result, events=events, observations=observations)))
