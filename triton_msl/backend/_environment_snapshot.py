"""Exact, read-only snapshots of the ordinary Python environment mapping.

The fast path compares a COPY of every encoded key/value, not just the known
policy keys or the mapping's identity/size. New injection variables therefore
invalidate it too. Nonstandard environment mappings retain their live behavior.
No watcher, private dict version counter, or mutation hook is installed.
"""
import os
import sys
import threading
from collections.abc import Mapping
from importlib.machinery import FrozenImporter
from itertools import repeat
from operator import is_
from types import CodeType, MappingProxyType, MethodType


def _frozen_codes(module="os"):
    """Recognize the interpreter's implementation, not a preinstalled override.

    On interpreters without frozen os, use the unoptimized live mapping. Do not
    read arbitrary files, execute another module, or guess from function names.
    """
    try:
        root = FrozenImporter.get_code(module)
    except ImportError:
        return {}
    codes = {}
    def visit(code):
        for value in code.co_consts:
            if isinstance(value, CodeType):
                if hasattr(value, "co_qualname"):
                    codes[value.co_qualname] = value
                visit(value)
    if isinstance(root, CodeType):
        visit(root)
    return codes

_type = type(os.environ)
_get = Mapping.get
_getitem = _type.__getitem__
_iter = _type.__iter__
_copy = getattr(_type, "copy", None)
_codecs = tuple(getattr(os.environ, name, None) for name in
                ("encodekey", "decodekey", "encodevalue", "decodevalue"))
_codes = _frozen_codes()
_encoding = sys.getfilesystemencoding()


def _standard_encoding(fn):
    closure = getattr(fn, "__closure__", None)
    return (closure is not None and len(closure) == 1
            and type(closure[0].cell_contents) is str
            and closure[0].cell_contents == _encoding)


_standard = (
    os.name == "posix"  # Other environment normalization rules use live fallback.
    and _codecs[0] is _codecs[2] and _codecs[1] is _codecs[3]
    and _standard_encoding(_codecs[0]) and _standard_encoding(_codecs[1])
    and
    getattr(_get, "__code__", None) is not None
    and _get.__code__ == _frozen_codes("_collections_abc").get("Mapping.get")
    and all(getattr(fn, "__code__", None) is not None
        and fn.__code__ == _codes.get(name)
        for fn, name in ((_getitem, "_Environ.__getitem__"),
                         (_iter, "_Environ.__iter__"), (_copy, "_Environ.copy")))
    and all(getattr(fn, "__code__", None) is not None
            and fn.__code__ == _codes.get(getattr(fn.__code__, "co_qualname", None))
            and getattr(fn.__code__, "co_qualname", "").startswith("_create_environ_mapping.<locals>.")
            for fn in _codecs)
)
_lock = threading.RLock()
_cached = None


def environment_snapshot():
    """Recheck all bytes each time; decode only after a change.

CPython's os._Environ exposes its encoded backing dict. This is an optional
optimization only: a foreign mapping/codec/getter uses the original mapping.
The result is immutable only on the recognized path.
"""
    global _cached
    env = os.environ
    getter, copier = getattr(env, "get", None), getattr(env, "copy", None)
    if (not _standard or type(env) is not _type or _type is not getattr(os, "_Environ", None)
            or _type.__getitem__ is not _getitem or _type.__iter__ is not _iter
            or type(getter) is not MethodType or getter.__func__ is not _get or getter.__self__ is not env
            or type(copier) is not MethodType or copier.__func__ is not _copy or copier.__self__ is not env
            or any(getattr(env, name, None) is not fn for name, fn in zip(
                ("encodekey", "decodekey", "encodevalue", "decodevalue"), _codecs))
            or not _standard_encoding(_codecs[0]) or not _standard_encoding(_codecs[1])
            or type(getattr(env, "_data", None)) is not dict):
        return env
    with _lock:
        raw = env._data.copy()
        # environb accepts bytes subclasses whose decode/equality can depend on
        # mutable state. Equal byte payloads alone do not prove those objects
        # immutable, so check exact types BEFORE the cached equality fast path.
        if (not all(map(is_, map(type, raw), repeat(bytes)))
                or not all(map(is_, map(type, raw.values()), repeat(bytes)))):
            return env
        if _cached is not None and _cached[0] is env and _cached[1] == raw:
            return _cached[2]
        # Decode the private copy, not the live mapping: A->B->A edits during
        # env.copy() could otherwise cache B under A's unchanged raw token.
        decodekey, decodevalue = _codecs[1], _codecs[3]
        values = {decodekey(key): decodevalue(value) for key, value in raw.items()}
        if env._data != raw:
            from triton_msl.errors import MetalNonRecoverableError
            raise MetalNonRecoverableError("environment changed while capturing cache policy; retry with stable settings")
        snapshot = MappingProxyType(values)
        _cached = env, raw, snapshot
        return snapshot


def is_snapshot(value):
    # A foreign MappingProxyType can wrap somebody else's mutable dictionary.
    # Only our own private copy proves immutability, never the wrapper type.
    cached = _cached
    return cached is not None and value is cached[2]


_standard_environment_snapshot = environment_snapshot
