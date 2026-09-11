"""Source-cache contract primitives (the outer-cache migration is separate).

Implementation identity is immutable for a Python process: editing installed
sources is an upgrade requiring restart, not a supported hot-reload operation.
Policy is deliberately NOT cached. Cache keys never use checkout paths/mtimes.
"""

import functools
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import weakref


SOURCE_SCHEMA = 3
_jit_guard_lock = threading.RLock()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _package_content(root):
    """Stable logical filenames plus bytes of runtime source/native inputs.

    Native build SOURCES are not runtime inputs; a rebuilt extension's bytes are.
    Symlink targets are read for their bytes, but absolute paths never enter the
    result. Missing/unreadable files propagate instead of hashing as 'unknown'.
    """
    suffixes = {".py", ".so", ".dylib", ".a", ".bc", ".metal", ".metallib"}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes)
    if not files or not (root / "__init__.py").is_file():
        raise RuntimeError("cannot fingerprint an incomplete triton_msl package")
    return [(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]


@functools.lru_cache(maxsize=1)
def implementation_identity():
    """Package implementation ONLY; not a claim to fingerprint the toolchain.

    The persistent compiler/SDK/dependency and packed-descriptor boundaries are
    explicitly still owed by the full B migration. This primitive is separated
    so callers cannot confuse a code identity with a complete binary identity.
    """
    return _digest(_package_content(Path(__file__).resolve().parents[1]))


def effective_policy():
    """Current settings used by codegen/eligibility, with existing truth rules.

    FORCE_PYTHON and USE_CPP are retained separately: the family predicate reads
    USE_CPP itself, so collapsing them to one boolean would lose that dependency.
    """
    result = {}
    for name in ("MEPT", "QUANT_MATMUL", "FAST_MATMUL", "COMPILE_SHADER", "FA_FAST"):
        result[name] = os.environ.get("TRITON_MSL_" + name, "1") != "0"
    for name in ("INFER_LAYOUT", "LEGACY", "USE_CPP", "FORCE_PYTHON"):
        result[name] = os.environ.get("TRITON_MSL_" + name) == "1"
    result["FA_HALF_ACCUM"] = os.environ.get("TRITON_MSL_FA_HALF_ACCUM", "0") in ("1", "true", "True")
    # Preserve exact skip spelling until the caller's split/strip contract is
    # centralized; this can over-invalidate but cannot merge distinct policies.
    result["CPP_SKIP"] = os.environ.get("TRITON_MSL_CPP_SKIP", "")
    return result


def source_contract():
    from triton_msl import CODEGEN_VERSION

    return {
        "schema": SOURCE_SCHEMA,
        "implementation": implementation_identity(),
        "frameworks": framework_identity(),
        "policy": effective_policy(),
        "label": CODEGEN_VERSION,
    }


def source_key(mod_text, options_hash, input_metadata=None):
    # Structured fields avoid concatenation ambiguity between text and options.
    return _digest({"source": mod_text, "options": options_hash, "contract": source_contract(),
                    "input_metadata": _metadata_payload(input_metadata) if input_metadata is not None else None})


def _source_digest(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _target_fields(value):
    if (type(value) is not dict or set(value) != {"backend", "arch", "warp_size"}
            or type(value["backend"]) is not str or not value["backend"]
            or type(value["arch"]) not in (str, int)
            or type(value["warp_size"]) is not int or value["warp_size"] <= 0):
        raise TypeError("unrepresentable native GPUTarget")
    return value


def _metadata_payload(metadata):
    """Closed codec for the one native object in Triton's compiler metadata.

    No general default=vars: unknown objects must disable caching, not silently
    lose their type. The complete initial metadata also participates in the key,
    so a hit cannot replace this compilation's target/hash/options with another's.
    """
    from triton.backends.compiler import GPUTarget

    result = dict(metadata)
    if "target" in result:
        target = result["target"]
        if type(target) is not GPUTarget:
            raise TypeError("source metadata target is not a native GPUTarget")
        result["target"] = _target_fields({"backend": target.backend, "arch": target.arch,
                                            "warp_size": target.warp_size})
    return result


def metadata_record(source, metadata, key, *, kind="msl-source-product"):
    # A non-JSON descriptor makes the WHOLE cache product non-cacheable. Dropping
    # individual fields (especially dictionary descriptors) loses semantics.
    payload = _metadata_payload(metadata)
    record = {
        "schema": SOURCE_SCHEMA,
        "kind": kind,
        "key": key,
        "contract": source_contract(),
        "source_sha256": _source_digest(source),
        "metadata": payload,
        "metadata_sha256": _digest(payload),
    }
    return json.dumps(record, sort_keys=True, allow_nan=False)


def decode_metadata(source, record, key, *, kind="msl-source-product"):
    """Validate an already-loaded record at memory AND disk boundaries.

    No guessed launch defaults. A missing/partial/legacy/corrupt partner is not
    proof that a cached shader has the current dispatch or output-copy contract.
    """
    try:
        if not isinstance(record, dict) or type(record.get("schema")) is not int:
            return None
        if (
            record["schema"] != SOURCE_SCHEMA
            or record.get("kind") != kind
            or record.get("key") != key
            or record.get("contract") != source_contract()
            or record.get("source_sha256") != _source_digest(source)
        ):
            return None
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or record.get("metadata_sha256") != _digest(metadata):
            return None
        metadata = dict(metadata)
        if "target" in metadata:
            from triton.backends.compiler import GPUTarget

            metadata["target"] = GPUTarget(**_target_fields(metadata["target"]))
        return metadata
    except (OSError, ValueError, TypeError):
        return None


def read_metadata(source, path, key):
    """Read a source-product envelope; stash records cannot impersonate it."""
    try:
        return decode_metadata(source, json.loads(Path(path).read_text()), key)
    except (OSError, ValueError, TypeError):
        return None


def toolchain_identity():
    from triton_msl.backend._toolchain_contract import toolchain_identity as implementation
    return implementation()


def framework_identity():
    from triton_msl.backend._framework_contract import framework_identity as implementation
    return implementation()


def execution_contract():
    """Small immutable producer stamp, separate from the still-versioning launch ABI."""
    return json.dumps({"schema": 1, "source": source_contract(), "toolchain": toolchain_identity()},
                      sort_keys=True, separators=(",", ":"))


def validate_execution_contract(stamp):
    """Old/unknown/current-policy-mismatched handles refuse before runtime effects.

    Canonical serialization equality also rejects JSON type aliases such as
    schema=true vs schema=1. This stamp is not invented at cache restoration.
    """
    from triton_msl.errors import MetalNonRecoverableError

    if not isinstance(stamp, str) or not stamp or stamp != execution_contract():
        raise MetalNonRecoverableError(
            "compiled kernel execution contract is missing, stale or incompatible; "
            "recompile/clear the outer kernel cache, and restart after installation or toolchain changes"
        )
    return stamp


class _JitPolicyGuard:
    """Use Triton's per-function public pre-run hook, never a global monkeypatch."""

    def __init__(self, fn, stamp):
        self.fn = weakref.ref(fn)
        self.stamp = stamp

    def __call__(self, *args, **kwargs):
        fn = self.fn()
        if fn is None:
            return
        current = execution_contract()
        with _jit_guard_lock:
            if current == self.stamp:
                return
            # JITFunction.run executes pre_run_hooks BEFORE retrieving these
            # per-device kernel/binder caches. Clear only this owning function.
            # Already-restored/direct handles still validate at launch itself.
            fn.device_caches.clear()
            self.stamp = current


def install_jit_policy_guard(fn, stamp):
    """Best available transparent recompile, with an unconditional launcher backstop.

    Warmup-only compiled handles may not have constructed a launcher yet, and
    non-JIT restored handles need not have this public hook. Those cases safely
    refuse a stale execution contract rather than claim transparent invalidation.
    """
    if fn is None or not callable(getattr(fn, "add_pre_run_hook", None)) or not isinstance(
        getattr(fn, "device_caches", None), dict
    ):
        return False
    with _jit_guard_lock:
        existing = getattr(fn, "_triton_msl_policy_guard", None)
        if existing is not None:
            from triton_msl.errors import MetalNonRecoverableError
            if not isinstance(existing, _JitPolicyGuard) or existing.fn() is not fn:
                raise MetalNonRecoverableError("unrecognized per-JIT Metal execution contract guard; restart")
            return True
        try:
            guard = _JitPolicyGuard(fn, stamp)
        except TypeError:
            return False  # no weakref support: the launch backstop still applies
        fn.add_pre_run_hook(guard)
        fn._triton_msl_policy_guard = guard
        return True


def binary_key(source, options_hash, route, flags):
    if route not in ("msl", "llir"):
        raise ValueError("unknown Metal compiler input route")
    return _digest({
        "kind": "metal-compiled-binary", "source": source,
        "options": options_hash, "route": route, "flags": list(flags),
        "contract": source_contract(), "toolchain": toolchain_identity(),
    })


def read_binary_product(path, source, key):
    """Partial, legacy, or content-mismatched binary/record pairs miss."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
        record = json.loads(Path(str(path) + ".meta.json").read_text())
        metadata = decode_metadata(source, record, key, kind="metallib-product")
        if not data or metadata != {"binary_sha256": hashlib.sha256(data).hexdigest()}:
            return None
        return data
    except (OSError, ValueError, TypeError):
        return None


def publish_binary_record(path, source, key, data):
    """Best-effort commit record, bound to the PRIVATE compiler output bytes.

    Publishing a record can fail without invalidating the freshly compiled
    result. It must never hash a possibly replaced public cache file and certify
    that as our output. A concurrent replacement then simply causes a cache miss.
    """
    tmp = None
    try:
        if not isinstance(data, bytes) or not data:
            raise ValueError("empty or invalid compiled binary")
        record = metadata_record(source, {"binary_sha256": hashlib.sha256(data).hexdigest()},
                                 key, kind="metallib-product")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=Path(path).parent,
                                         suffix=".metallib-record.tmp", delete=False) as handle:
            tmp = handle.name
            handle.write(record)
        os.replace(tmp, str(path) + ".meta.json")
        return True
    except OSError:
        return False
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
