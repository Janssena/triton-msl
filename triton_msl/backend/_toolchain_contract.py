"""Content identity for the selected Metal toolchain and SDK.

Installed package/toolchain/SDK contents are immutable within an interpreter:
upgrades require restarting it. Selector-environment changes are refused, not
silently combined with cached device/AIR probes. No persistent stat-only memo is
trusted. The first lookup reads actual bytes; subsequent lookups reuse that
process snapshot. Whole-tree inventory is intentionally conservative.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import threading


_SELECTION = ("PATH", "DEVELOPER_DIR", "SDKROOT", "TOOLCHAINS")
_EXTERNAL_SEARCH = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
                    "COMPILER_PATH", "GCC_EXEC_PREFIX", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
                    "DYLD_FRAMEWORK_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_FALLBACK_FRAMEWORK_PATH",
                    "DYLD_ROOT_PATH", "DYLD_IMAGE_SUFFIX", "DYLD_VERSIONED_LIBRARY_PATH",
                    "DYLD_VERSIONED_FRAMEWORK_PATH")
_lock = threading.RLock()
_snapshot = None
_environment_inputs = None


def _tree_manifest(root, *, ignore_bytecode=False, exclude_root_names=(), allow_empty=False):
    """Logical names + bytes, including directory-symlink topology and contents."""
    root = Path(root).resolve(strict=True)
    records = []

    def visit(path, rel, ancestors):
        before = path.stat()
        inode = before.st_dev, before.st_ino
        if stat.S_ISDIR(before.st_mode):
            if inode in ancestors:
                records.append((rel, 0, ancestors[inode], "directory-back-edge"))
                return
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                if not rel and child.name in exclude_root_names:
                    continue
                if ignore_bytecode and (child.name == "__pycache__" or child.suffix in (".pyc", ".pyo")):
                    continue
                visit(child, f"{rel}/{child.name}" if rel else child.name, ancestors | {inode: rel})
        elif stat.S_ISREG(before.st_mode):
            with path.open("rb") as handle:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
                digest = hasher.hexdigest()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ):
                raise RuntimeError(f"toolchain input changed while hashing: {path}")
            records.append((rel, before.st_size, digest, "file"))
        else:
            raise RuntimeError(f"unsupported toolchain input kind: {path}")

    visit(root, "", {})
    if not records and not allow_empty:
        raise RuntimeError(f"empty toolchain input tree: {root}")
    return records


def _toolchain_native_dependencies(roots, entries):
    """Bind subprocess entry points and executable helpers, not only their bundle."""
    from ._native_dependencies import dependency_manifest, is_native, parse_image
    import platform

    cpu = {"arm64": 0x0100000C, "x86_64": 0x01000007}.get(platform.machine())
    if cpu is None:
        raise RuntimeError("unproved native toolchain CPU architecture")
    entries = dict(entries)
    visited = set()
    for role in ("metal", "metallib"):
        root = roots[role]
        pending = [(root, "")]
        while pending:
            path, rel = pending.pop()
            status = path.stat()
            inode = status.st_dev, status.st_ino
            if inode in visited:
                continue
            visited.add(inode)
            if stat.S_ISDIR(status.st_mode):
                pending.extend((child, f"{rel}/{child.name}" if rel else child.name)
                               for child in reversed(sorted(path.iterdir())))
            elif stat.S_ISREG(status.st_mode) and path.suffix != ".a" and is_native(path):
                if parse_image(path, cpu).file_type == 2:  # MH_EXECUTE, not standalone dylib
                    entries[f"{role}-helper:{rel}"] = path
    return {role: dependency_manifest([path], cpu_type=cpu, executable=path)
            for role, path in sorted(entries.items())}


def _resolve_toolchain():
    resolver = shutil.which("xcrun")
    if resolver is None:
        raise RuntimeError("xcrun is unavailable; install/select the Metal toolchain")

    def query(*args):
        value = subprocess.check_output([resolver, "-sdk", "macosx", *args],
                                        text=True, stderr=subprocess.PIPE).strip()
        if not value:
            raise RuntimeError(f"empty toolchain query: {args}")
        return value

    roots, entries = {}, {"resolver": Path(resolver).resolve(strict=True)}
    for name in ("metal", "metallib"):
        selector = Path(query("--find", name)).resolve(strict=True)
        bundle = next((p for p in selector.parents if p.name.endswith(".xctoolchain")), None)
        if bundle is None:
            raise RuntimeError(f"unproved {name} toolchain layout: {selector}")
        roots[name] = bundle
        entries[name + "-selector"] = selector
    version = query("metal", "--version")
    installed = [line.removeprefix("InstalledDir: ").strip() for line in version.splitlines()
                 if line.startswith("InstalledDir: ")]
    if len(installed) != 1:
        raise RuntimeError("Metal compiler does not identify its actual InstalledDir")
    actual = (Path(installed[0]) / "metal").resolve(strict=True)
    if not actual.is_file() or not actual.is_relative_to(roots["metal"]):
        raise RuntimeError("actual Metal compiler is outside the inventoried toolchain")
    entries["actual-metal"] = actual
    roots["sdk"] = Path(query("--show-sdk-path")).resolve(strict=True)
    # InstalledDir is diagnostic, not identity; its bytes are included by logical
    # bundle name. Relocating identical SDK/toolchain contents must retain identity.
    metadata = {
        "schema": 2,
        "compiler_version": "\n".join(line for line in version.splitlines() if not line.startswith("InstalledDir:")),
        "sdk_build": query("--show-sdk-build-version"),
        "os_build": subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.osversion"], text=True).strip(),
        "xcrun_sha256": hashlib.sha256(Path(resolver).read_bytes()).hexdigest(),
    }
    if not metadata["os_build"]:
        raise RuntimeError("OS build identity is unavailable")
    metadata["native_providers"] = _toolchain_native_dependencies(roots, entries)
    return roots, metadata


def toolchain_identity():
    """Fail closed on incomplete identity or live selector changes, never 'unknown'."""
    from ._environment_snapshot import environment_snapshot
    return _identity_for_environment(environment_snapshot())


def _identity_for_environment(environment):
    """Shared implementation; internal callers capture the live environment first."""
    global _snapshot, _environment_inputs
    from triton_msl.errors import MetalNonRecoverableError
    from ._environment_snapshot import is_snapshot

    cached = _environment_inputs
    if (cached is not None and cached[0] is environment
            and cached[1:3] == (_SELECTION, _EXTERNAL_SEARCH)):
        selection, external = cached[3:]
    else:
        selection = tuple(environment.get(key) for key in _SELECTION)
        external = sorted({key for key in _EXTERNAL_SEARCH if environment.get(key)}
                          | {key for key in environment if key.startswith("DYLD_") and environment.get(key)})
        if is_snapshot(environment):
            _environment_inputs = environment, _SELECTION, _EXTERNAL_SEARCH, selection, external
    if external:
        raise MetalNonRecoverableError(
            "untracked compiler search/injection environment: " + ", ".join(external)
            + "; clear it before compiling with the Metal cache contract"
        )
    with _lock:
        if _snapshot is not None:
            if _snapshot[0] != selection:
                raise MetalNonRecoverableError("Metal toolchain selection changed in this process; restart before compiling")
            return _snapshot[1]
        try:
            roots, metadata = _resolve_toolchain()
            # Deduplicate physical roots only as a read-cost optimization; each
            # named compiler role retains the complete same manifest in the key.
            manifests = {}
            by_root = {}
            for name, path in sorted(roots.items()):
                resolved = Path(path).resolve(strict=True)
                if resolved not in by_root:
                    by_root[resolved] = _tree_manifest(resolved)
                manifests[name] = by_root[resolved]
            if tuple(os.environ.get(key) for key in _SELECTION) != selection:
                raise RuntimeError("toolchain selection changed during fingerprint construction")
            payload = json.dumps({"metadata": metadata, "inputs": manifests}, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(payload.encode()).hexdigest()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            raise MetalNonRecoverableError(f"cannot establish Metal compiler/SDK identity: {exc}") from exc
        _snapshot = selection, digest
        return digest


_standard_toolchain_identity = toolchain_identity
_standard_identity_for_environment = _identity_for_environment
