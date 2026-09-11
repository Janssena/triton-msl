"""Closed Mach-O provider graph for the selected runtime installation.

This is dependency identity, not executable authentication. OS-shipped images
are bound by the separate OS-build contract. Third-party images are hashed in
full. Unsupported loader semantics fail closed; no basename/search-path guess
is permitted. Constants/layouts follow the installed mach-o/loader.h and fat.h.
"""
from dataclasses import dataclass
import hashlib
import mmap
from pathlib import Path
import struct


_THIN = {b"\xce\xfa\xed\xfe": ("<", 28), b"\xcf\xfa\xed\xfe": ("<", 32),
         b"\xfe\xed\xfa\xce": (">", 28), b"\xfe\xed\xfa\xcf": (">", 32)}
_FAT = {b"\xca\xfe\xba\xbe": (">", False), b"\xbe\xba\xfe\xca": ("<", False),
        b"\xca\xfe\xba\xbf": (">", True), b"\xbf\xba\xfe\xca": ("<", True)}
_DEPENDENCIES = {0xC, 0x80000018, 0x8000001F, 0x80000023, 0x20}
# Obsolete FVM/prebound/sub-framework loading, environment commands and the new
# lazy-load-info trie are not proved by this parser. Ordinary segment/symbol/
# signing/build-version commands do not select a different provider.
_NON_PROVIDER = {0x1, 0x2, 0x3, 0x4, 0x5, 0x8, 0xA, 0xB, 0x11, 0x16, 0x17,
                 0x19, 0x1A, 0x1B, 0x1D, 0x1E, 0x21, 0x22, 0x80000022, 0x24,
                 0x25, 0x26, 0x80000028, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E,
                 0x2F, 0x30, 0x31, 0x32, 0x80000033, 0x80000034, 0x36, 0x37,
                 0x38, 0x39}
_STRINGS = _DEPENDENCIES | {0xD, 0xE, 0xF, 0x8000001C}


@dataclass(frozen=True)
class NativeImage:
    digest: str
    dependencies: tuple
    rpaths: tuple
    file_type: int
    install_name: str | None


def is_native(path):
    with Path(path).open("rb") as handle:
        magic = handle.read(4)
    return magic in _THIN or magic in _FAT


def _parse(data, cpu_type):
    length = len(data)
    if length < 4:
        raise ValueError("short native Mach-O header")
    offset, size = 0, length
    magic = data[:4]
    if magic in _FAT:
        endian, wide = _FAT[magic]
        if length < 8:
            raise ValueError("short native fat header")
        count = struct.unpack_from(endian + "I", data, 4)[0]
        entry_size = 32 if wide else 20
        table_end = 8 + count * entry_size
        if not count or table_end > length:
            raise ValueError("invalid native fat architecture table")
        matches, intervals = [], []
        for i in range(count):
            fields = struct.unpack_from(endian + ("IIQQII" if wide else "IIIII"), data, 8 + i * entry_size)
            cpu, _, start, span, align = fields[:5]
            if align > 63 or start < table_end or not span or start + span > length or start % (1 << align):
                raise ValueError("invalid native fat slice bounds/alignment")
            if any(start < end and begin < start + span for begin, end in intervals):
                raise ValueError("overlapping native fat slices")
            intervals.append((start, start + span))
            if cpu == cpu_type:
                matches.append((start, span))
        if len(matches) != 1:
            raise ValueError("missing or ambiguous native CPU slice")
        offset, size = matches[0]
        magic = data[offset:offset + 4]
    if magic not in _THIN:
        raise ValueError("unproved native Mach-O magic")
    endian, header = _THIN[magic]
    if size < header:
        raise ValueError("short native Mach-O header")
    _, cpu, _, file_type, count, command_bytes, _ = struct.unpack_from(endian + "7I", data, offset)
    if cpu != cpu_type:
        raise ValueError("native Mach-O CPU does not match selected architecture")
    end = offset + header + command_bytes
    if end > offset + size or count > command_bytes // 8:
        raise ValueError("native load command bounds exceed image")
    cursor, dependencies, rpaths = offset + header, [], []
    install_name = None
    for _ in range(count):
        if cursor + 8 > end:
            raise ValueError("short native load command")
        cmd, command_size = struct.unpack_from(endian + "II", data, cursor)
        if command_size < 8 or command_size % (8 if header == 32 else 4) or cursor + command_size > end:
            raise ValueError("invalid native load command size")
        if cmd not in _NON_PROVIDER and cmd not in _STRINGS:
            raise ValueError(f"unsupported native load command {cmd:#x}")
        if cmd in _STRINGS:
            fixed = 12 if cmd in (0xE, 0xF, 0x8000001C) else 24
            if command_size < fixed:
                raise ValueError("short native dependency command")
            string_offset = struct.unpack_from(endian + "I", data, cursor + 8)[0]
            weak = cmd == 0x80000018
            if fixed == 24 and struct.unpack_from(endian + "I", data, cursor + 12)[0] == 0x1A741800:
                fixed = 28  # dylib_use_command, macOS 15+
                if command_size < fixed:
                    raise ValueError("short native dylib-use command")
                flags = struct.unpack_from(endian + "I", data, cursor + 24)[0]
                if flags & ~0xF:
                    raise ValueError("unknown native dylib-use flags")
                weak = weak or bool(flags & 1)
            if not fixed <= string_offset < command_size:
                raise ValueError("invalid native command string offset")
            raw = data[cursor + string_offset:cursor + command_size]
            stop = raw.find(b"\0")
            if stop <= 0:
                raise ValueError("unterminated/empty native command string")
            try:
                value = raw[:stop].decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise ValueError("unproved native command string encoding") from exc
            if cmd == 0x8000001C:
                rpaths.append(value)
            elif cmd == 0xD:
                if file_type != 6 or install_name is not None:
                    raise ValueError("invalid or duplicate native dylib install name")
                install_name = value
            elif cmd not in (0xD, 0xF):
                dependencies.append((value, weak))
        cursor += command_size
    if cursor != end:
        raise ValueError("native load command count/size mismatch")
    return tuple(dependencies), tuple(rpaths), file_type, install_name


def parse_image(path, cpu_type):
    path = Path(path)
    before = path.stat()
    if not before.st_size:
        raise ValueError("empty native Mach-O image")
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        dependencies, rpaths, file_type, install_name = _parse(data, cpu_type)
        digest = hashlib.sha256(data).hexdigest()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise ValueError(f"native provider changed while reading: {path}")
    return NativeImage(digest, dependencies, rpaths, file_type, install_name)


def _system(path):
    # Normalize '..' before granting the system-build classification.
    import os
    value = os.path.normpath(str(path))
    return value.startswith(("/System/Library/", "/usr/lib/"))


def _expand(reference, loader, executable):
    for token, root in (("@loader_path", loader.parent), ("@executable_path", executable.parent)):
        if reference == token or reference.startswith(token + "/"):
            return root / reference[len(token):].lstrip("/")
    if reference.startswith("/"):
        return Path(reference)
    raise ValueError(f"unproved native dependency provider: {reference}")


def dependency_graph(seed_paths, *, cpu_type, executable, loaded_paths=()):
    """Ordered provider graph, with indices preserving each root/edge association.

Logical root order is supplied by the package inventory. Absolute resolved paths
are traversal guards only, not persistent identity. Separate RPATH contexts are
separate graph nodes; identical bytes in two directories cannot erase their
different dependency providers. Cycles terminate by path+deduplicated context.

For @rpath references dyld first checks already-loaded images by their full
install name (Loader::getLoader / JustInTimeLoader::matchesPath). The caller
must supply an actual stable dyld snapshot, not a package inventory or search
directory. Bind the selected image's bytes and transitive edges. Conflicting
loaded install names remain unproved; basename matches never qualify.
"""
    executable = Path(executable).resolve(strict=True)
    parsed, indices, pending, rows = {}, {}, [], []
    loaded_names = {}
    for raw in loaded_paths:
        path = Path(raw).resolve(strict=True)
        if path not in parsed:
            parsed[path] = parse_image(path, cpu_type)
        name = parsed[path].install_name
        if name is not None and name.startswith("@rpath/"):
            previous = loaded_names.setdefault(name, path)
            if previous != path:
                raise ValueError(f"ambiguous loaded native install name: {name}")

    def intern(path, inherited):
        path = Path(path).resolve(strict=True)
        if path not in parsed:
            parsed[path] = parse_image(path, cpu_type)
        image = parsed[path]
        own = tuple(_expand(ref, path, executable).resolve() for ref in image.rpaths)
        context = tuple(dict.fromkeys((*own, *inherited)))
        key = path, context
        if key not in indices:
            if len(indices) >= 20000:
                raise ValueError("native dependency graph exceeds proved traversal bound")
            indices[key] = len(pending)
            pending.append(key)
        return indices[key]

    for path in seed_paths:
        intern(path, ())
    cursor = 0
    while cursor < len(pending):
        path, context = pending[cursor]
        image, edges = parsed[path], []
        for reference, weak in image.dependencies:
            if reference.startswith("@rpath/") and reference in loaded_names:
                edges.append((reference, ("provider", intern(loaded_names[reference], context))))
                continue
            if reference.startswith("@rpath/"):
                candidates = [p / reference[len("@rpath/"):] for p in context]
                if not candidates:
                    raise ValueError(f"unproved native rpath provider: {reference}")
            else:
                candidates = [_expand(reference, path, executable)]
            resolved = {}
            for candidate in candidates:
                if _system(candidate):
                    resolved[str(candidate)] = "system-build"
                elif candidate.exists():
                    actual = candidate.resolve(strict=True)
                    resolved[str(actual)] = actual
                elif candidate.is_symlink():
                    raise ValueError(f"broken native dependency provider: {candidate}")
            if len(resolved) > 1:
                raise ValueError(f"ambiguous native rpath dependency provider: {reference}")
            if not resolved:
                if not weak:
                    raise ValueError(f"missing native dependency provider: {reference}")
                edges.append((reference, "weak-absent"))
            else:
                provider = next(iter(resolved.values()))
                target = "system-build" if provider == "system-build" else ("provider", intern(provider, context))
                edges.append((reference, target))
        rows.append((image.digest, tuple(edges)))
        cursor += 1
    # Reading a loaded image's install name is NOT proving its dependencies.
    # Only traversed nodes belong to the closure; other loaded images must still
    # be added as ambient roots by the caller and have every edge checked.
    return tuple(rows), frozenset(path for path, _context in pending)


def dependency_manifest(seed_paths, *, cpu_type, executable):
    return dependency_graph(seed_paths, cpu_type=cpu_type, executable=executable)[0]
