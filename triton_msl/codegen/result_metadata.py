"""Immutable representation facts, separate from signed operation semantics.

Stage 1a is additive. Unknown facts stay unknown; no emitter consumes these
records yet and the legacy type parser's defaults are not imported here.
"""

from dataclasses import dataclass
import re
from typing import Mapping


@dataclass(frozen=True)
class TypeFacts:
    raw: str
    kind: str = "unknown"
    elem: str | None = None
    width: int | None = None
    signed: bool | None = None
    shape: tuple[int | None, ...] | None = None
    is_tensor: bool = False
    layout: str | None = None
    pointee: "TypeFacts | None" = None
    address_space: int | None = None
    unknown_reason: str | None = None


@dataclass(frozen=True)
class ResultMeta:
    value_id: int
    type: TypeFacts
    kind: str  # result / entry_arg / callee_arg / block_arg
    producer_id: int | None
    result_index: int
    owner_id: int | None = None  # block identity, not a guessed producer op
    function_name: str | None = None
    schema_version: int = 1


def _parts(text):
    """Split top-level commas; validate nested type/layout delimiters."""
    stack, result, start = [], [], 0
    close = {">": "<", "}": "{", "]": "[", ")": "("}
    for i, c in enumerate(text):
        if c in "<{[(":
            stack.append(c)
        elif c in close:
            if not stack or stack.pop() != close[c]:
                raise ValueError("unbalanced type delimiters")
        elif c == "," and not stack:
            result.append(text[start:i].strip())
            start = i + 1
    if stack:
        raise ValueError("unbalanced type delimiters")
    return [*result, text[start:].strip()]


def _resolve_layout(text, aliases, seen=frozenset()):
    def replace(match):
        token = match.group()
        # An inline dialect attribute is not an alias. Its nested aliases are
        # visited separately by this same substitution.
        if text[match.end() :].lstrip().startswith("<"):
            return token
        if token in seen:
            raise ValueError(f"cyclic layout alias {token}")
        if token not in aliases:
            raise ValueError(f"unresolved layout alias {token}")
        return _resolve_layout(aliases[token], aliases, seen | {token})

    return re.sub(r"#[A-Za-z_][\w.]*", replace, text)


def layout_aliases(module_text: str) -> dict[str, str]:
    # MLIR's module printer emits each alias on one complete line. Keep the
    # value verbatim; recursive alias resolution happens once per cached type.
    return dict(re.findall(r"^\s*(#[A-Za-z_][\w.]*)\s*=\s*(.+?)\s*$", module_text, re.M))


def parse_type_facts(raw: str, aliases: Mapping[str, str] | None = None) -> TypeFacts:
    """Closed scalar/tensor/pointer facts; never equate index/pointer with i32.

    Signless iN has signed=None even when produced by a signed arithmetic op:
    its bits can subsequently feed both signed and unsigned operations.
    Triton's omitted pointer address space is 1 (Triton IR Types.cpp), but a
    pointer's bit width is deliberately unspecified without a target contract.
    """
    text = raw.strip()
    aliases = aliases or {}
    shape, is_tensor, layout, reason = (), False, None, None
    try:
        if text.startswith("tensor<"):
            if not text.endswith(">"):
                raise ValueError("unterminated tensor type")
            parts = _parts(text[7:-1])
            if len(parts) not in (1, 2):
                raise ValueError("unsupported tensor type fields")
            text, is_tensor = parts[0], True
            dims = []
            while (match := re.match(r"(\d+|\?)x", text)) is not None:
                dims.append(None if match.group(1) == "?" else int(match.group(1)))
                text = text[match.end() :]
            shape = tuple(dims)
            if len(parts) == 2:
                try:
                    layout = _resolve_layout(parts[1], aliases)
                except ValueError as exc:
                    reason = str(exc)
        if text.startswith("!tt.ptr<") and text.endswith(">"):
            parts = _parts(text[8:-1])
            if len(parts) not in (1, 2):
                raise ValueError("unsupported pointer type fields")
            pointee = parse_type_facts(parts[0], aliases)
            if pointee.kind not in ("float", "integer", "index") or pointee.is_tensor:
                raise ValueError("unsupported pointer pointee")
            space = 1 if len(parts) == 1 else int(parts[1])
            if space < 0:
                raise ValueError("negative pointer address space")
            return TypeFacts(
                raw,
                "pointer",
                shape=shape,
                is_tensor=is_tensor,
                layout=layout,
                pointee=pointee,
                address_space=space,
                unknown_reason=reason,
            )
        float_widths = {
            "f16": 16,
            "bf16": 16,
            "f32": 32,
            "f64": 64,
            "f8E4M3FN": 8,
            "f8E4M3FNUZ": 8,
            "f8E5M2": 8,
            "f8E5M2FNUZ": 8,
            "f8E4M3B11FNUZ": 8,
            "f8E8M0FNU": 8,
        }
        signed = None
        if text in float_widths:
            kind, width = "float", float_widths[text]
        elif (match := re.fullmatch(r"(s|u)?i([1-9][0-9]*)", text)) is not None:
            kind, width = "integer", int(match.group(2))
            signed = None if match.group(1) is None else match.group(1) == "s"
        elif text == "index":
            kind, width = "index", None
        else:
            raise ValueError(f"unsupported element/type spelling {text!r}")
        return TypeFacts(raw, kind, text, width, signed, shape, is_tensor, layout, unknown_reason=reason)
    except ValueError as exc:
        return TypeFacts(raw, unknown_reason=str(exc))
