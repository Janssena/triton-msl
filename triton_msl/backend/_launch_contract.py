"""Versioned producer-to-consumer packed descriptor, with no restore defaults.

This binds descriptor contents, not the dynamic tensor argument ABI or a foreign
pipeline's binary. Those boundaries must not infer protection from this record.
"""
import json

from triton_msl.errors import MetalNonRecoverableError


FIELDS = ("num_warps", "num_ctas", "shared", "block_size", "output_arg_indices",
          "needs_2d_grid", "mm_two_kernel", "fast_matmul", "quant_matmul",
          "flash_attention", "batched_dot_bounds", "device_assert")


def _refuse(reason):
    raise MetalNonRecoverableError("packed launch contract: " + reason + "; recompile the kernel")


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        _refuse(f"unrepresentable descriptor ({exc})")


def _record(metadata):
    required = ("name", "execution_contract", *FIELDS)
    if any(name not in metadata for name in required):
        _refuse("missing producer field")
    if type(metadata["name"]) is not str or not metadata["name"]:
        _refuse("invalid kernel name")
    for name in ("num_warps", "num_ctas", "block_size"):
        if type(metadata[name]) is not int or metadata[name] <= 0:
            _refuse(f"invalid {name}")
    if type(metadata["shared"]) is not int or metadata["shared"] < 0:
        _refuse("invalid shared-memory declaration")
    if type(metadata["needs_2d_grid"]) is not bool:
        _refuse("invalid grid mode")
    outputs = metadata["output_arg_indices"]
    assertion = metadata['device_assert']
    if assertion is not None:
        if (type(assertion) is not dict or set(assertion) != {'schema', 'messages', 'buffer_index'}
            or type(assertion['schema']) is not int or assertion['schema'] != 1
            or type(assertion['buffer_index']) is not int or not 0 <= assertion['buffer_index'] < 31
            or type(assertion['messages']) is not list or not assertion['messages']
            or any(type(m) is not str for m in assertion['messages'])):
            _refuse('invalid device assertion descriptor')
        if any(metadata[name] is not None for name in FIELDS[6:11]):
            _refuse('device assertions require the generic launch ABI')
    if outputs is not None and (
        type(outputs) not in (list, tuple)
        or any(type(i) is not int or i < 0 for i in outputs)
        or len(set(outputs)) != len(outputs)
    ):
        _refuse("invalid output indices")
    return {"schema": 2, "kind": "metal-packed-launch", "name": metadata["name"],
            "execution_contract": metadata["execution_contract"],
            "fields": {name: metadata[name] for name in FIELDS}}


def seal_launch_metadata(metadata):
    """Called ONLY after successful fresh compilation, never during restore.

    Explicit producer defaults retain existing fresh IRSource semantics. Missing
    fields in a restored record are never interpreted through these defaults.
    """
    if "num_warps" not in metadata or "num_ctas" not in metadata:
        _refuse("compiler omitted launch options")
    defaults = {"shared": 0, "block_size": metadata["num_warps"] * 32,
                "output_arg_indices": None, "needs_2d_grid": False,
                **{name: None for name in FIELDS[6:]}}
    for name, value in defaults.items():
        metadata.setdefault(name, value)
    metadata["launch_contract"] = _canonical(_record(metadata))


def validate_launch_metadata(metadata):
    """Validate a full producer record; return an immutable packed-value snapshot."""
    if isinstance(metadata, dict):
        values = metadata
    elif hasattr(metadata, "__dict__"):
        values = vars(metadata)
    elif hasattr(metadata, "_asdict"):
        values = metadata._asdict()
    else:
        _refuse("unrecognized metadata container")
    record = _canonical(_record(values))
    if type(values.get("launch_contract")) is not str or values["launch_contract"] != record:
        _refuse("missing, altered or incompatible producer record")
    # The serialized copy cannot alias a mutable list/dict in restored metadata.
    return _canonical([values[name] for name in FIELDS])


def validate_packed_launch(packed, expected):
    if type(packed) not in (tuple, list) or len(packed) != len(FIELDS):
        _refuse("wrong packed descriptor arity")
    if type(expected) is not str or _canonical(packed) != expected:
        _refuse("descriptor changed after launcher construction")
    # Hooks can hold references to the caller's tuple and nested lists. Execute
    # from an unaliased checked snapshot, not mutable data checked before a hook.
    return tuple(json.loads(expected))


def pack_launch_metadata(metadata):
    expected = validate_launch_metadata(metadata)
    # Version 2 appends the assertion descriptor; nested JSON arrays remain lists,
    # as they already are after Triton's persistent metadata restoration.
    return tuple(json.loads(expected))
