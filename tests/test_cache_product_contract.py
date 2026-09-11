"""CPU witnesses: cached source and launch metadata are one semantic product.

Synthetic emission tests the cache boundary, not GPU correctness. No version bump
or other unrelated key change is allowed between either side of a witness.
"""

import json

import pytest
import triton  # noqa: F401

from triton_msl.errors import MetalNonRecoverableError


@pytest.fixture
def product(monkeypatch, tmp_path):
    import triton_msl.backend.compiler as compiler

    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "1")
    calls = []

    def emit(mod, metadata, options):
        import os

        if os.environ["TRITON_MSL_INFER_LAYOUT"] != "1":
            raise MetalNonRecoverableError("synthetic strict-layout rejection")
        calls.append(str(mod))
        metadata.update(
            name="cache_contract",
            block_size=64,
            needs_2d_grid=True,
            output_arg_indices=[1],
            fast_matmul={"test_nested_descriptor": [32, 64]},
        )
        return "kernel void cache_contract() {}"

    monkeypatch.setattr("triton_msl.codegen.msl_emitter.emit_msl", emit)

    def compile():
        metadata = {"name": "cache_contract"}
        source = compiler.MetalBackend.make_msl("module", metadata, compiler.MetalOptions())
        return source, metadata

    return compile, calls, tmp_path


def test_msl_cache_rechecks_live_strict_policy(product, monkeypatch):
    compile, calls, _ = product
    compile()
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "0")
    with pytest.raises(MetalNonRecoverableError, match="synthetic strict-layout rejection"):
        compile()
    assert len(calls) == 1


def test_msl_cache_preserves_full_launch_metadata(product):
    compile, calls, _ = product
    first = compile()
    second = compile()
    assert first == second
    assert len(calls) == 1


def test_msl_cache_preserves_native_triton_target(product):
    """Real Triton compilation supplies a GPUTarget, not a JSON dictionary.

    A synthetic name-only metadata fixture misses this boundary. A non-cacheable
    product is safe but must not silently disable every ordinary source-cache hit.
    """
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend.compiler import MetalBackend, MetalOptions

    _, calls, directory = product
    target = GPUTarget("metal", "apple-m4", 32)
    options = MetalOptions()
    first = {"name": "cache_contract", "target": target, **vars(options)}
    second = dict(first)
    source = MetalBackend.make_msl("module", first, options)
    assert list(directory.glob("*.meta.json")), "native target prevented source-cache publication"
    assert MetalBackend.make_msl("module", second, options) == source
    assert len(calls) == 1, "second compile must hit the inner source cache"
    assert type(second["target"]) is GPUTarget
    assert second["target"] == target
    # Triton's own persistent metadata uses JSON/default=vars, which normalizes
    # option tuples to lists. Preserve every serialized field, while the native
    # target type itself is checked separately above.
    assert json.dumps(second, default=vars, sort_keys=True) == json.dumps(first, default=vars, sort_keys=True)


@pytest.mark.parametrize("bad_target", [None, {"backend": "metal"}, object()])
def test_source_cache_never_guesses_an_unknown_native_target(product, bad_target):
    from triton_msl.backend.compiler import MetalBackend, MetalOptions

    _, calls, directory = product
    for _ in range(2):
        MetalBackend.make_msl("module", {"name": "cache_contract", "target": bad_target}, MetalOptions())
    assert len(calls) == 2
    assert not list(directory.glob("*.meta.json"))


@pytest.mark.parametrize("field,value", [("hash", "second-outer-hash"), ("arch", "apple-m5"), ("warp_size", 16)])
def test_inner_source_hit_cannot_replace_current_compiler_context(product, field, value):
    from triton.backends.compiler import GPUTarget
    from triton_msl.backend.compiler import MetalBackend, MetalOptions

    _, calls, _ = product
    target = {"backend": "metal", "arch": "apple-m4", "warp_size": 32}
    first = {"name": "cache_contract", "target": GPUTarget(**target), "hash": "first-outer-hash"}
    second = dict(first)
    if field == "hash":
        second[field] = value
    else:
        target[field] = value
        second["target"] = GPUTarget(**target)
    expected_target, expected_hash = second["target"], second["hash"]
    MetalBackend.make_msl("module", first, MetalOptions())
    MetalBackend.make_msl("module", second, MetalOptions())
    assert len(calls) == 2
    assert second["target"] == expected_target and second["hash"] == expected_hash


@pytest.mark.parametrize("damage", ["missing", "malformed", "wrong_geometry"])
def test_msl_cache_damaged_metadata_is_a_miss(product, damage):
    compile, calls, directory = product
    first = compile()
    path = next(directory.glob("*.meta.json"))
    if damage == "missing":
        path.rename(path.with_suffix(".json.retained"))
    elif damage == "malformed":
        path.write_text("not json")
    else:
        # A sidecar from another/older writer must not be interpreted as current
        # metadata, even when its JSON is syntactically valid.
        path.write_text(json.dumps({"block_size": 1024, "needs_2d_grid": False}))
    second = compile()
    assert first == second
    assert len(calls) == 2


def test_msl_cache_source_digest_binds_metadata(product):
    compile, calls, directory = product
    first = compile()
    path = next(directory.glob("*.msl"))
    path.write_text("kernel void unrelated_kernel() {}")
    assert compile() == first
    assert len(calls) == 2


@pytest.mark.parametrize("damage", ["geometry", "missing_descriptor"])
def test_msl_cache_record_binds_its_metadata_payload(product, damage):
    compile, calls, directory = product
    first = compile()
    path = next(directory.glob("*.meta.json"))
    record = json.loads(path.read_text())
    if damage == "geometry":
        record["metadata"]["block_size"] = 1024
    else:
        record["metadata"].pop("fast_matmul")
    path.write_text(json.dumps(record))
    assert compile() == first
    assert len(calls) == 2


@pytest.mark.parametrize(
    "flag,on,off",
    [
        ("TRITON_MSL_INFER_LAYOUT", "1", "0"),
        ("TRITON_MSL_LEGACY", "1", "0"),
        ("TRITON_MSL_QUANT_MATMUL", "1", "0"),
        ("TRITON_MSL_FAST_MATMUL", "1", "0"),
        ("TRITON_MSL_USE_CPP", "1", "0"),
        ("TRITON_MSL_FORCE_PYTHON", "1", "0"),
        ("TRITON_MSL_CPP_SKIP", "dot", ""),
    ],
)
def test_each_semantic_policy_moves_inner_key(monkeypatch, flag, on, off):
    from triton_msl.backend.compiler import _msl_cache_key

    monkeypatch.setenv(flag, on)
    first = _msl_cache_key("module", "options")
    monkeypatch.setenv(flag, off)
    assert _msl_cache_key("module", "options") != first
