"""Memory and disk source stashes must validate the same semantic product."""
import json

import pytest
import triton  # noqa: F401 — complete backend discovery before importing compiler

from triton_msl.backend import compiler

KEY = "a" * 64
SOURCE = "kernel void stash_contract() {}"


@pytest.fixture(autouse=True)
def isolated_stash(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "1")
    monkeypatch.setattr(compiler, "_MSL_BY_KEY", {})
    return tmp_path


@pytest.mark.parametrize("disk", [False, True])
def test_stash_policy_transition_misses_in_memory_and_on_disk(disk, monkeypatch):
    compiler._stash_msl(SOURCE, KEY, 64)
    assert compiler._load_stashed_msl(KEY) == (SOURCE, 64)
    if disk:
        compiler._MSL_BY_KEY.clear()
    monkeypatch.setenv("TRITON_MSL_INFER_LAYOUT", "0")
    assert compiler._load_stashed_msl(KEY) is None


@pytest.mark.parametrize("disk", [False, True])
def test_stash_implementation_transition_misses(disk, monkeypatch):
    from triton_msl.backend import _cache_contract
    compiler._stash_msl(SOURCE, KEY, 64)
    if disk:
        compiler._MSL_BY_KEY.clear()
    monkeypatch.setattr(_cache_contract, "implementation_identity", lambda: "different-package")
    assert compiler._load_stashed_msl(KEY) is None


@pytest.mark.parametrize("damage", ["source", "geometry", "schema", "key", "missing_metadata", "legacy"])
def test_corrupted_stash_is_a_miss_not_default_geometry(isolated_stash, damage):
    compiler._stash_msl(SOURCE, KEY, 64)
    path = isolated_stash / f"{KEY}.mslstash"
    record = json.loads(path.read_text())
    if damage == "legacy":
        record = {"msl": SOURCE, "block_size": 64}
    elif damage == "source":
        record["msl"] = "kernel void other() {}"
    elif damage == "geometry":
        # Handle the old shape as well so the pre-fix witness fails for an
        # accepted wrong value, not merely for a nonexistent envelope field.
        record.setdefault("metadata", {})["block_size"] = 1024
        record["block_size"] = 1024
    elif damage == "schema":
        record["schema"] = -1
    elif damage == "key":
        record["key"] = "b" * 64
    else:
        record.pop("metadata", None)
    path.write_text(json.dumps(record))
    compiler._MSL_BY_KEY.clear()
    assert compiler._load_stashed_msl(KEY) is None


@pytest.mark.parametrize("size", [None, False, 0, -1, 1025, "64"])
def test_unproved_stash_threadgroup_size_is_not_published(size, isolated_stash):
    compiler._stash_msl(SOURCE, KEY, size)
    assert compiler._load_stashed_msl(KEY) is None
    assert not (isolated_stash / f"{KEY}.mslstash").exists()


def test_recompile_repairs_poisoned_disk_entry(isolated_stash):
    path = isolated_stash / f"{KEY}.mslstash"
    path.write_text('{"msl":"wrong", "block_size":1024}')
    compiler._stash_msl(SOURCE, KEY, 64)
    compiler._MSL_BY_KEY.clear()
    assert compiler._load_stashed_msl(KEY) == (SOURCE, 64)


def test_legacy_in_memory_tuple_is_not_a_trusted_product():
    compiler._MSL_BY_KEY[KEY] = (SOURCE, 64)
    assert compiler._load_stashed_msl(KEY) is None


def test_identical_source_different_keys_stay_independent():
    compiler._stash_msl(SOURCE, KEY, 64)
    compiler._stash_msl(SOURCE, "b" * 64, 128)
    assert compiler._load_stashed_msl(KEY) == (SOURCE, 64)
    assert compiler._load_stashed_msl("b" * 64) == (SOURCE, 128)


def test_stash_envelope_cannot_impersonate_source_product(isolated_stash):
    from triton_msl.backend._cache_contract import read_metadata
    compiler._stash_msl(SOURCE, KEY, 64)
    assert read_metadata(SOURCE, isolated_stash / f"{KEY}.mslstash", KEY) is None


def test_source_product_cannot_impersonate_stash(isolated_stash):
    from triton_msl.backend._cache_contract import metadata_record
    record = json.loads(metadata_record(SOURCE, {"block_size": 64}, KEY))
    record["msl"] = SOURCE
    (isolated_stash / f"{KEY}.mslstash").write_text(json.dumps(record))
    assert compiler._load_stashed_msl(KEY) is None


@pytest.mark.parametrize("key", [None, "short", "../outside", "A" * 64])
def test_invalid_stash_key_cannot_reach_disk_lookup(key, monkeypatch):
    def forbidden():
        pytest.fail("invalid cache identity reached the filesystem")
    monkeypatch.setattr(compiler, "_get_cache_dir", forbidden)
    compiler._stash_msl(SOURCE, key, 64)
    assert compiler._load_stashed_msl(key) is None


def test_same_process_concurrent_stash_writers_use_distinct_temporary_files(isolated_stash, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    real = compiler.os.replace
    barrier = threading.Barrier(2)
    names = []
    def replace(src, dst):
        names.append(src)
        barrier.wait(timeout=5)
        real(src, dst)
    monkeypatch.setattr(compiler.os, "replace", replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(compiler._stash_msl, SOURCE, KEY, 64) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert len(set(names)) == 2
    compiler._MSL_BY_KEY.clear()
    assert compiler._load_stashed_msl(KEY) == (SOURCE, 64)
    assert not list(isolated_stash.glob("*.tmp"))
