"""Array reuse requires a real, cross-thread synchronization boundary."""

import pytest

from triton_msl.codegen._lowerer_helpers import _alias_shared_memory


def emit(body):
    return (
        """kernel void probe(uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float a[128];
    threadgroup float b[128];
"""
        + body
        + "\n}"
    )


@pytest.mark.parametrize(
    "boundary,allowed",
    [
        ("threadgroup_barrier(mem_flags::mem_threadgroup);", True),
        ("threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);", True),
        ("", False),
        ("threadgroup_barrier(mem_flags::mem_device);", False),
        ("simdgroup_barrier(mem_flags::mem_threadgroup);", False),
        ("if (lid == 0) {\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n}", False),
        ("if (lid == 0)\nthreadgroup_barrier(mem_flags::mem_threadgroup);", False),
        ("for (int i=0; i<0; ++i) {\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n}", False),
        ("/* }\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n{ */", False),
        ('const char* diagnostic = "} threadgroup_barrier(mem_flags::mem_threadgroup); {";', False),
    ],
)
def test_reuse_requires_unconditional_threadgroup_memory_boundary(boundary, allowed):
    aliases = {}
    result = _alias_shared_memory(
        emit("a[lid] = 1;\nfloat old = a[127-lid];\n" + boundary + "\nb[lid] = old;"), allocation_aliases=aliases
    )
    assert ("threadgroup float b[128];" not in result) is allowed
    assert (aliases.get("b", "b") == aliases.get("a", "a")) is allowed


def test_loop_local_barrier_cannot_prove_backedge_reuse():
    body = """for (int step=0; step<4; ++step) {
        a[lid] = 1;
        float old = a[127-lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        b[lid] = old;
        float next = b[127-lid];
    }"""
    result = _alias_shared_memory(emit(body))
    assert "threadgroup float a[128];" in result
    assert "threadgroup float b[128];" in result
    # Forward and backedge synchronization together make the reuse legitimate.
    safe = body.replace(
        "float next = b[127-lid];", "float next = b[127-lid];\nthreadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    assert "threadgroup float b[128];" not in _alias_shared_memory(emit(safe))


def test_overlapping_lifetimes_do_not_alias_across_barrier():
    result = _alias_shared_memory(
        emit("""a[lid] = 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        b[lid] = a[lid];
        float value = a[lid] + b[lid];""")
    )
    assert "threadgroup float b[128];" in result


@pytest.mark.parametrize(
    "inside,expected",
    [
        ("a[lid]=1;\nfloat old=a[127-lid];\nthreadgroup_barrier(mem_flags::mem_threadgroup);", True),
        ("a[lid]=1;\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nfloat old=a[127-lid];", False),
        (
            "a[lid]=1;\nfloat old=a[127-lid];\nfor(int j=0;j<0;j++) {\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n}",
            False,
        ),
        ("a[lid]=1;\nfloat old=a[127-lid];\nif(i==2) break;\nthreadgroup_barrier(mem_flags::mem_threadgroup);", False),
        (
            "a[lid]=1;\nfloat old=a[127-lid];\nif(i==2) continue;\nthreadgroup_barrier(mem_flags::mem_threadgroup);",
            False,
        ),
    ],
)
def test_loop_exit_barrier_must_follow_last_read(inside, expected):
    source = emit("for(int i=0;i<4;i++) {\n" + inside + "\n}\nb[lid]=2;")
    assert ("threadgroup float b[128];" not in _alias_shared_memory(source)) is expected


def test_different_types_and_preprocessor_paths_do_not_alias():
    body = "a[lid] = 1;\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nb[lid] = 2;"
    typed = emit(body).replace("float b[128]", "int b[128]")
    assert "threadgroup int b[128];" in _alias_shared_memory(typed)
    conditional = emit(
        body.replace("threadgroup_barrier", "#if ENABLE\nthreadgroup_barrier").replace(";\nb[lid]", ";\n#endif\nb[lid]")
    )
    assert "threadgroup float b[128];" in _alias_shared_memory(conditional)
