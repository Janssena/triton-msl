"""Initialize the compiler's required module map before capturing product keys."""
from triton.backends.compiler import GPUTarget
from triton_msl.backend.compiler import MetalBackend


def test_required_module_map_is_ready_before_backend_hash(monkeypatch):
    from triton_msl.backend import _cache_contract as cache
    calls = []
    monkeypatch.setattr(MetalBackend, "get_module_map", lambda self: calls.append("module-map") or {})
    original = cache.source_contract
    def contract():
        assert calls, "backend identity captured before required compiler imports"
        return original()
    monkeypatch.setattr(cache, "source_contract", contract)
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    backend.hash()
    assert calls == ["module-map"]
