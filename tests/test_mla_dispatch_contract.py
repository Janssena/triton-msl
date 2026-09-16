"""Packet 107/108: the MLA replacement dispatch must reproduce the SOURCE kernel's ABI —
its explicit stride arguments, its compile-time head widths and its Z/H/N scalars — never
the host tensors' shapes or ``.stride()``.

On clean ``e2330ec`` (and the packet-106 candidate) ``_dispatch_mla`` concatenated each Q/K
part's WHOLE host tensor, passed V / Out's host ``.stride()``, and derived Z/H/N from the
concatenated tensor: a valid padded view (row stride 129 over a [.., 128] host view) or a
physically wider head (129 columns with DN = 128) silently changed the result (packet 107 §2,
§3: err 3.56 / 17.1 / 0.12 vs the source oracle). Now the descriptor carries the source ABI
(widths, six stride-ref tuples, element types, Z/H/N refs, the 32 q-block) and the dispatcher
resolves it from the kernel's arguments, proves the views fit the storage, builds exact
``as_strided`` views before concatenation, and passes the source strides for V and Out.
"""

import importlib.util
import math
import pathlib

import pytest

try:
    import torch
    import torch.nn.functional as F

    import triton_msl.autotuning._fa_dispatch as fa_dispatch
    from triton_msl.autotuning._fa_dispatch import _dispatch_mla
    from triton_msl.errors import MetalNonRecoverableError

    HAS = True
    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS = False
    HAS_GPU = False

requires = pytest.mark.skipif(not HAS, reason="triton_msl needed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"


def _kernel():
    """The canonical prescaled MLA kernel from the asymmetric-attention suite (loaded by path,
    so this file stays a leaf)."""
    p = pathlib.Path(__file__).with_name("test_fa_asymmetric.py")
    spec = importlib.util.spec_from_file_location("mla_contract_src", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod._mla_prescaled_fwd
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    return fn


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))


@pytest.fixture
def mla_spy(monkeypatch):
    hits = []
    real = fa_dispatch._dispatch_mla

    def spy(*a, **k):
        r = real(*a, **k)
        hits.append(bool(r))
        return r

    monkeypatch.setattr(fa_dispatch, "_dispatch_mla", spy)
    return hits


Z, H, N, DN, DR, DV = 1, 2, 64, 128, 64, 128


def _ref(qn, qr, kn, kr, v):
    scale = 1.0 / math.sqrt(DN + DR)
    q = torch.cat([qn, qr], dim=-1)
    k = torch.cat([kn, kr], dim=-1)
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)


def _run(fn, qn, qr, kn, kr, v, out, strides):
    """Launch with EXPLICIT source strides (a list of six 4-tuples), the full grid."""
    flat = [s for st in strides for s in st]
    try:
        fn[(N // 32, Z * H)](qn, qr, kn, kr, v, out, *flat, Z, H, N, 32, 32, DN, DR, DV, False)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return "refused"
    return "computed"


def _contig():
    torch.manual_seed(107)
    t = lambda d: torch.randn(Z, H, N, d, device=D, dtype=torch.float16)
    qn, qr, kn, kr, v = t(DN), t(DR), t(DN), t(DR), t(DV)
    out = torch.full((Z, H, N, DV), float("nan"), device=D, dtype=torch.float16)
    return qn, qr, kn, kr, v, out


def _st(t):
    return tuple(t.stride(i) for i in range(4))


@requires_gpu
def test_contiguous_positive_routes_and_matches(cold_gpu_caches, mla_spy):
    fn = _kernel()
    qn, qr, kn, kr, v, out = _contig()
    assert _run(fn, qn, qr, kn, kr, v, out, [_st(x) for x in (qn, qr, kn, kr, v, out)]) == "computed"
    assert mla_spy == [True]
    assert (out.float() - _ref(qn, qr, kn, kr, v)).abs().max().item() < 5e-2


@requires_gpu
@pytest.mark.parametrize("role", ["qn", "qr", "kn", "kr"])
def test_runtime_row_stride_is_the_source_semantics(cold_gpu_caches, mla_spy, role):
    """Bites on e2330ec: the host tensor is an exact-shape view with row stride D, but the
    kernel is passed row stride D + 1 (a valid padded view of the same storage, whose
    padding column is made dominant). The replacement concatenated the host view."""
    fn = _kernel()
    qn, qr, kn, kr, v, out = _contig()
    parts = {"qn": (qn, DN), "qr": (qr, DR), "kn": (kn, DN), "kr": (kr, DR)}
    _, d = parts[role]
    torch.manual_seed(108)
    storage = torch.randn(Z, H, N, d + 1, device=D, dtype=torch.float16)
    storage[..., d] = 64.0
    host = torch.as_strided(storage, (Z, H, N, d), (H * N * d, N * d, d, 1))  # exact host shape, stride d
    source = storage[..., :d]  # what the kernel reads with row stride d + 1
    tensors = dict(qn=qn, qr=qr, kn=kn, kr=kr)
    tensors[role] = host
    logical = dict(tensors)
    logical[role] = source
    strides = [_st(x) for x in (tensors["qn"], tensors["qr"], tensors["kn"], tensors["kr"], v, out)]
    strides[list(tensors).index(role)] = _st(source)  # the SOURCE contract: padded row stride
    state = _run(fn, tensors["qn"], tensors["qr"], tensors["kn"], tensors["kr"], v, out, strides)
    if state == "computed":
        exp = _ref(logical["qn"], logical["qr"], logical["kn"], logical["kr"], v)
        assert (out.float() - exp).abs().max().item() < 5e-2, "used the host view, not the kernel's strides"
    else:
        assert bool(out.isnan().all())


@requires_gpu
def test_v_runtime_stride_is_the_source_semantics(cold_gpu_caches, mla_spy):
    fn = _kernel()
    qn, qr, kn, kr, _, out = _contig()
    torch.manual_seed(109)
    storage = torch.randn(Z, H, N, DV + 1, device=D, dtype=torch.float16)
    storage[..., DV] = 64.0
    v_host = torch.as_strided(storage, (Z, H, N, DV), (H * N * DV, N * DV, DV, 1))
    v_source = storage[..., :DV]
    strides = [_st(qn), _st(qr), _st(kn), _st(kr), _st(v_source), _st(out)]
    state = _run(fn, qn, qr, kn, kr, v_host, out, strides)
    if state == "computed":
        assert (out.float() - _ref(qn, qr, kn, kr, v_source)).abs().max().item() < 5e-2
    else:
        assert bool(out.isnan().all())


@requires_gpu
def test_out_runtime_stride_is_the_source_semantics(cold_gpu_caches, mla_spy):
    """Out is written with the kernel's strides: a padded output view leaves its padding
    column untouched and lands every value where the kernel would."""
    fn = _kernel()
    qn, qr, kn, kr, v, _ = _contig()
    storage = torch.full((Z, H, N, DV + 1), float("nan"), device=D, dtype=torch.float16)
    out_host = torch.as_strided(storage, (Z, H, N, DV), (H * N * DV, N * DV, DV, 1))
    out_source = storage[..., :DV]
    strides = [_st(qn), _st(qr), _st(kn), _st(kr), _st(v), _st(out_source)]
    state = _run(fn, qn, qr, kn, kr, v, out_host, strides)
    if state == "computed":
        assert (out_source.float() - _ref(qn, qr, kn, kr, v)).abs().max().item() < 5e-2
        assert bool(storage[..., DV].isnan().all()), "wrote outside the kernel's output view"
    else:
        assert bool(storage.isnan().all())


@requires_gpu
def test_physical_head_width_uses_the_compile_time_width(cold_gpu_caches, mla_spy):
    """Bites on e2330ec: Qn/Kn physically 129 wide, DN = 128, row stride 129 — the kernel
    reads the first 128; the replacement concatenated all 129 (packet 107 §3)."""
    fn = _kernel()
    _, qr, _, kr, v, out = _contig()
    torch.manual_seed(110)
    qn = torch.randn(Z, H, N, DN + 1, device=D, dtype=torch.float16)
    kn = torch.randn(Z, H, N, DN + 1, device=D, dtype=torch.float16)
    qn[..., DN] = 64.0
    kn[..., DN] = 64.0
    strides = [_st(qn), _st(qr), _st(kn), _st(kr), _st(v), _st(out)]
    state = _run(fn, qn, qr, kn, kr, v, out, strides)
    if state == "computed":
        assert (out.float() - _ref(qn[..., :DN], qr, kn[..., :DN], kr, v)).abs().max().item() < 5e-2
    else:
        assert bool(out.isnan().all())


@requires_gpu
def test_padded_views_are_a_supported_positive(cold_gpu_caches, mla_spy):
    """A genuinely strided source view (padded rows on every part) still ROUTES to the MLA
    replacement and matches — the capability is kept, not refused."""
    fn = _kernel()
    torch.manual_seed(111)
    mk = lambda d: torch.randn(Z, H, N, d + 1, device=D, dtype=torch.float16)
    qn, qr, kn, kr, v = mk(DN), mk(DR), mk(DN), mk(DR), mk(DV)
    out = torch.full((Z, H, N, DV + 1), float("nan"), device=D, dtype=torch.float16)
    strides = [_st(x) for x in (qn, qr, kn, kr, v, out)]  # row strides d + 1
    assert _run(fn, qn, qr, kn, kr, v, out, strides) == "computed"
    assert mla_spy == [True]
    exp = _ref(qn[..., :DN], qr[..., :DR], kn[..., :DN], kr[..., :DR], v[..., :DV])
    assert (out[..., :DV].float() - exp).abs().max().item() < 5e-2
    assert bool(out[..., DV].isnan().all())


# ---------------------------------------------------------------- descriptor contract (fake rt)
class _FakeRT:
    def __init__(self):
        self.calls = []

    def is_unsupported(self, msl):
        return False

    def mark_unsupported(self, msl):
        pass

    def get_library(self, msl):
        return ("lib", msl)

    def dispatch(self, lib, name, kargs, *, threads, group_size):
        self.calls.append((name, threads, group_size, kargs))


def _desc(**over):
    base = dict(
        idx=(0, 1, 2, 3, 4, 5),
        z=30,
        h=31,
        n=32,
        bm=32,
        dims=(DN, DR, DV),
        # kargs layout (see _kargs): 6 tensors at 0..5, then FOUR stride args per tensor
        # (z, h, row, inner) at 6..29, then Z, H, N at 30..32; the inner stride is folded ("c1").
        strides={
            "q": (6, 7, 8, "c1"),
            "q_rope": (10, 11, 12, "c1"),
            "k": (14, 15, 16, "c1"),
            "k_rope": (18, 19, 20, "c1"),
            "v": (22, 23, 24, "c1"),
            "out": (26, 27, 28, "c1"),
        },
        elems=("f16",) * 6,
    )
    base.update(over)
    return (
        "mla",
        "#include <metal_stdlib>\n// mla",
        "mla_kernel",
        256,
        *base["idx"],
        base["z"],
        base["h"],
        base["n"],
        base["bm"],
        base["dims"],
        base["strides"],
        base["elems"],
    )


def _kargs(dtype=torch.float16, over=None):
    """kargs mirroring the prescaled kernel: 6 tensors, 24 strides (Z,H,N-major row strides,
    unit inner folded), Z, H, N. ``over`` = {kargs index: value} overrides."""
    t = lambda d: torch.randn(Z, H, N, d, device=D, dtype=dtype)
    ts = [t(DN), t(DR), t(DN), t(DR), t(DV), t(DV)]
    strides = []
    for x in ts:
        strides += [x.stride(0), x.stride(1), x.stride(2), 1]
    k = ts + strides + [Z, H, N]
    for i, v in (over or {}).items():
        k[i] = v
    return k


@requires_gpu
def test_descriptor_contract_dispatches_full_grid_from_refs():
    rt = _FakeRT()
    assert _dispatch_mla(rt, _desc(), _kargs(), grid=(2, Z * H, 1)) is True
    name, threads, group, bufs = rt.calls[0]
    assert threads == (2 * 256, Z * H, 1)
    assert tuple(bufs[-3:]) == (Z, H, N) and bufs[0].shape[-1] == DN + DR and bufs[0].is_contiguous()


@requires_gpu
@pytest.mark.parametrize(
    "case",
    [
        "bm16",
        "partial-grid",
        "no-grid",
        "legacy-descriptor",
        "z-ref-is-a-tensor",
        "n-ref-out-of-range",
        "dtype-mismatch",
        "stride-out-of-storage",
        "negative-stride",
    ],
)
def test_descriptor_contract_refuses(case):
    rt = _FakeRT()
    desc, kargs, grid = _desc(), _kargs(), (2, Z * H, 1)
    if case == "bm16":
        desc = _desc(bm=16)
        grid = (4, Z * H, 1)
    elif case == "partial-grid":
        grid = (1, Z * H, 1)
    elif case == "no-grid":
        grid = None
    elif case == "legacy-descriptor":
        desc = desc[:14]
    elif case == "z-ref-is-a-tensor":
        desc = _desc(z=0)
    elif case == "n-ref-out-of-range":
        desc = _desc(n=99)
    elif case == "dtype-mismatch":
        kargs = _kargs(dtype=torch.float32)
    elif case == "stride-out-of-storage":
        kargs = _kargs(over={8: 10 * DN})  # q row stride far beyond the tensor's storage
    elif case == "negative-stride":
        kargs = _kargs(over={8: -DN})
    assert _dispatch_mla(rt, desc, kargs, grid=grid) is False
    assert not rt.calls


@requires_gpu
def test_descriptor_contract_folded_c1_head_count():
    """H folded to the constant 1 (``"c1"``) resolves without reading any kernel arg."""
    rt = _FakeRT()
    desc = _desc(h="c1")
    kargs = _kargs()
    assert _dispatch_mla(rt, desc, kargs, grid=(2, Z * 1, 1)) is True
    assert rt.calls[0][1] == (2 * 256, Z * 1, 1)
