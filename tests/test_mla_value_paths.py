"""Packet 109/110: the MLA replacement must prove the WHOLE value path it replaces.

On ``e2330ec`` ``_lower_mla_attention_template`` emitted the MLA descriptor without calling
the value-path verifier the symmetric routes use, and ``_dispatch_mla`` checked each role's
dtype only against its own descriptor entry while the template declares ONE pointer type.
Seven classes were silently wrong on the GPU (packet 109 §1-§2):

  A  mixed element types across Qn/Qr/Kn/Kr/V/Out (fp32 or bf16 roles behind an fp16 template)
  B1 a Q-rope transform (``qr * 2``)          B2 a score bias after the two dots
  B3 a probability transform (``p + 0.25``)   B4 an Out epilogue (``acc + 1``)
  B5 a value mask on the Q-nope load (``m < 16``)   B6 a partial Out store mask (``m < 16``)

Now the MLA lowering requires all six roles to share the template's one type (f16 or f32) and
runs the shared verifier with the rope pair (same single scale, replayed transposes/casts
only), the score path from the ROPE dot, load masks restricted to the exact N_CTX boundary
form with a zero ``other``, and the output-mask guard. The same load-mask rule now covers the
symmetric simd route (last section), whose canonical kernel masks Q at the boundary.

Every positive asserts the replacement dispatch was actually taken (spy) so a generic
fallback cannot masquerade as a fix; every refusal asserts no dispatch happened.
"""

import importlib.util
import math
import pathlib
import sys

import pytest

try:
    import torch
    import torch.nn.functional as F
    import triton
    import triton.language as tl

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

Z, H, N, DN, DR, DV = 1, 2, 32, 128, 64, 128


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


@pytest.fixture
def fa_spy(monkeypatch):
    """Spy on the symmetric FA replacement dispatch (the driver imports the symbol at call time)."""
    hits = []
    real = fa_dispatch.dispatch_flash_attention

    def spy(*a, **k):
        r = real(*a, **k)
        hits.append(bool(r))
        return r

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)
    return hits


if HAS:

    @triton.jit
    def _mla_value_path(
        Qn,
        Qr,
        Kn,
        Kr,
        V,
        Out,
        qnz,
        qnh,
        qnm,
        qnk,
        qrz,
        qrh,
        qrm,
        qrk,
        knz,
        knh,
        knn,
        knk,
        krz,
        krh,
        krn,
        krk,
        vz,
        vh,
        vn,
        vk,
        oz,
        oh,
        om,
        ok_,
        Z,
        H,
        N_CTX,
        BM: tl.constexpr,
        BN: tl.constexpr,
        DN: tl.constexpr,
        DR: tl.constexpr,
        DV: tl.constexpr,
        MODE: tl.constexpr,
    ):
        """GPT's packet-109 harness: MODE 0 is the canonical MLA kernel; each other MODE adds
        exactly one semantic operation the replacement does not replay."""
        pm = tl.program_id(0)
        hz = tl.program_id(1)
        z = hz // H
        h = hz % H
        m = pm * BM + tl.arange(0, BM)
        n0 = tl.arange(0, BN)
        dn = tl.arange(0, DN)
        dr = tl.arange(0, DR)
        dv = tl.arange(0, DV)
        scale = 1.0 / tl.sqrt(float(DN + DR))
        qn_ptr = Qn + z * qnz + h * qnh + m[:, None] * qnm + dn[None, :] * qnk
        if MODE == 5:
            qn = tl.load(qn_ptr, mask=m[:, None] < 16, other=0.0) * scale
        else:
            qn = tl.load(qn_ptr) * scale
        qr = tl.load(Qr + z * qrz + h * qrh + m[:, None] * qrm + dr[None, :] * qrk) * scale
        if MODE == 1:
            qr = qr * 2.0
        acc = tl.zeros([BM, DV], dtype=tl.float32)
        mi = tl.full([BM], float("-inf"), tl.float32)
        li = tl.zeros([BM], tl.float32)
        for s in range(0, N_CTX, BN):
            kn = tl.load(Kn + z * knz + h * knh + (s + n0)[:, None] * knn + dn[None, :] * knk)
            kr = tl.load(Kr + z * krz + h * krh + (s + n0)[:, None] * krn + dr[None, :] * krk)
            qk = tl.dot(qn, tl.trans(kn).to(qn.dtype)) + tl.dot(qr, tl.trans(kr).to(qr.dtype))
            if MODE == 2:
                qk = qk + (s + n0)[None, :] * 0.01
            mij = tl.max(qk, 1)
            mnew = tl.maximum(mi, mij)
            al = tl.exp(mi - mnew)
            p = tl.exp(qk - mnew[:, None])
            if MODE == 3:
                p = p + 0.25
            li = li * al + tl.sum(p, 1)
            acc = acc * al[:, None]
            v = tl.load(V + z * vz + h * vh + (s + n0)[:, None] * vn + dv[None, :] * vk)
            acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
            mi = mnew
        acc = acc / li[:, None]
        if MODE == 4:
            acc = acc + 1.0
        out_ptr = Out + z * oz + h * oh + m[:, None] * om + dv[None, :] * ok_
        if MODE == 6:
            tl.store(out_ptr, acc.to(Out.dtype.element_ty), mask=m[:, None] < 16)
        else:
            tl.store(out_ptr, acc.to(Out.dtype.element_ty))

    @triton.jit
    def _sym_fa_masked(
        Q,
        K,
        V,
        Out,
        qz,
        qh,
        qm,
        qk_,
        kz,
        kh,
        kn,
        kk,
        vz,
        vh,
        vn,
        vk,
        oz,
        oh,
        om,
        ok_,
        Z,
        H,
        N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        MODE: tl.constexpr,
    ):
        """The canonical symmetric FA spelling (as tests/test_flash_attention.py) with the Q
        load mask varied: MODE 0 = the boundary mask the template applies itself, MODE 1 = a
        VALUE mask (rows >= 16 read as zero), MODE 2 = boundary mask with a nonzero ``other``,
        MODE 3 = a value mask on the K load."""
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        q_ptrs = Q + off_z * qz + off_h * qh + offs_m[:, None] * qm + offs_d[None, :] * qk_
        if MODE == 1:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < 16, other=0.0)
        elif MODE == 2:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=1.0)
        else:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        qk_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        q = q * qk_scale
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        for start_n in range(0, N_CTX, BLOCK_N):
            k_ptrs = K + off_z * kz + off_h * kh + (start_n + offs_n)[:, None] * kn + offs_d[None, :] * kk
            if MODE == 3:
                k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < 16, other=0.0)
            else:
                k = tl.load(k_ptrs)
            qk = tl.dot(q, tl.trans(k).to(q.dtype))
            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(V + off_z * vz + off_h * vh + (start_n + offs_n)[:, None] * vn + offs_d[None, :] * vk)
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new
        acc = acc / l_i[:, None]
        o_ptrs = Out + off_z * oz + off_h * oh + offs_m[:, None] * om + offs_d[None, :] * ok_
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty))


def _st(t):
    return [t.stride(i) for i in range(4)]


def _mla_expected(qn, qr, kn, kr, v, mode):
    """GPT's oracle for each MODE (what the SOURCE kernel computes)."""
    scale = 1.0 / math.sqrt(qn.shape[-1] + qr.shape[-1])
    if mode == 5:
        qn = qn.clone()
        qn[..., 16:, :] = 0
    if mode == 1:
        qr = qr * 2.0
    scores = (qn.float() @ kn.float().transpose(-2, -1) + qr.float() @ kr.float().transpose(-2, -1)) * scale
    if mode == 2:
        scores = scores + torch.arange(scores.shape[-1], device=D)[None, None, None, :] * 0.01
    if mode == 3:
        raw = torch.exp(scores - scores.max(-1, keepdim=True).values) + 0.25
        out = (raw @ v.float()) / raw.sum(-1, keepdim=True)
    elif mode == 2:
        out = torch.softmax(scores, -1) @ v.float()
    else:
        out = F.scaled_dot_product_attention(
            torch.cat([qn.float(), qr.float()], -1), torch.cat([kn.float(), kr.float()], -1), v.float(), scale=scale
        )
    if mode == 4:
        out = out + 1.0
    if mode == 6:
        out = out.clone()
        out[..., 16:, :] = 99.0
    return out


def _run_mla(fn, tensors, out, mode, dims=(DN, DR, DV)):
    qn, qr, kn, kr, v = tensors
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    try:
        fn[(N // 32, Z * H)](
            qn,
            qr,
            kn,
            kr,
            v,
            out,
            *_st(qn),
            *_st(qr),
            *_st(kn),
            *_st(kr),
            *_st(v),
            *_st(out),
            Z,
            H,
            N,
            32,
            32,
            *dims,
            mode,
        )
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return "refused"
    return "computed"


def _mla_tensors(seed, dtype=torch.float16):
    torch.manual_seed(seed)
    mk = lambda d, ty=dtype: torch.randn(Z, H, N, d, device=D, dtype=ty)
    return mk(DN), mk(DR), mk(DN), mk(DR), mk(DV)


# ---------------------------------------------------------------------------------------
# B1-B6 + the control: end-to-end on the GPU
# ---------------------------------------------------------------------------------------


@requires_gpu
def test_canonical_mla_routes_and_matches(cold_gpu_caches, mla_spy):
    tensors = _mla_tensors(1090)
    out = torch.full((Z, H, N, DV), float("nan"), device=D, dtype=torch.float16)
    assert _run_mla(_mla_value_path, tensors, out, 0) == "computed"
    assert mla_spy == [True], "the MLA replacement must be the path taken"
    assert (out.float() - _mla_expected(*tensors, 0)).abs().max().item() < 5e-2


@requires_gpu
@pytest.mark.parametrize(
    "mode,label",
    [
        (1, "B1 Q-rope transform"),
        (2, "B2 score bias after the two dots"),
        (3, "B3 probability transform"),
        (4, "B4 Out epilogue"),
        (5, "B5 value mask on the Q-nope load"),
        (6, "B6 partial Out store mask"),
    ],
)
def test_unreplayed_mla_semantics_refuse(cold_gpu_caches, mla_spy, mode, label):
    """Bites on e2330ec: every mode routed to the MLA replacement (one hit each) and stored
    the CANONICAL result — max err against the source oracle 0.91 / 0.10 / 1.12 / 1.00 /
    0.99 / 100.7 for modes 1-6 (packet 109 §3, reproduced to the digit on the clean tree).
    Now: refused before any dispatch."""
    tensors = _mla_tensors(1090 + mode)
    fill = 99.0 if mode == 6 else float("nan")
    out = torch.full((Z, H, N, DV), fill, device=D, dtype=torch.float16)
    state = _run_mla(_mla_value_path, tensors, out, mode)
    if state == "computed":
        # a route that computes must have computed the SOURCE semantics
        assert (out.float() - _mla_expected(*tensors, mode)).abs().max().item() < 5e-2, label
    else:
        assert mla_spy == [], f"{label}: refusal must happen before any dispatch"


# ---------------------------------------------------------------------------------------
# A: one template type for six roles
# ---------------------------------------------------------------------------------------


def _prescaled_kernel():
    p = pathlib.Path(__file__).with_name("test_fa_asymmetric.py")
    spec = importlib.util.spec_from_file_location("mla_value_src", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod._mla_prescaled_fwd
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    return fn


@requires_gpu
@pytest.mark.parametrize(
    "qk_dtype,v_dtype,out_dtype",
    [
        (torch.float32, torch.float16, torch.float16),  # fp32 Q/K roles behind an fp16 template
        (torch.bfloat16, torch.bfloat16, torch.float16),  # bf16 roles: the template has no bf16 variant
    ],
)
def test_mixed_role_dtypes_never_route_to_the_one_type_template(cold_gpu_caches, mla_spy, qk_dtype, v_dtype, out_dtype):
    """Bites on e2330ec: the bf16 row routed (one MLA hit) and reinterpreted the buffers, max
    err 2.22 against SDPA (packet 109 §2); the fp32-Q/K row returned NON-FINITE output there.
    Now: refused at the producer, no dispatch."""
    fn = _prescaled_kernel()
    n = 64
    torch.manual_seed(109)
    mk = lambda d, ty: torch.randn(Z, H, n, d, device=D, dtype=ty)
    qn, qr, kn, kr, v = mk(DN, qk_dtype), mk(DR, qk_dtype), mk(DN, qk_dtype), mk(DR, qk_dtype), mk(DV, v_dtype)
    out = torch.full((Z, H, n, DV), float("nan"), device=D, dtype=out_dtype)
    try:
        fn[(n // 32, Z * H)](
            qn,
            qr,
            kn,
            kr,
            v,
            out,
            *_st(qn),
            *_st(qr),
            *_st(kn),
            *_st(kr),
            *_st(v),
            *_st(out),
            Z,
            H,
            n,
            32,
            32,
            DN,
            DR,
            DV,
            False,
        )
        torch.mps.synchronize()
        state = "computed"
    except MetalNonRecoverableError:
        state = "refused"
    assert True not in mla_spy, "a mixed-type ABI must never reach the one-type template"
    if state == "computed":
        ref = F.scaled_dot_product_attention(
            torch.cat([qn.float(), qr.float()], -1),
            torch.cat([kn.float(), kr.float()], -1),
            v.float(),
            scale=1.0 / math.sqrt(DN + DR),
        )
        assert (out.float() - ref).abs().max().item() < 5e-2


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


def _kargs(dtypes):
    ts = [torch.randn(Z, H, N, d, device=D, dtype=ty) for d, ty in zip((DN, DR, DN, DR, DV, DV), dtypes)]
    strides = []
    for x in ts:
        strides += [x.stride(0), x.stride(1), x.stride(2), 1]
    return ts + strides + [Z, H, N]


@requires_gpu
@pytest.mark.parametrize(
    "elems,dtypes",
    [
        (("f16",) * 5 + ("bf16",), (torch.float16,) * 5 + (torch.bfloat16,)),  # Out bf16 behind fp16
        (("f32",) + ("f16",) * 5, (torch.float32,) + (torch.float16,) * 5),  # Q-nope fp32 behind fp16
        (("bf16",) * 6, (torch.bfloat16,) * 6),  # uniform bf16: no template variant
        (("f16",) * 5, (torch.float16,) * 6),  # a descriptor missing a role type
    ],
)
def test_dispatcher_requires_one_template_type_for_all_roles(elems, dtypes):
    """Bites on e2330ec: the dispatcher checked each role only against its own entry, so a
    per-role-consistent mixed descriptor dispatched. Now: False, no dispatch."""
    rt = _FakeRT()
    assert _dispatch_mla(rt, _desc(elems=elems), _kargs(dtypes), grid=(1, Z * H, 1)) is False
    assert not rt.calls


@requires_gpu
@pytest.mark.parametrize("elem,dtype", [("f16", torch.float16), ("f32", torch.float32)])
def test_dispatcher_uniform_supported_type_dispatches(elem, dtype):
    """The explicit policy: all-f16 or all-f32 are the two supported ABIs."""
    rt = _FakeRT()
    assert _dispatch_mla(rt, _desc(elems=(elem,) * 6), _kargs((dtype,) * 6), grid=(1, Z * H, 1)) is True
    assert len(rt.calls) == 1 and rt.calls[0][3][0].dtype == dtype


# ---------------------------------------------------------------------------------------
# Sibling: the symmetric simd route under the same load-mask rule
# ---------------------------------------------------------------------------------------


def _sym_expected(q, k, v, mode):
    if mode == 1:
        q = q.clone()
        q[..., 16:, :] = 0
    if mode == 3:
        k = k.clone()
        k[..., 16:, :] = 0
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float())


@requires_gpu
def test_symmetric_boundary_masked_q_routes_and_matches(cold_gpu_caches, fa_spy):
    """The canonical spelling masks Q at ``offs_m < N_CTX`` with ``other=0``: exactly the
    boundary the simd template applies itself — accepted and routed."""
    z, h, n, hd = 1, 2, 64, 128
    torch.manual_seed(1100)
    q, k, v = (torch.randn(z, h, n, hd, device=D, dtype=torch.float16) for _ in range(3))
    out = torch.full_like(q, float("nan"))
    if hasattr(_sym_fa_masked, "device_caches"):
        _sym_fa_masked.device_caches.clear()
    _sym_fa_masked[(n // 32, z * h)](q, k, v, out, *_st(q), *_st(k), *_st(v), *_st(out), z, h, n, 32, 32, hd, 0)
    torch.mps.synchronize()
    assert True in fa_spy, "the simd FA replacement must be the path taken"
    assert (out.float() - _sym_expected(q, k, v, 0)).abs().max().item() < 5e-2


@requires_gpu
@pytest.mark.parametrize(
    "mode,label",
    [(1, "value mask on Q (rows >= 16 zero)"), (2, "boundary mask with other=1.0"), (3, "value mask on K")],
)
def test_symmetric_non_boundary_load_masks_refuse_or_match(cold_gpu_caches, fa_spy, mode, label):
    """Bites on e2330ec (modes 1, 2): the simd route stored the unmasked result (err ~0.5 vs the
    source). Now: refused before any dispatch, or the exact source semantics."""
    z, h, n, hd = 1, 2, 64, 128
    torch.manual_seed(1100 + mode)
    q, k, v = (torch.randn(z, h, n, hd, device=D, dtype=torch.float16) for _ in range(3))
    out = torch.full_like(q, float("nan"))
    if hasattr(_sym_fa_masked, "device_caches"):
        _sym_fa_masked.device_caches.clear()
    try:
        _sym_fa_masked[(n // 32, z * h)](q, k, v, out, *_st(q), *_st(k), *_st(v), *_st(out), z, h, n, 32, 32, hd, mode)
        torch.mps.synchronize()
        state = "computed"
    except MetalNonRecoverableError:
        state = "refused"
    if state == "computed":
        assert (out.float() - _sym_expected(q, k, v, mode)).abs().max().item() < 5e-2, label
    else:
        assert True not in fa_spy, f"{label}: refusal must happen before any dispatch"
