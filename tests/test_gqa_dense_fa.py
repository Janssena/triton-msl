"""GQA/MQA (grouped-/multi-query attention: Llama, Mistral, Qwen) on the DENSE [Z,H,N,D]
FlashAttention path. K/V have H/group heads, indexed by off_h // group -- a DIFFERENT head
index than Q. The hd64/hd128 simd/tiled templates apply Q's head offset (h = zh % H) to
K/V, so routing GQA SILENT-WRONGS (measured max err ~1.1 at head_dim 128 before the guard).

The detector must refuse GQA (Q/K/V head-index mismatch) so it lowers on a path that honors
the real per-tensor head offset (generic at hd<=64; loud refuse -> CPU fallback at hd128).
Either way the OUTPUT must match the GQA reference. MHA (shared head offset) still routes.
"""
import math
import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _dense_gqa_fa(Q, K, V, O, sqz, sqh, sqm, sqk, skz, skh, skn, skk,
                  svz, svh, svn, svk, soz, soh, som, sok,
                  Z, H, GROUP, N, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    sm = tl.program_id(0); hz = tl.program_id(1); z = hz // H; h = hz % H
    hkv = h // GROUP                                  # GQA: fewer kv heads
    om = sm * BM + tl.arange(0, BM); on = tl.arange(0, BN); od = tl.arange(0, D)
    q = tl.load(Q + z * sqz + h * sqh + om[:, None] * sqm + od[None, :] * sqk, mask=om[:, None] < N, other=0.)
    q = q * (1.0 / math.sqrt(D))
    mi = tl.full([BM], -float("inf"), tl.float32); li = tl.zeros([BM], tl.float32); acc = tl.zeros([BM, D], tl.float32)
    for kn in range(0, N, BN):
        kk = kn + on
        k = tl.load(K + z * skz + hkv * skh + kk[:, None] * skn + od[None, :] * skk, mask=kk[:, None] < N, other=0.)
        qk = tl.dot(q, tl.trans(k))
        m2 = tl.maximum(mi, tl.max(qk, 1)); a = tl.exp(mi - m2); p = tl.exp(qk - m2[:, None])
        li = li * a + tl.sum(p, 1); acc = acc * a[:, None]
        v = tl.load(V + z * svz + hkv * svh + kk[:, None] * svn + od[None, :] * svk, mask=kk[:, None] < N, other=0.)
        acc += tl.dot(p, v); mi = m2
    tl.store(O + z * soz + h * soh + om[:, None] * som + od[None, :] * sok, acc / li[:, None], mask=om[:, None] < N)


def _ref_gqa(q, k, v, H, G, D):
    o = torch.zeros_like(q)
    for z in range(q.shape[0]):
        for h in range(H):
            hk = h // G
            sc = (q[z, h].float() @ k[z, hk].float().transpose(-2, -1)) / math.sqrt(D)
            o[z, h] = (torch.softmax(sc, -1) @ v[z, hk].float())
    return o


@requires_mps
@pytest.mark.parametrize("D", [32, 64, 128])
def test_dense_gqa_not_misrouted(D):
    dev = "mps"; torch.manual_seed(0)
    Z, H, Hkv, N = 1, 4, 2, 64
    G = H // Hkv
    q = torch.randn(Z, H, N, D, device=dev); k = torch.randn(Z, Hkv, N, D, device=dev)
    v = torch.randn(Z, Hkv, N, D, device=dev); o = torch.zeros(Z, H, N, D, device=dev)
    st = lambda t: t.stride()
    try:
        _dense_gqa_fa[(triton.cdiv(N, 32), Z * H)](
            q, k, v, o, *st(q), *st(k), *st(v), *st(o), Z, H, G, N, 32, 32, D)
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        return  # refused loudly — safe
    ref = _ref_gqa(q, k, v, H, G, D)
    err = (o - ref).abs().max().item()
    assert err < 1e-2, f"dense GQA hd{D} mis-computed (routed as MHA?): err {err:.2e}"


@triton.jit
def _dense_mha_fa(Q, K, V, O, sqz, sqh, sqm, sqk, skz, skh, skn, skk,
                  svz, svh, svn, svk, soz, soh, som, sok,
                  Z, H, N, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    sm = tl.program_id(0); hz = tl.program_id(1); z = hz // H; h = hz % H
    om = sm * BM + tl.arange(0, BM); on = tl.arange(0, BN); od = tl.arange(0, D)
    q = tl.load(Q + z * sqz + h * sqh + om[:, None] * sqm + od[None, :] * sqk, mask=om[:, None] < N, other=0.)
    q = q * (1.0 / math.sqrt(D))
    mi = tl.full([BM], -float("inf"), tl.float32); li = tl.zeros([BM], tl.float32); acc = tl.zeros([BM, D], tl.float32)
    for kn in range(0, N, BN):
        kk = kn + on
        k = tl.load(K + z * skz + h * skh + kk[:, None] * skn + od[None, :] * skk, mask=kk[:, None] < N, other=0.)
        qk = tl.dot(q, tl.trans(k))
        m2 = tl.maximum(mi, tl.max(qk, 1)); a = tl.exp(mi - m2); p = tl.exp(qk - m2[:, None])
        li = li * a + tl.sum(p, 1); acc = acc * a[:, None]
        v = tl.load(V + z * svz + h * svh + kk[:, None] * svn + od[None, :] * svk, mask=kk[:, None] < N, other=0.)
        acc += tl.dot(p, v); mi = m2
    tl.store(O + z * soz + h * soh + om[:, None] * som + od[None, :] * sok, acc / li[:, None], mask=om[:, None] < N)


@requires_mps
@pytest.mark.parametrize("D", [64, 128])
def test_dense_mha_still_routes_correct(D):
    # MHA (Q/K/V share the head offset) must NOT be over-refused by the GQA guard.
    dev = "mps"; torch.manual_seed(0)
    Z, H, N = 1, 4, 64
    q = torch.randn(Z, H, N, D, device=dev); k = torch.randn(Z, H, N, D, device=dev)
    v = torch.randn(Z, H, N, D, device=dev); o = torch.zeros(Z, H, N, D, device=dev)
    st = lambda t: t.stride()
    _dense_mha_fa[(triton.cdiv(N, 32), Z * H)](
        q, k, v, o, *st(q), *st(k), *st(v), *st(o), Z, H, N, 32, 32, D)
    torch.mps.synchronize()
    ref = torch.zeros_like(o)
    for h in range(H):
        sc = (q[0, h].float() @ k[0, h].float().transpose(-2, -1)) / math.sqrt(D)
        ref[0, h] = (torch.softmax(sc, -1) @ v[0, h].float())
    err = (o - ref).abs().max().item()
    assert err < 1e-2, f"MHA hd{D} wrong (guard over-refused/broke routing?): err {err:.2e}"
