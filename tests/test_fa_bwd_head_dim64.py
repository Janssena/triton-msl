"""Biased/triangle-attention BACKWARD at head_dim>32 (the #292 BK-subtile follow-up).

The scalar backward templates staged BLOCK*head_dim threadgroup buffers, so head_dim=64
overflowed Metal's 1024-thread / 32 KB threadgroup limits (a LOUD OutOfResources — never
silent-wrong). They now sub-tile the accumulation dimension so a large head_dim fits:
  * bwd_kv K-subtiles (KS) -> dK/dV work at head_dim 64 AND 128,
  * bwd_q  J-subtiles (JS) -> dQ works at head_dim 64; head_dim 128 REFUSES (the
    J-independent K/V staging exceeds the budget at any subtile),
  * bwd_b  de-stages Q -> dbias works at head_dim 64; head_dim 128 REFUSES.
head_dim<=32 stays a SINGLE pass (KS==BK / JS==BJ / Q staged) -> byte-identical to the
validated trifast path. These reuse the exact _bwd_{kv,q,b} kernels from the routing test.
"""
import math
import pytest
import torch
import triton

from triton_msl.errors import MetalNonRecoverableError
from .test_fa_bwd_routing import _bwd_kv, _bwd_q, _bwd_b, _reference

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@requires_mps
@pytest.mark.parametrize("DIM", [64, 128])
@pytest.mark.parametrize("Hc,Hh,I,N", [(4, 2, 3, 64), (1, 1, 1, 48)])
def test_bwd_kv_head_dim_gt32(Hc, Hh, I, N, DIM):
    dev = "mps"; BJ = BK = 32; NEG = -1e9
    r = _reference(Hc, Hh, I, N, DIM)
    st = lambda t: tuple(t.stride())
    dk = torch.zeros(Hc, I, N, DIM, device=dev)
    dv = torch.zeros(Hc, I, N, DIM, device=dev)
    q, k, v, bias, mask, do, lse, delta, sm = (
        r["q"], r["k"], r["v"], r["bias"], r["mask"], r["do"], r["lse"], r["delta"], r["sm"])
    _bwd_kv[(triton.cdiv(N, BK), I, Hc)](
        delta, *st(delta), q, *st(q), k, *st(k), v, *st(v), bias, *st(bias),
        lse, *st(lse), mask, *st(mask), do, *st(do), dk, *st(dk), dv, *st(dv),
        sm, NEG, N, Hh, DIM, N, BJ, BK)
    torch.mps.synchronize()
    assert (dk - r["dk_ref"]).abs().max().item() < 3e-3
    assert (dv - r["dv_ref"]).abs().max().item() < 3e-3


def _bwd_q_launch(Hc, Hh, I, N, DIM):
    dev = "mps"; BJ = BK = 32; NEG = -1e9
    torch.manual_seed(0)
    sm = 1.0 / math.sqrt(DIM)
    q = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    k = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    v = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    bias = torch.randn(Hc, N, N, device=dev, requires_grad=True)
    mask = (torch.rand(Hc // Hh, I, N, device=dev) < 0.15).to(torch.uint8)
    do = torch.randn(Hc, I, N, DIM, device=dev)
    qk = torch.einsum("hijd,hikd->hijk", q, k) * sm
    mh = mask[torch.arange(Hc, device=dev) // Hh]
    raw = (qk + bias[:, None, :, :]).masked_fill(mh[:, :, None, :].bool(), float("-inf"))
    p = torch.softmax(raw, dim=-1)
    o = torch.einsum("hijk,hikd->hijd", p, v)
    o.retain_grad(); o.backward(do)
    dq_ref = q.grad.detach()
    lse = torch.logsumexp(raw, dim=-1).detach()
    o_det = o.detach().contiguous()
    delta_ref = (o_det * do).sum(-1)
    st = lambda t: tuple(t.stride())
    qd, kd, vd, bd = q.detach(), k.detach(), v.detach(), bias.detach()
    dq = torch.zeros(Hc, I, N, DIM, device=dev)
    delta = torch.zeros(Hc, I, N, device=dev)
    _bwd_q[(triton.cdiv(N, BJ), I, Hc)](
        delta, *st(delta), qd, *st(qd), kd, *st(kd), vd, *st(vd), bd, *st(bd),
        lse, *st(lse), mask, *st(mask), o_det, *st(o_det), do, *st(do), dq, *st(dq),
        sm, NEG, N, Hh, DIM, N, BJ, BK)
    torch.mps.synchronize()
    return dq, dq_ref, delta, delta_ref


@requires_mps
@pytest.mark.parametrize("Hc,Hh,I,N", [(4, 2, 3, 64), (1, 1, 1, 48)])
def test_bwd_q_head_dim64(Hc, Hh, I, N):
    dq, dq_ref, delta, delta_ref = _bwd_q_launch(Hc, Hh, I, N, 64)
    assert (dq - dq_ref).abs().max().item() < 3e-3
    assert (delta - delta_ref).abs().max().item() < 3e-3


@requires_mps
def test_bwd_q_head_dim128_refuses():
    # dQ can't stage 2*BLOCK_K*128 K/V within 32 KB at any j-subtile -> LOUD refuse.
    with pytest.raises(MetalNonRecoverableError):
        _bwd_q_launch(1, 1, 1, 48, 128)


def _bwd_b_launch(Hc, Hh, N, DIM):
    dev = "mps"; I = N; BJ = BK = 32; NEG = -1e9
    torch.manual_seed(0)
    sm = 1.0 / math.sqrt(DIM)
    q = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    k = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    v = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    bias = torch.randn(Hc, N, N, device=dev, requires_grad=True)
    mask = (torch.rand(Hc // Hh, I, N, device=dev) < 0.15).to(torch.uint8)
    do = torch.randn(Hc, I, N, DIM, device=dev)
    qk = torch.einsum("hijd,hikd->hijk", q, k) * sm
    mh = mask[torch.arange(Hc, device=dev) // Hh]
    raw = (qk + bias[:, None, :, :]).masked_fill(mh[:, :, None, :].bool(), float("-inf"))
    p = torch.softmax(raw, dim=-1)
    o = torch.einsum("hijk,hikd->hijd", p, v)
    o.retain_grad(); o.backward(do)
    db_ref = bias.grad.detach()
    lse = torch.logsumexp(raw, dim=-1).detach()
    delta = (o.detach() * do).sum(-1).contiguous()
    st = lambda t: tuple(t.stride())
    qd, kd, vd, bd = q.detach(), k.detach(), v.detach(), bias.detach()
    db = torch.zeros(Hc, N, N, device=dev)
    _bwd_b[(triton.cdiv(N, BJ), triton.cdiv(N, BK), Hc)](
        delta, *st(delta), qd, *st(qd), kd, *st(kd), vd, *st(vd), bd, *st(bd),
        lse, *st(lse), mask, *st(mask), do, *st(do), db, *st(db),
        sm, NEG, Hh, N, DIM, N, BJ, BK)
    torch.mps.synchronize()
    return db, db_ref


@requires_mps
@pytest.mark.parametrize("Hc,Hh,N", [(4, 2, 32), (1, 1, 48)])
def test_bwd_b_head_dim64(Hc, Hh, N):
    db, db_ref = _bwd_b_launch(Hc, Hh, N, 64)
    assert (db - db_ref).abs().max().item() < 5e-3


@requires_mps
def test_bwd_b_head_dim128_refuses():
    with pytest.raises(MetalNonRecoverableError):
        _bwd_b_launch(1, 1, 32, 128)
