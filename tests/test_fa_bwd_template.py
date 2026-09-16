"""Standalone validation of the FA-2 backward templates for biased/triangle
attention vs torch autograd. Route-only ABI: baked strides, pointers + runtime
scale as buffers, 3-D grid, bias shared across i, cross-head mask. These lock the
backward MATH; the detector/routing that makes trifast's _bwd_* kernels auto-route
is layered on top.
"""

import math
import pytest
import torch

from triton_msl.codegen._msl_templates import make_flash_attention_bwd_kv_kernel

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


def _u(x):
    return f"{x}u"


@requires_mps
@pytest.mark.parametrize("Hc,Hh,I,N,DIM", [(4, 2, 3, 64, 32), (2, 2, 4, 32, 32)])
def test_bwd_kv_matches_autograd(Hc, Hh, I, N, DIM):
    dev = "mps"
    torch.manual_seed(0)
    batch = Hc // Hh
    sm = 1.0 / math.sqrt(DIM)
    BJ = BK = 32
    q = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    k = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    v = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    bias = torch.randn(Hc, N, N, device=dev, requires_grad=True)
    mask = (torch.rand(batch, I, N, device=dev) < 0.15).to(torch.uint8)
    do = torch.randn(Hc, I, N, DIM, device=dev)

    qk = torch.einsum("hijd,hikd->hijk", q, k) * sm
    mh = mask[torch.arange(Hc, device=dev) // Hh]
    raw = (qk + bias[:, None, :, :]).masked_fill(mh[:, :, None, :].bool(), float("-inf"))
    p = torch.softmax(raw, dim=-1)
    o_ref = torch.einsum("hijk,hikd->hijd", p, v)
    o_ref.backward(do)
    dk_ref, dv_ref = k.grad.detach().clone(), v.grad.detach().clone()

    qd, kd, vd, bd = q.detach(), k.detach(), v.detach(), bias.detach()
    lse = torch.logsumexp(raw.detach(), dim=-1).contiguous()
    delta = (o_ref.detach() * do).sum(-1).contiguous()
    dk = torch.zeros(Hc, I, N, DIM, device=dev)
    dv = torch.zeros(Hc, I, N, DIM, device=dev)

    S = lambda t: t.stride()
    qs, ks, vs, bs, ms, ls, ds, dos, dks, dvs = (
        S(qd),
        S(kd),
        S(vd),
        S(bd),
        S(mask),
        S(lse),
        S(delta),
        S(do),
        S(dk),
        S(dv),
    )
    arg_decls = [
        "    device float* DK [[buffer(0)]]",
        "    device float* DV [[buffer(1)]]",
        "    device const float* Q [[buffer(2)]]",
        "    device const float* K [[buffer(3)]]",
        "    device const float* V [[buffer(4)]]",
        "    device const float* Bias [[buffer(5)]]",
        "    device const uchar* Mask [[buffer(6)]]",
        "    device const float* Lse [[buffer(7)]]",
        "    device const float* Delta [[buffer(8)]]",
        "    device const float* dO [[buffer(9)]]",
        "    constant float& arg_scale [[buffer(10)]]",
    ]
    bindings = {
        "q_sz": _u(qs[0]),
        "q_sh": _u(qs[1]),
        "q_sm": _u(qs[2]),
        "q_sk": _u(qs[3]),
        "k_sz": _u(ks[0]),
        "k_sh": _u(ks[1]),
        "k_sn": _u(ks[2]),
        "k_sk": _u(ks[3]),
        "v_sz": _u(vs[0]),
        "v_sh": _u(vs[1]),
        "v_sn": _u(vs[2]),
        "v_sk": _u(vs[3]),
        "b_sz": _u(bs[0]),
        "b_sh": "0u",
        "b_sm": _u(bs[1]),
        "b_sn": _u(bs[2]),
        "mask_sz": _u(ms[0]),
        "mask_sh": _u(ms[1]),
        "mask_sn": _u(ms[2]),
        "lse_sz": _u(ls[0]),
        "lse_sh": _u(ls[1]),
        "lse_sm": _u(ls[2]),
        "dlt_sz": _u(ds[0]),
        "dlt_sh": _u(ds[1]),
        "dlt_sm": _u(ds[2]),
        "do_sz": _u(dos[0]),
        "do_sh": _u(dos[1]),
        "do_sm": _u(dos[2]),
        "do_sk": _u(dos[3]),
        "dk_sz": _u(dks[0]),
        "dk_sh": _u(dks[1]),
        "dk_sn": _u(dks[2]),
        "dk_sk": _u(dks[3]),
        "dv_sz": _u(dvs[0]),
        "dv_sh": _u(dvs[1]),
        "dv_sn": _u(dvs[2]),
        "dv_sk": _u(dvs[3]),
        "H": _u(Hh),
        "N_CTX": _u(N),
        "scale": "arg_scale",
    }
    src = make_flash_attention_bwd_kv_kernel(
        DIM, BJ, BK, out_dtype="fp32", arg_decls=arg_decls, bindings=bindings, grid_3d=True, mask_batch_div=_u(Hh)
    )
    lib = torch.mps.compile_shader(src)
    TPG = BK * DIM
    n_k = (N + BK - 1) // BK
    lib.flash_attention_bwd_kv(
        dk, dv, qd, kd, vd, bd, mask, lse, delta, do, float(sm), threads=(n_k * TPG, I, Hc), group_size=(TPG, 1, 1)
    )
    torch.mps.synchronize()
    assert (dk - dk_ref).abs().max().item() < 3e-3
    assert (dv - dv_ref).abs().max().item() < 3e-3


@requires_mps
@pytest.mark.parametrize("Hc,Hh,I,N,DIM", [(4, 2, 3, 64, 32), (2, 2, 4, 32, 32)])
def test_bwd_q_matches_autograd(Hc, Hh, I, N, DIM):
    from triton_msl.codegen._msl_templates import make_flash_attention_bwd_q_kernel

    dev = "mps"
    torch.manual_seed(0)
    batch = Hc // Hh
    sm = 1.0 / math.sqrt(DIM)
    BJ = BK = 32
    q = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    k = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    v = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    bias = torch.randn(Hc, N, N, device=dev, requires_grad=True)
    mask = (torch.rand(batch, I, N, device=dev) < 0.15).to(torch.uint8)
    do = torch.randn(Hc, I, N, DIM, device=dev)
    qk = torch.einsum("hijd,hikd->hijk", q, k) * sm
    mh = mask[torch.arange(Hc, device=dev) // Hh]
    raw = (qk + bias[:, None, :, :]).masked_fill(mh[:, :, None, :].bool(), float("-inf"))
    pp = torch.softmax(raw, dim=-1)
    o_ref = torch.einsum("hijk,hikd->hijd", pp, v)
    o_ref.backward(do)
    dq_ref = q.grad.detach().clone()

    qd, kd, vd, bd = q.detach(), k.detach(), v.detach(), bias.detach()
    o_det = o_ref.detach().contiguous()
    lse = torch.logsumexp(raw.detach(), dim=-1).contiguous()
    delta_ref = (o_det * do).sum(-1)
    dq = torch.zeros(Hc, I, N, DIM, device=dev)
    delta = torch.zeros(Hc, I, N, device=dev)
    S = lambda t: t.stride()
    qs, ks, vs, bs, ms, ls, os_, dos, dqs, dls = (
        S(qd),
        S(kd),
        S(vd),
        S(bd),
        S(mask),
        S(lse),
        S(o_det),
        S(do),
        S(dq),
        S(delta),
    )
    arg_decls = [
        "    device float* DQ [[buffer(0)]]",
        "    device float* Delta [[buffer(1)]]",
        "    device const float* Q [[buffer(2)]]",
        "    device const float* K [[buffer(3)]]",
        "    device const float* V [[buffer(4)]]",
        "    device const float* Bias [[buffer(5)]]",
        "    device const uchar* Mask [[buffer(6)]]",
        "    device const float* Lse [[buffer(7)]]",
        "    device const float* O [[buffer(8)]]",
        "    device const float* dO [[buffer(9)]]",
        "    constant float& arg_scale [[buffer(10)]]",
    ]
    bindings = {
        "q_sz": _u(qs[0]),
        "q_sh": _u(qs[1]),
        "q_sm": _u(qs[2]),
        "q_sk": _u(qs[3]),
        "k_sz": _u(ks[0]),
        "k_sh": _u(ks[1]),
        "k_sn": _u(ks[2]),
        "k_sk": _u(ks[3]),
        "v_sz": _u(vs[0]),
        "v_sh": _u(vs[1]),
        "v_sn": _u(vs[2]),
        "v_sk": _u(vs[3]),
        "b_sz": _u(bs[0]),
        "b_sh": "0u",
        "b_sm": _u(bs[1]),
        "b_sn": _u(bs[2]),
        "mask_sz": _u(ms[0]),
        "mask_sh": _u(ms[1]),
        "mask_sn": _u(ms[2]),
        "lse_sz": _u(ls[0]),
        "lse_sh": _u(ls[1]),
        "lse_sm": _u(ls[2]),
        "o_sz": _u(os_[0]),
        "o_sh": _u(os_[1]),
        "o_sm": _u(os_[2]),
        "o_sk": _u(os_[3]),
        "do_sz": _u(dos[0]),
        "do_sh": _u(dos[1]),
        "do_sm": _u(dos[2]),
        "do_sk": _u(dos[3]),
        "dq_sz": _u(dqs[0]),
        "dq_sh": _u(dqs[1]),
        "dq_sm": _u(dqs[2]),
        "dq_sk": _u(dqs[3]),
        "dlt_sz": _u(dls[0]),
        "dlt_sh": _u(dls[1]),
        "dlt_sm": _u(dls[2]),
        "H": _u(Hh),
        "N_CTX": _u(N),
        "scale": "arg_scale",
    }
    src = make_flash_attention_bwd_q_kernel(
        DIM, BJ, BK, out_dtype="fp32", arg_decls=arg_decls, bindings=bindings, grid_3d=True, mask_batch_div=_u(Hh)
    )
    lib = torch.mps.compile_shader(src)
    TPG = BJ * DIM
    n_j = (N + BJ - 1) // BJ
    lib.flash_attention_bwd_q(
        dq, delta, qd, kd, vd, bd, mask, lse, o_det, do, float(sm), threads=(n_j * TPG, I, Hc), group_size=(TPG, 1, 1)
    )
    torch.mps.synchronize()
    assert (dq - dq_ref).abs().max().item() < 3e-3
    assert (delta - delta_ref).abs().max().item() < 3e-3


@requires_mps
@pytest.mark.parametrize("Hc,Hh,N,DIM", [(4, 2, 32, 32), (2, 1, 32, 32)])
def test_bwd_b_matches_autograd(Hc, Hh, N, DIM):
    """dbias sums dS over the triangle-i axis (I == N, per trifast)."""
    from triton_msl.codegen._msl_templates import make_flash_attention_bwd_b_kernel

    dev = "mps"
    torch.manual_seed(0)
    I = N  # _bwd_b loops i over N
    batch = Hc // Hh
    sm = 1.0 / math.sqrt(DIM)
    BJ = BK = 32
    q = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    k = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    v = torch.randn(Hc, I, N, DIM, device=dev, requires_grad=True)
    bias = torch.randn(Hc, N, N, device=dev, requires_grad=True)
    mask = (torch.rand(batch, I, N, device=dev) < 0.15).to(torch.uint8)
    do = torch.randn(Hc, I, N, DIM, device=dev)
    qk = torch.einsum("hijd,hikd->hijk", q, k) * sm
    mh = mask[torch.arange(Hc, device=dev) // Hh]
    raw = (qk + bias[:, None, :, :]).masked_fill(mh[:, :, None, :].bool(), float("-inf"))
    pp = torch.softmax(raw, dim=-1)
    o_ref = torch.einsum("hijk,hikd->hijd", pp, v)
    o_ref.backward(do)
    db_ref = bias.grad.detach().clone()

    qd, kd, vd, bd = q.detach(), k.detach(), v.detach(), bias.detach()
    o_det = o_ref.detach()
    lse = torch.logsumexp(raw.detach(), dim=-1).contiguous()
    delta = (o_det * do).sum(-1).contiguous()
    db = torch.zeros(Hc, N, N, device=dev)
    S = lambda t: t.stride()
    qs, ks, vs, bs, ms, ls, ds, dos, dbs = (S(qd), S(kd), S(vd), S(bd), S(mask), S(lse), S(delta), S(do), S(db))
    arg_decls = [
        "    device float* DB [[buffer(0)]]",
        "    device const float* Q [[buffer(1)]]",
        "    device const float* K [[buffer(2)]]",
        "    device const float* V [[buffer(3)]]",
        "    device const float* Bias [[buffer(4)]]",
        "    device const uchar* Mask [[buffer(5)]]",
        "    device const float* Lse [[buffer(6)]]",
        "    device const float* Delta [[buffer(7)]]",
        "    device const float* dO [[buffer(8)]]",
        "    constant float& arg_scale [[buffer(9)]]",
    ]
    bindings = {
        "q_sh": _u(qs[0]),
        "q_si": _u(qs[1]),
        "q_sm": _u(qs[2]),
        "q_sk": _u(qs[3]),
        "k_sh": _u(ks[0]),
        "k_si": _u(ks[1]),
        "k_sn": _u(ks[2]),
        "k_sk": _u(ks[3]),
        "v_sh": _u(vs[0]),
        "v_si": _u(vs[1]),
        "v_sn": _u(vs[2]),
        "v_sk": _u(vs[3]),
        "b_sh": _u(bs[0]),
        "b_sm": _u(bs[1]),
        "b_sn": _u(bs[2]),
        "mask_sz": _u(ms[0]),
        "mask_si": _u(ms[1]),
        "mask_sn": _u(ms[2]),
        "lse_sh": _u(ls[0]),
        "lse_si": _u(ls[1]),
        "lse_sm": _u(ls[2]),
        "dlt_sh": _u(ds[0]),
        "dlt_si": _u(ds[1]),
        "dlt_sm": _u(ds[2]),
        "do_sh": _u(dos[0]),
        "do_si": _u(dos[1]),
        "do_sm": _u(dos[2]),
        "do_sk": _u(dos[3]),
        "db_sh": _u(dbs[0]),
        "db_sm": _u(dbs[1]),
        "db_sn": _u(dbs[2]),
        "H": _u(Hh),
        "N_CTX": _u(N),
        "scale": "arg_scale",
    }
    src = make_flash_attention_bwd_b_kernel(
        DIM, BJ, BK, out_dtype="fp32", arg_decls=arg_decls, bindings=bindings, mask_batch_div=_u(Hh)
    )
    lib = torch.mps.compile_shader(src)
    TPG = BJ * BK
    n_j = (N + BJ - 1) // BJ
    n_k = (N + BK - 1) // BK
    lib.flash_attention_bwd_b(
        db, qd, kd, vd, bd, mask, lse, delta, do, float(sm), threads=(n_j * TPG, n_k, Hc), group_size=(TPG, 1, 1)
    )
    torch.mps.synchronize()
    assert (db - db_ref).abs().max().item() < 5e-3
