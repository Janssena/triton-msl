"""Regression: dense FA must not template-route mismatched K/V batch offsets."""

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
def _dense_fa_k_batch_shift(
    Q,
    K,
    V,
    O,
    sqz,
    sqh,
    sqm,
    sqd,
    skz,
    skh,
    skn,
    skd,
    svz,
    svh,
    svn,
    svd,
    soz,
    soh,
    som,
    sod,
    Z,
    H,
    N,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    start_m = tl.program_id(0)
    zh = tl.program_id(1)
    z = zh // H
    h = zh % H
    z_k = (z + 1) % Z
    om = start_m * BM + tl.arange(0, BM)
    on = tl.arange(0, BN)
    od = tl.arange(0, D)

    q = tl.load(
        Q + z * sqz + h * sqh + om[:, None] * sqm + od[None, :] * sqd,
        mask=om[:, None] < N,
        other=0.0,
    )
    q = q * SCALE
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for start_n in range(0, N, BN):
        kn = start_n + on
        k = tl.load(
            K + z_k * skz + h * skh + kn[:, None] * skn + od[None, :] * skd,
            mask=kn[:, None] < N,
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k).to(q.dtype))
        qk = tl.where(kn[None, :] < N, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            V + z * svz + h * svh + kn[:, None] * svn + od[None, :] * svd,
            mask=kn[:, None] < N,
            other=0.0,
        )
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new

    tl.store(
        O + z * soz + h * soh + om[:, None] * som + od[None, :] * sod,
        acc / l_i[:, None],
        mask=om[:, None] < N,
    )


@requires_mps
def test_dense_fa_shifted_k_batch_correct_or_refuse(monkeypatch):
    import triton_msl.autotuning._fa_dispatch as fa_dispatch

    route = {"calls": 0, "tag": None}
    real = fa_dispatch.dispatch_flash_attention

    def spy(*args, **kwargs):
        route["calls"] += 1
        if len(args) > 1 and isinstance(args[1], (tuple, list)) and args[1]:
            route["tag"] = args[1][0]
        return real(*args, **kwargs)

    monkeypatch.setattr(fa_dispatch, "dispatch_flash_attention", spy)

    device = "mps"
    torch.manual_seed(8301)
    Z, H, N, D = 2, 2, 64, 128
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(Z, H, N, D, device=device)
    k = torch.randn(Z, H, N, D, device=device)
    v = torch.randn(Z, H, N, D, device=device)
    out = torch.full_like(q, float("nan"))
    strides = lambda tensor: tuple(tensor.stride())

    try:
        _dense_fa_k_batch_shift[(triton.cdiv(N, 32), Z * H)](
            q,
            k,
            v,
            out,
            *strides(q),
            *strides(k),
            *strides(v),
            *strides(out),
            Z,
            H,
            N,
            SCALE=scale,
            BM=32,
            BN=32,
            D=D,
        )
        torch.mps.synchronize()
    except MetalNonRecoverableError:
        print(f"DENSE_FA_BATCH_PROBE refused route={route}")
        return

    ref = torch.empty_like(out)
    for z in range(Z):
        for h in range(H):
            scores = (q[z, h].float() * scale) @ k[(z + 1) % Z, h].float().T
            ref[z, h] = torch.softmax(scores, -1) @ v[z, h].float()
    err = (out - ref).abs().max().item()
    print(f"DENSE_FA_BATCH_PROBE route={route} err={err}")
    assert err < 1e-3, (
        "dense FA violated correct-or-refuse for a shifted K batch offset: "
        f"route={route}, max_err={err}"
    )
