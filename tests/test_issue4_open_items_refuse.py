"""Correct-or-refuse guard for the three issue-#4 items that are recognized-but-not-yet-
supported (#4, #6a, #6b). Each must fail LOUDLY (a raised error), never silently miscompute.
These pin the integrity contract on the open items so a future change can't quietly turn
one into a silent-wrong. When an item is genuinely *made to work*, flip its test from
"refuses" to a correctness check.

Verified current behavior (2026-08-24, branch fix/trifast-issue4):
  #4  MEPT 2-D accumulator, fewer threads than tile -> MetalCompilationError (loud).
  #6a scale applied to a tt.dot RESULT (not folded into Q) -> MetalNonRecoverableError.
  #6b FA that also stores lse (1-D store of a loop-carried row reduce) ->
      MetalNonRecoverableError (the loop-carried-row-reduce LAYOUT guard, which protects a
      real "collapses every row to the first" silent-wrong -- so #6b is entangled with that
      guard, NOT a trivial spelling change on our side).
"""
import math
import pytest
import torch
import triton
import triton.language as tl

from triton_msl.errors import MetalNonRecoverableError, MetalCompilationError

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


@triton.jit
def _fa_scale_on_qk(Q, K, V, O, sqz, sqh, sqm, sqk, skz, skh, skn, skk,
                    svz, svh, svn, svk, soz, soh, som, sok, Z, H, N,
                    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    sm = tl.program_id(0); hz = tl.program_id(1); z = hz // H; h = hz % H
    om = sm * BM + tl.arange(0, BM); on = tl.arange(0, BN); od = tl.arange(0, D)
    q = tl.load(Q + z*sqz + h*sqh + om[:, None]*sqm + od[None, :]*sqk, mask=om[:, None] < N, other=0.)
    mi = tl.full([BM], -float("inf"), tl.float32); li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for kn in range(0, N, BN):
        kk = kn + on
        k = tl.load(K + z*skz + h*skh + kk[:, None]*skn + od[None, :]*skk, mask=kk[:, None] < N, other=0.)
        qk = tl.dot(q, tl.trans(k))
        qk = qk * (1.0 / math.sqrt(D))          # scale on the dot RESULT (#6a)
        m2 = tl.maximum(mi, tl.max(qk, 1)); a = tl.exp(mi - m2); p = tl.exp(qk - m2[:, None])
        li = li*a + tl.sum(p, 1); acc = acc*a[:, None]
        v = tl.load(V + z*svz + h*svh + kk[:, None]*svn + od[None, :]*svk, mask=kk[:, None] < N, other=0.)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32)); mi = m2
    tl.store(O + z*soz + h*soh + om[:, None]*som + od[None, :]*sok, (acc/li[:, None]), mask=om[:, None] < N)


@triton.jit
def _fa_with_lse(Q, K, V, O, L, sqz, sqh, sqm, sqk, skz, skh, skn, skk,
                 svz, svh, svn, svk, soz, soh, som, sok, slz, slh, slm, Z, H, N,
                 BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    sm = tl.program_id(0); hz = tl.program_id(1); z = hz // H; h = hz % H
    om = sm * BM + tl.arange(0, BM); on = tl.arange(0, BN); od = tl.arange(0, D)
    q = tl.load(Q + z*sqz + h*sqh + om[:, None]*sqm + od[None, :]*sqk, mask=om[:, None] < N, other=0.)
    q = q * (1.0 / math.sqrt(D))
    mi = tl.full([BM], -float("inf"), tl.float32); li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for kn in range(0, N, BN):
        kk = kn + on
        k = tl.load(K + z*skz + h*skh + kk[:, None]*skn + od[None, :]*skk, mask=kk[:, None] < N, other=0.)
        qk = tl.dot(q, tl.trans(k))
        m2 = tl.maximum(mi, tl.max(qk, 1)); a = tl.exp(mi - m2); p = tl.exp(qk - m2[:, None])
        li = li*a + tl.sum(p, 1); acc = acc*a[:, None]
        v = tl.load(V + z*svz + h*svh + kk[:, None]*svn + od[None, :]*svk, mask=kk[:, None] < N, other=0.)
        acc += tl.dot(p.to(tl.float32), v.to(tl.float32)); mi = m2
    tl.store(O + z*soz + h*soh + om[:, None]*som + od[None, :]*sok, (acc/li[:, None]), mask=om[:, None] < N)
    tl.store(L + z*slz + h*slh + om*slm, mi + tl.log(li), mask=om < N)   # 1-D lse store (#6b)


@triton.jit
def _mept_2d_acc(inp, out, M: tl.constexpr, N: tl.constexpr, STEPS: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for _ in range(0, STEPS):
        acc += tl.load(inp + rm[:, None] * N + rn[None, :])
    tl.store(out + rm[:, None] * N + rn[None, :], acc)


@requires_mps
def test_6a_scale_on_dot_result_refuses():
    Z, H, N, D = 1, 2, 64, 64
    dev = "mps"
    q = torch.randn(Z, H, N, D, device=dev); k = torch.randn(Z, H, N, D, device=dev)
    v = torch.randn(Z, H, N, D, device=dev); o = torch.zeros(Z, H, N, D, device=dev)
    st = lambda t: t.stride()
    with pytest.raises(MetalNonRecoverableError):
        _fa_scale_on_qk[(triton.cdiv(N, 32), Z*H)](q, k, v, o, *st(q), *st(k), *st(v), *st(o), Z, H, N, 32, 32, D)


@requires_mps
def test_6b_fa_lse_store_refuses():
    Z, H, N, D = 1, 2, 64, 64
    dev = "mps"
    q = torch.randn(Z, H, N, D, device=dev); k = torch.randn(Z, H, N, D, device=dev)
    v = torch.randn(Z, H, N, D, device=dev); o = torch.zeros(Z, H, N, D, device=dev)
    lse = torch.zeros(Z, H, N, device=dev)
    st = lambda t: t.stride()
    with pytest.raises(MetalNonRecoverableError):
        _fa_with_lse[(triton.cdiv(N, 32), Z*H)](q, k, v, o, lse, *st(q), *st(k), *st(v), *st(o), *st(lse), Z, H, N, 32, 32, D)


@requires_mps
def test_4_mept_2d_acc_fewer_threads_loud():
    # Loud compile error (not silent-wrong) when the accumulator tile exceeds the threadgroup.
    dev = "mps"
    inp = torch.randn(16, 16, device=dev); out = torch.zeros(16, 16, device=dev)
    with pytest.raises((MetalCompilationError, MetalNonRecoverableError)):
        _mept_2d_acc[(1,)](inp, out, 16, 16, 3, num_warps=1)
