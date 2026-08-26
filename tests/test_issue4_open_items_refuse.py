"""Correct-or-refuse guard for the issue-#4 open items (#4, #6a, #6b). Each must either
compute correctly or fail LOUDLY (a raised error), never silently miscompute. These pin the
integrity contract so a future change can't quietly turn one into a silent-wrong. When an
item is genuinely *made to work*, its test flips from "refuses" to a correctness check.

Current behavior (2026-08-25, branch fix/trifast-issue4):
  #4  MEPT 2-D accumulator, fewer threads than tile -> FIXED: lowers via the per-element
      scalar carry inside the wrap loop (test below asserts correctness at all num_warps).
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


@triton.jit
def _mept_two_accs(inp, out1, out2, M: tl.constexpr, N: tl.constexpr, STEPS: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N)
    a = tl.zeros((M, N), dtype=tl.float32)
    b = tl.full((M, N), 1.0, dtype=tl.float32)
    for _ in range(0, STEPS):
        v = tl.load(inp + rm[:, None] * N + rn[None, :])
        a += v
        b = b * 0.5 + v * 2.0
    tl.store(out1 + rm[:, None] * N + rn[None, :], a)
    tl.store(out2 + rm[:, None] * N + rn[None, :], b)


@triton.jit
def _mept_nested_acc(inp, out, M: tl.constexpr, N: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for _i in range(0, S1):
        for _j in range(0, S2):
            acc += tl.load(inp + rm[:, None] * N + rn[None, :])
    tl.store(out + rm[:, None] * N + rn[None, :], acc)


@requires_mps
def test_4_two_carried_accumulators():
    # Two independent 2-D carried accumulators in one loop, both diverted to per-element
    # scalars under the wrap regime.
    dev = "mps"; M, N, S = 16, 16, 3
    torch.manual_seed(0)
    x = torch.randn(M, N, device=dev)
    o1 = torch.zeros(M, N, device=dev); o2 = torch.zeros(M, N, device=dev)
    _mept_two_accs[(1,)](x, o1, o2, M, N, S, num_warps=1)
    torch.mps.synchronize()
    rb = torch.full_like(x, 1.0)
    for _ in range(S):
        rb = rb * 0.5 + x * 2.0
    assert (o1 - x * S).abs().max().item() < 1e-4
    assert (o2 - rb).abs().max().item() < 1e-4


@requires_mps
def test_4_nested_loop_carried_accumulator():
    # The accumulator carried through NESTED loops: the taint walk propagates through the
    # inner scf.for's init->block-arg/result instead of refusing.
    dev = "mps"; M, N = 16, 16
    torch.manual_seed(0)
    x = torch.randn(M, N, device=dev); o = torch.zeros(M, N, device=dev)
    _mept_nested_acc[(1,)](x, o, M, N, 2, 3, num_warps=1)
    torch.mps.synchronize()
    assert (o - x * 6).abs().max().item() < 1e-4


@requires_mps
@pytest.mark.parametrize("nw", [1, 2, 4, 8])
def test_4_mept_2d_acc_fewer_threads_computes(nw):
    # #4 FIXED (was: loud compile error): a 2-D accumulator carried across a loop with
    # fewer threads than tile elements now LOWERS via the per-element scalar carry inside
    # the `_loop_e` wrap loop (the smem representation is diverted in the wrap regime,
    # where its cooperative init/accumulate would nest per-element). Correct at every
    # num_warps, including the previously-broken 1/2/4.
    dev = "mps"
    torch.manual_seed(0)
    inp = torch.randn(16, 16, device=dev); out = torch.zeros(16, 16, device=dev)
    _mept_2d_acc[(1,)](inp, out, 16, 16, 3, num_warps=nw)
    torch.mps.synchronize()
    err = (out - inp * 3).abs().max().item()
    assert err < 1e-4, f"2-D loop-carried accumulator wrong at num_warps={nw}: err {err:.2e}"
