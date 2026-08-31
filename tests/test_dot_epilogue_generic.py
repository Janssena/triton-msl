"""Dot-recovery stage 1a (2026-08-30): single-tile dot + epilogue kernels.

Two changes pinned here:

1. GENERIC-ROUTE CAPABILITY: the five upstream test_dot epilogue tiers
   (add-matrix / add-rows / add-cols / softmax / chain-dot) used to REFUSE at five
   distinct matmul-guard sites. Inside the generic dot path's PROVEN envelope
   (non-looped, every dot M == N == K in {16, 32}, uniform tensor shapes) the guards
   now fall through to the generic lowerer, which computes all tiers correctly
   (probe matrix 2026-08-30: 36/36 rows, fp32 + fp16-in fp16/fp32-out, including the
   upstream form where the bias tile ALIASES the output buffer). Outside the
   envelope every guard still refuses: the shapes probe showed the generic mapping
   silently mis-tiles NON-UNIFORM shapes (M32xN64 err 0.2..4.0).

2. DATAFLOW POINTER ROLES (silent-wrong fix, pre-existing at 9c2c416): the fused
   matmul-softmax/epilogue templates resolved X/Y/Z POSITIONALLY
   (a=ptr[0], b=ptr[1], c=ptr[-1]). A claimed kernel whose arg order differed —
   e.g. an extra pointer arg AFTER the output — computed a numerically perfect
   result and stored it into the WRONG BUFFER, leaving the real output untouched
   (GPU: M32N64 softmax landed in the unused 4th arg at 5.6e-09). Roles now come
   from _resolve_dot_ptr_roles (dot operands + the store target); unresolvable
   roles decline to the loud #157 catch-all. The softmax detector also gained the
   epilogue detector's single-store guard.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal

    from triton_msl.errors import MetalNonRecoverableError

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + triton needed")

if HAS:

    @triton.jit
    def _dot_ep(X, Y, Z, W, S: tl.constexpr, EP: tl.constexpr):
        # Upstream test_dot shape: single tile, bias vectors/tile loaded FROM Z
        # (the output buffer) exactly like unit/language/test_core.py::test_dot.
        om = tl.arange(0, S)
        on = tl.arange(0, S)
        ok = tl.arange(0, S)
        x = tl.load(X + om[:, None] * S + ok[None, :])
        y = tl.load(Y + ok[:, None] * S + on[None, :])
        z = tl.dot(x, y)
        if EP == 1:  # add-matrix (aliases Z)
            z += tl.load(Z + om[:, None] * S + on[None, :])
        if EP == 2:  # add-rows (aliases Z column 0)
            z += tl.load(Z + om * S)[:, None]
        if EP == 3:  # add-cols (aliases Z row 0)
            z += tl.load(Z + on)[None, :]
        if EP == 4:  # softmax
            zmax = tl.max(z, 1)
            z = z - zmax[:, None]
            num = tl.exp(z)
            den = tl.sum(num, 1)
            z = num / den[:, None]
        if EP == 5:  # chain-dot
            w = tl.load(W + on[:, None] * S + on[None, :])
            z = tl.dot(z.to(w.dtype), w)
        tl.store(Z + om[:, None] * S + on[None, :], z)

    def _ref(x, y, w, z0, ep):
        z = x.float() @ y.float()
        if ep == 1:
            z = z + z0.float()
        elif ep == 2:
            z = z + z0.float()[:, 0][:, None]
        elif ep == 3:
            z = z + z0.float()[0, :][None, :]
        elif ep == 4:
            z = torch.softmax(z, dim=-1)
        elif ep == 5:
            z = z.to(w.dtype).float() @ w.float()
        return z


_EPS = {1: "add-matrix", 2: "add-rows", 3: "add-cols", 4: "softmax", 5: "chain-dot"}


@requires
@pytest.mark.parametrize("size", [16, 32])
@pytest.mark.parametrize("ep", sorted(_EPS))
@pytest.mark.parametrize("in_dt,out_dt", [("float32", "float32"), ("float16", "float16"), ("float16", "float32")])
def test_dot_epilogue_computes(size, ep, in_dt, out_dt):
    idt = getattr(torch, in_dt)
    odt = getattr(torch, out_dt)
    torch.manual_seed(17)
    x = (torch.randn(size, size, device="mps") * 0.1).to(idt)
    y = (torch.randn(size, size, device="mps") * 0.1).to(idt)
    w = (torch.randn(size, size, device="mps") * 0.1).to(idt)
    z = torch.randn(size, size, device="mps").to(odt)
    z0 = z.clone()
    _dot_ep[(1,)](x, y, z, w, S=size, EP=ep, num_warps=4)
    torch.mps.synchronize()
    r = _ref(x, y, w, z0, ep)
    tol = 1e-4 if idt == torch.float32 else 2e-2
    err = (z.float() - r).abs().max().item()
    assert err == err and err < tol, f"{_EPS[ep]} S={size} {in_dt}->{out_dt}: err {err}"


if HAS:

    @triton.jit
    def _dot_softmax_extra_arg(X, Y, Z, EXTRA, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        # Template-claimed shape (M != N so the generic envelope does NOT apply)
        # with an extra pointer arg AFTER the output — the positional c=ptr[-1]
        # pick stored the whole result into EXTRA and left Z untouched.
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        y = tl.load(Y + ok[:, None] * N + on[None, :])
        z = tl.dot(x, y)
        zmax = tl.max(z, 1)
        z = z - zmax[:, None]
        num = tl.exp(z)
        den = tl.sum(num, 1)
        z = num / den[:, None]
        tl.store(Z + om[:, None] * N + on[None, :], z)


@requires
def test_template_stores_to_dataflow_output_not_last_ptr_arg():
    # SILENT-WRONG pin (pre-existing at 9c2c416): fused-template pointer roles were
    # positional; with EXTRA trailing the arg list, the numerically perfect softmax
    # landed in EXTRA (err 5.6e-09) and Z stayed untouched. Roles must come from
    # the store's dataflow.
    torch.manual_seed(17)
    M, N, K = 32, 64, 32
    x = torch.randn(M, K, device="mps") * 0.1
    y = torch.randn(K, N, device="mps") * 0.1
    z = torch.full((M, N), float("nan"), device="mps")
    extra = torch.full((M * N,), 7.0, device="mps")
    _dot_softmax_extra_arg[(1,)](x, y, z, extra, M=M, N=N, K=K, num_warps=4)
    torch.mps.synchronize()
    ref = torch.softmax(x @ y, dim=-1)
    err = (z - ref).abs().max().item()
    assert err == err and err < 1e-4, f"output not in Z (err {err}) — wrote to the wrong buffer?"
    assert bool((extra == 7.0).all()), "EXTRA was written — positional c=ptr[-1] regressed"


if HAS:

    @triton.jit
    def _dot_addmatrix_nonuniform(X, Y, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        om = tl.arange(0, M)
        on = tl.arange(0, N)
        ok = tl.arange(0, K)
        x = tl.load(X + om[:, None] * K + ok[None, :])
        y = tl.load(Y + ok[:, None] * N + on[None, :])
        z = tl.dot(x, y)
        z += tl.load(Z + om[:, None] * N + on[None, :])
        tl.store(Z + om[:, None] * N + on[None, :], z)


@requires
def test_nonuniform_fused_accumulator_still_refuses():
    # FAIL-CLOSED pin: outside the proven envelope (M != N) the fused-accumulator
    # guard must STILL refuse — the shapes probe showed the generic mapping
    # silently mis-tiles non-uniform shapes (M32xN64 add-matrix err 3.96).
    torch.manual_seed(17)
    M, N, K = 32, 64, 32
    x = torch.randn(M, K, device="mps") * 0.1
    y = torch.randn(K, N, device="mps") * 0.1
    z = torch.randn(M, N, device="mps")
    with pytest.raises(MetalNonRecoverableError, match="accumulator"):
        _dot_addmatrix_nonuniform[(1,)](x, y, z, M=M, N=N, K=K, num_warps=4)
