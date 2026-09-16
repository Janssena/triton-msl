"""Packet 170 (158 row 4) — the tensor-N atomic-RETURN sweep: regression controls.

Census of every atomic-with-return spelling the emitter takes (`_lower_atomic_rmw`, `_lower_atomic_cas`
and their tensor / masked / looped consumers), each run against a torch reference of the OLD values
and of the memory after the atomic. The census found no defect (13 / 13 on the 168 tree); these rows
keep it that way. Facts the sweep established, pinned here so a later change cannot drift them:

- a tensor<N> atomic with N > the warp count's thread budget is NOT under-covered on Metal: the
  dispatcher launches BLOCK threads (up to 1024), so N = 256 with num_warps = 1 and N = 1024 with
  num_warps = 4 each return every lane's old value;
- masked-off lanes return the emitter's zero initializer. Upstream leaves that value unspecified,
  so a program that consumes it (the full-width reduce row) gets a backend-specific number on every
  backend; the row pins ours so that a change is noticed, not because 0 is a contract;
- a float `atomic_max` (the sign-split int / uint pair the frontend emits) returns the old float
  bits correctly for negative and positive olds;
- every lane hitting ONE address returns N distinct old values (the atomics are per-lane, not
  wavefront-merged);
- returns accumulate correctly across a loop.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl

    import triton_msl

    HAS_GPU = torch.backends.mps.is_available()
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU needed")
D = "mps"


@pytest.fixture
def cold_gpu_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_MSL_CACHE_DIR", str(tmp_path / "msl"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")


@triton.jit
def _k_rmw(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.atomic_add(x_ptr + offs, 1))


@triton.jit
def _k_rmw_masked(x_ptr, out_ptr, n, N: tl.constexpr):
    offs = tl.arange(0, N)
    m = offs < n
    old = tl.atomic_add(x_ptr + offs, 1, mask=m)
    tl.store(out_ptr + offs, old, mask=m)


@triton.jit
def _k_rmw_masked_reduced(x_ptr, out_ptr, n, N: tl.constexpr):
    offs = tl.arange(0, N)
    m = offs < n
    old = tl.atomic_add(x_ptr + offs, 1, mask=m)
    tl.store(out_ptr, tl.sum(old, 0))


@triton.jit
def _k_rmw_2d(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    tl.store(out_ptr + r * C + c, tl.atomic_add(x_ptr + r * C + c, 1))


@triton.jit
def _k_xchg(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.atomic_xchg(x_ptr + offs, offs * 10))


@triton.jit
def _k_fmax(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.atomic_max(x_ptr + offs, 0.5))


@triton.jit
def _k_rmw_loop(x_ptr, out_ptr, N: tl.constexpr, ITERS: tl.constexpr):
    offs = tl.arange(0, N)
    acc = tl.zeros((N,), tl.int32)
    for _ in range(ITERS):
        acc += tl.atomic_add(x_ptr + offs, 1)
    tl.store(out_ptr + offs, acc)


@triton.jit
def _k_rmw_same_addr(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.atomic_add(x_ptr + offs * 0, 1))


def _x0(n):
    return (torch.arange(n, device=D) % 7).to(torch.int32)


def _run(fn, x, out, *args, **launch):
    if hasattr(fn, "device_caches"):
        fn.device_caches.clear()
    fn[(1,)](x, out, *args, **launch)
    torch.mps.synchronize()


@requires_gpu
@pytest.mark.parametrize(
    "n,num_warps",
    [(32, 1), (8, 1), (256, 1), (256, 4), (1024, 4)],
    ids=["full-32", "under-8", "over-256-nw1", "over-256-nw4", "over-1024-nw4"],
)
def test_tensor_rmw_returns_every_lanes_old_value(cold_gpu_caches, n, num_warps):
    x0 = _x0(n)
    x = x0.clone()
    out = torch.full((n,), -99, device=D, dtype=torch.int32)
    _run(_k_rmw, x, out, N=n, num_warps=num_warps)
    assert torch.equal(out.cpu(), x0.cpu()) and torch.equal(x.cpu(), x0.cpu() + 1)


@requires_gpu
def test_masked_rmw_returns_old_on_active_lanes_only(cold_gpu_caches):
    n, k = 32, 20
    x0 = _x0(n)
    x = x0.clone()
    out = torch.full((n,), -99, device=D, dtype=torch.int32)
    _run(_k_rmw_masked, x, out, k, N=n, num_warps=1)
    m = torch.arange(n) < k
    assert torch.equal(out.cpu()[m], x0.cpu()[m]) and bool((out.cpu()[~m] == -99).all())
    assert torch.equal(x.cpu(), x0.cpu() + m.to(torch.int32))


@requires_gpu
def test_masked_rmw_return_on_inactive_lanes_is_the_zero_initializer(cold_gpu_caches):
    """Upstream leaves masked-off returns unspecified; ours are 0. Pinned so a change is noticed."""
    n, k = 32, 20
    x0 = _x0(n)
    x = x0.clone()
    out = torch.full((1,), -99, device=D, dtype=torch.int32)
    _run(_k_rmw_masked_reduced, x, out, k, N=n, num_warps=1)
    m = torch.arange(n) < k
    assert out.item() == x0.cpu()[m].sum().item()


@requires_gpu
@pytest.mark.parametrize("R,C,num_warps", [(8, 8, 2), (16, 16, 1)])
def test_2d_rmw_returns_old_values(cold_gpu_caches, R, C, num_warps):
    n = R * C
    x0 = _x0(n)
    x = x0.clone()
    out = torch.full((n,), -99, device=D, dtype=torch.int32)
    _run(_k_rmw_2d, x, out, R=R, C=C, num_warps=num_warps)
    assert torch.equal(out.cpu(), x0.cpu()) and torch.equal(x.cpu(), x0.cpu() + 1)


@requires_gpu
def test_xchg_returns_old_values(cold_gpu_caches):
    n = 32
    x0 = _x0(n)
    x = x0.clone()
    out = torch.full((n,), -99, device=D, dtype=torch.int32)
    _run(_k_xchg, x, out, N=n, num_warps=1)
    assert torch.equal(out.cpu(), x0.cpu()) and torch.equal(x.cpu(), (torch.arange(n) * 10).to(torch.int32))


@requires_gpu
def test_float_max_returns_old_values_across_the_sign_split(cold_gpu_caches):
    n = 32
    x0 = (_x0(n).float() - 3) * 0.25
    x = x0.clone()
    out = torch.full((n,), -99.0, device=D)
    _run(_k_fmax, x, out, N=n, num_warps=1)
    assert torch.equal(out.cpu(), x0.cpu()) and torch.equal(x.cpu(), torch.maximum(x0.cpu(), torch.tensor(0.5)))


@requires_gpu
def test_rmw_returns_accumulate_across_a_loop(cold_gpu_caches):
    n = 32
    x0 = _x0(n)
    x = x0.clone()
    out = torch.full((n,), -99, device=D, dtype=torch.int32)
    _run(_k_rmw_loop, x, out, N=n, ITERS=4, num_warps=1)
    assert torch.equal(out.cpu(), x0.cpu() * 4 + 6) and torch.equal(x.cpu(), x0.cpu() + 4)


@requires_gpu
def test_one_address_returns_n_distinct_old_values(cold_gpu_caches):
    n = 32
    x = torch.zeros(n, device=D, dtype=torch.int32)
    out = torch.full((n,), -99, device=D, dtype=torch.int32)
    _run(_k_rmw_same_addr, x, out, N=n, num_warps=1)
    assert sorted(out.cpu().tolist()) == list(range(n)) and x[0].item() == n
