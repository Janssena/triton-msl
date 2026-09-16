"""Exceptional-value coverage for variance / Welford reductions with NaN / +-Inf INPUT.
Classifications are SOURCE-DERIVED (probed 2026-09-10, evidence
2026-09-10-exceptional-checker/wf_classify.json) and rejected against wrong controls:

  * layernorm: sum -> centering -> variance -> 1/sqrt makes ANY non-finite input contaminate
    the whole row to NaN (a +Inf feeds Inf-Inf in centering; var becomes NaN; inv*xc = NaN).
    So the poisoned row is ALL NaN for NaN/+Inf/-Inf alike -- NOT merely "non-finite" (an
    all-+Inf row is impossible here and must be rejected). Finite sibling rows are unchanged,
    checked BITWISE via an integer view (distinguishes +0/-0), and against a float64 oracle.
  * single-pass Welford recurrence: the tree reduction forms Inf-Inf, so a non-finite input
    yields mean=NaN and m2=NaN on this backend. The order-robust bound asserted is: mean
    non-finite, and m2 in {NaN, +Inf} and NEVER negative/-Inf (squared-delta * positive weight
    is >= 0). Clean reference uses float64.
"""

import math
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("Metal GPU required", allow_module_level=True)
import triton
import triton.language as tl


# --------------------------------------------------------------------------- layernorm
@triton.jit
def _layernorm(X, O, ncols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < ncols
    x = tl.load(X + row * ncols + cols, mask=m, other=0.0)
    mean = tl.sum(x, 0) / ncols
    xc = tl.where(m, x - mean, 0.0)
    var = tl.sum(xc * xc, 0) / ncols
    inv = 1.0 / tl.sqrt(var + eps)
    tl.store(O + row * ncols + cols, xc * inv, mask=m)


def _run_ln(inp):
    R, C = inp.shape
    o = torch.zeros(R, C, device="mps")
    _layernorm[(R,)](inp.to("mps"), o, C, 1e-6, BLOCK=64)
    torch.mps.synchronize()
    return o.cpu()


def _ln_oracle_f64(row, eps=1e-6):
    x = row.double()
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return ((x - mean) / (var + eps).sqrt()).float()


def _bit_equal(a, b):
    # true bitwise equality (distinguishes +0.0 / -0.0), for finite rows
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def test_layernorm_clean_matches_float64_oracle():
    torch.manual_seed(0)
    clean = torch.randn(4, 64)
    o = _run_ln(clean)
    for r in range(4):
        assert torch.allclose(o[r], _ln_oracle_f64(clean[r]), atol=1e-3), r


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_layernorm_nonfinite_input_makes_row_all_nan_and_siblings_exact(bad):
    torch.manual_seed(0)
    clean = torch.randn(4, 64)
    oc = _run_ln(clean)
    poison = clean.clone()
    poison[1, 7] = bad
    op = _run_ln(poison)
    # source-exact: the poisoned row is ALL NaN (rejects an all-+Inf or any-finite row)
    assert torch.isnan(op[1]).all(), f"poisoned row must be all-NaN for this source; got {op[1].tolist()}"
    # finite sibling rows: BITWISE identical to the clean run and matching the float64 oracle
    for r in (0, 2, 3):
        assert _bit_equal(op[r], oc[r]), f"row {r} perturbed by poison (bitwise)"
        assert torch.allclose(op[r], _ln_oracle_f64(clean[r]), atol=1e-3)


# --------------------------------------------------------------------------- Welford recurrence
@triton.jit
def _welford_combine(mean_a, m2_a, w_a, mean_b, m2_b, w_b):
    delta = mean_b - mean_a
    w = w_a + w_b
    ratio = tl.where(w == 0.0, 0.0, w_b / w)
    mean = mean_a + delta * ratio
    m2 = m2_a + m2_b + delta * delta * w_a * ratio
    return mean, m2, w


@triton.jit
def _welford(X, OM, OV, N: tl.constexpr):
    off = tl.arange(0, N)
    v = tl.load(X + off).to(tl.float32)
    m2 = tl.full((N,), 0.0, tl.float32)
    w = tl.full((N,), 1.0, tl.float32)
    mean, m2, w = tl.reduce((v, m2, w), 0, _welford_combine)
    tl.store(OM, mean)
    tl.store(OV, m2)  # RAW m2 (sum of squared deviations), so the assertion sees the recurrence


def _run_welford(x):
    om = torch.zeros(1, device="mps")
    ov = torch.zeros(1, device="mps")
    _welford[(1,)](x.to("mps"), om, ov, N=x.numel())
    torch.mps.synchronize()
    return om.cpu().item(), ov.cpu().item()


def _m2_admissible_nonfinite(m2):
    # m2 = sum of (squared delta * positive weight) => >= 0; a non-finite input makes it
    # NaN or +Inf, but NEVER negative or -Inf. This is the order-robust bound.
    return (math.isnan(m2) or m2 == float("inf")) and not (m2 < 0)


def test_welford_clean_matches_float64_reference():
    torch.manual_seed(0)
    x = torch.randn(64)
    mean, m2 = _run_welford(x)
    xd = x.double()
    n = x.numel()
    assert abs(mean - xd.mean().item()) < 1e-4
    assert abs(m2 - (xd.var(unbiased=False) * n).item()) < 1e-2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_welford_nonfinite_input_mean_nonfinite_and_m2_never_negative(bad):
    torch.manual_seed(0)
    x = torch.randn(64)
    x[5] = bad
    mean, m2 = _run_welford(x)
    # mean is non-finite (observed NaN on this backend; +Inf also admissible by order)
    assert not math.isfinite(mean), f"mean should be non-finite, got {mean}"
    # m2 is NaN or +Inf, never negative / -Inf (the classification Astra required not pass)
    assert _m2_admissible_nonfinite(m2), f"m2 must be NaN or +Inf, never negative/-Inf; got {m2}"


# ------------------------------------------------------------------ checker self-validation
def test_checkers_reject_wrong_classifications():
    # layernorm predicate (isnan-all) must REJECT an all-+Inf row and a finite-beside-bad row.
    allnan = torch.full((64,), float("nan"))
    allposinf = torch.full((64,), float("inf"))
    finite_beside = allnan.clone()
    finite_beside[0] = 1.234
    assert torch.isnan(allnan).all()  # correct classification passes
    assert not torch.isnan(allposinf).all()  # all-+Inf must FAIL (Astra's gap)
    assert not torch.isnan(finite_beside).all()  # wrong-finite must FAIL
    # Welford m2 predicate must REJECT m2 = -Inf and any negative m2.
    assert _m2_admissible_nonfinite(float("nan"))
    assert _m2_admissible_nonfinite(float("inf"))
    assert not _m2_admissible_nonfinite(float("-inf"))  # Astra's gap: -Inf must FAIL
    assert not _m2_admissible_nonfinite(-1.0)
    # bitwise sibling check must distinguish +0.0 from -0.0
    pz = torch.zeros(4)
    nz = torch.tensor([-0.0, 0.0, 0.0, 0.0])
    assert torch.equal(pz, nz)  # value-equal (the weaker check Astra flagged)
    assert not _bit_equal(pz, nz)  # bitwise-distinct (the check we now use)
