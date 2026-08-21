"""Standalone parity tests for the biased/triangle-attention extension of the
head-dim-tiled FA2 template (``make_flash_attention_kernel_tiled`` with
``bias``/``mask``/``lse``/``runtime_scale``).

trifast-style triangle attention adds, on top of standard FA: an additive bias
into the QK scores, a loaded mask -> -inf, a per-query log-sum-exp output, and a
*runtime* softmax scale. The generic op-by-op lowering cannot express a dot-result
epilogue (see generic_lowerer ``_dot_result_scaled``), so these fuse into the
validated online-softmax loop instead. His kernel's ``inv_ln2`` + ``exp2`` is the
base-2 reformulation of natural-exp softmax over ``(scale*QK + bias)`` — proven
numerically identical here, so the template uses natural exp and stores the
natural-log lse.

The template returns an MSL STRING; these tests compile it with
``torch.mps.compile_shader`` and run on the Metal GPU vs a torch reference.
Metal's 31-buffer limit means the full per-stride ABI does not fit, so strides
are baked as constants and only the 7 pointers + runtime scale are buffers (the
routed path packs the overflow scalars via the #7 argument-buffer path).
"""

import math
import pytest
import torch

from triton_msl.codegen._msl_templates import make_flash_attention_kernel_tiled

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


def _run_biased(q, k, v, bias, mask, sm_scale, D, BM, BN, Dc, causal, out_dtype):
    """Compile + launch the biased tiled FA template; return (out, lse)."""
    Z, H, N = q.shape[0], q.shape[1], q.shape[2]
    elem = "float" if out_dtype in ("fp32", "f32") else "half"
    tdt = torch.float32 if elem == "float" else torch.float16
    out = torch.zeros(Z, H, N, D, device="mps", dtype=tdt)
    lse = torch.zeros(Z, H, N, device="mps", dtype=torch.float32)

    arg_decls = [
        f"    device const {elem}* Q [[buffer(0)]]",
        f"    device const {elem}* K [[buffer(1)]]",
        f"    device const {elem}* V [[buffer(2)]]",
        f"    device {elem}* Out [[buffer(3)]]",
        f"    device const {elem}* Bias [[buffer(4)]]",
        "    device const uchar* Mask [[buffer(5)]]",
        "    device float* Lse [[buffer(6)]]",
        "    constant float& arg_scale [[buffer(7)]]",
    ]
    qs, ks, vs, os_, bs, ms, ls = (
        q.stride(), k.stride(), v.stride(), out.stride(), bias.stride(), mask.stride(), lse.stride(),
    )
    bindings = {
        "q_sz": f"{qs[0]}u", "q_sh": f"{qs[1]}u", "q_sm": f"{qs[2]}u", "q_sk": f"{qs[3]}u",
        "k_sz": f"{ks[0]}u", "k_sh": f"{ks[1]}u", "k_sn": f"{ks[2]}u", "k_sk": f"{ks[3]}u",
        "v_sz": f"{vs[0]}u", "v_sh": f"{vs[1]}u", "v_sn": f"{vs[2]}u", "v_sk": f"{vs[3]}u",
        "o_sz": f"{os_[0]}u", "o_sh": f"{os_[1]}u", "o_sm": f"{os_[2]}u", "o_sk": f"{os_[3]}u",
        "Z": f"{Z}u", "H": f"{H}u", "N_CTX": f"{N}u",
        "b_sz": f"{bs[0]}u", "b_sh": f"{bs[1]}u", "b_sm": f"{bs[2]}u", "b_sn": f"{bs[3]}u",
        "mask_sz": f"{ms[0]}u", "mask_sh": f"{ms[1]}u", "mask_sn": f"{ms[2]}u",
        "lse_sz": f"{ls[0]}u", "lse_sh": f"{ls[1]}u", "lse_sm": f"{ls[2]}u",
        "scale": "arg_scale",
    }
    src = make_flash_attention_kernel_tiled(
        D, BM, BN, Dc=Dc, causal=causal, out_dtype=out_dtype,
        bias=True, mask=True, lse=True, runtime_scale=True,
        arg_decls=arg_decls, bindings=bindings,
    )
    lib = torch.mps.compile_shader(src)
    tpg = BM * BN
    threads = ((N // BM) * tpg, Z * H)
    group_size = (tpg, 1)
    lib.flash_attention(q, k, v, out, bias, mask, lse, float(sm_scale),
                        threads=threads, group_size=group_size)
    torch.mps.synchronize()
    return out, lse


def _assert_close(out, o_ref, lse, lse_ref, o_tol, l_tol):
    """Compare out everywhere; compare lse only where the reference is finite (a
    fully-masked query row has lse = -inf in BOTH kernel and reference, and
    -inf - -inf = nan, so exclude those and separately assert they agree at -inf)."""
    assert (out.float() - o_ref).abs().max().item() < o_tol
    finite = torch.isfinite(lse_ref)
    if finite.any():
        assert (lse[finite] - lse_ref[finite]).abs().max().item() < l_tol
    if (~finite).any():
        # reference row is -inf (fully masked) -> kernel must be strongly negative too
        assert (lse[~finite] < -1e30).all(), lse[~finite]


def _ref(q, k, v, bias, mask, sm_scale, causal):
    raw = sm_scale * (q.float() @ k.float().transpose(-2, -1)) + bias.float()
    raw = raw.masked_fill(mask[:, :, None, :].bool(), float("-inf"))
    if causal:
        N = raw.shape[-1]
        tri = torch.triu(torch.ones(N, N, device=raw.device), diagonal=1).bool()
        raw = raw.masked_fill(tri[None, None], float("-inf"))
    p = torch.softmax(raw, dim=-1)
    o = torch.nan_to_num(p, nan=0.0) @ v.float()
    lse = torch.logsumexp(raw, dim=-1)
    return o, lse


@requires_mps
@pytest.mark.parametrize("Z,H,N,D", [(1, 2, 64, 32), (1, 1, 96, 64), (2, 2, 64, 32)])
def test_biased_fa_fp32(Z, H, N, D):
    torch.manual_seed(0)
    Dc = D if D <= 64 else 64
    sm = 1.0 / math.sqrt(D)
    q = torch.randn(Z, H, N, D, device="mps")
    k = torch.randn(Z, H, N, D, device="mps")
    v = torch.randn(Z, H, N, D, device="mps")
    bias = torch.randn(Z, H, N, N, device="mps")
    mask = (torch.rand(Z, H, N, device="mps") < 0.25).to(torch.uint8)
    out, lse = _run_biased(q, k, v, bias, mask, sm, D, 32, 32, Dc, False, "fp32")
    o_ref, lse_ref = _ref(q, k, v, bias, mask, sm, False)
    _assert_close(out, o_ref, lse, lse_ref, 1e-3, 1e-3)


@requires_mps
def test_biased_fa_causal_composes():
    torch.manual_seed(1)
    Z, H, N, D = 1, 2, 64, 32
    sm = 1.0 / math.sqrt(D)
    q = torch.randn(Z, H, N, D, device="mps")
    k = torch.randn(Z, H, N, D, device="mps")
    v = torch.randn(Z, H, N, D, device="mps")
    bias = torch.randn(Z, H, N, N, device="mps")
    # No extra mask here (causal is the mask); keep a few masked kv to exercise both.
    mask = (torch.rand(Z, H, N, device="mps") < 0.1).to(torch.uint8)
    out, lse = _run_biased(q, k, v, bias, mask, sm, D, 32, 32, D, True, "fp32")
    o_ref, lse_ref = _ref(q, k, v, bias, mask, sm, True)
    _assert_close(out, o_ref, lse, lse_ref, 1e-3, 1e-3)


@requires_mps
def test_biased_fa_fp16():
    torch.manual_seed(2)
    Z, H, N, D = 1, 2, 64, 32
    sm = 1.0 / math.sqrt(D)
    q = torch.randn(Z, H, N, D, device="mps", dtype=torch.float16)
    k = torch.randn(Z, H, N, D, device="mps", dtype=torch.float16)
    v = torch.randn(Z, H, N, D, device="mps", dtype=torch.float16)
    bias = torch.randn(Z, H, N, N, device="mps", dtype=torch.float16)
    mask = (torch.rand(Z, H, N, device="mps") < 0.25).to(torch.uint8)
    out, lse = _run_biased(q, k, v, bias, mask, sm, D, 32, 32, D, False, "fp16")
    o_ref, lse_ref = _ref(q, k, v, bias, mask, sm, False)
    # fp16 in/out, fp32 compute: looser tol on out; lse computed in fp32.
    _assert_close(out, o_ref, lse, lse_ref, 2e-2, 5e-2)


def test_flags_off_emits_no_bias_mask_lse():
    """bias/mask/lse=False (default) must emit NO Bias/Mask/Lse/arg_scale — the
    standard-FA emission stays behaviorally identical."""
    src = make_flash_attention_kernel_tiled(128, 32, 32, Dc=64, causal=False, out_dtype="fp32")
    for tok in ("Bias", "Mask", "Lse", "arg_scale", "bias_base", "lse_base"):
        assert tok not in src, f"plain FA leaked biased token: {tok!r}"
    assert "const float scale = " in src  # baked scale, not runtime


def test_biased_flags_emit_expected_tokens():
    src = make_flash_attention_kernel_tiled(
        32, 32, 32, Dc=32, causal=False, out_dtype="fp32",
        bias=True, mask=True, lse=True, runtime_scale=True,
    )
    for tok in (
        "device const float* Bias", "device const uchar* Mask", "device float* Lse",
        "const float scale = arg_scale;", "s += float(Bias[bias_base",
        "if (Mask[mask_base", "Lse[lse_base + q_row * lse_sm] = (l_val > 0.0f) ? (tg_m[r] + log(l_val)) : -INFINITY;",
    ):
        assert tok in src, f"biased FA missing token: {tok!r}"
