"""Standalone validation of make_varlen_flash_attention (route-only template): build the
arg_decls/bindings by hand for a packed [total, H, D] layout, dispatch via the runtime,
and check EXACT vs the per-sequence reference across seqlens / causal / H. This validates
the template's ABI + FA2 algorithm independently of the (separate) routing detector.
"""
import math
import pytest
import torch

from triton_msl.codegen._msl_templates import make_varlen_flash_attention

requires_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + compile_shader",
)


def _packed_bindings(D):
    # packed [total, H, D]: token stride = H*D, head stride = D, dim stride = 1.
    hd = f"(H * {D}u)"
    return {
        "Q": "Q", "K": "K", "V": "V", "O": "Out", "CUQ": "cu_q", "CUK": "cu_k", "H": "H",
        "q_st": hd, "q_sh": f"{D}u", "q_sk": "1u",
        "k_st": hd, "k_sh": f"{D}u", "k_sk": "1u",
        "v_st": hd, "v_sh": f"{D}u", "v_sk": "1u",
        "o_st": hd, "o_sh": f"{D}u", "o_sk": "1u",
    }


_ARG_DECLS = [
    "    device const float* Q [[buffer(0)]]",
    "    device const float* K [[buffer(1)]]",
    "    device const float* V [[buffer(2)]]",
    "    device float* Out [[buffer(3)]]",
    "    device const int* cu_q [[buffer(4)]]",
    "    device const int* cu_k [[buffer(5)]]",
    "    constant uint& H [[buffer(6)]]",
]


def _ref(q, k, v, cu, H, D, causal):
    out = torch.zeros_like(q)
    for b in range(len(cu) - 1):
        s, e = int(cu[b]), int(cu[b + 1]); n = e - s
        for h in range(H):
            sc = (q[s:e, h].float() @ k[s:e, h].float().transpose(-2, -1)) / math.sqrt(D)
            if causal:
                m = torch.tril(torch.ones(n, n, device=q.device, dtype=torch.bool))
                sc = sc.masked_fill(~m, float("-inf"))
            out[s:e, h] = (torch.softmax(sc, -1) @ v[s:e, h].float()).to(out.dtype)
    return out


@requires_mps
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lens,H,D", [
    ([48, 32], 2, 64),
    ([64, 16, 48], 4, 64),
    ([128, 128], 8, 64),
    ([100, 50, 30], 2, 64),   # ragged, non-block-multiple
])
def test_varlen_fa_template_exact(lens, H, D, causal):
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    dev = "mps"; torch.manual_seed(0)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
    T = int(cu[-1])
    q = torch.randn(T, H, D, device=dev); k = torch.randn(T, H, D, device=dev)
    v = torch.randn(T, H, D, device=dev); o = torch.zeros(T, H, D, device=dev)
    msl = make_varlen_flash_attention(D, causal, arg_decls=_ARG_DECLS, bindings=_packed_bindings(D))
    rt = CompileShaderRuntime()
    lib = rt.get_library(msl)
    BM = 32
    nmb = math.ceil(max(lens) / BM)
    bh = len(lens) * H
    rt.dispatch(lib, "varlen_fa", [q, k, v, o, cu, cu, H],
                threads=(nmb * BM, bh, 1), group_size=(BM, 1, 1))
    torch.mps.synchronize()
    assert (o - _ref(q, k, v, cu, H, D, causal)).abs().max().item() < 2e-3


@requires_mps
@pytest.mark.parametrize("dt,elem,tol", [
    (torch.float16, "half", 2e-2),
    (torch.bfloat16, "bfloat", 6e-2),
])
def test_varlen_fa_template_dtypes(dt, elem, tol):
    # The scalar template's non-fp32 dtype paths (half / bfloat) — reached when a fp16/bf16
    # varlen kernel is MMA-INELIGIBLE (non-%8 head_dim, or the tg budget). Route-free so it
    # exercises the codegen directly (bf16 @jit kernels often won't compile at odd head_dims).
    dev = "mps"; torch.manual_seed(0); H, D = 2, 64; lens = [48, 32]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
    T = int(cu[-1])
    q = torch.randn(T, H, D, device=dev, dtype=dt); k = torch.randn(T, H, D, device=dev, dtype=dt)
    v = torch.randn(T, H, D, device=dev, dtype=dt); o = torch.zeros(T, H, D, device=dev, dtype=dt)
    decls = [f"    device const {elem}* Q [[buffer(0)]]", f"    device const {elem}* K [[buffer(1)]]",
             f"    device const {elem}* V [[buffer(2)]]", f"    device {elem}* Out [[buffer(3)]]",
             "    device const int* cu_q [[buffer(4)]]", "    device const int* cu_k [[buffer(5)]]",
             "    constant uint& H [[buffer(6)]]"]
    out_dtype = "fp16" if dt == torch.float16 else "bf16"
    msl = make_varlen_flash_attention(D, False, out_dtype=out_dtype, arg_decls=decls, bindings=_packed_bindings(D))
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    rt = CompileShaderRuntime(); lib = rt.get_library(msl); BM = 32
    rt.dispatch(lib, "varlen_fa", [q, k, v, o, cu, cu, H],
                threads=(math.ceil(max(lens) / BM) * BM, len(lens) * H, 1), group_size=(BM, 1, 1))
    torch.mps.synchronize()
    assert (o.float() - _ref(q, k, v, cu, H, D, False).float()).abs().max().item() < tol
