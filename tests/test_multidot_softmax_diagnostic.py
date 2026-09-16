import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_msl.backend.compiler import MetalBackend
from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _mlp(X, W1, W2, Z, D: tl.constexpr):
    r = tl.arange(0, D)
    x = tl.load(X + r[:, None] * D + r[None, :])
    w1 = tl.load(W1 + r[:, None] * D + r[None, :])
    h = tl.maximum(tl.dot(x, w1), 0.0)
    w2 = tl.load(W2 + r[:, None] * D + r[None, :])
    y = tl.dot(h, w2)
    y = tl.exp(y - tl.max(y, axis=1)[:, None])
    y = y / tl.sum(y, axis=1)[:, None]
    tl.store(Z + r[:, None] * D + r[None, :], y)


def test_full_native_mlp128_refusal_does_not_claim_attention_pointer_roles():
    backend = MetalBackend(GPUTarget("metal", "apple-m4", 32))
    options = backend.parse_options({"num_warps": 4})
    ctx = ir.context()
    ir.load_dialects(ctx)
    module = ASTSource(
        _mlp,
        {"X": "*fp32", "W1": "*fp32", "W2": "*fp32", "Z": "*fp32", "D": "constexpr"},
        {"D": 128},
    ).make_ir(
        backend.target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        ctx,
    )
    module = backend.make_ttir(module, {}, options)
    module = backend.make_ttgir(module, {}, options)
    with pytest.raises(MetalNonRecoverableError) as caught:
        backend.make_msl(module, {}, options)
    message = str(caught.value)
    assert "multi-dot softmax-shaped kernel" in message
    assert "FlashAttention" not in message
    assert "Q pointer" not in message and "K pointer" not in message and "V pointer" not in message


def test_direct_real_fa_detector_retains_specific_unresolved_role_diagnostic():
    from tests.test_fa_detect import _build_fa_lowerer

    lowerer = _build_fa_lowerer(causal=False, head_dim=128, block=32)
    q_stride = next(arg.id for arg in lowerer.graph.args if arg.name == "stride_qm")
    for op in lowerer.graph.ops:
        if op.op == "tt.splat" and op.operand_ids == [q_stride]:
            op.operand_ids = []
    with pytest.raises(MetalNonRecoverableError, match="FlashAttention recognized but the Q pointer/stride chain"):
        lowerer._detect_flash_attention()
