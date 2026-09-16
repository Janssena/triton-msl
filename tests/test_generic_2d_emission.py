"""Native IR/emission only; no Metal compilation, device or dispatch."""
import pytest
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir
from triton_msl.backend.compiler import MetalBackend
from triton_msl.codegen.msl_emitter import emit_msl
from tests.test_generic_2d_gpu import affine_2d,integer_2d,integer_source,checked_accumulate_2d


def test_integer_public_source_signature_is_explicit():
    source=integer_source()
    assert source.signature['BIAS']=='i64' and source.signature['UNSIGNED']=='u32'


@pytest.mark.parametrize('fn,signature',[
    (affine_2d,dict(X='*fp32',O='*fp32',N='i32',SCALE='fp32')),
    (integer_2d,dict(O='*i64',N='i32',BIAS='i64',UNSIGNED='u32')),
    (checked_accumulate_2d,dict(X='*fp32',O='*fp32',N='i32',LIMIT='i32'))])
def test_native_generic_2d_metadata_and_scalar_abi(fn,signature,monkeypatch):
    monkeypatch.setenv('TRITON_MSL_USE_CPP','0')
    target=GPUTarget('metal','apple-m4',32);backend=MetalBackend(target)
    options=backend.parse_options(dict(num_warps=4,debug=True))
    source=integer_source() if fn is integer_2d else ASTSource(fn,signature,constexprs=dict(BLOCK=32))
    ctx=ir.context();ir.load_dialects(ctx)
    module=source.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),ctx)
    metadata={};module=backend.make_ttir(module,metadata,options);module=backend.make_ttgir(module,metadata,options)
    msl=emit_msl(module,metadata,options)
    assert metadata['needs_2d_grid'] is True and 1<=metadata['block_size']<=1024
    assert all(metadata.get(name) is None for name in ['mm_two_kernel','fast_matmul','quant_matmul','flash_attention','batched_dot_bounds'])
    assert 'pid3.y' in msl and 'UNKNOWN' not in msl
    if fn is integer_2d:
        # Native MLIR integers are signless at the parameter boundary. The
        # declared u32 bits must be zero-extended by the actual unsigned cast.
        assert 'constant long& BIAS' in msl and 'constant int& UNSIGNED' in msl
        assert 'static_cast<ulong>(static_cast<uint>(UNSIGNED))' in msl
    if fn is checked_accumulate_2d:
        assert 'generic 2d input bound' in metadata['device_assert']['messages']
