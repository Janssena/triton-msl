"""The source's load padding and its score mask are different contracts."""

import pytest
import torch
import triton
import triton.language as tl

from tests.cache_helpers import patch_live_singleton_method

from triton_msl.errors import MetalNonRecoverableError


@triton.jit
def _tail_attention(
    Q, K, V, Out,
    sqz, sqh, sqm, sqk, skz, skh, skn, skk,
    svz, svh, svn, svk, soz, soh, som, sok, Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    MASK: tl.constexpr, START: tl.constexpr, STEP: tl.constexpr,
    FILL_K: tl.constexpr, FILL_V: tl.constexpr,
    PLAIN_K: tl.constexpr, PLAIN_V: tl.constexpr,
):
    pid = tl.program_id(0)
    zh = tl.program_id(1)
    z, h = zh // H, zh % H
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rd = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + z*sqz + h*sqh + rm[:, None]*sqm + rd[None, :]*sqk,
                rm[:, None] < N_CTX, 0.)
    q = q * (1. / tl.sqrt(float(HEAD_DIM)))
    m = tl.full([BLOCK_M], float('-inf'), tl.float32)
    l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    hi = N_CTX
    if MASK == 3:
        hi = tl.minimum((pid + 1) * BLOCK_M, N_CTX)
    for j in range(START, hi, STEP):
        kn = j + rn
        kp = K + z*skz + h*skh + kn[:, None]*skn + rd[None, :]*skk
        if PLAIN_K:
            k = tl.load(kp)
        else:
            k = tl.load(kp, kn[:, None] < N_CTX, FILL_K)
        s = tl.dot(q, tl.trans(k).to(q.dtype))
        if MASK == 1:
            s = tl.where(kn[None, :] < N_CTX, s, float('-inf'))
        elif MASK == 2:
            s = tl.where(rm[:, None] < N_CTX, s, float('-inf'))
        elif MASK == 3 or MASK == 6:
            s = tl.where(rm[:, None] >= kn[None, :], s, float('-inf'))
        elif MASK == 4:
            s = tl.where(kn[:, None] < N_CTX, s, float('-inf'))
        elif MASK == 5:
            s = tl.where(rm[None, :] < N_CTX, s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, 1))
        a = tl.exp(m - mn)
        p = tl.exp(s - mn[:, None])
        l = l*a + tl.sum(p, 1)
        vp = V + z*svz + h*svh + kn[:, None]*svn + rd[None, :]*svk
        if PLAIN_V:
            v = tl.load(vp)
        else:
            v = tl.load(vp, kn[:, None] < N_CTX, FILL_V)
        acc = acc*a[:, None] + tl.dot(p, v.to(p.dtype))
        m = mn
    tl.store(Out + z*soz + h*soh + rm[:, None]*som + rd[None, :]*sok,
             (acc/l[:, None]).to(Out.dtype.element_ty), rm[:, None] < N_CTX)


def _graph(dtype, dim, mask, start=0, step=32, fill_k=0., fill_v=0., plain_k=False,
           plain_v=False, n_ctx_constant=None):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.codegen.mlir_walker import walk_ttgir

    target = GPUTarget('metal', 'apple-m4', 32)
    backend = MetalBackend(target)
    opts = backend.parse_options({})
    constants = dict(Z=1, sqk=1, skk=1, svk=1, sok=1, BLOCK_M=32,
                     BLOCK_N=32, HEAD_DIM=dim, MASK=mask, START=start, STEP=step,
                     FILL_K=fill_k, FILL_V=fill_v, PLAIN_K=plain_k, PLAIN_V=plain_v)
    if n_ctx_constant is not None:
        constants['N_CTX'] = n_ctx_constant
    if dim == 128:
        del constants['sqk']  # A real noncontiguous Q selects the tiled route.
    signature = {n: ('*'+dtype if n in ('Q', 'K', 'V', 'Out') else 'i32')
                 for n in _tail_attention.arg_names if n not in constants}
    source = ASTSource(_tail_attention, signature, constexprs=constants)
    context = ir.context()
    ir.load_dialects(context)
    mod = source.make_ir(target, opts, backend.get_codegen_implementation(opts),
                         backend.get_module_map(), context)
    meta = {}
    mod = backend.make_ttir(mod, meta, opts)
    mod = backend.make_ttgir(mod, meta, opts)
    return walk_ttgir(mod, opts), opts


@pytest.fixture
def executed(monkeypatch):
    """Observe both real launchers; a tiled host route is not compile_shader."""
    from triton_msl.backend.driver import _get_compile_shader_runtime, _get_utils
    hits = []
    # Earlier instance spies can leave bound attributes after undo; patching
    # either class then misses real dispatches. Observe BOTH exact singletons
    # consumed by MetalLauncher, without changing which route executes.
    def observe_dispatch(runtime, dispatch):
        def record_dispatch(lib, *args, **kwargs):
            sources = [s for s, entry in runtime._lib_cache.items() if entry is lib]
            assert len(sources) == 1
            result = dispatch(lib, *args, **kwargs)
            hits.append(dict(msl=sources[0]))
            return result

        return record_dispatch

    def observe_launch(_utils, launch):
        def record_launch(pipeline, *args, **kwargs):
            result = launch(pipeline, *args, **kwargs)
            hits.append(dict(pipeline=pipeline))
            return result

        return record_launch

    patch_live_singleton_method(
        monkeypatch, _get_compile_shader_runtime, "dispatch", observe_dispatch
    )
    patch_live_singleton_method(monkeypatch, _get_utils, "launch", observe_launch)
    return hits


def _executed_source(hits, compiled):
    assert len(hits) == 1
    if 'msl' in hits[0]:
        return hits[0]['msl']
    assert hits[0]['pipeline'] is compiled.function
    return compiled.asm['msl']


# Fifteen rows, combining fresh lowering contracts with hardware/source equivalence.
# The fp32 D64 controls retain tail residues 1, 15, and 31 (N=33, 47, 63),
# promoted from the independent packet-346 sweep without multiplying the full
# dtype/dimension/mask matrix.
# N=1 is a separate constexpr-specialized graph and must compute, not merely
# satisfy correct-or-refuse through the generic forward-FA guard.
# D128 uses a stride-two Q to exercise the scalar/tiled caller, not a forced maker.
@pytest.mark.parametrize('dtype,dim,mask,n,plain', [
    ('fp32', 64, 0, 1, ''), ('fp16', 64, 0, 1, ''),
    ('fp32', 64, 0, 33, ''), ('fp32', 64, 1, 33, ''),
    ('fp32', 64, 2, 33, ''),
    ('fp32', 64, 3, 33, ''), ('fp32', 64, 0, 47, ''), ('fp32', 64, 0, 63, ''),
    ('fp32', 64, 0, 64, ''), ('fp16', 64, 0, 33, ''),
    ('bf16', 64, 1, 33, ''),
    ('fp32', 128, 0, 33, ''), ('fp16', 128, 2, 33, ''),
    ('fp32', 64, 0, 33, 'K'), ('fp32', 128, 0, 33, 'V'),
])
def test_dense_tail_preserves_source(dtype, dim, mask, n, plain, monkeypatch, executed):
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    graph, opts = _graph(dtype, dim, mask, plain_k=plain == 'K', plain_v=plain == 'V',
                         n_ctx_constant=1 if n == 1 else None)
    lower = GenericLowerer(graph, opts)
    msl = lower.lower()  # Cached JIT execution cannot hide this contract.
    route_marker = ('Head-dim-tiled FlashAttention-2'
                    if dim == 128 or dtype == 'bf16'
                    else 'simdgroup-MMA FlashAttention-2')
    assert route_marker in msl
    if dtype == 'bf16':
        assert 'device const bfloat* Q' in msl
        assert 'simdgroup_multiply_accumulate' not in msl
    if mask in (0, 2):
        assert 'source KV tail: 32' in msl
    else:
        assert 'source KV tail:' not in msl
    if not torch.backends.mps.is_available():
        pytest.skip('GPU equivalence requires MPS; lowering contract ran')
    monkeypatch.setenv('TRITON_MSL_COMPILE_SHADER', '1')
    torch.manual_seed(289)
    td = {'fp32': torch.float32, 'fp16': torch.float16,
          'bf16': torch.bfloat16}[dtype]
    padded = triton.cdiv(n, 32)*32
    q = torch.randn(1, 2, n, dim).to(td)
    k, v = [torch.randn(1, 2, padded, dim).to(td) for _ in range(2)]
    v += 1
    kp, vp = k.float().clone(), v.float().clone()
    if plain != 'K':
        kp[:, :, n:] = 0
    if plain != 'V':
        vp[:, :, n:] = 0
    scores = (q.float() * (dim**-.5)) @ kp.transpose(-1, -2)
    if mask == 1:
        scores[:, :, :, n:] = -float('inf')
    elif mask == 3:
        scores.masked_fill_(torch.arange(padded)[None, :] > torch.arange(n)[:, None],
                            -float('inf'))
    ref = (scores.softmax(-1) @ vp).to(td)
    q, k, v = [x.to('mps') for x in (q, k, v)]
    if dim == 128:
        q_view = torch.empty(1, 2, n, 2*dim, device='mps', dtype=td)[..., ::2]
        q_view.copy_(q)
        q = q_view
    out = torch.full_like(q, float('nan'))
    compiled = _tail_attention[(triton.cdiv(n, 32), 2)](
        q, k, v, out, *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        1, 2, n, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=dim, MASK=mask,
        START=0, STEP=32, FILL_K=0., FILL_V=0.,
        PLAIN_K=plain == 'K', PLAIN_V=plain == 'V',
    )
    torch.mps.synchronize()
    actual_msl = _executed_source(executed, compiled)
    assert route_marker in actual_msl
    if mask in (0, 2):
        assert 'source KV tail: 32' in actual_msl
    tolerance = .02 if dtype == 'bf16' else .002
    torch.testing.assert_close(out.cpu(), ref, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('change', [dict(start=32), dict(step=64),
                                 dict(fill_k=1.), dict(fill_v=1.), dict(mask=4), dict(mask=5)])
def test_dense_tail_rejects_unreplayed_loop_or_fill(change):
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    graph, opts = _graph('fp32', 64, **(dict(mask=0) | change))
    with pytest.raises(MetalNonRecoverableError):
        GenericLowerer(graph, opts).lower()


@pytest.mark.parametrize('term', ['kn[:, None]*skn', 'kn[:, None]*svn',
                                'rm[:, None]*sqm', 'rm[:, None]*som'])
def test_dense_addresses_must_match_the_proved_coordinates(term, monkeypatch):
    import sys
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    fn = triton.JITFunction(_tail_attention.fn)
    assert fn.src.count(term) == 1
    fn._unsafe_update_src(fn.src.replace(term, '(' + term[:2] + '+32)' + term[2:]))
    monkeypatch.setattr(sys.modules[__name__], '_tail_attention', fn)
    graph, opts = _graph('fp32', 64, 0)
    with pytest.raises(MetalNonRecoverableError):
        GenericLowerer(graph, opts).lower()


@pytest.mark.parametrize('dim,full', [(64, False), (128, False), (64, True), (128, True)])
def test_dense_causal_loop_visits_exact_source_rows(dim, full, monkeypatch, executed):
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    mask = 6 if full else 3
    graph, opts = _graph('fp32', dim, mask)
    msl = GenericLowerer(graph, opts).lower()
    route_marker = 'Head-dim-tiled FlashAttention-2' if dim == 128 else 'simdgroup-MMA FlashAttention-2'
    assert route_marker in msl
    assert ('// source KV loop stop' in msl) == (not full)
    if full:
        assert '// causal: skip fully-masked blocks' not in msl
    if not torch.backends.mps.is_available():
        pytest.skip('GPU equivalence requires MPS; lowering contract ran')
    monkeypatch.setenv('TRITON_MSL_COMPILE_SHADER', '1')
    torch.manual_seed(291)
    q, k, v = [torch.randn(1, 2, 96, dim) for _ in range(3)]
    v[:, :, 80 if full else 40] = float('nan')
    # A NaN in an unvisited source iteration must not contaminate earlier tiles.
    expected_nan = torch.ones(1, 2, 96, dtype=torch.bool)
    if not full:
        expected_nan[:, :, :32] = False
    s = (q[:, :, :32] * dim**-.5) @ k[:, :, :32].transpose(-1, -2)
    s.masked_fill_(torch.arange(32)[None, :] > torch.arange(32)[:, None], -float('inf'))
    finite_ref = s.softmax(-1) @ v[:, :, :32]
    q, k, v = [x.to('mps') for x in (q, k, v)]
    if dim == 128:
        q_view = torch.empty(1, 2, 96, 2*dim, device='mps')[..., ::2]
        q_view.copy_(q)
        q = q_view
    out = torch.empty_like(q)
    compiled = _tail_attention[(3, 2)](q, k, v, out, *q.stride(), *k.stride(), *v.stride(), *out.stride(),
                           1, 2, 96, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=dim, MASK=mask,
                           START=0, STEP=32, FILL_K=0., FILL_V=0., PLAIN_K=False, PLAIN_V=False)
    torch.mps.synchronize()
    got = out.cpu()
    assert route_marker in _executed_source(executed, compiled)
    assert torch.equal(torch.isnan(got).all(-1), expected_nan)
    assert torch.isfinite(got[~expected_nan]).all()
    if not full:
        torch.testing.assert_close(got[:, :, :32], finite_ref, atol=.002, rtol=.002)
