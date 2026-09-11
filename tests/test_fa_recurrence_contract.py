"""A templated forward loop must prove the values it carries, not merely ancestry."""
import pytest
import triton

from triton_msl.errors import MetalNonRecoverableError


@pytest.mark.parametrize('block', [32, 64])
@pytest.mark.parametrize('old,new', [
    ('l_i = l_i * alpha + tl.sum(p, 1)', 'l_i = l_i * alpha + 2. * tl.sum(p, 1)'),
    ('l_i = l_i * alpha + tl.sum(p, 1)', 'l_i = l_i * (alpha * 2.) + tl.sum(p, 1)'),
    ('acc = acc * alpha[:, None]', 'acc = acc * (alpha[:, None] * 2.)'),
    ('tl.sum(p, 1)', 'tl.max(p, 1)'),
], ids=['denominator-sum', 'denominator-carry', 'numerator-carry', 'denominator-reducer'])
def test_unreplayed_recurrence_refuses_at_lowering(block, old, new, monkeypatch):
    import test_fa_query_subtiles as source
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    fn = triton.JITFunction(source._flash_attn_fwd.fn)
    assert fn.src.count(old) == 1
    fn._unsafe_update_src(fn.src.replace(old, new))
    monkeypatch.setattr(source, '_flash_attn_fwd', fn)
    graph, options = source._graph('fp32', block, block, False)
    # These eight sources previously emitted canonical shaders and returned
    # incorrect results. This is containment, not support for these recurrences.
    with pytest.raises(MetalNonRecoverableError, match='recurrence'):
        GenericLowerer(graph, options).lower()


@pytest.mark.parametrize('block', [32, 64])
def test_finite_initializer_is_replayed_not_over_refused(block, monkeypatch):
    import test_fa_query_subtiles as source
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    fn = triton.JITFunction(source._flash_attn_fwd.fn)
    old = 'l_i = tl.zeros([BLOCK_M], dtype=tl.float32)'
    assert fn.src.count(old) == 1
    fn._unsafe_update_src(fn.src.replace(old, 'l_i = tl.full([BLOCK_M], 1., dtype=tl.float32)'))
    monkeypatch.setattr(source, '_flash_attn_fwd', fn)
    graph, options = source._graph('fp32', block, block, False)
    msl = GenericLowerer(graph, options).lower()
    assert 'tg_l[lid]=1.0f;' in msl.replace(' ', '')


@pytest.mark.parametrize('combiner', ['arith.addf', 'arith.maxnumf'], ids=['sum-return', 'max-return'])
def test_recurrence_reducer_must_return_its_combiner(combiner):
    import test_fa_query_subtiles as source
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    graph, options = source._graph('fp32', 32, 32, False)
    loop, = [op for op in graph.ops if op.op == 'scf.for']
    red, = [op for op in loop.region_ops if op.op == 'tt.reduce'
            and any(body.op == combiner for body in op.region_ops)]
    # Direct-IR contract: a legal reduction body may return its first argument
    # instead of the otherwise-present add/max. The template must inspect that
    # return just as generic lowering does, not credit an unused combiner.
    red.attrs['return_ids'] = [red.attrs['block_arg_ids'][0]]
    with pytest.raises(MetalNonRecoverableError, match='recurrence'):
        GenericLowerer(graph, options).lower()


@pytest.mark.parametrize('old,new', [
    ('l = l*a + tl.sum(p, 1)', 'l = l*a + 2. * tl.sum(p, 1)'),
    ('l = l*a + tl.sum(p, 1)', 'l = l*a + tl.max(p, 1)'),
    ('acc/l[:, None]', 'acc/(l[:, None] + 1.)'),
], ids=['singleton-denominator-scale', 'singleton-denominator-reducer',
        'singleton-output-denominator'])
def test_singleton_unrolled_recurrence_still_requires_source_values(old, new, monkeypatch):
    import test_fa_tail_semantics as source
    from triton_msl.codegen.generic_lowerer import GenericLowerer

    fn = triton.JITFunction(source._tail_attention.fn)
    assert fn.src.count(old) == 1
    fn._unsafe_update_src(fn.src.replace(old, new))
    monkeypatch.setattr(source, '_tail_attention', fn)
    graph, options = source._graph('fp32', 64, 0, n_ctx_constant=1)
    with pytest.raises(MetalNonRecoverableError):
        GenericLowerer(graph, options).lower()
