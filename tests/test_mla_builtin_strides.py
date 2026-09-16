"""CPU-only current-descriptor equivalence and fast-resolution mechanism controls."""
import ast
import builtins
import copy
import os
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(os.environ.get('MLA_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
REFERENCE = Path(__file__).parent / 'fixtures/mla_parent635_dispatch.py'


def load(root, reference=False):
    path = REFERENCE if reference else root / 'triton_msl/autotuning/_fa_dispatch.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    class PostSubmitError(Exception):
        pass
    scope = {'PostSubmitError': PostSubmitError}
    sub = ast.parse((root / 'triton_msl/autotuning/_submission.py').read_text())
    exec(compile(ast.Module(body=[n for n in sub.body if isinstance(n, ast.ClassDef)], type_ignores=[]), 'submission', 'exec'), scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def exercise(scope, mode='normal', profile=False):
    events = []
    buffers = []
    resolutions = []
    def event(name, value=None):
        events.append((name, value))
    class Scalar:
        def __init__(self, value):
            self.value = value
        def __int__(self):
            event('int', self.value)
            if mode == 'raising_scalar':
                raise ValueError('scalar failure')
            return self.value
    class IntSubclass(int):
        def __int__(self):
            event('subclass_int', int.__int__(self))
            return int.__int__(self)
    class RefSubclass(str):
        def __eq__(self, other):
            event('ref_eq', str(self))
            return str.__eq__(self, other)
    class Refs(list):
        def __iter__(self):
            event('refs_iter')
            return super().__iter__()
    class Args(list):
        def __getitem__(self, index):
            event('args_get', index)
            return super().__getitem__(index)
    class Meta(type):
        def __eq__(self, other):
            event('metaclass_eq', other.__name__)
            return super().__eq__(other)
    class MetaArgs(Args, metaclass=Meta):
        pass
    class MetaRefs(Refs, metaclass=Meta):
        pass
    class Storage:
        def __init__(self, tensor):
            self.tensor = tensor
        def nbytes(self):
            event('nbytes', self.tensor.role)
            return self.tensor.width * 64 * 2
    class Tensor:
        def __init__(self, role, width):
            self.role, self.width = role, width
        @property
        def data_ptr(self):
            event('data_ptr', self.role)
            return lambda: 0
        @property
        def device(self):
            event('device', self.role)
            return 'mps'
        @property
        def dtype(self):
            event('dtype', self.role)
            if mode == 'mutate_later_stride' and self.role == 'q_rope':
                args[12] = Scalar(128)
            if mode == 'runtime_bool_binding' and self.role == 'q':
                patches['bool'] = scope.get('bool')
                scope['bool'] = int
            return 'f32' if mode == 'wrong_dtype' and self.role == 'out' else 'f16'
        def untyped_storage(self):
            event('storage', self.role)
            return Storage(self)
        def element_size(self):
            event('element_size', self.role)
            return 2
        def storage_offset(self):
            event('offset', self.role)
            return 64 * self.width if mode == 'offset_oob' and self.role == 'out' else 0
        def contiguous(self):
            event('contiguous', self.role)
            if mode == 'contiguous_failure':
                raise RuntimeError('contiguous failure')
            return self
        def stride(self):
            event('stride', self.role)
            return 64 * self.width, 32 * self.width, self.width, 1
    def as_strided(t, shape, strides):
        event('as_strided', (t.role, shape, strides))
        if mode == 'view_failure' and t.role == 'v':
            raise RuntimeError('view failure')
        return t
    def cat(ts, dim):
        event('cat', (tuple(t.role for t in ts), dim))
        return Tensor('cat_' + ts[0].role, sum(t.width for t in ts))
    class RT:
        def is_unsupported(self, source):
            event('unsupported')
            return False
        def mark_unsupported(self, source):
            event('mark')
        def get_library(self, source):
            event('library')
            return None
        def dispatch(self, lib, name, values, **kwargs):
            event('dispatch')
            buffers.append((tuple(t.role for t in values[:4]), values[4:], kwargs))
            if mode == 'dispatch_failure':
                raise RuntimeError('dispatch failure')
    names = ('q', 'q_rope', 'k', 'k_rope', 'v', 'out')
    widths = (128, 64, 128, 64, 128, 128)
    args = [Tensor(n, w) for n, w in zip(names, widths)]
    for w in widths:
        args.extend((64 * w, 32 * w, w))
    args.extend((2, 32))
    refs = {n: [6 + i * 3, 7 + i * 3, 8 + i * 3, 'c1'] for i, n in enumerate(names)}
    desc = ['mla', 'synthetic shader', '_mla_value_path', 256, 0, 1, 2, 3, 4, 5, 'c1', 24, 25, 32, [128, 64, 128], refs, ['f16'] * 6]
    if mode in ('scalar', 'raising_scalar'):
        args[7] = Scalar(args[7])
    if mode == 'int_subclass':
        args[7] = IntSubclass(args[7])
    if mode == 'ref_subclass':
        refs['q'][3] = RefSubclass('c1')
    if mode == 'refs_subclass':
        refs['q'] = Refs(refs['q'])
    if mode == 'args_subclass':
        args = Args(args)
    if mode == 'args_metaclass':
        args = MetaArgs(args)
    if mode == 'refs_metaclass':
        refs['q'] = MetaRefs(refs['q'])
    if mode == 'negative':
        args[7] = -1
    if mode == 'oob':
        args[8] = 100000
    if mode == 'bool_ref':
        refs['q'][1] = True
    if mode == 'bool_scalar':
        args[7] = True
    if mode == 'nonfolded_inner':
        refs['q'][3] = 25
    if mode == 'bad_ref':
        refs['q'][1] = 99
    if mode == 'short_refs':
        refs['q'] = refs['q'][:3]
    if mode == 'tuple_args':
        args = tuple(args)
    if mode == 'tuple_refs':
        refs = {n: tuple(r) for n, r in refs.items()}
        desc[15] = refs
    patches = {}
    if mode.startswith('builtin_'):
        name = mode.removeprefix('builtin_')
        original = getattr(builtins, name)
        def replacement(*values):
            event('builtin_callback', name)
            return original(*values)
        patches[name] = scope.get(name, None)
        scope[name] = replacement
    torch = types.SimpleNamespace(float16='f16', float32='f32', as_strided=as_strided, cat=cat)
    old_torch = sys.modules.get('torch')
    sys.modules['torch'] = torch
    def callback(frame, event_name, arg):
        if event_name == 'call' and frame.f_code.co_name == '_res':
            resolutions.append(1)
    old_profile = sys.getprofile()
    try:
        if profile:
            sys.setprofile(callback)
        try:
            value = scope['_dispatch_mla'](RT(), desc, args, grid=(2, 2, 1) if mode == 'wrong_grid' else (1, 2, 1), launch_exit_hook=lambda m: event('exit'))
            outcome = ('return', value)
        except Exception as error:
            outcome = ('raise', type(error).__name__, str(error), str(error.__cause__))
    finally:
        sys.setprofile(old_profile)
        if old_torch is None:
            del sys.modules['torch']
        else:
            sys.modules['torch'] = old_torch
        for name, previous in patches.items():
            if previous is None:
                del scope[name]
            else:
                scope[name] = previous
    return outcome, events, buffers, len(resolutions)


@pytest.mark.parametrize('mode', [
    'normal', 'scalar', 'raising_scalar', 'int_subclass', 'ref_subclass', 'refs_subclass', 'args_subclass',
    'negative', 'oob', 'offset_oob', 'wrong_dtype', 'wrong_grid', 'view_failure', 'contiguous_failure',
    'dispatch_failure', 'mutate_later_stride', 'bool_ref', 'bool_scalar', 'nonfolded_inner', 'bad_ref',
    'short_refs', 'tuple_args', 'tuple_refs', 'builtin_int', 'builtin_hasattr', 'builtin_isinstance',
    'builtin_len', 'builtin_tuple', 'runtime_bool_binding', 'args_metaclass', 'refs_metaclass',
])
def test_parent_callback_and_buffer_equivalence(mode):
    assert exercise(load(ROOT), mode)[:3] == exercise(load(ROOT, reference=True), mode)[:3]


def test_builtin_stride_resolution_avoids_reentering_scalar_callbacks():
    current = exercise(load(ROOT), profile=True)
    parent = exercise(load(ROOT, reference=True), profile=True)
    assert current[:3] == parent[:3]
    assert current[0] == ('return', True)
    assert parent[3] == 27
    assert current[3] == 3
