"""On-device replay checks for noncanonical dot-accumulator bias addresses."""
import pytest
import torch

from tests.test_matmul_accumulator_bias_address import _bias_dot
from tests.test_fa_tail_semantics import executed, _executed_source  # noqa: F401


@pytest.mark.parametrize('step,offset,masked', [(1,0,False), (2,0,False),
                                               (1,3,False), (1,0,True), (2,3,True)])
def test_accumulator_bias_address_computes_source(step, offset, masked, executed):
    # Dyadic values keep products/sums exactly representable: this isolates the
    # source address/mask without a loose matmul tolerance hiding a bias error.
    a_cpu = (torch.arange(1024, dtype=torch.float32).reshape(32,32) % 5 - 2) / 8
    b_cpu = (torch.arange(1024, dtype=torch.float32).reshape(32,32) % 7 - 3) / 8
    bias_cpu = (torch.arange(96, dtype=torch.float32) - 23) / 4
    indices = offset + torch.arange(32) * step
    selected = bias_cpu[indices].clone()
    if masked:
        selected[16:] = -3
    reference = torch.relu(a_cpu.double() @ b_cpu.double() + selected.double()[None,:]).float()
    a, b, bias = a_cpu.to('mps'), b_cpu.to('mps'), bias_cpu.to('mps')
    backing = torch.full((1032,), -8192.0, dtype=torch.float32, device='mps')
    output = backing[4:-4].reshape(32,32)
    handle = _bias_dot[(1,)](a, b, bias, output, M=32, N=32, K=32,
                            STEP=step, OFFSET=offset, MASKED=masked, num_warps=4)
    torch.mps.synchronize()
    source = _executed_source(executed, handle)
    assert source == handle.asm['msl'], 'executed source differs from compiled source'
    if (step, offset, masked) != (1,0,False):
        assert 'Bias[col]' not in source, 'unproved bias address routed to unit-stride template'
    assert torch.equal(output.cpu().view(torch.int32), reference.view(torch.int32)), 'bias address/mask result differs'
    for actual, original in ((a,a_cpu), (b,b_cpu), (bias,bias_cpu)):
        assert torch.equal(actual.cpu().view(torch.int32), original.view(torch.int32)), 'input modified'
    retained = backing.cpu()
    assert bool((retained[:4] == -8192).all()) and bool((retained[-4:] == -8192).all()), 'output canary modified'
