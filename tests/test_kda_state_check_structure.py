"""CPU pins for the producer-owned KDA state diagnostic and replay ordering."""

import pytest

from triton_msl.codegen._msl_templates import make_kda_kernel


@pytest.mark.parametrize("fp16", [False, True])
def test_state_check_stays_with_owning_update_and_preserves_barriers(fp16):
    source = make_kda_kernel(fp16=fp16)
    update = "float Bl=B[(C-1u)*D+l]; S[l*D+j]=Bl*(S[l*D+j]+d);"
    check = "if(!isfinite(S[l*D+j])) atomic_store_explicit(&replay,1u,memory_order_relaxed);"
    assert source.count(update) == source.count(check) == 1
    assert update + "\n      " + check + " }" in source
    # Check the stored float32 value, including for half-I/O shaders, with no
    # arithmetic reassociation or reload of the complete state after a barrier.
    assert "threadgroup float S[64*64]" in source
    assert "if(!isfinite(S[idx]))" not in source
    expected_tail = """    // The diagnostic reads Out through different lanes than the MMA store.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for(uint idx=lid;idx<C*D;idx+=NT)
      if(!isfinite(W[idx]) || !isfinite(float(Out[hb+base*D+idx])))
        atomic_store_explicit(&replay,1u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(atomic_load_explicit(&replay,memory_order_relaxed)){"""
    assert source.index(check) < source.index(expected_tail)
    assert "for(uint idx=lid; idx<D*D; idx+=NT){ uint l=idx/D, j=idx%D; float d=0.0f;" in source
    # Each S entry has one writer; the local predicate inspects that writer's
    # final value before the same existing uniform replay decision.
    owned = [idx for lane in range(256) for idx in range(lane, 64 * 64, 256)]
    assert len(owned) == len(set(owned)) == 64 * 64
    assert set(owned) == set(range(64 * 64))
    assert all((idx // 64) * 64 + idx % 64 == idx for idx in owned)
